#include <caching_allocator.h>
#include <shm.h>
#include <util.h>

#include <algorithm>
#include <unordered_set>

#include <boost/unordered_map.hpp>

#include <glog/logging.h>


namespace mpool {

MappingRegion::MappingRegion(SharedMemory &shared_memory, PagesPool &page_pool,
                             Belong belong, std::string log_prefix,
                             size_t va_range_scale)
    : log_prefix(log_prefix), mem_block_nbytes(page_pool.config.PAGE_NBYTES),
      mem_block_num(page_pool.config.POOL_NBYTES /
                    page_pool.config.PAGE_NBYTES),
      va_range_scale(va_range_scale), belong(belong),
      shared_memory_(shared_memory), page_pool_(page_pool) {
  CU_CALL(cuMemAddressReserve(reinterpret_cast<CUdeviceptr *>(&base_ptr_),
                              mem_block_nbytes * mem_block_num * va_range_scale,
                              mem_block_nbytes, 0, 0));
  LOG(INFO) << log_prefix << "dev_ptr = " << base_ptr_ << ".";
}

CachingAllocator::CachingAllocator(SharedMemory &shared_memory,
                                   PagesPool &page_pool,
                                   CachingAllocatorConfig config,
                                   __attribute__((unused)) bool first_init)
    : config(std::move(config)), page_pool_(page_pool),
      shared_memory_(shared_memory),
      mapping_region_(shared_memory_, page_pool, this->config.belong,
                      this->config.log_prefix, config.va_range_scale),
      all_block_list_(
          *shared_memory_->find_or_construct<bip_list<shm_handle<MemBlock>>>(
              "CA_all_block_list")(shared_memory_->get_segment_manager())),
      global_stream_context_(*shared_memory_->find_or_construct<StreamContext>(
          "CA_global_stream_context")(
          reinterpret_cast<cudaStream_t>(0), shared_memory_, mapping_region_,
          all_block_list_, this->config.small_block_nbytes)),
      stream_context_map_(
          *shared_memory_->find_or_construct<
              bip_unordered_map<cudaStream_t, shm_handle<StreamContext>>>(
              "CA_stream_context_map")(
              shared_memory_->get_segment_manager())){};

CachingAllocator::~CachingAllocator() {}

MemBlock *CachingAllocator::Alloc(size_t nbytes, cudaStream_t cuda_stream,
                                  bool try_expand_VA) {
  nbytes = (nbytes + config.align_nbytes - 1) & ~(config.align_nbytes - 1);
  auto &stream_context = GetStreamContext(cuda_stream);
  auto *block = AllocWithContext(nbytes, stream_context);
  if (block == nullptr) {
    block = AllocWithContext(nbytes, global_stream_context_);
  }
  if (block == nullptr && try_expand_VA) {
    CHECK(!MORE_MORE_CHECK_STATE || CheckState());
    block = stream_context.stream_block_list.CreateEntryExpandVA(nbytes);
    LOG(INFO) << block;
    CHECK(!MORE_MORE_CHECK_STATE || CheckState());
    stream_context.stream_free_list.PushBlock(block);
    CHECK(!MORE_MORE_CHECK_STATE || CheckState());
    block = AllocWithContext(nbytes, stream_context);
    // LOG(INFO) << block;
  }
  CHECK(!MORE_CHECK_STATE || CheckState());
  if (block != nullptr) {
    mapping_region_.EnsureBlockWithPage(block, all_block_list_);
  }
  CHECK(!CHECK_STATE || CheckState());
  return block;
}

void CachingAllocator::Free(MemBlock *block) {
  auto &context = GetStreamContext(block->stream);
  CHECK(!block->is_free) << block;
  CHECK(!MORE_MORE_CHECK_STATE || CheckState());
  if (block->is_small) {
    block = context.stream_free_list.PushBlock(block);
    CHECK(!MORE_MORE_CHECK_STATE || CheckState());
    block = context.stream_free_list.MaybeMergeAdj(block);
    CHECK(!MORE_MORE_CHECK_STATE || CheckState());
  } else {
    block = context.stream_free_list.PushBlock(block);
    if (auto *prev_entry = context.stream_block_list.GetPrevEntry(block);
        prev_entry && prev_entry->is_small && prev_entry->is_free &&
        prev_entry->unalloc_pages == 0) {
      size_t prev_entry_nbytes = prev_entry->nbytes;
      auto *maybe_merged_entry =
          context.stream_free_list.MaybeMergeAdj(prev_entry);
      CHECK(!MORE_MORE_CHECK_STATE || CheckState());
      if (maybe_merged_entry->nbytes > prev_entry_nbytes) {
        block = maybe_merged_entry;
      }
    }
    if (auto *next_entry = context.stream_block_list.GetNextEntry(block);
        next_entry && next_entry->is_small && next_entry->is_free &&
        next_entry->unalloc_pages == 0) {
      size_t next_entry_nbytes = next_entry->nbytes;
      auto *maybe_merged_entry =
          context.stream_free_list.MaybeMergeAdj(next_entry);
      CHECK(!MORE_MORE_CHECK_STATE || CheckState());
      if (maybe_merged_entry->nbytes > next_entry_nbytes) {
        block = maybe_merged_entry;
      }
    }
  }
  CHECK(!CHECK_STATE || CheckState());
}
void CachingAllocator::EmptyCache(__attribute__((unused))
                                  cudaStream_t cuda_stream) {
  LOG_IF(INFO, VERBOSE) << config.log_prefix << "Release free physical memory.";
  // auto &context = GetStreamContext(cuda_stream);
  // context.stream_block_list.EmptyCache();
  mapping_region_.EmptyCache(all_block_list_);
  for (auto &[_, handle] : stream_context_map_) {
    auto stream_context = handle.ptr(shared_memory_);
    stream_context->MoveFreeBlockTo(global_stream_context_);
  }
  CHECK(!CHECK_STATE || CheckState());
};


StreamContext &CachingAllocator::GetStreamContext(cudaStream_t cuda_stream) {
  auto iter = stream_context_map_.find(cuda_stream);
  if (iter == stream_context_map_.end()) {
    LOG(INFO) << "Init Stream context";
    auto *context =
        new (shared_memory_->allocate(sizeof(StreamContext))) StreamContext{
            cuda_stream,
            shared_memory_,
            mapping_region_,
            all_block_list_,
            config.small_block_nbytes,
        };
    auto [insert_iter, insert_succ] = stream_context_map_.insert(
        std::make_pair(cuda_stream, shm_handle{context, shared_memory_}));
    CHECK(insert_succ);
    iter = insert_iter;
  }
  return *iter->second.ptr(shared_memory_);
}

MemBlock *CachingAllocator::AllocWithContext(size_t nbytes,
                                             StreamContext &stream_context) {
  bool is_small = nbytes <= config.small_block_nbytes;
  CHECK(!MORE_CHECK_STATE || CheckState());
  auto *free_block =
      stream_context.stream_free_list.PopBlock(is_small, nbytes, 50);
  CHECK(!MORE_CHECK_STATE || CheckState());
  // LOG(INFO) << free_block << " is small " << is_small;
  if (free_block == nullptr && is_small) {
    free_block = stream_context.stream_free_list.PopBlock(
        false, config.small_block_nbytes, 50);
    if (free_block != nullptr) {
      free_block->is_small = true;
      free_block = stream_context.stream_free_list.PushBlock(free_block);
      free_block = stream_context.stream_free_list.PopBlock(true, nbytes, 0);
    }
  }
  CHECK(!MORE_CHECK_STATE || CheckState());
  return free_block;
}
bool CachingAllocator::CheckState() {
  bool ret = true;
  ret &= global_stream_context_.stream_block_list.CheckState(true);
  ret &= global_stream_context_.stream_free_list.CheckState();
  for (auto &&[cuda_stream, context] : stream_context_map_) {
    ret &= context.ptr(shared_memory_)->stream_block_list.CheckState();
    ret &= context.ptr(shared_memory_)->stream_free_list.CheckState();
  }
  return ret;
}
} // namespace mpool