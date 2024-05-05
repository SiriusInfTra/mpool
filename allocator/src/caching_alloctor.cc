#include "shm.h"
#include <caching_allocator.h>

#include <glog/logging.h>
#include <ostream>

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

std::pair<index_t, index_t>
MappingRegion::MappingPageRange(ptrdiff_t addr_offset, size_t nbytes) const {
  index_t index_begin = addr_offset / mem_block_num;
  index_t index_end =
      (addr_offset + nbytes + mem_block_num - 1) / mem_block_num;
  return {index_begin, index_end};
}

void MappingRegion::EnsureBlockWithPage(MemBlock *block) {
  auto [index_begin, index_end] =
      MappingPageRange(block->addr_offset, block->nbytes);
  if (block->unalloc_pages == 0) {
    return;
  }
  std::vector<index_t> missing_pages;
  if (mapping_pages_.size() < index_end) {
    mapping_pages_.resize(index_end, nullptr);
  }
  for (index_t index = index_begin; index < index_end; ++index) {
    if (mapping_pages_[index] == nullptr) {
      missing_pages.push_back(index);
    }
  }
  CHECK_EQ(block->unalloc_pages, missing_pages.size());
  std::vector<index_t> phy_pages;
  {
    auto lock = page_pool_.Lock();
    phy_pages = page_pool_.AllocDisPages(belong, missing_pages.size(), lock);
  }
  for (size_t k = 0; k < missing_pages.size(); ++k) {
    CU_CALL(cuMemMap(reinterpret_cast<CUdeviceptr>(
                         base_ptr_ + missing_pages[k] * mem_block_nbytes),
                     mem_block_nbytes, 0,
                     page_pool_.PagesView()[phy_pages[k]].cu_handle, 0));
  }
  ptrdiff_t addr_offset;
  size_t nbytes;
  if (std::adjacent_find(missing_pages.begin(), missing_pages.end(),
                         [](index_t a, index_t b) { return b != a + 1; }) ==
      missing_pages.end()) {
    addr_offset = missing_pages.front() * mem_block_nbytes;
    nbytes = missing_pages.size() * mem_block_nbytes;
  } else {
    addr_offset = index_begin * mem_block_nbytes;
    nbytes = (index_end - index_begin) * mem_block_nbytes;
  }
  CUmemAccessDesc acc_desc = {
      .location = {.type = CU_MEM_LOCATION_TYPE_DEVICE, .id = 0},
      .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
  CU_CALL(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(base_ptr_ + addr_offset),
                         nbytes, &acc_desc, 1));
}

unsigned MappingRegion::GetUnallocPages(ptrdiff_t addr_offset, size_t nbytes) {
  auto [index_begin, index_end] = MappingPageRange(addr_offset, nbytes);
  unsigned unalloc_pages = 0;
  for (index_t index = index_begin; index < index_end; ++index) {
    if (mapping_pages_[index] == nullptr) {
      unalloc_pages++;
    }
  }
  return unalloc_pages;
}

StreamBlockList::StreamBlockList(cudaStream_t cuda_stream,
                                 SharedMemory &shared_memory,
                                 MappingRegion &mapping_region,
                                 bip_list<shm_handle<MemBlock>> &all_block_list,
                                 size_t small_block_nbytes)
    : current_stream_(cuda_stream), mapping_region_(mapping_region),
      shared_memory_(shared_memory), all_block_list_(all_block_list),
      stream_block_list_(shared_memory->get_segment_manager()),
      small_block_nbytes_(small_block_nbytes) {}

MemBlock *StreamBlockList::CreateEntryExpandVA(size_t nbytes) {
  ptrdiff_t addr_offset;
  if (all_block_list_.empty()) {
    addr_offset = 0;
  } else {
    auto *last_block = std::prev(all_block_list_.cend())->ptr(shared_memory_);
    addr_offset = last_block->addr_offset + last_block->nbytes;
  }
  auto *block = new (shared_memory_->allocate(sizeof(MemBlock)))
      MemBlock{.addr_offset = addr_offset,
               .nbytes = nbytes,
               .stream = current_stream_,
               .is_free = false,
               .is_small = nbytes < small_block_nbytes_};
  shm_handle handle{block, shared_memory_};
  block->iter_all_block_list =
      all_block_list_.insert(all_block_list_.cend(), handle);
  block->iter_stream_block_list =
      stream_block_list_.insert(stream_block_list_.cend(), handle);
  return block;
}

MemBlock *StreamBlockList::GetPrevEntry(MemBlock *entry) {
  auto iter = entry->iter_all_block_list;
  if (iter == all_block_list_.cbegin()) {
    return nullptr;
  }
  return std::prev(iter)->ptr(shared_memory_);
}

MemBlock *StreamBlockList::GetNextEntry(MemBlock *entry) {
  auto iter = std::next(entry->iter_all_block_list);
  if (iter == all_block_list_.cend()) {
    return nullptr;
  }
  return iter->ptr(shared_memory_);
}

MemBlock *StreamBlockList::SplitEntry(MemBlock *origin_entry, size_t remain) {
  CHECK_GT(origin_entry->nbytes, remain);
  /* [origin: remain] [insert_after_entry: nbytes - remain] */
  auto *insert_after_entry = new (shared_memory_->allocate(sizeof(MemBlock)))
      MemBlock{.addr_offset =
                   origin_entry->addr_offset + static_cast<ptrdiff_t>(remain),
               .nbytes = origin_entry->nbytes - remain,
               .stream = origin_entry->stream,
               .is_free = origin_entry->is_free,
               .is_small = origin_entry->is_small};
  shm_handle insert_after_entry_handle{insert_after_entry, shared_memory_};
  insert_after_entry->iter_all_block_list = all_block_list_.insert(
      std::next(origin_entry->iter_all_block_list), insert_after_entry_handle);
  insert_after_entry->iter_stream_block_list =
      stream_block_list_.insert(std::next(origin_entry->iter_stream_block_list),
                                insert_after_entry_handle);

  origin_entry->nbytes = remain;

  if (origin_entry->unalloc_pages > 0) {
    insert_after_entry->unalloc_pages = mapping_region_.GetUnallocPages(
        insert_after_entry->addr_offset, insert_after_entry->nbytes);
    origin_entry->unalloc_pages = mapping_region_.GetUnallocPages(
        origin_entry->addr_offset, origin_entry->nbytes);
  }

  return insert_after_entry;
}

MemBlock *StreamBlockList::MergeMemEntry(MemBlock *first_block,
                                         MemBlock *secound_block) {
  CHECK_EQ(first_block->addr_offset + first_block->nbytes,
           secound_block->addr_offset);
  CHECK_EQ(first_block->is_free, secound_block->is_free);
  CHECK_EQ(first_block->is_small, secound_block->is_small);

  first_block->nbytes += secound_block->nbytes;
  if (first_block->unalloc_pages > 0 || secound_block->unalloc_pages > 0) {
    first_block->unalloc_pages = mapping_region_.GetUnallocPages(
        first_block->addr_offset, first_block->nbytes);
  }

  all_block_list_.erase(secound_block->iter_all_block_list);
  stream_block_list_.erase(secound_block->iter_stream_block_list);
  memset(secound_block, 63, sizeof(MemBlock));
  shared_memory_->deallocate(secound_block);

  return first_block;
}
StreamFreeList::StreamFreeList(SharedMemory &shared_memory,
                               cudaStream_t cuda_stream,
                               MappingRegion &mapping_region,
                               StreamBlockList &stream_block_list)
    : shared_memory_(shared_memory), current_stream_(cuda_stream),
      mapping_region_(mapping_region), stream_block_list_(stream_block_list),
      free_block_list_{bip_multimap<size_t, shm_handle<MemBlock>>(
                           shared_memory->get_segment_manager()),
                       bip_multimap<size_t, shm_handle<MemBlock>>(
                           shared_memory->get_segment_manager())} {}

MemBlock *StreamFreeList::PopBlock(bool is_small, size_t nbytes,
                                   size_t find_optimal_retry) {
  auto &free_list = free_block_list_[is_small];
  auto iter = free_list.lower_bound(nbytes);
  if (iter == free_list.cend()) {
    return nullptr;
  }
  auto *block = iter->second.ptr(shared_memory_);
  if (block->unalloc_pages > 0 && find_optimal_retry > 0) {
    // try to minimize the number of unallocated pages
    auto *optimal_block = block;
    auto optimal_iter = iter;
    for (auto iter1 = iter; iter1 != free_list.cend() && find_optimal_retry > 0;
         ++iter1, --find_optimal_retry) {
      if (auto *block1 = iter1->second.ptr(shared_memory_);
          block1->unalloc_pages < optimal_block->unalloc_pages) {
        optimal_block = block1;
        optimal_iter = iter1;
      }
    }
    block = optimal_block;
    iter = optimal_iter;
  }
  LOG(INFO)  << "is small " << block->is_small << "free_list " << free_list.size();
  block->is_free = false;
  free_list.erase(iter);
  if (block->nbytes > nbytes) {
    auto *split_block =
        stream_block_list_.SplitEntry(block, block->nbytes - nbytes);
    split_block->iter_free_block_list = free_list.insert(std::make_pair(
        split_block->nbytes, shm_handle{split_block, shared_memory_}));
  }
  return block;
}

MemBlock *StreamFreeList::PopBlock(MemBlock *block) {
  CHECK(block->is_free);
  free_block_list_[block->is_small].erase(block->iter_free_block_list);
  block->is_free = false;
  return block;
}

MemBlock *StreamFreeList::PushBlock(MemBlock *block) {
  auto &free_list = free_block_list_[block->is_small];
  block->is_free = true;

  if (auto prev_block = stream_block_list_.GetPrevEntry(block);
      prev_block && prev_block->is_free &&
      prev_block->is_small == block->is_small &&
      prev_block->stream == current_stream_) {
    LOG(INFO)  << "is small " << block->is_small << "free_list " << free_list.size();
    free_list.erase(prev_block->iter_free_block_list);
    block = stream_block_list_.MergeMemEntry(prev_block, block);
  }
  if (auto next_block = stream_block_list_.GetNextEntry(block);
      next_block && next_block->is_free &&
      next_block->is_small == block->is_small &&
      next_block->stream == current_stream_) {
    LOG(INFO)  << "is small " << block->is_small << "free_list " << free_list.size();
    free_list.erase(next_block->iter_free_block_list);
    block = stream_block_list_.MergeMemEntry(block, next_block);
  }

   block->iter_free_block_list =
      free_list.insert(std::make_pair(block->nbytes, shm_handle{block, shared_memory_}));
    LOG(INFO)  << "is small " << block->is_small << "free_list " << free_list.size();
  return block;
}

MemBlock *StreamFreeList::MaybeMergeAdj(MemBlock *entry) {
  if (entry->unalloc_pages > 0) {
    return entry;
  }
  CHECK(entry->is_free) << entry;
  CHECK(entry->is_small) << entry;
  auto *prev_block = stream_block_list_.GetPrevEntry(entry);
  DCHECK(prev_block == nullptr || !prev_block->is_free ||
         prev_block->unalloc_pages > 0)
      << prev_block;
  auto *next_block = stream_block_list_.GetNextEntry(entry);
  DCHECK(next_block == nullptr || !next_block->is_free ||
         next_block->unalloc_pages > 0)
      << next_block;
  bool put_free_list_large = true;
  size_t total_nbytes = entry->nbytes;
  for (auto *adj_block : {prev_block, next_block}) {
    if (adj_block == nullptr) {
      continue;
    }
    put_free_list_large &= adj_block->is_small;
    total_nbytes += adj_block->nbytes;
  }
  if (put_free_list_large && total_nbytes >= 2_MB) {
    entry = PopBlock(entry);
    entry->is_small = false;
    entry = PushBlock(entry);
  }
  return entry;
}

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
    block = stream_context.stream_block_list.CreateEntryExpandVA(nbytes);
    LOG(INFO) << block;
    stream_context.stream_free_list.PushBlock(block);
    block = AllocWithContext(nbytes, stream_context);
    // LOG(INFO) << block;
  }
  if (block != nullptr) {
    mapping_region_.EnsureBlockWithPage(block);
  }
  // LOG(INFO) << block;
  return block;
}

void CachingAllocator::Free(MemBlock *block) {
  CHECK(block->is_free);
  auto &context = GetStreamContext(block->stream);
  if (block->is_small) {
    block = context.stream_free_list.PushBlock(block);
    block = context.stream_free_list.MaybeMergeAdj(block);
  } else {
    block = context.stream_free_list.PushBlock(block);
    if (auto *prev_entry = context.stream_block_list.GetPrevEntry(block);
        prev_entry && prev_entry->is_small && prev_entry->is_free &&
        prev_entry->unalloc_pages == 0) {
      size_t prev_entry_nbytes = prev_entry->nbytes;
      auto *maybe_merged_entry =
          context.stream_free_list.MaybeMergeAdj(prev_entry);
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
      if (maybe_merged_entry->nbytes > next_entry_nbytes) {
        block = maybe_merged_entry;
      }
    }
  }
}
void CachingAllocator::EmptyCache(cudaStream_t cuda_stream) {
  LOG_IF(INFO, VERBOSE) << config.log_prefix << "Release free physical memory.";
  auto &context = GetStreamContext(cuda_stream);
  context.stream_block_list.EmptyCache();
};

void StreamBlockList::DumpStreamBlockList(std::ostream &out) {
  DumpMemBlockColumns(out);
  for (auto handle : stream_block_list_) {
    DumpMemBlock(out, handle.ptr(shared_memory_));
  }
}
void StreamBlockList::DumpMemBlockColumns(std::ostream &out) {
  out << "start,len,next,prev,unalloc_pages,is_free,is_small"
      << "\n";
}
void StreamBlockList::DumpMemBlock(std::ostream &out, MemBlock *block) {
  auto *prev = GetPrevEntry(block);
  auto *next = GetNextEntry(block);
  out << block->addr_offset << "," << block->nbytes << ","
      << (next ? next->addr_offset : -1) << ","
      << (prev ? prev->addr_offset : -1) << "," << block->unalloc_pages << ","
      << block->is_free << "," << block->is_small << "\n";
}
} // namespace mpool