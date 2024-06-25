#include "mapping_region.h"
#include <array>
#include <caching_allocator.h>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <mem_block.h>
#include <pages.h>
#include <shm.h>
#include <util.h>
#include <fstream>

#include <boost/unordered_map.hpp>

#include <glog/logging.h>
#include <utility>

namespace mpool {

CachingAllocator::CachingAllocator(SharedMemory &shared_memory,
                                   PagesPool &page_pool,
                                   CachingAllocatorConfig config,
                                   bool first_init)
    : belong(
          page_pool.GetBelongRegistry().GetOrCreateBelong(config.belong_name)),
      config(std::move(config)), page_pool(page_pool),
      shared_memory_(shared_memory),
      mapping_region_(shared_memory_, page_pool, belong,
                      this->config.log_prefix, this->config.va_range_scale, [&](int device_id, cudaStream_t cuda_stream, OOMReason reason){ ReportOOM(device_id, cuda_stream, reason); }),
      all_block_list_(
          *shared_memory_->find_or_construct<bip_list<shm_ptr<MemBlock>>>(
              "CA_all_block_list")(shared_memory_->get_segment_manager())),
      process_local_{page_pool, shared_memory, mapping_region_,
                     all_block_list_},
      global_stream_context_(*shared_memory_->find_or_construct<StreamContext>(
          "CA_global_stream_context")(process_local_,
                                      page_pool.config.device_id, nullptr,
                                      this->config.small_block_nbytes)),
      stream_context_map_(
          *shared_memory_->find_or_construct<
              bip_unordered_map<cudaStream_t, shm_ptr<StreamContext>>>(
              "CA_stream_context_map")(shared_memory_->get_segment_manager())) {
  LOG_IF(INFO, VERBOSE_LEVEL >= 1)
      << this->config.log_prefix << "Init CachingAllocator";
};

CachingAllocator::~CachingAllocator() {
  LOG_IF(INFO, VERBOSE_LEVEL >= 1)
      << config.log_prefix << "Release CachingAllocator";
}

MemBlock *CachingAllocator::Alloc(size_t nbytes, cudaStream_t cuda_stream,
                                  bool try_expand_VA) {
  LOG_IF(INFO, VERBOSE_LEVEL >= 1)
      << "Alloc " << ByteDisplay(nbytes) << ", stream = " << cuda_stream
      << ", try_expand_VA = " << try_expand_VA << ".";
  CHECK_GT(nbytes, 0);
  bool tried_global_stream = false, tried_expand_va = false;
  bip::scoped_lock lock{shared_memory_.GetMutex()};
  CHECK(CHECK_LEVEL < 1 || CheckStateInternal(lock));
  nbytes = (nbytes + config.align_nbytes - 1) & ~(config.align_nbytes - 1);
  auto &stream_context = GetStreamContext(cuda_stream, lock);
  auto *block = AllocWithContext(nbytes, stream_context, lock);
  if (block == nullptr) {
    block = AllocWithContext(nbytes, global_stream_context_, lock);
    tried_global_stream = true;
  }
  if (block == nullptr && try_expand_VA) {
    CHECK(CHECK_LEVEL < 3 || CheckStateInternal(lock));
    block = stream_context.stream_block_list.CreateEntryExpandVA(process_local_,
                                                                 nbytes);
    LOG_IF(INFO, VERBOSE_LEVEL >= 3) << block;
    CHECK(CHECK_LEVEL < 3 || CheckStateInternal(lock));
    stream_context.stream_free_list.PushBlock(process_local_, block);
    CHECK(CHECK_LEVEL < 3 || CheckStateInternal(lock));
    block = AllocWithContext(nbytes, stream_context, lock);
    tried_expand_va = true;
  }
  CHECK(CHECK_LEVEL < 2 || CheckStateInternal(lock));
  if (block != nullptr) {
    mapping_region_.EnsureMemBlockWithMappings(block, all_block_list_);
    block->ref_count += 1;
  } else {
    LOG(WARNING) << config.log_prefix
                 << "OOM: Cannot find free memory block, tried_global_stream = "
                 << tried_global_stream
                 << ", tried_expand_va = " << tried_expand_va << ".";
    ReportOOM(page_pool.config.device_id, cuda_stream, OOMReason::NO_MEMORY_BLOCK);
  }
  CHECK(CHECK_LEVEL < 1 || CheckStateInternal(lock));
  return block;
}

void CachingAllocator::Free(const MemBlock *block0) {
  auto *block = const_cast<MemBlock *>(block0);
  LOG_IF(INFO, VERBOSE_LEVEL >= 1)
      << config.log_prefix << "Free block " << block;
  bip::scoped_lock lock{shared_memory_.GetMutex()};
  CHECK(CHECK_LEVEL < 1 || CheckStateInternal(lock));
  if (--block->ref_count > 0) {
    return;
  }
  auto &context = GetStreamContext(block->stream, lock);
  CHECK(!block->is_free) << block;
  CHECK(CHECK_LEVEL < 3 || CheckStateInternal(lock));
  if (block->is_small) {
    block = context.stream_free_list.PushBlock(process_local_, block);
    CHECK(CHECK_LEVEL < 3 || CheckStateInternal(lock));
    block = context.stream_free_list.MaybeMergeAdj(process_local_, block);
    CHECK(CHECK_LEVEL < 3 || CheckStateInternal(lock));
  } else {
    block = context.stream_free_list.PushBlock(process_local_, block);
    if (auto *prev_entry =
            context.stream_block_list.GetPrevEntry(process_local_, block);
        prev_entry && prev_entry->is_small && prev_entry->is_free &&
        prev_entry->unalloc_pages == 0) {
      size_t prev_entry_nbytes = prev_entry->nbytes;
      auto *maybe_merged_entry =
          context.stream_free_list.MaybeMergeAdj(process_local_, prev_entry);
      CHECK(CHECK_LEVEL < 3 || CheckStateInternal(lock));
      if (maybe_merged_entry->nbytes > prev_entry_nbytes) {
        block = maybe_merged_entry;
      }
    }
    if (auto *next_entry =
            context.stream_block_list.GetNextEntry(process_local_, block);
        next_entry && next_entry->is_small && next_entry->is_free &&
        next_entry->unalloc_pages == 0) {
      size_t next_entry_nbytes = next_entry->nbytes;
      auto *maybe_merged_entry =
          context.stream_free_list.MaybeMergeAdj(process_local_, next_entry);
      CHECK(CHECK_LEVEL < 3 || CheckStateInternal(lock));
      if (maybe_merged_entry->nbytes > next_entry_nbytes) {
        block = maybe_merged_entry;
      }
    }
  }
  CHECK(CHECK_LEVEL < 1 || CheckStateInternal(lock));
}
void CachingAllocator::EmptyCache() {
  LOG_IF(INFO, VERBOSE_LEVEL)
      << config.log_prefix << "Release free physical memory.";
  // auto &context = GetStreamContext(cuda_stream);
  // context.stream_block_list.EmptyCache();
  bip::scoped_lock lock{shared_memory_.GetMutex()};
  mapping_region_.EmptyCache(all_block_list_);
  for (auto &[_, handle] : stream_context_map_) {
    auto stream_context = handle.ptr(shared_memory_);
    stream_context->MoveFreeBlockTo(process_local_, global_stream_context_);
  }
  CHECK(CHECK_LEVEL < 1 || CheckStateInternal(lock));
};

StreamContext &
CachingAllocator::GetStreamContext(cudaStream_t cuda_stream,
                                   const bip::scoped_lock<bip_mutex> &lock) {
  auto iter = stream_context_map_.find(cuda_stream);
  if (iter == stream_context_map_.end()) {
    LOG(INFO) << "Init Stream context";
    auto *context =
        new (shared_memory_->allocate(sizeof(StreamContext))) StreamContext{
            process_local_,
            page_pool.config.device_id,
            cuda_stream,
            config.small_block_nbytes,
        };
    auto [insert_iter, insert_succ] = stream_context_map_.insert(
        std::make_pair(cuda_stream, shm_ptr{context, shared_memory_}));
    CHECK(insert_succ);
    iter = insert_iter;
  }
  return *iter->second.ptr(shared_memory_);
}

MemBlock *
CachingAllocator::AllocWithContext(size_t nbytes, StreamContext &stream_context,
                                   const bip::scoped_lock<bip_mutex> &lock) {
  bool is_small = nbytes <= config.small_block_nbytes;
  CHECK(CHECK_LEVEL < 2 || CheckStateInternal(lock));
  auto *free_block = stream_context.stream_free_list.PopBlock(
      process_local_, is_small, nbytes, 50);
  CHECK(CHECK_LEVEL < 2 || CheckStateInternal(lock));
  // LOG(INFO) << free_block << " is small " << is_small;
  if (free_block == nullptr && is_small) {
    free_block = stream_context.stream_free_list.PopBlock(
        process_local_, false, config.small_block_nbytes, 50);
    if (free_block != nullptr) {
      free_block->is_small = true;
      free_block =
          stream_context.stream_free_list.PushBlock(process_local_, free_block);
      free_block = stream_context.stream_free_list.PopBlock(process_local_,
                                                            true, nbytes, 0);
    }
  }
  CHECK(CHECK_LEVEL < 2 || CheckStateInternal(lock));
  return free_block;
}
bool CachingAllocator::CheckStateInternal(
    const bip::scoped_lock<bip_mutex> &lock) {
  bool ret = true;
  ret &=
      global_stream_context_.stream_block_list.CheckState(process_local_, true);
  ret &= global_stream_context_.stream_free_list.CheckState(process_local_);
  for (auto &&[cuda_stream, context] : stream_context_map_) {
    ret &= context.ptr(shared_memory_)
               ->stream_block_list.CheckState(process_local_);
    ret &= context.ptr(shared_memory_)
               ->stream_free_list.CheckState(process_local_);
  }
  return ret;
}

MemBlock *CachingAllocator::ReceiveMemBlock(shm_ptr<MemBlock> handle) {
  bip::scoped_lock lock{shared_memory_.GetMutex()};
  auto *mem_block = handle.ptr(shared_memory_);
  LOG_IF(INFO, VERBOSE_LEVEL >= 0)
      << "Receive MemBlock: " << handle << " -> " << *mem_block << ".";
  mapping_region_.EnsureMemBlockWithMappings(mem_block, all_block_list_);
  CHECK_GE(mem_block->addr_offset, 0) << "Invalid handle";
  return mem_block;
}
shm_ptr<MemBlock> CachingAllocator::SendMemBlock(MemBlock *mem_block) {
  shm_ptr<MemBlock> handle{mem_block, shared_memory_};
  LOG_IF(INFO, VERBOSE_LEVEL >= 0)
      << "Send MemBlock: " << mem_block << " -> " << handle << ".";
  bip::scoped_lock lock{shared_memory_.GetMutex()};
  mem_block->ref_count++;
  return handle;
}
void CachingAllocator::ReportOOM(int device_id, cudaStream_t cuda_stream, OOMReason reason) {
  for (auto *oom_observer : oom_observers_) {
    (*oom_observer)(device_id, cuda_stream, reason);
  }
  for (auto &&[stream, handle] : stream_context_map_) {
    for (auto &&[is_small, label] : {std::make_pair(false, "large"), std::make_pair(true, "small")}) {
      LOG(INFO) << "~~~~~~~~~~ Stream " << stream << " (" << label << ") used / free ~~~~~~~~~~";
      auto stream_context = handle.ptr(shared_memory_);
      std::array<size_t, 2> cnt = {0, 0}, sum = {0, 0}, max = {0, 0}, min = {std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()}, avg; // used / free
      for (auto [iter, end] = stream_context->stream_block_list.Iterators(); iter != end; ++iter) {
        auto *block = iter->ptr(shared_memory_);
        if (block->is_small != is_small) { continue; }
        cnt[block->is_free] += 1;
        sum[block->is_free] += block->nbytes;
        max[block->is_free] = std::max(max[block->is_free], block->nbytes);
        min[block->is_free] = std::min(min[block->is_free], block->nbytes);
      }
      for (bool is_free : {false, true}) {
        if (sum[is_free] == 0) {
          avg[is_free] = 0;
        } else {
          avg[is_free] = sum[is_free] / cnt[is_free];
        }
      }
      for (auto &min_value : min) {
        if (min_value == std::numeric_limits<size_t>::max()) {
          min_value = 0;
        }
      }
      LOG(INFO) << "cnt: " << cnt[0] << " / " << cnt[1];
      LOG(INFO) << "sum: " << ByteDisplay(sum[0]) << " / " << ByteDisplay(sum[1]);
      LOG(INFO) << "avg: " << ByteDisplay(avg[0]) << " / " << ByteDisplay(avg[1]);
      LOG(INFO) << "max: " << ByteDisplay(max[0]) << " / " << ByteDisplay(max[1]);
      LOG(INFO) << "min: " << ByteDisplay(min[0]) << " / " << ByteDisplay(min[1]);
    }
  }
  LOG(FATAL) << config.log_prefix << "Abort.";
}
void CachingAllocator::DumpState() {
  auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "mempool_dump_" <<  
        std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S") << "_" << ms.count();
  auto output_dir = std::filesystem::temp_directory_path() / ss.str();
  CHECK(std::filesystem::create_directory(output_dir));
  for (auto &[stream, handle] : stream_context_map_) {
    auto stream_context = handle.ptr(shared_memory_);
    {
      std::ofstream o_handle{output_dir / (std::string{"block_list_"} + std::to_string(reinterpret_cast<size_t>(stream)))};
      CHECK(o_handle.is_open());
      stream_context->stream_block_list.DumpStreamBlockList(process_local_, o_handle);
    }
    {
      std::ofstream o_handle{output_dir / (std::string{"free_list_small_"} + std::to_string(reinterpret_cast<size_t>(stream)))};
      CHECK(o_handle.is_open());
      stream_context->stream_free_list.DumpFreeBlockList(process_local_, true, o_handle);
    }
    {
      std::ofstream o_handle{output_dir / (std::string{"free_list_large_"} + std::to_string(reinterpret_cast<size_t>(stream)))};
      CHECK(o_handle.is_open());
      stream_context->stream_free_list.DumpFreeBlockList(process_local_, false, o_handle);
    }
  }
}
} // namespace mpool