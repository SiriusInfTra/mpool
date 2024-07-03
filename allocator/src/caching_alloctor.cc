#include "stats.h"
#include <caching_allocator.h>
#include <mapping_region.h>
#include <pages.h>
#include <shm.h>
#include <util.h>

#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mem_block.h>
#include <utility>

#include <boost/unordered_map.hpp>

#include <glog/logging.h>

namespace mpool {

StreamContext &
CachingAllocator::GetStreamContext(cudaStream_t cuda_stream,
                                   const bip::scoped_lock<bip_mutex> &lock) {
  auto iter = stream_context_map_.find(cuda_stream);
  if (iter == stream_context_map_.end()) {
    LOG(INFO) << "Init Stream context";
    auto *context = new (shared_memory_->allocate(sizeof(StreamContext)))
        StreamContext{process_local_, page_pool.config.device_id, cuda_stream,
                      config.small_block_nbytes, stats};
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
      stats.SetBlockIsSmall(free_block, true);
      free_block =
          stream_context.stream_free_list.PushBlock(process_local_, free_block);
      free_block = stream_context.stream_free_list.PopBlock(process_local_,
                                                            true, nbytes, 0);
    }
  }
  CHECK(CHECK_LEVEL < 2 || CheckStateInternal(lock));
  return free_block;
}


MemBlock *
CachingAllocator::Alloc0(size_t nbytes, cudaStream_t cuda_stream,
                         bool try_expand_VA,
                         bip::scoped_lock<bip::interprocess_mutex> &lock) {
  LOG_IF(INFO, VERBOSE_LEVEL >= 1)
      << "Alloc " << ByteDisplay(nbytes) << ", stream = " << cuda_stream
      << ", try_expand_VA = " << try_expand_VA << ".";
  CHECK_GT(nbytes, 0);
  bool tried_global_stream = false, tried_expand_va = false;
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
    mapping_region_.AllocMappingsAndUpdateFlags(block, all_block_list_);
    block->ref_count += 1;
  } else {
    LOG(WARNING) << config.log_prefix
                 << "OOM: Cannot find free memory block, tried_global_stream = "
                 << tried_global_stream
                 << ", tried_expand_va = " << tried_expand_va << ".";
    ReportOOM(page_pool.config.device_id, cuda_stream,
              OOMReason::NO_MEMORY_BLOCK);
  }
  CHECK(CHECK_LEVEL < 1 || CheckStateInternal(lock));
  return block;
}

MemBlock *CachingAllocator::Alloc(size_t nbytes, size_t alignment,
                                  cudaStream_t cuda_stream, size_t flags) {
  bip::scoped_lock lock{shared_memory_.GetMutex()};
  return Alloc0(nbytes, cuda_stream,
                flags & VMMAllocator::ALLOC_TRY_EXPAND_VA, lock);
}

MemBlock *CachingAllocator::Realloc(MemBlock *block, size_t nbytes,
                                    cudaStream_t cuda_stream, size_t flags) {
  bool fallback_memcpy = flags & REALLOC_FALLBACK_MEMCPY;
  bip::scoped_lock lock{shared_memory_.GetMutex()};
  CHECK_EQ(block->stream, cuda_stream);
  auto &context = GetStreamContext(cuda_stream, lock);
  auto *resized_block =
      context.stream_free_list.ResizeBlock(process_local_, block, nbytes);
  if (!fallback_memcpy || resized_block != nullptr) {
    return resized_block;
  }
  resized_block = Alloc0(nbytes, cuda_stream, true, lock);
  auto *dst = mapping_region_.GetBasePtr() + resized_block->addr_offset;
  auto *src = mapping_region_.GetBasePtr() + block->addr_offset;
  size_t count = block->nbytes;
  Free0(block, lock);
  CUDA_CALL(
      cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, cuda_stream));
  return resized_block;
}

void CachingAllocator::Free(const MemBlock *block0, size_t flags) {
  bip::scoped_lock lock{shared_memory_.GetMutex()};
  return Free0(const_cast<MemBlock *>(block0), lock);
}

MemBlock *CachingAllocator::ReceiveMemBlock(shm_ptr<MemBlock> handle) {
  bip::scoped_lock lock{shared_memory_.GetMutex()};
  auto *mem_block = handle.ptr(shared_memory_);
  LOG_IF(INFO, VERBOSE_LEVEL >= 0)
      << "Receive MemBlock: " << handle << " -> " << *mem_block << ".";
  mapping_region_.AllocMappingsAndUpdateFlags(mem_block, all_block_list_);
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

void CachingAllocator::EmptyCache() {
  LOG_IF(INFO, VERBOSE_LEVEL)
      << config.log_prefix << "Release free physical memory.";
  // auto &context = GetStreamContext(cuda_stream);
  // context.stream_block_list.EmptyCache();
  bip::scoped_lock lock{shared_memory_.GetMutex()};
  mapping_region_.EmptyCacheAndUpdateFlags(all_block_list_);
  for (auto &[_, handle] : stream_context_map_) {
    auto stream_context = handle.ptr(shared_memory_);
    stream_context->MoveFreeBlockTo(process_local_, global_stream_context_);
  }
  CHECK(CHECK_LEVEL < 1 || CheckStateInternal(lock));
};
CachingAllocator::CachingAllocator(SharedMemory &shared_memory,
                                   PagesPool &page_pool,
                                   CachingAllocatorConfig config,
                                   bool first_init)
    : VMMAllocator(shared_memory, page_pool, config, first_init),
      stream_context_map_(
          *shared_memory_->find_or_construct<
              bip_unordered_map<cudaStream_t, shm_ptr<StreamContext>>>(
              "CA_stream_context_map")(shared_memory_->get_segment_manager())) {
}

void CachingAllocator::DumpState() {
  auto now = std::chrono::system_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) %
            1000;
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << "mempool_dump_"
     << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S") << "_"
     << ms.count();
  auto output_dir = std::filesystem::temp_directory_path() / ss.str();
  CHECK(std::filesystem::create_directory(output_dir));
  for (auto &[stream, handle] : stream_context_map_) {
    auto stream_context = handle.ptr(shared_memory_);
    {
      std::ofstream o_handle{
          output_dir / (std::string{"block_list_"} +
                        std::to_string(reinterpret_cast<size_t>(stream)))};
      CHECK(o_handle.is_open());
      stream_context->stream_block_list.DumpStreamBlockList(process_local_,
                                                            o_handle);
    }
    {
      std::ofstream o_handle{
          output_dir / (std::string{"free_list_small_"} +
                        std::to_string(reinterpret_cast<size_t>(stream)))};
      CHECK(o_handle.is_open());
      stream_context->stream_free_list.DumpFreeBlockList(process_local_, true,
                                                         o_handle);
    }
    {
      std::ofstream o_handle{
          output_dir / (std::string{"free_list_large_"} +
                        std::to_string(reinterpret_cast<size_t>(stream)))};
      CHECK(o_handle.is_open());
      stream_context->stream_free_list.DumpFreeBlockList(process_local_, false,
                                                         o_handle);
    }
  }
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
  ret &= CheckStats();
  return ret;
}


void CachingAllocator::ReportOOM(int device_id, cudaStream_t cuda_stream,
                                 OOMReason reason) {
  for (auto *oom_observer : oom_observers_) {
    (*oom_observer)(device_id, cuda_stream, reason);
  }
  for (auto &&[stream, handle] : stream_context_map_) {
    for (auto &&[is_small, label] :
         {std::make_pair(false, "large"), std::make_pair(true, "small")}) {
      LOG(INFO) << "~~~~~~~~~~ Stream " << stream << " (" << label
                << ") used / free ~~~~~~~~~~";
      auto stream_context = handle.ptr(shared_memory_);
      std::array<size_t, 2> cnt = {0, 0}, sum = {0, 0}, max = {0, 0},
                            min = {std::numeric_limits<size_t>::max(),
                                   std::numeric_limits<size_t>::max()},
                            avg; // used / free
      for (auto [iter, end] = stream_context->stream_block_list.Iterators();
           iter != end; ++iter) {
        auto *block = iter->ptr(shared_memory_);
        if (block->is_small != is_small) {
          continue;
        }
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
      LOG(INFO) << "sum: " << ByteDisplay(sum[0]) << " / "
                << ByteDisplay(sum[1]);
      LOG(INFO) << "avg: " << ByteDisplay(avg[0]) << " / "
                << ByteDisplay(avg[1]);
      LOG(INFO) << "max: " << ByteDisplay(max[0]) << " / "
                << ByteDisplay(max[1]);
      LOG(INFO) << "min: " << ByteDisplay(min[0]) << " / "
                << ByteDisplay(min[1]);
    }
  }
  LOG(FATAL) << config.log_prefix << "Abort.";
}


} // namespace mpool