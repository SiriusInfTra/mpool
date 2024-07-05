#include "vmm_allocator.h"
#include <caching_allocator.h>
#include <mapping_region.h>
#include <pages.h>
#include <shm.h>
#include <stats.h>
#include <util.h>

#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <mem_block.h>
#include <utility>

#include <boost/unordered_map.hpp>

#include <glog/logging.h>

namespace mpool {

CachingAllocator::CachingAllocator(SharedMemory &shared_memory,
                                   PagesPool &page_pool,
                                   VMMAllocatorConfig config,
                                   bool first_init)
    : VMMAllocator(shared_memory, page_pool, config, first_init),
      mapping_region_(
          shared_memory_, page_pool, belong, this->config.log_prefix,
          this->config.va_range_scale,
          [&](int device_id, cudaStream_t cuda_stream, OOMReason reason) {
            ReportOOM(cuda_stream, reason, true);
          }),
      process_local_{page_pool, shared_memory, mapping_region_, all_block_list_,
                     all_block_map_},
      stream_context_map_(
          *shared_memory_->find_or_construct<
              bip_unordered_map<cudaStream_t, shm_ptr<StreamContext>>>(
              "CA_stream_context_map")(shared_memory_->get_segment_manager())) {
}

StreamContext &
CachingAllocator::GetStreamContext(cudaStream_t cuda_stream,
                                   const bip::scoped_lock<bip_mutex> &lock) {
  auto iter = stream_context_map_.find(cuda_stream);
  if (iter == stream_context_map_.end()) {
    LOG(INFO) << "Init Stream context";
    auto *context = new (shared_memory_->allocate(sizeof(StreamContext)))
        StreamContext{process_local_.shared_memory_, page_pool.config.device_id,
                      cuda_stream, config.small_block_nbytes, stats};
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

MemBlock *CachingAllocator::AllocWithLock(
    size_t nbytes_with_alignment, cudaStream_t cuda_stream, bool try_expand_VA,
    bip::scoped_lock<bip::interprocess_mutex> &lock) {
  bool tried_global_stream = false, tried_expand_va = false;
  CHECK(CHECK_LEVEL < 1 || CheckStateInternal(lock));

  auto &stream_context = GetStreamContext(cuda_stream, lock);
  auto *block = AllocWithContext(nbytes_with_alignment, stream_context, lock);
  if (block == nullptr) {
    block =
        AllocWithContext(nbytes_with_alignment, global_stream_context_, lock);
    tried_global_stream = true;
  }
  if (block == nullptr && try_expand_VA) {
    CHECK(CHECK_LEVEL < 3 || CheckStateInternal(lock));
    block = stream_context.stream_block_list.CreateEntryExpandVA(
        process_local_, nbytes_with_alignment);
    LOG_IF(INFO, VERBOSE_LEVEL >= 3) << block;
    CHECK(CHECK_LEVEL < 3 || CheckStateInternal(lock));
    stream_context.stream_free_list.PushBlock(process_local_, block);
    CHECK(CHECK_LEVEL < 3 || CheckStateInternal(lock));
    block = AllocWithContext(nbytes_with_alignment, stream_context, lock);
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
    ReportOOM(cuda_stream, OOMReason::NO_MEMORY_BLOCK, true);
  }
  CHECK(CHECK_LEVEL < 1 || CheckStateInternal(lock));
  return block;
}

void CachingAllocator::FreeWithLock(
    MemBlock *block, bip::scoped_lock<bip::interprocess_mutex> &lock) {
  LOG_IF(INFO, VERBOSE_LEVEL >= 1)
      << config.log_prefix << "Free block " << block;
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

MemBlock *CachingAllocator::Alloc(size_t nbytes, size_t alignment,
                                  cudaStream_t cuda_stream, size_t flags) {
  LOG_IF(INFO, VERBOSE_LEVEL >= 1)
      << "Alloc " << ByteDisplay(nbytes) << ", stream = " << cuda_stream
      << ", flags = " << flags << ".";
  CHECK(IsPower2(alignment));
  CHECK_GT(nbytes, 0);
  size_t nbytes_with_alignment = AlignNbytes(nbytes, alignment);
  bip::scoped_lock lock{shared_memory_.GetMutex()};
  CHECK_GT(nbytes_with_alignment, 0);
  return AllocWithLock(nbytes_with_alignment, cuda_stream,
                       flags & VMMAllocator::ALLOC_TRY_EXPAND_VA, lock);
}

MemBlock *CachingAllocator::Realloc(MemBlock *block, size_t nbytes,
                                    size_t alignment, cudaStream_t cuda_stream,
                                    size_t flags) {
  LOG_IF(INFO, VERBOSE_LEVEL >= 1)
      << "Realloc from" << block->nbytes << " to " << ByteDisplay(nbytes)
      << ", alignment = " << alignment << ", stream = " << cuda_stream
      << ", flags = " << flags << ".";
  bool fallback_memcpy = flags & REALLOC_FALLBACK_MEMCPY;
  CHECK_GT(nbytes, 0);
  CHECK(IsPower2(alignment));
  size_t nbytes_with_alignment = AlignNbytes(nbytes, alignment);
  bip::scoped_lock lock{shared_memory_.GetMutex()};
  CHECK_EQ(block->stream, cuda_stream);
  auto &context = GetStreamContext(cuda_stream, lock);
  auto *resized_block =
      context.stream_free_list.ResizeBlock(process_local_, block, nbytes_with_alignment);
  if (!fallback_memcpy || resized_block != nullptr) {
    return resized_block;
  }
  resized_block = AllocWithLock(nbytes_with_alignment, cuda_stream, true, lock);
  auto *dst = mapping_region_.GetBasePtr() + resized_block->addr_offset;
  auto *src = mapping_region_.GetBasePtr() + block->addr_offset;
  size_t count = block->nbytes;
  FreeWithLock(block, lock);
  lock.unlock();
  CUDA_CALL(
      cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, cuda_stream));
  return resized_block;
}

void CachingAllocator::Free(const MemBlock *block, size_t flags) {
  bip::scoped_lock lock{shared_memory_.GetMutex()};
  return FreeWithLock(const_cast<MemBlock *>(block), lock);
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

void CachingAllocator::ReportOOM(cudaStream_t cuda_stream, OOMReason reason,
                                 bool force_abort) {
  VMMAllocator::ReportOOM(cuda_stream, reason);
  page_pool.PrintStats();
  for (auto &&[_, handle] : stream_context_map_) {
    PrintStreamStats(*handle.ptr(shared_memory_));
  }
  LOG_IF(FATAL, force_abort) << config.log_prefix << "Abort.";
}

} // namespace mpool