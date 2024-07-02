#pragma once
#include "stats.h"
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <cstddef>
#include <iterator>
#include <string>

#include <belong.h>
#include <mapping_region.h>
#include <mem_block.h>
#include <pages.h>
#include <pages_pool.h>
#include <shm.h>
#include <stream_context.h>
#include <util.h>

#include <cuda_runtime_api.h>

namespace mpool {

struct CachingAllocatorConfig {
  std::string log_prefix;
  std::string shm_name;
  size_t shm_nbytes;
  size_t va_range_scale;
  std::string belong_name;
  size_t small_block_nbytes;
  size_t align_nbytes;
};

class CachingAllocator {
public:
  const Belong belong;
  const CachingAllocatorConfig config;
  PagesPool &page_pool;

  static bool RemoveShm(const CachingAllocatorConfig &config) {
    return bip::shared_memory_object::remove(config.shm_name.c_str());
  }

private:
  SharedMemory &shared_memory_;
  CachingAllocatorStats &stats;
  MappingRegion mapping_region_;
  bip_list<shm_ptr<MemBlock>> &all_block_list_;
  ProcessLocalData process_local_;

  StreamContext &global_stream_context_;

  bip_unordered_map<cudaStream_t, shm_ptr<StreamContext>> &stream_context_map_;

  std::vector<OOMObserver *> oom_observers_;

  StreamContext &GetStreamContext(cudaStream_t cuda_stream,
                                  const bip::scoped_lock<bip_mutex> &lock);

  MemBlock *AllocWithContext(size_t nbytes, StreamContext &stream_context,
                             const bip::scoped_lock<bip_mutex> &lock);

  bool CheckStateInternal(const bip::scoped_lock<bip_mutex> &lock);

  bool CheckStats() {
    CachingAllocatorStats stats;
    for (auto ptr : all_block_list_) {
      auto *block = ptr.ptr(shared_memory_);
      stats.mem_block_nbytes[block->is_small].allocated_free[block->is_free] +=
          block->nbytes;
      stats.mem_block_nbytes[block->is_small].current += block->nbytes;
      stats.mem_block_count[block->is_small].allocated_free[block->is_free] +=
          1;
      stats.mem_block_count[block->is_small].current += 1;
    }
    for (bool is_small : {false, true}) {
      stats.mem_block_count[is_small].peak =
          this->stats.mem_block_count[is_small].peak;
      stats.mem_block_nbytes[is_small].peak =
          this->stats.mem_block_nbytes[is_small].peak;
    }
    CHECK_EQ(stats.mem_block_count[false], this->stats.mem_block_count[false]);
    CHECK_EQ(stats.mem_block_nbytes[false],
             this->stats.mem_block_nbytes[false]);
    CHECK_EQ(stats.mem_block_count[true], this->stats.mem_block_count[true]);
    CHECK_EQ(stats.mem_block_nbytes[true], this->stats.mem_block_nbytes[true]);
    return true;
  }

  void
  Free0(MemBlock *block,
                          bip::scoped_lock<bip::interprocess_mutex> &lock) {
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

  MemBlock *Alloc0(size_t nbytes, cudaStream_t cuda_stream, bool try_expand_VA,
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
      block = stream_context.stream_block_list.CreateEntryExpandVA(
          process_local_, nbytes);
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
      LOG(WARNING)
          << config.log_prefix
          << "OOM: Cannot find free memory block, tried_global_stream = "
          << tried_global_stream << ", tried_expand_va = " << tried_expand_va
          << ".";
      ReportOOM(page_pool.config.device_id, cuda_stream,
                OOMReason::NO_MEMORY_BLOCK);
    }
    CHECK(CHECK_LEVEL < 1 || CheckStateInternal(lock));
    return block;
  }

public:
  CachingAllocator(SharedMemory &shared_memory, PagesPool &page_pool,
                   CachingAllocatorConfig config, bool first_init);

  ~CachingAllocator();

  std::byte *GetBasePtr() const { return mapping_region_.GetBasePtr(); }

  std::byte *GetEndPtr() const { return mapping_region_.GetEndPtr(); }

  bool IsAllocatedPtr(std::byte *ptr) const {
    return ptr >= mapping_region_.GetBasePtr() &&
           ptr < mapping_region_.GetEndPtr();
  }

  MemBlock *Alloc(size_t nbytes, cudaStream_t cuda_stream,
                  bool try_expand_VA = true);

  MemBlock *Realloc(MemBlock *block, size_t nbytes, cudaStream_t cuda_stream) {
    bip::scoped_lock lock{shared_memory_.GetMutex()};
    CHECK_EQ(block->stream, cuda_stream);
    auto &context = GetStreamContext(cuda_stream, lock);
    auto *resized_block =
        context.stream_free_list.ResizeBlock(process_local_, block, nbytes);
    if (resized_block != nullptr) {
      return resized_block;
    }
    resized_block = Alloc0(nbytes, cuda_stream, true, lock);
    auto *dst = mapping_region_.GetBasePtr() + resized_block->addr_offset;
    auto *src = mapping_region_.GetBasePtr() + block->addr_offset;
    size_t count = block->nbytes;
    Free0(block, lock);
    CUDA_CALL(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice,
                              cuda_stream));
    return resized_block;
  }

  MemBlock *ReceiveMemBlock(shm_ptr<MemBlock> handle);

  shm_ptr<MemBlock> SendMemBlock(MemBlock *mem_block);

  void Free(const MemBlock *block);

  void EmptyCache();

  size_t GetDeviceFreeNbytes() const {
    size_t free_nbytes =
        page_pool.GetBelongRegistry().GetFreeBelong().GetPagesNum() *
        page_pool.config.page_nbytes;
    free_nbytes += belong.GetPagesNum() * page_pool.config.page_nbytes -
                   belong.GetAllocatedNbytes();
    return free_nbytes;
  }

  void AddOOMObserver(OOMObserver *observer) {
    oom_observers_.push_back(observer);
  }

  void RemoveOOMObserver(OOMObserver *observer) {
    oom_observers_.erase(
        std::remove(oom_observers_.begin(), oom_observers_.end(), observer));
  }

  void ReportOOM(int device_id, cudaStream_t cuda_stream, OOMReason reason);

  void DumpState();

  const CachingAllocatorStats &GetStats() const { return stats; }

  std::pair<bip_list_iterator<MemBlock>, bip_list_iterator<MemBlock>>
  GetAllBlocks() const {
    return {{all_block_list_.begin(), shared_memory_},
            {all_block_list_.end(), shared_memory_}};
  }

  void ResetPeakStats() { stats.ResetPeakStats(); }
};

} // namespace mpool