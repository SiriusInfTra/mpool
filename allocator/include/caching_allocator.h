#pragma once
#include "stats.h"
#include <cstddef>
#include <string>

#include <stream_context.h>
#include <mapping_region.h>
#include <belong.h>
#include <pages.h>
#include <pages_pool.h>
#include <shm.h>
#include <util.h>
#include <mem_block.h>

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

  bip_unordered_map<cudaStream_t, shm_ptr<StreamContext>>
      &stream_context_map_;

  std::vector<OOMObserver*> oom_observers_;

  StreamContext &GetStreamContext(cudaStream_t cuda_stream, const bip::scoped_lock<bip_mutex> &lock);

  MemBlock *AllocWithContext(size_t nbytes, StreamContext &stream_context, const bip::scoped_lock<bip_mutex> &lock);

  bool CheckStateInternal(const bip::scoped_lock<bip_mutex> &lock);

  bool CheckStats() {
    CachingAllocatorStats stats;
    for (auto ptr : all_block_list_) {
      auto *block = ptr.ptr(shared_memory_);
      if (block->is_free) {
        stats.free[block->is_small].AddBlock(block->nbytes);
      } else {
        stats.allocated[block->is_small].AddBlock(block->nbytes);
      }
    }
    CHECK_EQ(stats.allocated[false], this->stats.allocated[false]);
    CHECK_EQ(stats.allocated[true], this->stats.allocated[true]);
    CHECK_EQ(stats.free[false], this->stats.free[false]);
    CHECK_EQ(stats.free[true], this->stats.free[true]);
    return true;
  }
public:
  CachingAllocator(SharedMemory &shared_memory, PagesPool &page_pool,
                   CachingAllocatorConfig config, bool first_init);
  
  ~CachingAllocator();

  std::byte *GetBasePtr() const { return mapping_region_.GetBasePtr(); }

  std::byte *GetEndPtr() const { return mapping_region_.GetEndPtr(); }

  bool IsAllocatedPtr(std::byte *ptr) const {
    return ptr >= mapping_region_.GetBasePtr() && ptr < mapping_region_.GetEndPtr();
  }

  MemBlock *Alloc(size_t nbytes, cudaStream_t cuda_stream,
                  bool try_expand_VA = true);

  MemBlock *ReceiveMemBlock(shm_ptr<MemBlock> handle);

  shm_ptr<MemBlock> SendMemBlock(MemBlock *mem_block);

  void Free(const MemBlock *block);

  void EmptyCache();

  size_t GetDeviceFreeNbytes() const {
    size_t free_nbytes = page_pool.GetBelongRegistry().GetFreeBelong().GetPagesNum() * page_pool.config.page_nbytes;
    free_nbytes += belong.GetPagesNum() * page_pool.config.page_nbytes - belong.GetAllocatedNbytes();
    return free_nbytes;
  }

  void AddOOMObserver(OOMObserver *observer) {
    oom_observers_.push_back(observer);
  }

  void RemoveOOMObserver(OOMObserver *observer) {
    oom_observers_.erase(std::remove(oom_observers_.begin(), oom_observers_.end(), observer));
  }

  void ReportOOM(int device_id, cudaStream_t cuda_stream, OOMReason reason);

  void DumpState();

  const CachingAllocatorStats &GetStats() const {
    return stats;
  }

};


}