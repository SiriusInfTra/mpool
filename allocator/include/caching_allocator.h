#pragma once

#include <cstddef>
#include <string>

#include <belong.h>
#include <mapping_region.h>
#include <mem_block.h>
#include <pages.h>
#include <pages_pool.h>
#include <shm.h>
#include <stream_context.h>
#include <util.h>
#include <allocator.h>
#include <stats.h>

#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

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

class CachingAllocator : public IAllocator {
public:
  const Belong belong;
  const CachingAllocatorConfig config;
  PagesPool &page_pool;

  static bool RemoveShm(const CachingAllocatorConfig &config) {
    return bip::shared_memory_object::remove(config.shm_name.c_str());
  }

  const constexpr static size_t ALLOC_TRY_EXPAND_VA = 1;
  const constexpr static size_t REALLOC_FALLBACK_MEMCPY = 1;

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

  bool CheckStats();

  void Free0(MemBlock *block, bip::scoped_lock<bip::interprocess_mutex> &lock);

  MemBlock *Alloc0(size_t nbytes, cudaStream_t cuda_stream, bool try_expand_VA,
                   bip::scoped_lock<bip::interprocess_mutex> &lock);

public:
  CachingAllocator(SharedMemory &shared_memory, PagesPool &page_pool,
                   CachingAllocatorConfig config, bool first_init);

  ~CachingAllocator();

  std::byte *GetBasePtr() const override {
    return mapping_region_.GetBasePtr();
  }


  MemBlock *Alloc(size_t nbytes, size_t alignment, cudaStream_t cuda_stream,
                  size_t flags = 0) override;

  MemBlock *Realloc(MemBlock *block, size_t nbytes, cudaStream_t cuda_stream,
                    size_t flags = 0) override;

  MemBlock *ReceiveMemBlock(shm_ptr<MemBlock> handle);

  shm_ptr<MemBlock> SendMemBlock(MemBlock *mem_block);

  void Free(const MemBlock *block, size_t flags = 0) override;

  void EmptyCache();

  size_t GetDeviceFreeNbytes() const;

  void AddOOMObserver(OOMObserver *observer);

  void RemoveOOMObserver(OOMObserver *observer);

  void ReportOOM(int device_id, cudaStream_t cuda_stream, OOMReason reason);

  void DumpState();

  const CachingAllocatorStats &GetStats() const { return stats; }

  std::pair<bip_list_iterator<MemBlock>, bip_list_iterator<MemBlock>>
  GetAllBlocks() const;

  void ResetPeakStats();
};

} // namespace mpool