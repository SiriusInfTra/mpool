#pragma once

#include <cstddef>
#include <string>

#include <mpool/allocator.h>
#include <mpool/belong.h>
#include <mpool/mapping_region.h>
#include <mpool/mem_block.h>
#include <mpool/pages.h>
#include <mpool/pages_pool.h>
#include <mpool/shm.h>
#include <mpool/stats.h>
#include <mpool/stream_context.h>
#include <mpool/util.h>

#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include <cuda_runtime_api.h>

namespace mpool {

struct VMMAllocatorConfig {
  std::string log_prefix;
  std::string shm_name;
  size_t shm_nbytes;
  size_t va_range_scale;
  std::string belong_name;
  size_t small_block_nbytes;
  size_t align_nbytes;
};

class VMMAllocator : public IAllocator {
public:
  const Belong belong;
  const VMMAllocatorConfig config;
  PagesPool &page_pool;

  static bool RemoveShm(const VMMAllocatorConfig &config) {
    return bip::shared_memory_object::remove(config.shm_name.c_str());
  }

  const constexpr static size_t ALLOC_TRY_EXPAND_VA = 1;
  const constexpr static size_t REALLOC_FALLBACK_MEMCPY = 2;
  const constexpr static size_t SKIP_ZERO_FILLING = 4;

protected:
  SharedMemory &shared_memory_;
  CachingAllocatorStats &stats;
  bip_list<shm_ptr<MemBlock>> &all_block_list_;
  bip_map<ptrdiff_t, shm_ptr<MemBlock>> &all_block_map_;
  cudaStream_t zero_filing_stream_;


  StreamContext &global_stream_context_;
  ProcessLocalData process_local_;

  std::vector<std::shared_ptr<OOMObserver>> oom_observers_;

  bool CheckStats();

  void SetZero(MemBlock *block, cudaStream_t stream);

public:
  VMMAllocator(SharedMemory &shared_memory, PagesPool &page_pool,
               VMMAllocatorConfig config, bool first_init);

  ~VMMAllocator();
  size_t GetDeviceFreeNbytes() const;

  virtual StreamContext &GetStreamContext(cudaStream_t cuda_stream,
                                const bip::scoped_lock<bip_mutex> &lock) = 0;

  void AddOOMObserver(std::shared_ptr<OOMObserver> observer);

  void RemoveOOMObserver(std::shared_ptr<OOMObserver> observer);

  virtual void ReportOOM(cudaStream_t cuda_stream, OOMReason reason, 
                         bool force_abort = true) {
    for (auto ptr : oom_observers_) {
      auto &oom_observer = *ptr.get();
      oom_observer(page_pool.config.device_id, cuda_stream, reason);
    }
  }

  const CachingAllocatorStats &GetStats() const { return stats; }

  void PrintStreamStats(StreamContext &stream_context);

  std::pair<bip_list_iterator<MemBlock>, bip_list_iterator<MemBlock>>
  GetAllBlocks() const;
  virtual MemBlock *RetainMemBlock(std::byte *ptr) override {
    bip::scoped_lock lock{shared_memory_.GetMutex()};
    auto iter = all_block_map_.find(ptr - GetBasePtr());
    if (iter == all_block_map_.end()) {
      return nullptr;
    }
    return iter->second.ptr(shared_memory_);
  }

  void ResetPeakStats();

  void DumpState(std::vector<StreamContext*> stream_contexts);

  void AllocMappingsAndUpdateFlags(MemBlock *block, 
    bip::scoped_lock<bip::interprocess_mutex> &lock);
};

} // namespace mpool