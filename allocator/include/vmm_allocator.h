#pragma once

#include <cstddef>
#include <string>

#include <allocator.h>
#include <belong.h>
#include <mapping_region.h>
#include <mem_block.h>
#include <pages.h>
#include <pages_pool.h>
#include <shm.h>
#include <stats.h>
#include <stream_context.h>
#include <util.h>

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

class VMMAllocator : public IAllocator {
public:
  const Belong belong;
  const CachingAllocatorConfig config;
  PagesPool &page_pool;

  static bool RemoveShm(const CachingAllocatorConfig &config) {
    return bip::shared_memory_object::remove(config.shm_name.c_str());
  }

  const constexpr static size_t ALLOC_TRY_EXPAND_VA = 1;
  const constexpr static size_t REALLOC_FALLBACK_MEMCPY = 1;

protected:
  SharedMemory &shared_memory_;
  CachingAllocatorStats &stats;
  bip_list<shm_ptr<MemBlock>> &all_block_list_;


  StreamContext &global_stream_context_;

  std::vector<OOMObserver *> oom_observers_;

  bool CheckStats();

public:
  VMMAllocator(SharedMemory &shared_memory, PagesPool &page_pool,
               CachingAllocatorConfig config, bool first_init);

  ~VMMAllocator();
  size_t GetDeviceFreeNbytes() const;

  void AddOOMObserver(OOMObserver *observer);

  void RemoveOOMObserver(OOMObserver *observer);

  const CachingAllocatorStats &GetStats() const { return stats; }

  std::pair<bip_list_iterator<MemBlock>, bip_list_iterator<MemBlock>>
  GetAllBlocks() const;

  void ResetPeakStats();
};

} // namespace mpool