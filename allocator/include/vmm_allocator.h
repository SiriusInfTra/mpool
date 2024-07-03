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

  virtual void ReportOOM(cudaStream_t cuda_stream, OOMReason reason, bool force_abort = true) {
    for (auto *oom_observer : oom_observers_) {
      (*oom_observer)(page_pool.config.device_id, cuda_stream, reason);
    }
  }

  const CachingAllocatorStats &GetStats() const { return stats; }


  void PrintStreamStats(StreamContext &stream_context) {
    for (auto &&[is_small, label] :
         {std::make_pair(false, "large"), std::make_pair(true, "small")}) {
      LOG(INFO) << "~~~~~~~~~~ Stream " << stream_context.cuda_stream << " (" << label
                << ") used / free ~~~~~~~~~~";
      std::array<size_t, 2> cnt = {0, 0}, sum = {0, 0}, max = {0, 0},
                            min = {std::numeric_limits<size_t>::max(),
                                   std::numeric_limits<size_t>::max()},
                            avg; // used / free
      for (auto [iter, end] = stream_context.stream_block_list.Iterators();
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

  std::pair<bip_list_iterator<MemBlock>, bip_list_iterator<MemBlock>>
  GetAllBlocks() const;

  void ResetPeakStats();
};

} // namespace mpool