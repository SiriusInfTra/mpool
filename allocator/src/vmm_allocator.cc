#include <mpool/stats.h>
#include <mpool/caching_allocator.h>
#include <mpool/mapping_region.h>
#include <mpool/pages.h>
#include <mpool/shm.h>
#include <mpool/util.h>
#include <mpool/mem_block.h>

#include <cstddef>
#include <utility>

#include <boost/unordered_map.hpp>

#include <glog/logging.h>

namespace mpool {
VMMAllocator::VMMAllocator(SharedMemory &shared_memory, PagesPool &page_pool,
                           VMMAllocatorConfig config, bool first_init)
    : belong(
          page_pool.GetBelongRegistry().GetOrCreateBelong(config.belong_name)),
      config(std::move(config)), page_pool(page_pool),
      shared_memory_(shared_memory),
      stats(*shared_memory_->find_or_construct<CachingAllocatorStats>(
          "CA_stats")()),

      all_block_list_(
          *shared_memory_->find_or_construct<bip_list<shm_ptr<MemBlock>>>(
              "CA_all_block_list")(shared_memory_->get_segment_manager())),
      all_block_map_(
          *shared_memory_
               ->find_or_construct<bip_map<ptrdiff_t, shm_ptr<MemBlock>>>(
                   "CA_all_block_map")(shared_memory_->get_segment_manager())),
      global_stream_context_(*shared_memory_->find_or_construct<StreamContext>(
          "CA_global_stream_context")(shared_memory_,
                                      page_pool.config.device_id, nullptr,
                                      this->config.small_block_nbytes, stats)) {
  LOG_IF(INFO, VERBOSE_LEVEL >= 1)
      << this->config.log_prefix << "Init VMMAllocator";
};

VMMAllocator::~VMMAllocator() {
  LOG_IF(INFO, VERBOSE_LEVEL >= 1)
      << config.log_prefix << "Release VMMAllocator";
}

bool VMMAllocator::CheckStats() {
  CachingAllocatorStats stats;
  for (auto ptr : all_block_list_) {
    auto *block = ptr.ptr(shared_memory_);
    stats.mem_block_nbytes[block->is_small].allocated_free[block->is_free] +=
        block->nbytes;
    stats.mem_block_nbytes[block->is_small].current += block->nbytes;
    stats.mem_block_count[block->is_small].allocated_free[block->is_free] += 1;
    stats.mem_block_count[block->is_small].current += 1;
  }
  for (bool is_small : {false, true}) {
    stats.mem_block_count[is_small].peak =
        this->stats.mem_block_count[is_small].peak;
    stats.mem_block_nbytes[is_small].peak =
        this->stats.mem_block_nbytes[is_small].peak;
  }
  CHECK_EQ(stats.mem_block_count[false], this->stats.mem_block_count[false]);
  CHECK_EQ(stats.mem_block_nbytes[false], this->stats.mem_block_nbytes[false]);
  CHECK_EQ(stats.mem_block_count[true], this->stats.mem_block_count[true]);
  CHECK_EQ(stats.mem_block_nbytes[true], this->stats.mem_block_nbytes[true]);
  return true;
}
size_t VMMAllocator::GetDeviceFreeNbytes() const {
  size_t free_nbytes =
      page_pool.GetBelongRegistry().GetFreeBelong().GetPagesNum() *
      page_pool.config.page_nbytes;
  free_nbytes += belong.GetPagesNum() * page_pool.config.page_nbytes -
                 belong.GetAllocatedNbytes();
  return free_nbytes;
}
void VMMAllocator::AddOOMObserver(std::shared_ptr<OOMObserver> observer) {
  oom_observers_.push_back(observer);
}
void VMMAllocator::RemoveOOMObserver(std::shared_ptr<OOMObserver> observer) {
  oom_observers_.erase(
      std::remove(oom_observers_.begin(), oom_observers_.end(), observer));
}
std::pair<bip_list_iterator<MemBlock>, bip_list_iterator<MemBlock>>
VMMAllocator::GetAllBlocks() const {
  return {{all_block_list_.begin(), shared_memory_},
          {all_block_list_.end(), shared_memory_}};
}
void VMMAllocator::ResetPeakStats() { stats.ResetPeakStats(); }

void VMMAllocator::PrintStreamStats(StreamContext &stream_context) {
  for (auto &&[is_small, label] :
       {std::make_pair(false, "large"), std::make_pair(true, "small")}) {
    LOG(INFO) << "~~~~~~~~~~ Stream " << stream_context.cuda_stream << " ("
              << label << ") used / free ~~~~~~~~~~";
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
    LOG(INFO) << "sum: " << ByteDisplay(sum[0]) << " / " << ByteDisplay(sum[1]);
    LOG(INFO) << "avg: " << ByteDisplay(avg[0]) << " / " << ByteDisplay(avg[1]);
    LOG(INFO) << "max: " << ByteDisplay(max[0]) << " / " << ByteDisplay(max[1]);
    LOG(INFO) << "min: " << ByteDisplay(min[0]) << " / " << ByteDisplay(min[1]);
  }
}
} // namespace mpool