#include "stats.h"
#include <caching_allocator.h>
#include <mapping_region.h>
#include <pages.h>
#include <shm.h>
#include <util.h>

#include <array>
#include <cstddef>
#include <mem_block.h>
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
void VMMAllocator::AddOOMObserver(OOMObserver *observer) {
  oom_observers_.push_back(observer);
}
void VMMAllocator::RemoveOOMObserver(OOMObserver *observer) {
  oom_observers_.erase(
      std::remove(oom_observers_.begin(), oom_observers_.end(), observer));
}
std::pair<bip_list_iterator<MemBlock>, bip_list_iterator<MemBlock>>
VMMAllocator::GetAllBlocks() const {
  return {{all_block_list_.begin(), shared_memory_},
          {all_block_list_.end(), shared_memory_}};
}
void VMMAllocator::ResetPeakStats() { stats.ResetPeakStats(); }

} // namespace mpool