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
VMMAllocator::VMMAllocator(SharedMemory &shared_memory,
                                   PagesPool &page_pool,
                                   CachingAllocatorConfig config,
                                   bool first_init)
    : belong(
          page_pool.GetBelongRegistry().GetOrCreateBelong(config.belong_name)),
      config(std::move(config)), page_pool(page_pool),
      shared_memory_(shared_memory),
      stats(*shared_memory_->find_or_construct<CachingAllocatorStats>(
          "CA_stats")()),
      mapping_region_(
          shared_memory_, page_pool, belong, this->config.log_prefix,
          this->config.va_range_scale,
          [&](int device_id, cudaStream_t cuda_stream, OOMReason reason) {
            // ReportOOM(device_id, cuda_stream, reason);
          }),
      all_block_list_(
          *shared_memory_->find_or_construct<bip_list<shm_ptr<MemBlock>>>(
              "CA_all_block_list")(shared_memory_->get_segment_manager())),
      process_local_{page_pool, shared_memory, mapping_region_,
                     all_block_list_},
      global_stream_context_(*shared_memory_->find_or_construct<StreamContext>(
          "CA_global_stream_context")(process_local_,
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

void CachingAllocator::Free0(MemBlock *block,
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
}