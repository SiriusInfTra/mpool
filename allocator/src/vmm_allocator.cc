#include "mpool/vmm_allocator.h"
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <filesystem>
#include <mpool/stats.h>
#include <mpool/caching_allocator.h>
#include <mpool/mapping_region.h>
#include <mpool/pages.h>
#include <mpool/shm.h>
#include <mpool/util.h>
#include <mpool/mem_block.h>

#include <cstddef>
#include <utility>
#include <fstream>

#include <boost/unordered_map.hpp>

#include <mpool/logging_is_spdlog.h>

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
                                      this->config.small_block_nbytes, stats)),
      process_local_{page_pool, shared_memory, nullptr,
                     all_block_list_, all_block_map_} {
  LOG_IF(INFO, VERBOSE_LEVEL >= 1)
      << this->config.log_prefix << "Init VMMAllocator";
  CUDA_CALL(cudaStreamCreate(&zero_filing_stream_));
};

VMMAllocator::~VMMAllocator() {
  // LOG_IF(INFO, VERBOSE_LEVEL >= 1)
  //     << config.log_prefix << "Release VMMAllocator";
}

struct Recorder {
  const char *name;
  std::atomic<long> cnt = 0;
  std::atomic<long> us = 0;

  void inc() {
    cnt += 1;
    if (cnt % 1000 == 0) {
      LOG(INFO) << name << ": " << cnt << " times, " << 1.0 * us / cnt
                << " us / time";;
    }
  }

  ~Recorder() {
    std::cout << name << ": " << cnt << " times, " << 1.0 * us / cnt << " us / time";
  }
};
static Recorder set_zero_func{"SetZeroFunc"};
static Recorder fill_zero{"FillZero"};
static Recorder skip_fill{"SkipFill"};

void VMMAllocator::SetZero(MemBlock *block,
                              cudaStream_t stream) {
  if (!block || block->nbytes == 0) { return; }
  return;
  if (stream != nullptr) { return; }
  auto *mapping_region = process_local_.mapping_region_;
  auto t0 = std::chrono::steady_clock::now();
  for (index_t i = mapping_region->ByteOffsetToIndex(block->addr_offset);
       i < mapping_region->ByteOffsetToIndex(block->addr_offset + block->nbytes + mapping_region->mem_block_nbytes - 1);
       ++i) {
    auto page = mapping_region->GetMutableSelfPageTable()[i];
    
    if (*page->last_belong != *page->belong && *page->last_belong != page_pool.GetBelongRegistry().GetFreeBelong().GetHandle()) {
      *page->last_belong = *page->belong;
      auto begin = std::chrono::steady_clock::now();
      if (stream != nullptr) {
        CUDA_CALL(cudaMemsetAsync(GetBasePtr() + i * mapping_region->mem_block_nbytes, 0, mapping_region->mem_block_nbytes, zero_filing_stream_));
        CUDA_CALL(cudaStreamSynchronize(zero_filing_stream_));
      }

      auto dur = std::chrono::steady_clock::now() - begin;
      fill_zero.us += std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
      fill_zero.inc();
      // LOG(INFO) << config.log_prefix
      //   << "SetZero: " << ByteDisplay(mapping_region->mem_block_nbytes)
      //   << ", stream = " << stream << ", dur_us = " << std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
    } else {
      skip_fill.inc();
      // LOG(INFO) << config.log_prefix
      //     << "SkipSet: " << ByteDisplay(mapping_region->mem_block_nbytes)
      //     << ", stream = " << stream;
    }
  }
  auto dur2 = std::chrono::steady_clock::now() - t0;
  set_zero_func.us += std::chrono::duration_cast<std::chrono::microseconds>(dur2).count();
  set_zero_func.inc();
  // LOG(INFO) << config.log_prefix
  //           << "VMMAllocator: " << ByteDisplay(block->nbytes)
  //           << ", stream = " << stream << ", dur_us = "
  //           << std::chrono::duration_cast<std::chrono::microseconds>(dur).count();

  
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

void VMMAllocator::ResetPeakStats() { 
  stats.ResetPeakStats(); 
}

void VMMAllocator::PrintStreamStats(StreamContext &stream_context) {
  for (auto &&[is_small, label] :
       {std::make_pair(false, "large"), std::make_pair(true, "small")}) {
    LOG(INFO) << "~~~~~~~~~~ Stream " << stream_context.cuda_stream 
              << " (" << label << ") used / free ~~~~~~~~~~";
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
void VMMAllocator::DumpState(std::vector<StreamContext*> stream_contexts) {
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
  LOG(WARNING) << "Dump state to " << output_dir << ".";
  CHECK(std::filesystem::create_directory(output_dir));
  for (auto *stream_context : stream_contexts) {
    {
      std::ofstream o_handle{
          output_dir / (std::string{"block_list_"} +
                        std::to_string(reinterpret_cast<size_t>(stream_context->cuda_stream)))};
      CHECK(o_handle.is_open());
      stream_context->stream_block_list.DumpStreamBlockList(process_local_,
                                                            o_handle);
    }
    {
      std::ofstream o_handle{
          output_dir / (std::string{"free_list_small_"} +
                        std::to_string(reinterpret_cast<size_t>(stream_context->cuda_stream)))};
      CHECK(o_handle.is_open());
      stream_context->stream_free_list.DumpFreeBlockList(process_local_, true,
                                                         o_handle);
    }
    {
      std::ofstream o_handle{
          output_dir / (std::string{"free_list_large_"} +
                        std::to_string(reinterpret_cast<size_t>(stream_context->cuda_stream)))};
      CHECK(o_handle.is_open());
      stream_context->stream_free_list.DumpFreeBlockList(process_local_, false,
                                                         o_handle);
    }
  }
}
void VMMAllocator::AllocMappingsAndUpdateFlags(MemBlock *block, 
  bip::scoped_lock<bip::interprocess_mutex> &lock) {
  if (block->addr_offset + block->nbytes > page_pool.config.page_nbytes *
                                               page_pool.page_num *
                                               config.va_range_scale) {
    LOG(WARNING) << "OOM: Cannot reserve VA for block: " << block
                 << ", addr_offset = " << block->addr_offset
                 << ", nbytes = " << block->nbytes << ".";
    ReportOOM(block->stream, OOMReason::NO_VIRTUAL_SPACE);
    return;
  }
  std::vector<index_t> missing_va_mapping_i = process_local_.mapping_region_->AllocMappings(block);

  /* 4. update memory block unalloc_pages flag */
  if (!missing_va_mapping_i.empty()) {
    block->unalloc_pages = 0;
    MemBlock *next_block = block;
    while ((next_block = global_stream_context_
        .stream_block_list.GetNextEntry(
          process_local_, next_block)) != nullptr) {
      if (process_local_.mapping_region_->ByteOffsetToIndex(
        next_block->addr_offset) > missing_va_mapping_i.back()) {
        break;
      }
      DCHECK_GE(next_block->unalloc_pages, 1);
      next_block->unalloc_pages--;
      DCHECK_EQ(
          next_block->unalloc_pages,
          process_local_.mapping_region_->CalculateUnallocFlags(
            next_block->addr_offset, next_block->nbytes))
          << next_block;
      auto &context = GetStreamContext(next_block->stream, lock);
      next_block = context.stream_free_list.PopBlock(process_local_, next_block);
      next_block = context.stream_free_list.PushBlock(process_local_, next_block);
    }
    MemBlock *prev_block = block;
    while ((prev_block = global_stream_context_
      .stream_block_list.GetPrevEntry(
        process_local_, prev_block)) != nullptr) {
      if (process_local_.mapping_region_->ByteOffsetToIndex(
          prev_block->addr_offset + prev_block->nbytes - 1) <
          missing_va_mapping_i.front()) {
        break;
      }
      DCHECK_GE(prev_block->unalloc_pages, 1);
      prev_block->unalloc_pages--;
      DCHECK_EQ(
          prev_block->unalloc_pages,
          process_local_.mapping_region_->CalculateUnallocFlags(
            prev_block->addr_offset, prev_block->nbytes))
          << prev_block;
      auto &context = GetStreamContext(prev_block->stream, lock);
      prev_block = context.stream_free_list.PopBlock(process_local_, prev_block);
      prev_block = context.stream_free_list.PushBlock(process_local_, prev_block);
    }
  }
}
} // namespace mpool