#pragma once

#include "mem_block.h"
#include "pages_pool.h"
#include "util.h"
#include "vmm_allocator.h"
#include <algorithm>
#include <allocator.h>
#include <cstddef>
namespace mpool {

using DirectAllocatorConfig = CachingAllocatorConfig;

class DirectAllocator : public VMMAllocator {
public:
  DirectAllocator(SharedMemory &shared_memory, PagesPool &page_pool,
                  DirectAllocatorConfig config, bool first_init): VMMAllocator(shared_memory, page_pool, config, first_init) {
                    
                  }
  virtual ~DirectAllocator() = default;
  virtual MemBlock *Alloc(size_t request_nbytes, size_t alignment,
                          cudaStream_t cuda_stream, size_t flags) override {
    CHECK_GT(request_nbytes, 0);
    LOG_IF(INFO, VERBOSE_LEVEL >= 1)
        << "Alloc " << ByteDisplay(request_nbytes)
        << ", stream = " << cuda_stream << ", flags = " << flags << ".";
    if (request_nbytes >= page_pool.config.page_nbytes) {
      alignment = std::max(alignment, page_pool.config.page_nbytes);
    }
    request_nbytes = (request_nbytes + alignment - 1) / alignment * alignment;
    if (request_nbytes < page_pool.config.page_nbytes) {
      auto *mem_block = global_stream_context_.stream_free_list.PopBlock(
          process_local_, true, request_nbytes, 0);
      if (mem_block != nullptr) {
        return mem_block;
      } else {
        std::vector<index_t> pages_index;
        {
          auto lock = page_pool.Lock();
          pages_index = page_pool.AllocDisPages(belong, 1, lock);
        }
        if (pages_index.empty()) {
          LOG(WARNING) << config.log_prefix
                       << "OOM: Cannot find any physical page.";
          // ReportOOM(page_pool.config.device_id, cuda_stream,
          //           OOMReason::NO_PHYSICAL_PAGES);
        }
        ptrdiff_t addr_offset = pages_index[0] * page_pool.config.page_nbytes;
        size_t nbytes = page_pool.config.page_nbytes;
        auto *mem_block = global_stream_context_.stream_free_list.PopBlock(
            process_local_, false, addr_offset, nbytes);
        stats.SetBlockIsSmall(mem_block, true);
        if (request_nbytes < nbytes) {
          global_stream_context_.stream_free_list.PushBlock(process_local_,
                                                            mem_block);
          mem_block = global_stream_context_.stream_free_list.PopBlock(
              process_local_, true, request_nbytes, 0);
        }
        return mem_block;
      }
    } else {
      size_t pages_cnt = request_nbytes / page_pool.config.page_nbytes;
      index_t page_begin;
      {
        auto lock = page_pool.Lock();
        page_begin = page_pool.AllocConPages(belong, pages_cnt, lock);
      }
      if (page_begin == INVALID_INDEX) {
        LOG(WARNING)
            << config.log_prefix
            << "OOM: Cannot find enough continuous physical pages: require = "
            << pages_cnt << ".";
        // ReportOOM(page_pool.config.device_id, cuda_stream,
        //           OOMReason::NO_PHYSICAL_PAGES);
        return nullptr;
      }
      ptrdiff_t addr_offset = page_begin * page_pool.config.page_nbytes;
      size_t nbytes = pages_cnt * page_pool.config.page_nbytes;
      return global_stream_context_.stream_free_list.PopBlock(
          process_local_, false, addr_offset, nbytes);
    }
  }
  virtual MemBlock *Realloc(MemBlock *block, size_t nbytes,
                            cudaStream_t cuda_stream, size_t flags) override {
    return nullptr;
  }
  virtual void Free(const MemBlock *block, size_t flags) override {
    if (!block->is_small) {
      index_t page_begin =
          mapping_region_.ByteOffsetToIndex(block->addr_offset);
      num_t pages_cnt = block->nbytes / page_pool.config.page_nbytes;
      global_stream_context_.stream_free_list.PushBlock(
          process_local_, const_cast<MemBlock *>(block));
      std::vector<index_t> pages;
      for (size_t i = page_begin; i < page_begin + pages_cnt; ++i) {
        pages.push_back(i);
      }
      {
        auto lock = page_pool.Lock();
        page_pool.FreePages(pages, belong, lock);
      }
    } else {
      index_t range_l = mapping_region_.ByteOffsetToIndex(block->addr_offset);
      index_t range_r = mapping_region_.ByteOffsetToIndex(
                            block->addr_offset + block->nbytes - 1 +
                            page_pool.config.page_nbytes - 1) +
                        1;
      block = global_stream_context_.stream_free_list.PushBlock(
          process_local_, const_cast<MemBlock *>(block));
      range_l = std::max(range_l,
                         mapping_region_.ByteOffsetToIndex(block->addr_offset));
      range_r = std::min(range_r, mapping_region_.ByteOffsetToIndex(
                                      block->addr_offset + block->nbytes - 1) +
                                      1);
      std::vector<index_t> pages;
      for (size_t i = range_l; i < range_r; ++i) {
        if (belong == *page_pool.PagesView()[i].belong) {
          pages.push_back(i);
        }
      }
      {
        auto lock = page_pool.Lock();
        page_pool.FreePages(pages, belong, lock);
      }
    }
  }
};
} // namespace mpool