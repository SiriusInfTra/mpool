#include "mpool/vmm_allocator.h"
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <mpool/util.h>
#include <mpool/direct_allocator.h>

#include <mpool/logging_is_spdlog.h>

namespace mpool {
DirectAllocator::DirectAllocator(SharedMemory &shared_memory,
                                 PagesPool &page_pool,
                                 VMMAllocatorConfig config, bool first_init)
    : VMMAllocator(shared_memory, page_pool, config, first_init),
      mapping_region_(
        shared_memory_, page_pool, belong, this->config.log_prefix,
        this->config.va_range_scale,
        [&](int device_id, cudaStream_t cuda_stream, OOMReason reason) 
        { ReportOOM(cuda_stream, reason, true); }
) {
  process_local_.mapping_region_ = &mapping_region_;
  auto *mem_block =
      global_stream_context_.stream_block_list.CreateEntryExpandVA(
          process_local_, mapping_region_.GetVARangeNBytes());
  LOG(WARNING) << "DirectAllocator::DirectAllocator " << ByteDisplay(mapping_region_.GetVARangeNBytes());
  global_stream_context_.stream_free_list.PushBlock(process_local_, mem_block);

}

MemBlock *DirectAllocator::Alloc(size_t request_nbytes, size_t alignment,
                                 cudaStream_t cuda_stream, size_t flags) {
  CHECK_GT(request_nbytes, 0ULL);
  LOG_IF(INFO, VERBOSE_LEVEL >= 1)
      << "Alloc " << ByteDisplay(request_nbytes) << ", stream = " << cuda_stream
      << ", flags = " << flags << ".";
  // if (request_nbytes >= page_pool.config.page_nbytes) {
  //   alignment = std::max(alignment, page_pool.config.page_nbytes);
  // }
  // request_nbytes <  page_pool.config.page_nbytes: small pool
  // request_nbytes >= page_pool.config.page_nbytes: large pool
  request_nbytes = AlignToPages(request_nbytes);
  bip::scoped_lock lock{shared_memory_.GetMutex()};
  auto *mem_block = AllocWithLock(request_nbytes, alignment, cuda_stream,
                                  flags & VMMAllocator::ALLOC_TRY_EXPAND_VA, lock);
  EnsureInit();
  lock.unlock();
  if ((flags & VMMAllocator::SKIP_ZERO_FILLING) == 0) {
    SetZero(mem_block, cuda_stream);
  }
  return mem_block;
}

void DirectAllocator::Free(const MemBlock *block, size_t flags) {
 bip::scoped_lock lock{shared_memory_.GetMutex()};
  if (!block->is_small) {
    index_t page_begin = mapping_region_.ByteOffsetToIndex(block->addr_offset);
    num_t pages_cnt = block->nbytes / page_pool.config.page_nbytes;
    CHECK_EQ(block->addr_offset % page_pool.config.page_nbytes, 0);
    CHECK_EQ(block->nbytes % page_pool.config.page_nbytes, 0);
    if (ENABLE_STRICT_MAPPING) {
      CUmemAccessDesc acc_desc = {
        .location = {.type = CU_MEM_LOCATION_TYPE_DEVICE ,
                    .id = page_pool.config.device_id},
        .flags = CU_MEM_ACCESS_FLAGS_PROT_NONE,
      };
      CU_CALL(cuMemUnmap(reinterpret_cast<CUdeviceptr>(mapping_region_.GetBasePtr() + block->addr_offset),
                        block->nbytes));
      for (size_t i = 0; i < pages_cnt; ++i) {
        auto & page = page_pool.PagesView()[page_begin + i];
        CHECK_EQ(page.index, page_begin + i);
          CU_CALL(cuMemMap(
            reinterpret_cast<CUdeviceptr>(
              mapping_region_.GetBasePtr() + block->addr_offset + i * page_pool.config.page_nbytes),
                page_pool.config.page_nbytes, 0, page.cu_handle, 0));
      }
      CU_CALL(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(mapping_region_.GetBasePtr() + block->addr_offset),
                            block->nbytes, &acc_desc, 1));
    }
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
    block = global_stream_context_.stream_free_list.PushBlock(
        process_local_, const_cast<MemBlock *>(block));
    if (block->nbytes == mapping_region_.mem_block_nbytes) {
      block = global_stream_context_.stream_free_list.PopBlock(
          process_local_, const_cast<MemBlock *>(block));
      stats.SetBlockIsSmall(const_cast<MemBlock *>(block), false);
      {
        auto lock = page_pool.Lock();
        page_pool.FreePages({mapping_region_.ByteOffsetToIndex(block->addr_offset)}, 
                            belong, lock);
      }
      global_stream_context_.stream_free_list.PushBlock(
          process_local_, const_cast<MemBlock *>(block));
    }
  }
}

void DirectAllocator::Unmap(MemBlock *block) {
  bip::scoped_lock lock{shared_memory_.GetMutex()};
  // CHECK_EQ(block->nbytes, page_pool.config.page_nbytes);
  CHECK_EQ((block->addr_offset & (page_pool.config.page_nbytes - 1)), 0);
  CHECK_EQ(block->nbytes % page_pool.config.page_nbytes, 0);
  CHECK_EQ(block->unalloc_pages, 0);

  int num_pages = block->nbytes / page_pool.config.page_nbytes;

  auto vpage_idx = mapping_region_.ByteOffsetToIndex(block->addr_offset);
  auto & page_table = mapping_region_.GetMutableSelfPageTable();
  std::vector<index_t> pages;
  for (int i = 0; i < num_pages; i++) {
    CHECK(page_table.at(vpage_idx + i) != nullptr);
    // index_t page_begin = page_table.at(vpage_idx)->index;
    pages.push_back(page_table.at(vpage_idx + i)->index);
  }

  {
    auto page_pool_lk = page_pool.Lock();
    // page_pool.FreePages({page_begin}, belong, page_pool_lk);
    page_pool.FreePages(pages, belong, page_pool_lk);
  }

  auto addr = reinterpret_cast<CUdeviceptr>(
      mapping_region_.GetBasePtr() + block->addr_offset);
  // CU_CALL(cuMemUnmap(addr, page_pool.config.page_nbytes));
  CU_CALL(cuMemUnmap(addr, block->nbytes));

  block->unalloc_pages = num_pages;

  for (int i = 0; i < num_pages; i++) {
    page_table.at(vpage_idx + i) = nullptr;
  }
}

void DirectAllocator::Map(MemBlock* block) {
  // return nullptr;
  bip::scoped_lock lock{shared_memory_.GetMutex()};
  // CHECK_EQ(block->nbytes, page_pool.config.page_nbytes);
  CHECK_EQ(block->nbytes % page_pool.config.page_nbytes, 0);
  CHECK_EQ((block->addr_offset & (page_pool.config.page_nbytes - 1)), 0);

  int num_pages = block->nbytes / page_pool.config.page_nbytes;
  CHECK_EQ(block->unalloc_pages, num_pages);

  // index_t page_begin = mapping_region_.ByteOffsetToIndex(block->addr_offset);
  auto & page_table = mapping_region_.GetMutableSelfPageTable();
  auto addr = mapping_region_.GetBasePtr() + block->addr_offset;
  auto vpage_idx = mapping_region_.ByteOffsetToIndex(block->addr_offset);
  for (int i = 0; i < num_pages; i++) {
    CHECK(page_table.at(vpage_idx + i) == nullptr);
  }
  
  std::vector<index_t> new_page_index;
  {
    auto page_pool_lk = page_pool.Lock();
    new_page_index = page_pool.AllocDisPages(belong, num_pages, page_pool_lk);
  }
  if (new_page_index.size() != static_cast<uint64_t>(num_pages)) {
    LOG(WARNING) << config.log_prefix
                 << "[Map] OOM: Cannot find enough physical page, "
                 << "block: " << block << ", "
                 << "num_pages: " << num_pages << ".";
    ReportOOM(0, OOMReason::NO_PHYSICAL_PAGES, true);
  }

  for (int i = 0; i < num_pages; i++) {
    auto & page = page_pool.PagesView()[new_page_index[i]];
    CU_CALL(cuMemMap(
        reinterpret_cast<CUdeviceptr>(addr) + i * page_pool.config.page_nbytes,
        page_pool.config.page_nbytes, 0, page.cu_handle, 0));
  }

  CUmemAccessDesc acc_desc = {
    .location = {.type = CU_MEM_LOCATION_TYPE_DEVICE ,
                 .id = page_pool.config.device_id},
    .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
  };
  CU_CALL(cuMemSetAccess(
      reinterpret_cast<CUdeviceptr>(addr), 
      block->nbytes, &acc_desc, 1));
      // page_pool.config.page_nbytes, &acc_desc, 1));

  block->unalloc_pages = 0;

  for (int i = 0; i < num_pages; i++) {
    auto & page = page_pool.PagesView()[new_page_index[i]];
    page_table.at(vpage_idx + i) = &page;
  }
  // page_table.at(vpage_idx) = &page;
}

void DirectAllocator::ReportOOM(cudaStream_t cuda_stream, OOMReason reason,
                                bool force_abort) {
  VMMAllocator::ReportOOM(cuda_stream, reason, force_abort);
  page_pool.PrintStats();
  PrintStreamStats(global_stream_context_);
  DumpStateWithLock();
  LOG_IF(FATAL, force_abort) << config.log_prefix << "Abort.";
}

void DirectAllocator::DumpStateWithLock() {
  VMMAllocator::DumpState({&global_stream_context_});
}
MemBlock *DirectAllocator::AllocWithLock(
    size_t request_nbytes, size_t alignment, cudaStream_t cuda_stream, bool try_expand_VA,
    const bip::scoped_lock<bip::interprocess_mutex> &lock) {
  CHECK(lock.owns());
  if (request_nbytes < page_pool.config.page_nbytes) {
    auto *mem_block = global_stream_context_.stream_free_list.PopBlock(
        process_local_, true, request_nbytes, 0);
    if (mem_block != nullptr) {
      CHECK_EQ(mem_block->nbytes, request_nbytes);
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
        ReportOOM(cuda_stream, OOMReason::NO_PHYSICAL_PAGES, true);
      }
      ptrdiff_t addr_offset = pages_index[0] * page_pool.config.page_nbytes;
      size_t nbytes = page_pool.config.page_nbytes;
      auto *mem_block = global_stream_context_.stream_free_list.PopBlock(
          process_local_, false, addr_offset, nbytes);
      stats.SetBlockIsSmall(mem_block, true);
      if (request_nbytes < nbytes) {
        mem_block = global_stream_context_.stream_free_list.PushBlock(
            process_local_, mem_block);
        mem_block = global_stream_context_.stream_free_list.PopBlock(
            process_local_, true, request_nbytes, 0);
      }
      CHECK_EQ(mem_block->nbytes, request_nbytes);
      return mem_block;
    }
  } else {
    size_t pages_cnt = request_nbytes / page_pool.config.page_nbytes;
    size_t aligned_pages_cnt = alignment / page_pool.config.page_nbytes;
    index_t page_begin;
    {
      auto lock = page_pool.Lock();
      if (aligned_pages_cnt > 0) {
        page_begin = page_pool.AllocConPagesAlign(
            belong, pages_cnt, aligned_pages_cnt, lock);
      } else {
        page_begin = page_pool.AllocConPages(belong, pages_cnt, lock);
      }
    }
    if (page_begin == INVALID_INDEX) {
      LOG(WARNING)
          << config.log_prefix
          << "OOM: Cannot find enough continuous physical pages: require = "
          << pages_cnt << ".";
      ReportOOM(cuda_stream, OOMReason::NO_PHYSICAL_PAGES, true);
      return nullptr;
    }
    ptrdiff_t addr_offset = page_begin * page_pool.config.page_nbytes;
    size_t nbytes = pages_cnt * page_pool.config.page_nbytes;
    auto *mem_block = global_stream_context_.stream_free_list.PopBlock(
        process_local_, false, addr_offset, nbytes);
    CHECK_EQ(mem_block->nbytes, request_nbytes);
    return mem_block;
  }
}
} // namespace mpool