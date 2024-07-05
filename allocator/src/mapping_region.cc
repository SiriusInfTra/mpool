#include "pages.h"
#include "util.h"
#include <algorithm>
#include <boost/interprocess/containers/vector.hpp>
#include <cuda.h>
#include <functional>
#include <glog/logging.h>
#include <limits>
#include <mapping_region.h>

namespace mpool {


/** Implementation Details: Ensure MemBlock with valid mappings.
  * If the virtual address range is not allocated with physical pages, allocate
  * physical pages. If the virtual address range is allocated with physical
  * pages remotely, retain those pages. Otherwise, the virtual address range
  * has valid mappings. So just do nothing.
  */
void DynamicMappingRegion::AllocMappingsAndUpdateFlags(
    MemBlock *block, bip_list<shm_ptr<MemBlock>> &all_block_list) {
  if (block->addr_offset + block->nbytes > mem_block_nbytes * mem_block_num * va_range_scale) {
    LOG(WARNING) << "OOM: Cannot reserve VA for block: " << block
                 << ", addr_offset = " << block->addr_offset
                 << ", nbytes = " << block->nbytes << ".";
    ReportOOM(page_pool_.config.device_id, block->stream, OOMReason::NO_VIRTUAL_SPACE);
    return;
  }

  index_t va_range_l_i = PAGE_INDEX_L(block);
  index_t va_range_r_i = PAGE_INDEX_R(block);

  /* 1. find missing physical pages . */
  std::vector<index_t> missing_va_mapping_i;
  if (block->unalloc_pages > 0) {
    if (shared_global_mappings_.size() < va_range_r_i) {
      shared_global_mappings_.resize(va_range_r_i, INVALID_INDEX);
    }
    for (index_t index = va_range_l_i; index < va_range_r_i; ++index) {
      if (shared_global_mappings_[index] == INVALID_INDEX) {
        missing_va_mapping_i.push_back(index);
      }
    }
    CHECK_EQ(block->unalloc_pages, missing_va_mapping_i.size()) << "Mismatch unalloc pages: " << block << ".";
  }

  /* 2. alloc physical pages if necessary */
  if (!missing_va_mapping_i.empty()) {
    std::vector<index_t> new_allocated_pages_index;
    {
      auto lock = page_pool_.Lock();
      new_allocated_pages_index =
          page_pool_.AllocDisPages(belong, missing_va_mapping_i.size(), lock);
    }
    if (new_allocated_pages_index.size() < missing_va_mapping_i.size()) {
      LOG(WARNING) << "OOM: Cannot allocate enough pages for block: " << block
              << ", unalloc_pages = " << block->unalloc_pages
              << ", allocated_pages = " << missing_va_mapping_i.size() << ".";
      {
        auto lock = page_pool_.Lock();
        page_pool_.FreePages(new_allocated_pages_index, belong, lock);
      }
      ReportOOM(page_pool_.config.device_id, block->stream, OOMReason::NO_PHYSICAL_PAGES);
      return;
    }
    for (index_t k = 0; k < new_allocated_pages_index.size(); ++k) {
      shared_global_mappings_[missing_va_mapping_i[k]] = new_allocated_pages_index[k];
    }
  }

  /* 3. modify local page table if necessary */
  if (self_page_table_.size() < va_range_r_i) {
    self_page_table_.resize(va_range_r_i, nullptr);
  }
  index_t min_va_i = std::numeric_limits<index_t>::max();
  index_t max_va_i = std::numeric_limits<index_t>::min();
  for (index_t k = va_range_l_i; k < va_range_r_i; ++k) {
    if (self_page_table_[k] != nullptr &&
        self_page_table_[k]->index != shared_global_mappings_[k]) {
      CU_CALL(cuMemUnmap(reinterpret_cast<CUdeviceptr>(base_ptr_ + k * mem_block_nbytes), mem_block_nbytes));
      self_page_table_[k] = nullptr;
    }
    if (self_page_table_[k] == nullptr) {
      auto &page = page_pool_.PagesView()[shared_global_mappings_[k]];
      self_page_table_[k] = &page;
      CHECK_EQ(belong, Belong(*page.belong, shared_memory_));
      CU_CALL(cuMemMap(
          reinterpret_cast<CUdeviceptr>(base_ptr_ + k * mem_block_nbytes),
          mem_block_nbytes, 0, page.cu_handle, 0));
      min_va_i = std::min(min_va_i, k);
      max_va_i = std::max(max_va_i, k);
    }
    CHECK_EQ(self_page_table_[k]->index, shared_global_mappings_[k]);
  }
  if (min_va_i < std::numeric_limits<index_t>::max()) {
    auto *dev_ptr = base_ptr_ + min_va_i * mem_block_nbytes;
    size_t nbytes = (max_va_i - min_va_i + 1) * mem_block_nbytes;
    CUmemAccessDesc acc_desc = {
        .location = {.type = CU_MEM_LOCATION_TYPE_DEVICE, .id = page_pool_.config.device_id},
        .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
    CU_CALL(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(dev_ptr), nbytes,
                           &acc_desc, 1));
  }

  /* 4. update memory block unalloc_pages flag */
  if (!missing_va_mapping_i.empty()) {
    block->unalloc_pages = 0;
    auto next_iter = std::next(block->iter_all_block_list);
    while (next_iter != all_block_list.cend()) {
      auto *next_block = next_iter->ptr(shared_memory_);
      if (ByteOffsetToIndex(next_block->addr_offset) >
          missing_va_mapping_i.back()) {
        break;
      }
      DCHECK_GE(next_block->unalloc_pages, 1);
      next_block->unalloc_pages--;
      DCHECK_EQ(next_block->unalloc_pages,
                CalculateUnallocFlags(next_block->addr_offset, next_block->nbytes))
          << next_block;
      next_iter++;
    }
    auto prev_iter = block->iter_all_block_list;
    while (prev_iter != all_block_list.cbegin()) {
      --prev_iter;
      auto *prev_block = prev_iter->ptr(shared_memory_);
      if (ByteOffsetToIndex(prev_block->addr_offset + prev_block->nbytes - 1) <
          missing_va_mapping_i.front()) {
        break;
      }
      DCHECK_GE(prev_block->unalloc_pages, 1);
      prev_block->unalloc_pages--;
      DCHECK_EQ(prev_block->unalloc_pages,
                CalculateUnallocFlags(prev_block->addr_offset, prev_block->nbytes))
          << prev_block;
    }
  }  
  // LOG(INFO) << "EnsureMemBlockWithMappings: " << self_page_table_;
}

void DynamicMappingRegion::ReleasePages(const std::vector<index_t> &release_pages) {
  auto lock = page_pool_.Lock();
  page_pool_.FreePages(release_pages, belong, lock);
}

void DynamicMappingRegion::UnMapPages(const std::vector<index_t> &unmap_pages) {
  for (auto index : unmap_pages) {
    self_page_table_[index] = nullptr;
  }
  auto iter = unmap_pages.cbegin();
  do {
    auto iter_dis =
        std::adjacent_find(iter, unmap_pages.cend(),
                           [](index_t a, index_t b) { return a + 1 != b; });
    if (iter != iter_dis) {
      /* unmap continuous physical pages */
      CU_CALL(cuMemUnmap(
          reinterpret_cast<CUdeviceptr>(base_ptr_ + *iter * mem_block_nbytes),
          std::distance(iter, iter_dis) * mem_block_nbytes));
    }
    if (iter_dis != unmap_pages.cend()) {
      /* unmap discontinuous physical pages */
      CU_CALL(cuMemUnmap(reinterpret_cast<CUdeviceptr>(
                             base_ptr_ + *iter_dis * mem_block_nbytes),
                         mem_block_nbytes));
      iter = std::next(iter_dis);
    } else {
      break;
    }
  } while (iter != unmap_pages.cend());
}



void DynamicMappingRegion::EmptyCacheAndUpdateFlags(bip_list<shm_ptr<MemBlock>> &block_list) {
  LOG_IF(INFO, VERBOSE_LEVEL >= 0) << "EmptyCache";
  std::vector<index_t> release_pages;
  std::vector<index_t> unmap_pages;
  auto iter = block_list.begin();
  auto block = iter->ptr(shared_memory_);
  for (index_t mapping_index = 0; mapping_index < self_page_table_.size();) {
    /* 1. ignore dummy cases. */
    // ignore virtual pages with no physical block.
    if (shared_global_mappings_[mapping_index] == INVALID_INDEX) {
      if (self_page_table_[mapping_index] != nullptr) {
        unmap_pages.push_back(mapping_index);
      }
      mapping_index++;
      continue;
    }

    // ignore logical memory block with no physical block. 
    if (index_t block_mapping_index_begin = PAGE_INDEX_L(block);
        block_mapping_index_begin > mapping_index) {
      mapping_index = block_mapping_index_begin;
      continue;
    }

    while (mapping_index >= PAGE_INDEX_R(block)) {
      iter++;
      block = iter->ptr(shared_memory_);
    }
    // CHECK_LE(GetMappingPage(block->addr_offset), mapping_index);

    /* 2. check whether mapping_index-th is free. */
    bool page_is_free = true;
    std::vector<MemBlock *> blocks_with_pages;
    while (true) {
      blocks_with_pages.push_back(block);
      if (!block->is_free) {
        page_is_free = false;
        break;
      }
      if (block->addr_offset + block->nbytes - 1 <
          mapping_index * mem_block_nbytes + mem_block_nbytes - 1) {
        iter++;
        if (iter != block_list.cend()) {
          block = iter->ptr(shared_memory_);
        } else {
          break;
        }

      } else {
        break;
      }
    }
    /* 3. release page if free and update block's unalloc page. */
    if (page_is_free) {
      release_pages.push_back(shared_global_mappings_[mapping_index]);
      shared_global_mappings_[mapping_index] = INVALID_INDEX;
      for (auto block1 : blocks_with_pages) {
        block1->unalloc_pages++;
      }
    }
    mapping_index++;
  }
  ReleasePages(release_pages);
  // UnMapPages(unmap_pages);
}

IMappingRegion::IMappingRegion(
    SharedMemory &shared_memory, PagesPool &page_pool, Belong belong,
    std::string log_prefix, size_t va_range_scale,
    std::function<void(int device_id, cudaStream_t cuda_stream,
                       OOMReason reason)>
        ReportOOM) : log_prefix(log_prefix), mem_block_nbytes(page_pool.config.page_nbytes),
      mem_block_num(page_pool.config.pool_nbytes /
                    page_pool.config.page_nbytes),
      va_range_scale(va_range_scale), belong(belong),
      shared_memory_(shared_memory),
      shared_global_mappings_(
          *shared_memory->find_or_construct<bip_vector<index_t>>(
              "ME_shared_global_mappings")(
              shared_memory->get_segment_manager())),
      page_pool_(page_pool), ReportOOM(ReportOOM) {
  CU_CALL(cuMemAddressReserve(reinterpret_cast<CUdeviceptr *>(&base_ptr_),
                              mem_block_nbytes * mem_block_num * va_range_scale,
                              mem_block_nbytes, 0, 0));
  LOG(INFO) << log_prefix << "dev_ptr = " << base_ptr_
            << ", mem_block_nbytes = " << mem_block_nbytes
            << ", mem_block_num = " << mem_block_num
            << ", va_range_scale = " << va_range_scale << ".";
}
int IMappingRegion::CalculateUnallocFlags(ptrdiff_t addr_offset,
                                          size_t nbytes) {
  index_t va_range_l_i = ByteOffsetToIndex(addr_offset);
  index_t va_range_r_i = ByteOffsetToIndex(addr_offset + nbytes - 1) + 1;
  int unalloc_pages = 0;
  for (index_t index = va_range_l_i; index < va_range_r_i; ++index) {
    if (index >= shared_global_mappings_.size() ||
        shared_global_mappings_[index] == INVALID_INDEX) {
      unalloc_pages++;
    }
  }
  DCHECK_LE(unalloc_pages,
            (nbytes + mem_block_nbytes - 1) / mem_block_nbytes + 1)
      << ByteDisplay(nbytes);
  return unalloc_pages;
}
bool StaticMappingRegion::CanMerge(const MemBlock *block_a,
                                   const MemBlock *block_b) {
  return !block_a->is_small || ByteOffsetToIndex(block_a->addr_offset) ==
         ByteOffsetToIndex(block_b->addr_offset + block_b->nbytes - 1);
};
bool DynamicMappingRegion::CanMerge(const MemBlock *block_a,
                                    const MemBlock *block_b) {
  return true;
};
StaticMappingRegion::StaticMappingRegion(
    SharedMemory &shared_memory, PagesPool &page_pool, Belong belong,
    std::string log_prefix, size_t va_range_scale,
    std::function<void(int device_id, cudaStream_t cuda_stream,
                       OOMReason reason)>
        ReportOOM)
    : IMappingRegion(shared_memory, page_pool, belong, log_prefix,
                     va_range_scale, ReportOOM) {
  if (va_range_scale > 1) {
    LOG(WARNING) << log_prefix
                 << "va_range_scale > 1 is not supported for static mapping "
                    "region, so reset to 1.";
  }
  for (auto &page : page_pool_.PagesView()) {
      CU_CALL(cuMemMap(
          reinterpret_cast<CUdeviceptr>(base_ptr_ + page.index * mem_block_nbytes),
          mem_block_nbytes, 0, page.cu_handle, 0));
  }
  CUmemAccessDesc acc_desc = {
      .location = {.type = CU_MEM_LOCATION_TYPE_DEVICE, .id = page_pool_.config.device_id},
      .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
  CU_CALL(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(base_ptr_), mem_block_nbytes * mem_block_num,
                          &acc_desc, 1));
  for (size_t k = 0; k < mem_block_num; k++) {
    self_page_table_.push_back(&page_pool.PagesView()[k]);
  }
  if (shared_global_mappings_.empty()) {
    for (size_t k = 0; k < mem_block_num; k++) {
      shared_global_mappings_.push_back(k);
    }
  }

}
DynamicMappingRegion::DynamicMappingRegion(
    SharedMemory &shared_memory, PagesPool &page_pool, Belong belong,
    std::string log_prefix, size_t va_range_scale,
    std::function<void(int device_id, cudaStream_t cuda_stream,
                       OOMReason reason)>
        ReportOOM)
    : IMappingRegion(shared_memory, page_pool, belong, log_prefix,
                     va_range_scale, ReportOOM) {}
} // namespace mpool