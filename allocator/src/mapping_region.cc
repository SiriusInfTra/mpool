#include <mapping_region.h>

#include <cuda.h>
#include <glog/logging.h>

namespace mpool {
std::pair<index_t, index_t>
MappingRegion::MappingPageRange(ptrdiff_t addr_offset, size_t nbytes) const {
  index_t index_begin = addr_offset / mem_block_nbytes;
  index_t index_end =
      (addr_offset + nbytes + mem_block_nbytes - 1) / mem_block_nbytes;
  return {index_begin, index_end};
}


index_t MappingRegion::GetMappingPage(ptrdiff_t addr_offset) const {
  return addr_offset / mem_block_nbytes;
}

void MappingRegion::EnsureBlockWithPage(
    MemBlock *block, bip_list<shm_handle<MemBlock>> &all_block_list) {
  CHECK_LE(block->addr_offset + block->nbytes,
           mem_block_nbytes * mem_block_num * va_range_scale);
  /* 1. check whether the block is already mapped. */
  if (block->unalloc_pages == 0) {
    return;
  }
  auto [index_begin, index_end] =
      MappingPageRange(block->addr_offset, block->nbytes);
  
  /* 2. find missing pages mapping index. */
  std::vector<index_t> missing_pages_mapping_index;
  if (mapping_pages_.size() < index_end) {
    mapping_pages_.resize(index_end, nullptr);
  }
  for (index_t index = index_begin; index < index_end; ++index) {
    if (mapping_pages_[index] == nullptr) {
      missing_pages_mapping_index.push_back(index);
    }
  }
  CHECK_EQ(block->unalloc_pages, missing_pages_mapping_index.size());

  /* 3. alloc physical pages */
  std::vector<index_t> assign_phy_pages_index;
  {
    auto lock = page_pool_.Lock();
    assign_phy_pages_index = page_pool_.AllocDisPages(
        belong, missing_pages_mapping_index.size(), lock);
  }
  CHECK_EQ(assign_phy_pages_index.size(), missing_pages_mapping_index.size());
  for (size_t k = 0; k < missing_pages_mapping_index.size(); ++k) {
    CU_CALL(cuMemMap(
        reinterpret_cast<CUdeviceptr>(
            base_ptr_ + missing_pages_mapping_index[k] * mem_block_nbytes),
        mem_block_nbytes, 0,
        page_pool_.PagesView()[assign_phy_pages_index[k]].cu_handle, 0));
    mapping_pages_[missing_pages_mapping_index[k]] =
        &page_pool_.PagesView()[assign_phy_pages_index[k]];
  }

  /* 4. modify page table */
  ptrdiff_t addr_offset;
  size_t nbytes;
  if (std::adjacent_find(missing_pages_mapping_index.begin(),
                         missing_pages_mapping_index.end(),
                         [](index_t a, index_t b) { return b != a + 1; }) ==
      missing_pages_mapping_index.end()) {
    addr_offset = missing_pages_mapping_index.front() * mem_block_nbytes;
    nbytes = missing_pages_mapping_index.size() * mem_block_nbytes;
  } else {
    addr_offset = index_begin * mem_block_nbytes;
    nbytes = (index_end - index_begin) * mem_block_nbytes;
  }
  CUmemAccessDesc acc_desc = {
      .location = {.type = CU_MEM_LOCATION_TYPE_DEVICE, .id = 0},
      .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
  CU_CALL(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(base_ptr_ + addr_offset),
                         nbytes, &acc_desc, 1));
  
  /* 5. update memory block unalloc_pages flag */
    CHECK(!missing_pages_mapping_index.empty())
      << missing_pages_mapping_index.size();
  block->unalloc_pages = 0;
  auto next_iter = std::next(block->iter_all_block_list);
  while (next_iter != all_block_list.cend()) {
    auto *next_block = next_iter->ptr(shared_memory_);
    if (GetMappingPage(next_block->addr_offset) > missing_pages_mapping_index.back()) {
      break;
    }
    DCHECK_GE(next_block->unalloc_pages, 1);
    next_block->unalloc_pages--;
    DCHECK_EQ(next_block->unalloc_pages, GetUnallocPages(next_block->addr_offset,
                                             next_block->nbytes)) <<
                                             next_block;
    next_iter++;
  }
  auto prev_iter = block->iter_all_block_list;
  while (prev_iter != all_block_list.cbegin()) {
    --prev_iter;
    auto *prev_block = prev_iter->ptr(shared_memory_);
    if (GetMappingPage(prev_block->addr_offset + prev_block->nbytes - 1) < missing_pages_mapping_index.front()) {
      break;
    }
    DCHECK_GE(prev_block->unalloc_pages, 1);
    prev_block->unalloc_pages--;
    DCHECK_EQ(prev_block->unalloc_pages, GetUnallocPages(prev_block->addr_offset,
                                             prev_block->nbytes)) <<
                                             prev_block;
  }
}

int32_t MappingRegion::GetUnallocPages(ptrdiff_t addr_offset, size_t nbytes) {
  auto [index_begin, index_end] = MappingPageRange(addr_offset, nbytes);
  size_t ub = std::min(index_end, mapping_pages_.size());
  int32_t unalloc_pages = std::min(index_end - index_begin, index_end - ub);
  for (index_t index = index_begin; index < ub; ++index) {
    if (mapping_pages_[index] == nullptr) {
      unalloc_pages++;
    }
  }
  DCHECK_LE(unalloc_pages,
            (nbytes + mem_block_nbytes - 1) / mem_block_nbytes + 1)
      << ByteDisplay(nbytes);
  return unalloc_pages;
}
void MappingRegion::UnMapPages(const std::vector<index_t> &release_pages) {
  /* release physical page to mem pool */
  {
    std::vector<index_t> pages;
    for (index_t mapping_index : release_pages) {
      index_t page_index = mapping_pages_[mapping_index]->index;
      mapping_pages_[mapping_index] = nullptr;

      pages.push_back(page_index);
    }
    auto lock = page_pool_.Lock();
    page_pool_.FreePages(pages, belong, lock);
  }

  /* unmap corresponding physical pages */
  auto iter = release_pages.cbegin();
  do {
    auto iter_dis =
        std::adjacent_find(iter, release_pages.cend(),
                           [](index_t a, index_t b) { return a + 1 != b; });
    if (iter != iter_dis) {
      /* unmap continuous physical pages */
      CU_CALL(cuMemUnmap(
          reinterpret_cast<CUdeviceptr>(base_ptr_ + *iter * mem_block_nbytes),
          std::distance(iter, iter_dis) * mem_block_nbytes));
    }
    if (iter_dis != release_pages.cend()) {
      /* unmap discontinuous physical pages */
      CU_CALL(cuMemUnmap(reinterpret_cast<CUdeviceptr>(
                             base_ptr_ + *iter_dis * mem_block_nbytes),
                         mem_block_nbytes));
      iter = std::next(iter_dis);
    } else {
      break;
    }
  } while (iter != release_pages.cend());
}
void MappingRegion::EmptyCache(bip_list<shm_handle<MemBlock>> &block_list) {
  std::vector<index_t> release_pages;
  auto iter = block_list.begin();
  auto block = iter->ptr(shared_memory_);
  for (index_t mapping_index = 0; mapping_index < mapping_pages_.size();) {
    if (mapping_pages_[mapping_index] == nullptr) {
      mapping_index++;
      continue;
    }

    if (index_t block_mapping_index_begin = GetMappingPage(block->addr_offset);
        block_mapping_index_begin > mapping_index) {
      mapping_index = block_mapping_index_begin;
      continue;
    }

    while (GetMappingPage(block->addr_offset + block->nbytes - 1) <
           mapping_index) {
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
      release_pages.push_back(mapping_index);
      for (auto block1 : blocks_with_pages) {
        block1->unalloc_pages++;
      }
    }

    mapping_index++;
  }
  UnMapPages(release_pages);
}
} // namespace mpool