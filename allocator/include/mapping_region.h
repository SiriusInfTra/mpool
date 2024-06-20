#pragma once 

#include "pages.h"
#include <atomic>
#include <cstddef>
#include <string>
#include <vector>

#include <shm.h>
#include <util.h>
#include <belong.h>
#include <pages_pool.h>
#include <mem_block.h>

#define PAGE_INDEX_L(block) ((block)->addr_offset / mem_block_nbytes)
#define PAGE_INDEX_R(block) ((((block)->addr_offset + (block)->nbytes - 1) / mem_block_nbytes) + 1)


namespace mpool {

class MappingRegion {
public:
  const std::string log_prefix;
  const size_t mem_block_nbytes;
  const size_t mem_block_num;
  const size_t va_range_scale;
  const Belong belong;

private:
  SharedMemory &shared_memory_;
  std::byte *base_ptr_;
  std::vector<const PhyPage *> self_page_table_;
  bip_vector<index_t> &shared_global_mappings_;
  PagesPool &page_pool_;

public:
  MappingRegion(SharedMemory &shared_memory, PagesPool &page_pool,
                Belong belong, std::string log_prefix, size_t va_range_scale);

  index_t GetVirtualIndex(ptrdiff_t addr_offset) const;

  /** Ensure MemBlock with valid mappings. 
    * If the virtual address range is not allocated with physical pages, allocate physical pages.
    * If the virtual address range is allocated with physical pages remotely, retain those pages.
    * Otherwise, the virtual address range has valid mappings. So just do nothing. 
    */
  void EnsureMemBlockWithMappings(MemBlock *block,
                           bip_list<shm_handle<MemBlock>> &all_block_list);

  int32_t GetUnallocPages(ptrdiff_t addr_offset, size_t nbytes);

  void UnMapPages(const std::vector<index_t> &unmap_pages);

  void ReleasePages(const std::vector<index_t> &release_pages);

  void EmptyCache(bip_list<shm_handle<MemBlock>> &block_list);

  std::byte *GetBasePtr() const { return base_ptr_; }

  std::byte *GetEndPtr() const {
    return base_ptr_ + mem_block_nbytes * self_page_table_.size();
  }

};

struct ProcessLocalData {
  PagesPool &page_pool_;
  SharedMemory &shared_memory_;
  MappingRegion &mapping_region_;
};

class Operation {
public:
  virtual void Call() = 0;
};

struct CUMemMapOperation {
  const PhyPage *page;
  const CUdeviceptr dev_ptr;
};

struct CUMemUnMapOperation {
  const CUdeviceptr dev_ptr;
  const size_t nbytes;
};

struct CUMemSetAccessOperation {
  const CUdeviceptr dev_ptr;
  const size_t nbytes;
};

struct PageTableEntry {
  index_t phy_page_index;
  bool dirty;
};



// class MappingRegionRSM {
// public:
//   const size_t mem_block_nbytes;
//   const std::byte *base_ptr;
// private:
//   PagesPool &pages_pool_;
//   Belong self_belong_;
//   // Represents the current actual page table used by the GPU.
//   std::vector<const PhyPage *> curr_gpu_page_table_;
//   // Represents the page table that will match the GPU's actual page table 
//   // once all asynchronous operations are completed.
//   std::vector<index_t> pending_gpu_page_table_; 
//   // Represents the global page table shared among multiple processes. 
//   // The currentGPUPageTable and pendingGPUPageTable are part of this global page table.
//   bip::vector<index_t> &shared_global_page_table_; 
//   std::vector<CUMemMapOperation> plan_memmap_ops_;
//   std::vector<CUMemMapOperation> plan_memunmap_ops_;
//   std::vector<CUMemSetAccessOperation> plan_memsetaccess_ops_;

// public:
//   // BatchOperations(SharedMemory &shared_memory, size_t mem_block_nbytes): mem_block_nbytes(mem_block_nbytes), shared_global_page_table_(*shared_memory->find_or_construct<bip::vector<index_t>>("BO_shared_global_page_table")(shared_memory->get_free_memory())) {
    
//   // }

//   void EnsureMemBlockWithMapping(MemBlock *block) {
//     index_t page_range_l = PAGE_INDEX_L(block);
//     index_t page_range_r = PAGE_INDEX_R(block);
//     for (index_t i = page_range_l; i <= page_range_r; i++) {
//       auto *virtual_addr = base_ptr + i * mem_block_nbytes;
//       if (pending_gpu_page_table_[i].phy_page_index != INVALID_INDEX) {
//         continue;
//       }
//       AddLocalPendingMemMap(pages_pool_.RetainPage(shared_global_page_table_[i], self_belong_), virtual_addr);
//     }
//   }

//   void ApplyRemotePageTable(MemBlock *block) {
//     index_t page_range_l = PAGE_INDEX_L(block);
//     index_t page_range_r = PAGE_INDEX_R(block);
//     for (index_t i = page_range_l; i <= page_range_r; i++) {
//       index_t remote_page_index = shared_global_page_table_[i];
//       index_t local_page_index = pending_gpu_page_table_[i].phy_page_index;
//       if (remote_page_index == local_page_index) {
//         continue;
//       }
//       auto *virtual_addr = base_ptr + i * mem_block_nbytes;
//       if (local_page_index != INVALID_INDEX) {
//         AddLocalPendMemUnmap(virtual_addr, mem_block_nbytes);
//       }
//       if (remote_page_index != INVALID_INDEX) {
//         AddLocalPendingMemMap(pages_pool_.RetainPage(remote_page_index, self_belong_), virtual_addr);
//       }
//       AddLocalMemSetAccess(base_ptr + page_range_l * mem_block_nbytes, (page_range_r - page_range_l) * mem_block_nbytes);
//     }
//   }

//   void ApplyLocalMemMap(MemBlock *block) {
//     index_t page_range_l = PAGE_INDEX_L(block);
//     index_t page_range_r = PAGE_INDEX_R(block);
//     for (index_t i = page_range_l; i <= page_range_r; i++) {
//       auto *virtual_addr = base_ptr + i * mem_block_nbytes;
//       if (pending_gpu_page_table_[i].phy_page_index != INVALID_INDEX) {
//         continue;
//       }
//       AddLocalPendingMemMap(pages_pool_.RetainPage(shared_global_page_table_[i], self_belong_), virtual_addr);
//     }
//   }


//   void AddLocalPendingMemMap(PhyPage *page, const std::byte *virtual_addr) {
//     plan_memmap_ops_.push_back(CUMemMapOperation{page, reinterpret_cast<const CUdeviceptr>(virtual_addr)});
//     auto &page_entry = pending_gpu_page_table_[(virtual_addr - base_ptr) / mem_block_nbytes];
//     page_entry.phy_page_index = page->index;
//     page_entry.dirty = true;
//   }

//   void AddLocalPendMemUnmap(const std::byte *virtual_addr, size_t nbytes) {
//     plan_memunmap_ops_.emplace_back(virtual_addr, nbytes);
//   }

//   void AddLocalMemSetAccess(const std::byte *virtual_addr, size_t nbytes) {
//     CUmemAccessDesc acc_desc = {
//       .location = {.type = CU_MEM_LOCATION_TYPE_DEVICE, .id = 0},
//       .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
//     plan_memsetaccess_ops_.emplace_back(virtual_addr, nbytes, acc_desc);
//   }

//   void Flush() {
//     for (auto &op : plan_memunmap_ops_) {
//       op.Call();
//     }
//     for (auto &op : plan_memmap_ops_) {
//       op.Call();
//     }
//     for (auto &op : plan_memsetaccess_ops_) {
//       op.Call();
//     }
//   }
// };


}