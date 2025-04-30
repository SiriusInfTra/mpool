#pragma once

#include <cstddef>
#include <functional>
#include <string>

#include <mpool/belong.h>
#include <mpool/mem_block.h>
#include <mpool/pages.h>
#include <mpool/pages_pool.h>
#include <mpool/shm.h>
#include <mpool/util.h>
#include <vector>

#define PAGE_INDEX_L(block) ((block)->addr_offset / mem_block_nbytes)
#define PAGE_INDEX_R(block)                                                    \
  ((((block)->addr_offset + (block)->nbytes - 1) / mem_block_nbytes) + 1)

enum class OOMReason { NO_PHYSICAL_PAGES, NO_VIRTUAL_SPACE, NO_MEMORY_BLOCK };



namespace mpool {

using OOMObserver =
    std::function<void(int device_id, cudaStream_t stream, OOMReason reason)>;

class IMappingRegion {
public:
  const std::string log_prefix;
  const size_t mem_block_nbytes;
  const size_t mem_block_num;
  const size_t va_range_scale;
  const Belong belong;

protected:
  SharedMemory &shared_memory_;
  std::byte *base_ptr_;
  std::vector<const PhyPage *> self_page_table_;
  bip_vector<index_t> &shared_global_mappings_;
  PagesPool &page_pool_;
  std::function<void(int device_id, cudaStream_t cuda_stream, OOMReason reason)>
      ReportOOM;

public:
  IMappingRegion(SharedMemory &shared_memory, PagesPool &page_pool,
                 Belong belong, std::string log_prefix, size_t va_range_scale,
                 std::function<void(int device_id, cudaStream_t cuda_stream,
                                    OOMReason reason)>
                     ReportOOM);

  virtual ~IMappingRegion() = default;

  /**
   * @brief Empties the cache and updates the flags for the given block list.
   *
   * This function is responsible for emptying the cache and updating the flags
   * for the all block list.
   *
   * @param block_list The list of shared memory pointers to memory blocks.
   */
  virtual void
  EmptyCacheAndUpdateFlags(bip_list<shm_ptr<MemBlock>> &all_block_list) = 0;

  /**
   * @brief Allocates mappings for a memory block.
   *
   * This function is responsible for allocating mappings for the
   * specified memory block and updating the flags accordingly. The allocated
   * mappings are added to the provided block list.
   *
   * @param block The memory block for which mappings need to be allocated.
   */
  virtual std::vector<index_t> AllocMappings(MemBlock *block) = 0;

  /**
   * Calculates the unallocated flags for a given address offset and number of
   * bytes.
   *
   * @param addr_offset The offset of the address from the start of the mapping
   * region.
   * @param nbytes The number of bytes to calculate the unallocated flags for.
   * @return The unsigned value representing the unallocated flags (count).
   */
  virtual int CalculateUnallocFlags(ptrdiff_t addr_offset, size_t nbytes) = 0;

  /**
   * Calculates the unallocated flags for a given memory block.
   *
   * @param mem_block The memory block for which to calculate the unallocated
   * flags.
   * @return The unallocated flags for the memory block.
   */
  unsigned CalculateUnallocFlags(const MemBlock *mem_block) {
    return CalculateUnallocFlags(mem_block->addr_offset, mem_block->nbytes);
  }

  /**
   * Determines whether two adjacent memory blocks can be merged together.
   *
   * @param block_a The first memory block.
   * @param block_b The second memory block.
   * @return True if the memory blocks can be merged, false otherwise.
   */
  virtual bool CanMerge(const MemBlock *block_a, const MemBlock *block_b) = 0;

  /**
   * Calculates the index corresponding to a given byte offset.
   *
   * @param addr_offset The byte offset to calculate the index for.
   * @return The index corresponding to the given byte offset.
   */
  index_t ByteOffsetToIndex(ptrdiff_t addr_offset) const {
    return addr_offset / mem_block_nbytes;
  }

  /**
   * @brief Get the base pointer of the mapping region.
   *
   * This function returns the base pointer of the mapping region.
   *
   * @return The base pointer of the mapping region.
   */
  std::byte *GetBasePtr() const { return base_ptr_; }

  /**
   * Returns the total number of bytes in the virtual address range.
   *
   * This function calculates the total number of bytes in the virtual address
   * range by calculate the peak highest mappings address.
   *
   * @return The total number of bytes in the virtual address range.
   */
  size_t GetVARangeNBytes() const {
    return mem_block_nbytes * self_page_table_.size();
  }

  std::vector<const PhyPage *> & GetMutableSelfPageTable() { 
    return self_page_table_; 
  }
};

class DynamicMappingRegion : public IMappingRegion {
private:
  void UnMapPages(const std::vector<index_t> &unmap_pages);

  void ReleasePages(const std::vector<index_t> &release_pages);

public:
  DynamicMappingRegion(
      SharedMemory &shared_memory, PagesPool &page_pool, Belong belong,
      std::string log_prefix, size_t va_range_scale,
      std::function<void(int device_id, cudaStream_t cuda_stream,
                         OOMReason reason)>
          ReportOOM);

  std::vector<index_t> AllocMappings(MemBlock *block) override;
  
  void
  EmptyCacheAndUpdateFlags(bip_list<shm_ptr<MemBlock>> &block_list) override;

  bool CanMerge(const MemBlock *block_a, const MemBlock *block_b) override;

  int CalculateUnallocFlags(ptrdiff_t addr_offset, size_t nbytes) override;
};

class StaticMappingRegion : public IMappingRegion {
public:
  StaticMappingRegion(
      SharedMemory &shared_memory, PagesPool &page_pool, Belong belong,
      std::string log_prefix, size_t va_range_scale,
      std::function<void(int device_id, cudaStream_t cuda_stream,
                         OOMReason reason)>
          ReportOOM);
  std::vector<index_t> AllocMappings(MemBlock *block) override {
    return {};
  }

  void
  EmptyCacheAndUpdateFlags(bip_list<shm_ptr<MemBlock>> &block_list) override {}

  bool CanMerge(const MemBlock *block_a, const MemBlock *block_b) override;

  int CalculateUnallocFlags(ptrdiff_t addr_offset, size_t nbytes) override;
};

struct ProcessLocalData {
  PagesPool &page_pool_;
  SharedMemory &shared_memory_;
  IMappingRegion *mapping_region_;
  bip_list<shm_ptr<MemBlock>> &all_block_list_;
  bip_map<ptrdiff_t, shm_ptr<MemBlock>> &all_block_map_;
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
//   // The currentGPUPageTable and pendingGPUPageTable are part of this global
//   page table. bip::vector<index_t> &shared_global_page_table_;
//   std::vector<CUMemMapOperation> plan_memmap_ops_;
//   std::vector<CUMemMapOperation> plan_memunmap_ops_;
//   std::vector<CUMemSetAccessOperation> plan_memsetaccess_ops_;

// public:
//   // BatchOperations(SharedMemory &shared_memory, size_t mem_block_nbytes):
//   mem_block_nbytes(mem_block_nbytes),
//   shared_global_page_table_(*shared_memory->find_or_construct<bip::vector<index_t>>("BO_shared_global_page_table")(shared_memory->get_free_memory()))
//   {

//   // }

//   void EnsureMemBlockWithMapping(MemBlock *block) {
//     index_t page_range_l = PAGE_INDEX_L(block);
//     index_t page_range_r = PAGE_INDEX_R(block);
//     for (index_t i = page_range_l; i <= page_range_r; i++) {
//       auto *virtual_addr = base_ptr + i * mem_block_nbytes;
//       if (pending_gpu_page_table_[i].phy_page_index != INVALID_INDEX) {
//         continue;
//       }
//       AddLocalPendingMemMap(pages_pool_.RetainPage(shared_global_page_table_[i],
//       self_belong_), virtual_addr);
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
//         AddLocalPendingMemMap(pages_pool_.RetainPage(remote_page_index,
//         self_belong_), virtual_addr);
//       }
//       AddLocalMemSetAccess(base_ptr + page_range_l * mem_block_nbytes,
//       (page_range_r - page_range_l) * mem_block_nbytes);
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
//       AddLocalPendingMemMap(pages_pool_.RetainPage(shared_global_page_table_[i],
//       self_belong_), virtual_addr);
//     }
//   }

//   void AddLocalPendingMemMap(PhyPage *page, const std::byte *virtual_addr) {
//     plan_memmap_ops_.push_back(CUMemMapOperation{page, reinterpret_cast<const
//     CUdeviceptr>(virtual_addr)}); auto &page_entry =
//     pending_gpu_page_table_[(virtual_addr - base_ptr) / mem_block_nbytes];
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

} // namespace mpool