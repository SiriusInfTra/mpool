#pragma once
#include <cstddef>
#include <iterator>

#include <mpool/stats.h>
#include <mpool/mapping_region.h>
#include <mpool/mem_block.h>
#include <mpool/shm.h>

#include <cuda_runtime_api.h>

namespace mpool {

/**
 * @class StreamBlockList
 * @brief Represents a list of memory blocks associated with a stream.
 *
 * The StreamBlockList class is used to manage a list of memory blocks that are
 * associated with a specific stream. It provides low-level methods for
 * creating, merging, splitting and deconstructing memory blocks within the
 * list.
 *
 * @note This class is designed to be used in conjunction with the StreamContext
 * class.
 */
class StreamBlockList {
  friend class StreamContext;

private:
  int device_id;
  cudaStream_t current_stream_;

  bip_list<shm_ptr<MemBlock>> stream_block_list_;

  shm_ptr<CachingAllocatorStats> stats_;

  size_t small_block_nbytes_;

public:
  StreamBlockList(int device_id, cudaStream_t cuda_stream,
                  SharedMemory &shared_memory, size_t small_block_nbytes, CachingAllocatorStats &stats);
  StreamBlockList &operator=(const StreamBlockList &) = delete;
  StreamBlockList(const StreamBlockList &) = delete;

  MemBlock *CreateEntryExpandVA(ProcessLocalData &local, size_t nbytes);

  MemBlock *GetPrevEntry(ProcessLocalData &local, MemBlock *entry);

  MemBlock *GetNextEntry(ProcessLocalData &local, MemBlock *entry);

  MemBlock *SplitBlock(ProcessLocalData &local, MemBlock *origin_entry,
                       size_t remain);

  MemBlock *MergeMemEntry(ProcessLocalData &local, MemBlock *first_block,
                          MemBlock *secound_block);
  
  MemBlock *SearchBlock(ProcessLocalData &local, ptrdiff_t addr_offset) {
    if (local.all_block_map_.empty()) {
      return nullptr;
    }
    auto iter = local.all_block_map_.lower_bound(addr_offset);
    if (iter == local.all_block_map_.cend()) {
      auto *mem_block = std::prev(iter)->second.ptr(local.shared_memory_);
      if (mem_block->addr_offset + static_cast<ptrdiff_t>(mem_block->nbytes) - 1 >= addr_offset) {
        return mem_block;
      } else {
        return nullptr;
      }
    } else {
      auto *mem_block = iter->second.ptr(local.shared_memory_);
      if (mem_block->addr_offset == addr_offset) {
        return mem_block;
      } else if (mem_block->iter_all_block_map != local.all_block_map_.cbegin()) {
        return std::prev(iter)->second.ptr(local.shared_memory_);
      } else {
        return nullptr;
      }
    }
  }

  std::pair<bip_list<shm_ptr<MemBlock>>::const_iterator,
            bip_list<shm_ptr<MemBlock>>::const_iterator>
  Iterators() const;

  void DumpMemBlockColumns(std::ostream &out);
  void DumpMemBlock(ProcessLocalData &local, std::ostream &out,
                    MemBlock *block);

  void DumpStreamBlockList(ProcessLocalData &local,
                           std::ostream &out = std::cout);

  bool CheckState(ProcessLocalData &local,
                  bool check_global_block_list = false);

  size_t Size() const { return stream_block_list_.size(); }
};

/**
 * @class StreamFreeList
 * @brief Manages the free memory blocks for a CUDA stream.
 *
 * The StreamFreeList class is responsible for managing the free memory blocks
 * for a specific CUDA stream. It contains two memblocks pool: small & large. It
 provides methods to allocate(pop) and deallocate(push)
 * memory blocks from the free list, as well as perform operations like merging
 * adjacent memory blocks between small pool & large pool.

 * @note This class is designed to be used in conjunction with the StreamContext
 class.
 */
class StreamFreeList {
  friend class StreamContext;
  static const constexpr size_t LARGE = false;
  static const constexpr size_t SMALL = true;

private:
  int device_id;
  cudaStream_t current_stream_;

  shm_ptr<StreamBlockList> stream_block_list_;
  shm_ptr<CachingAllocatorStats> stats_;

  std::array<bip_multimap<size_t, shm_ptr<MemBlock>>, 2> free_block_list_;

public:
  StreamFreeList(int device_id, cudaStream_t cuda_stream,
                 SharedMemory &shared_memory,
                 StreamBlockList &stream_block_list, CachingAllocatorStats &stats);

  StreamFreeList &operator=(const StreamFreeList &) = delete;
  StreamFreeList(const StreamFreeList &) = delete;

  MemBlock *PopBlock(ProcessLocalData &local, bool is_small, size_t nbytes,
                     size_t find_optimal_retry);

  MemBlock *PopBlock(ProcessLocalData &local, bool is_small,
                     ptrdiff_t addr_offset, size_t nbytes);

  MemBlock *PopBlock(ProcessLocalData &local, MemBlock *block);

  MemBlock *ResizeBlock(ProcessLocalData &local, MemBlock *block, size_t nbytes);

  MemBlock *PushBlock(ProcessLocalData &local, MemBlock *block);

  MemBlock *MaybeMergeAdj(ProcessLocalData &local, MemBlock *entry);

  bool CheckState(ProcessLocalData &local);

  void DumpFreeBlockList(ProcessLocalData &local, bool is_small,
                         std::ostream &out = std::cout);
};

class StreamContext {
public:
  int device_id;
  const cudaStream_t cuda_stream;
  StreamBlockList stream_block_list;
  StreamFreeList stream_free_list;

  StreamContext(SharedMemory &shared_memory, int device_id, 
                cudaStream_t cuda_stream, size_t small_block_nbytes, 
                CachingAllocatorStats &stats)
      : device_id(device_id), cuda_stream{cuda_stream},
        stream_block_list{device_id, cuda_stream, shared_memory,
                          small_block_nbytes, stats},
        stream_free_list{device_id, cuda_stream, shared_memory,
                         stream_block_list, stats} {}
  StreamContext &operator=(const StreamContext &) = delete;
  StreamContext(const StreamContext &) = delete;

  void MoveFreeBlockTo(ProcessLocalData &local, StreamContext &other_context);
};

} // namespace mpool