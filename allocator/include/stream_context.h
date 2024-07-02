#pragma once
#include <cstddef>
#include <stats.h>
#include <mapping_region.h>
#include <mem_block.h>
#include <shm.h>

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

  MemBlock *PopBlock(ProcessLocalData &local, bool is_small, ptrdiff_t addr_offset, size_t nbytes) {
    auto free_list = free_block_list_[is_small];
    
    // TODO log(n) algorithm
    MemBlock *mem_block = nullptr;
    for (auto iter = free_list.lower_bound(nbytes); iter != free_list.end(); ++iter) {
      auto *iter_block = iter->second.ptr(local.shared_memory_);
      if (iter_block->addr_offset <= addr_offset && iter_block->addr_offset + iter_block->nbytes >= addr_offset + nbytes) {
        mem_block = iter_block;
        break;
      }
    }
    mem_block = PopBlock(local, mem_block);
    auto *stream_block_list = stream_block_list_.ptr(local.shared_memory_);
    CHECK(mem_block != nullptr);
    if (mem_block->addr_offset < addr_offset) {
      auto remain = addr_offset - mem_block->addr_offset;
      auto *right_mem_block = stream_block_list->SplitBlock(local, mem_block, remain);
      PushBlock(local, mem_block);
      mem_block = right_mem_block;
    }
    if (mem_block->addr_offset + mem_block->nbytes > addr_offset + nbytes) {
      auto remain = mem_block->nbytes - nbytes;
      auto *right_mem_block = stream_block_list->SplitBlock(local, mem_block, remain);
      PushBlock(local, right_mem_block);
    }
    auto *stat = stats_.ptr(local.shared_memory_);
    stat->SetBlockFree(mem_block, false);
    free_list.erase(mem_block->iter_free_block_list);
  }
  

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

  StreamContext(ProcessLocalData &local, int device_id, 
                cudaStream_t cuda_stream, size_t small_block_nbytes, CachingAllocatorStats &stats)
      : device_id(device_id), cuda_stream{cuda_stream},
        stream_block_list{device_id, cuda_stream, local.shared_memory_,
                          small_block_nbytes, stats},
        stream_free_list{device_id, cuda_stream, local.shared_memory_,
                         stream_block_list, stats} {}
  StreamContext &operator=(const StreamContext &) = delete;
  StreamContext(const StreamContext &) = delete;

  void MoveFreeBlockTo(ProcessLocalData &local, StreamContext &other_context);
};

} // namespace mpool