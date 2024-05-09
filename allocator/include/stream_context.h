#pragma once

#include <unordered_set>

#include <mapping_region.h>
#include <mem_block.h>
#include <shm.h>

#include <cuda_runtime_api.h>

namespace mpool {

class StreamBlockList {
  friend class StreamContext;

private:
  cudaStream_t current_stream_;
  MappingRegion &mapping_region_;
  SharedMemory &shared_memory_;

  bip_list<shm_handle<MemBlock>> &all_block_list_;
  bip_list<shm_handle<MemBlock>> stream_block_list_;

  size_t small_block_nbytes_;

public:
  StreamBlockList(cudaStream_t cuda_stream, SharedMemory &shared_memory,
                  MappingRegion &mapping_region,
                  bip_list<shm_handle<MemBlock>> &all_block_list,
                  size_t small_block_nbytes);
  StreamBlockList &operator=(const StreamBlockList &) = delete;
  StreamBlockList(const StreamBlockList &) = delete;

  MemBlock *CreateEntryExpandVA(size_t nbytes);

  MemBlock *GetPrevEntry(MemBlock *entry);

  MemBlock *GetNextEntry(MemBlock *entry);

  MemBlock *SplitBlock(MemBlock *origin_entry, size_t remain);

  MemBlock *MergeMemEntry(MemBlock *first_block, MemBlock *secound_block);

  std::pair<bip_list<shm_handle<MemBlock>>::const_iterator,
            bip_list<shm_handle<MemBlock>>::const_iterator>
  Iterators() const;

  void DumpMemBlockColumns(std::ostream &out);
  void DumpMemBlock(std::ostream &out, MemBlock *block);

  void DumpStreamBlockList(std::ostream &out = std::cout);

  bool CheckState(bool check_global_block_list = false);
};

class StreamFreeList {
  friend class StreamContext;
  static const constexpr size_t LARGE = false;
  static const constexpr size_t SMALL = true;

private:
  SharedMemory &shared_memory_;
  cudaStream_t current_stream_;

  MappingRegion &mapping_region_;
  StreamBlockList &stream_block_list_;

  std::array<bip_multimap<size_t, shm_handle<MemBlock>>, 2> free_block_list_;

public:
  StreamFreeList(SharedMemory &shared_memory, cudaStream_t cuda_stream,
                 MappingRegion &mapping_region,
                 StreamBlockList &stream_block_list);

  StreamFreeList &operator=(const StreamFreeList &) = delete;
  StreamFreeList(const StreamFreeList &) = delete;

  MemBlock *PopBlock(bool is_small, size_t nbytes, size_t find_optimal_retry);

  MemBlock *PopBlock(MemBlock *block);

  MemBlock *PushBlock(MemBlock *block);

  MemBlock *MaybeMergeAdj(MemBlock *entry);

  bool CheckState();

  void DumpFreeBlockList(bool is_small, std::ostream &out = std::cout);
};

class StreamContext {
public:
  const cudaStream_t cuda_stream;
  StreamBlockList stream_block_list;
  StreamFreeList stream_free_list;

  StreamContext(cudaStream_t cuda_stream, SharedMemory &shared_memory,
                MappingRegion &mapping_region,
                bip_list<shm_handle<MemBlock>> &all_block_list,
                size_t small_block_nbytes)
      : cuda_stream{cuda_stream},
        stream_block_list{cuda_stream, shared_memory, mapping_region,
                          all_block_list, small_block_nbytes},
        stream_free_list{shared_memory, cuda_stream, mapping_region,
                         stream_block_list} {}
  StreamContext &operator=(const StreamContext &) = delete;
  StreamContext(const StreamContext &) = delete;

  void MoveFreeBlockTo(StreamContext &other_context);
};

} // namespace mpool