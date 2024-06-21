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
  int device_id;
  cudaStream_t current_stream_;

  bip_list<shm_ptr<MemBlock>> stream_block_list_;

  size_t small_block_nbytes_;

public:
  StreamBlockList(int device_id, cudaStream_t cuda_stream, SharedMemory &shared_memory,
                  size_t small_block_nbytes);
  StreamBlockList &operator=(const StreamBlockList &) = delete;
  StreamBlockList(const StreamBlockList &) = delete;

  MemBlock *CreateEntryExpandVA(ProcessLocalData &local, size_t nbytes);

  MemBlock *GetPrevEntry(ProcessLocalData &local, MemBlock *entry);

  MemBlock *GetNextEntry(ProcessLocalData &local, MemBlock *entry);

  MemBlock *SplitBlock(ProcessLocalData &local, MemBlock *origin_entry, size_t remain);

  MemBlock *MergeMemEntry(ProcessLocalData &local, MemBlock *first_block, MemBlock *secound_block);

  std::pair<bip_list<shm_ptr<MemBlock>>::const_iterator,
            bip_list<shm_ptr<MemBlock>>::const_iterator>
  Iterators() const;

  void DumpMemBlockColumns(std::ostream &out);
  void DumpMemBlock(ProcessLocalData &local, std::ostream &out, MemBlock *block);

  void DumpStreamBlockList(ProcessLocalData &local, std::ostream &out = std::cout);

  bool CheckState(ProcessLocalData &local, bool check_global_block_list = false);
};

class StreamFreeList {
  friend class StreamContext;
  static const constexpr size_t LARGE = false;
  static const constexpr size_t SMALL = true;

private:
  int device_id;
  cudaStream_t current_stream_;

  shm_ptr<StreamBlockList> stream_block_list_;

  std::array<bip_multimap<size_t, shm_ptr<MemBlock>>, 2> free_block_list_;

public:
  StreamFreeList(int device_id, cudaStream_t cuda_stream, SharedMemory &shared_memory, 
                 StreamBlockList &stream_block_list);

  StreamFreeList &operator=(const StreamFreeList &) = delete;
  StreamFreeList(const StreamFreeList &) = delete;

  MemBlock *PopBlock(ProcessLocalData &local, bool is_small, size_t nbytes, size_t find_optimal_retry);

  MemBlock *PopBlock(ProcessLocalData &local, MemBlock *block);

  MemBlock *PushBlock(ProcessLocalData &local, MemBlock *block);

  MemBlock *MaybeMergeAdj(ProcessLocalData &local, MemBlock *entry);

  bool CheckState(ProcessLocalData &local);

  void DumpFreeBlockList(ProcessLocalData &local, bool is_small, std::ostream &out = std::cout);
};

class StreamContext {
public:
  int device_id;
  const cudaStream_t cuda_stream;
  StreamBlockList stream_block_list;
  StreamFreeList stream_free_list;

  StreamContext(ProcessLocalData &local, int device_id, cudaStream_t cuda_stream,
                size_t small_block_nbytes)
      : device_id(device_id), cuda_stream{cuda_stream},
        stream_block_list{device_id, cuda_stream, local.shared_memory_, small_block_nbytes},
        stream_free_list{device_id, cuda_stream, local.shared_memory_, stream_block_list} {}
  StreamContext &operator=(const StreamContext &) = delete;
  StreamContext(const StreamContext &) = delete;

  void MoveFreeBlockTo(ProcessLocalData &local, StreamContext &other_context);
};

} // namespace mpool