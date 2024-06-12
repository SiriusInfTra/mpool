#pragma once
#include <cstddef>
#include <iterator>
#include <string>
#include <iterator>

#include <stream_context.h>
#include <mapping_region.h>
#include <belong.h>
#include <pages.h>
#include <pages_pool.h>
#include <shm.h>
#include <util.h>
#include <mem_block.h>

#include <cuda_runtime_api.h>


namespace mpool {



struct CachingAllocatorConfig {
  std::string log_prefix;
  std::string shm_name;
  size_t shm_nbytes;
  size_t va_range_scale;
  std::string belong_name;
  size_t small_block_nbytes;
  size_t align_nbytes;
};



class CachingAllocator {
public:
  const Belong belong;
  const CachingAllocatorConfig config;

  static bool RemoveShm(const CachingAllocatorConfig &config) {
    return bip::shared_memory_object::remove(config.shm_name.c_str());
  }


private:
  PagesPool &page_pool_;
  SharedMemory &shared_memory_;
  MappingRegion mapping_region_;

  bip_list<shm_handle<MemBlock>> &all_block_list_;
  StreamContext &global_stream_context_;

  bip_unordered_map<cudaStream_t, shm_handle<StreamContext>>
      &stream_context_map_;

  StreamContext &GetStreamContext(cudaStream_t cuda_stream);

public:
  CachingAllocator(SharedMemory &shared_memory, PagesPool &page_pool,
                   CachingAllocatorConfig config, bool first_init);

  std::byte *GetBasePtr() const { return mapping_region_.GetBasePtr(); }

  std::byte *GetEndPtr() const { return mapping_region_.GetEndPtr(); }

  bool IsAllocatedPtr(std::byte *ptr) {
    return ptr >= mapping_region_.GetBasePtr() && ptr < mapping_region_.GetEndPtr();
  }

  ~CachingAllocator();

  MemBlock *AllocWithContext(size_t nbytes, StreamContext &stream_context);

  MemBlock *Alloc(size_t nbytes, cudaStream_t cuda_stream,
                  bool try_expand_VA = true);
  
  MemBlock *ReceiveMemBlock(shm_handle<MemBlock> handle) {
    auto *mem_block = handle.ptr(shared_memory_);
    CHECK_GE(mem_block->addr_offset, 0) << "Invalid handle";
    mem_block->ref_count++;
    return mem_block;
  }

  shm_handle<MemBlock> SendMemBlock(MemBlock *mem_block) {
    return {mem_block, shared_memory_};
  }

  void Free(const MemBlock *block);

  void EmptyCache();

  bool CheckState();
};


}