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

const static constexpr bool VERBOSE = true;
const static constexpr bool CHECK_STATE = true;
const static constexpr bool MORE_CHECK_STATE = true;
const static constexpr bool MORE_MORE_CHECK_STATE = true;

struct CachingAllocatorConfig {
  std::string log_prefix;
  std::string shm_name;
  size_t shm_nbytes;
  size_t va_range_scale;
  Belong belong;
  size_t small_block_nbytes;
  size_t align_nbytes;
};



class CachingAllocator {
public:
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

  ~CachingAllocator();

  MemBlock *AllocWithContext(size_t nbytes, StreamContext &stream_context);

  MemBlock *Alloc(size_t nbytes, cudaStream_t cuda_stream,
                  bool try_expand_VA = true);

  void Free(MemBlock *block);

  void EmptyCache(cudaStream_t cuda_stream);

  bool CheckState();
};
}