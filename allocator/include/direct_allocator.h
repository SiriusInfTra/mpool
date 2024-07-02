#pragma once

#include "pages_pool.h"
#include <allocator.h>
namespace mpool {

struct DirectAllocatorConfig {
  std::string log_prefix;
  std::string shm_name;
  size_t shm_nbytes;
  size_t va_range_scale;
  std::string belong_name;
  size_t small_block_nbytes;
  size_t align_nbytes;
};

class DirectAllocator : public IAllocator {
public:
  DirectAllocator(SharedMemory &shared_memory,
                                   PagesPool &page_pool,
                                   DirectAllocatorConfig config,
                                   bool first_init);
  virtual ~DirectAllocator() = default;
  virtual MemBlock *Alloc(size_t nbytes, size_t alignment,
                          cudaStream_t cuda_stream, size_t flags) override;
  virtual MemBlock *Realloc(MemBlock *block, size_t nbytes,
                            cudaStream_t cuda_stream, size_t flags) override;
  virtual void Free(MemBlock *block) override;
  virtual std::byte *GetBasePtr() const override;
};
} // namespace mpool