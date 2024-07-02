#pragma once

#include <mem_block.h>

namespace mpool {

class IAllocator {
public:
  virtual ~IAllocator() = default;
  virtual MemBlock *Alloc(size_t nbytes, size_t alignment,
                          cudaStream_t cuda_stream, size_t flags) = 0;
  virtual MemBlock *Realloc(MemBlock *block, size_t nbytes,
                            cudaStream_t cuda_stream, size_t flags) = 0;
  virtual void Free(const MemBlock *block, size_t flags) = 0;
  virtual std::byte *GetBasePtr() const = 0;
};
} // namespace mpool