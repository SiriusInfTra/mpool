#pragma once

#include <mem_block.h>

namespace mpool {

class IAllocator {
public:
  virtual ~IAllocator() = default;

  /**
   * @brief Allocates a memory block with the specified size and alignment.
   *
   * This function is used to allocate a memory block with the specified
   * number of bytes and alignment. The allocation can be performed on a
   * specific CUDA stream and with additional flags.
   *
   * @param nbytes The number of bytes to allocate.
   * @param alignment The alignment of the memory block.
   * @param cuda_stream The CUDA stream on which the allocation should be
   * performed.
   * @param flags Additional flags for the allocation.
   * @return A pointer to the allocated memory block, or nullptr if the
   * allocation failed.
   */
  virtual MemBlock *Alloc(size_t nbytes, size_t alignment,
                          cudaStream_t cuda_stream, size_t flags) = 0;

  /**
   * @brief Reallocates a memory block.
   *
   * This function reallocates the memory block pointed to by `block` to a new
   * size specified by `nbytes`. The reallocation is performed asynchronously on
   * the CUDA stream `cuda_stream`. The `flags` parameter specifies additional
   * options for the reallocation.
   *
   * @param block The memory block to be reallocated.
   * @param nbytes The new size of the memory block in bytes.
   * @param cuda_stream The CUDA stream on which the reallocation should be
   * performed.
   * @param flags Additional options for the reallocation.
   * @return A pointer to the reallocated memory block, or nullptr if the
   * reallocation failed.
   */
  virtual MemBlock *Realloc(MemBlock *block, size_t nbytes,
                            cudaStream_t cuda_stream, size_t flags) = 0;

  /**
   * @brief Frees a memory block.
   *
   * This function is used to deallocate a previously allocated memory block.
   *
   * @param block A pointer to the memory block to be freed.
   * @param flags Additional flags that control the deallocation process.
   *              These flags are implementation-specific.
   */
  virtual void Free(const MemBlock *block, size_t flags) = 0;

  /**
   * @brief Returns the base pointer of the allocator.
   *
   * This function returns the base pointer of the allocator, which is the
   * starting address of the allocated memory block.
   *
   * @return A pointer to the base address of the allocator.
   */
  virtual std::byte *GetBasePtr() const = 0;
};
} // namespace mpool