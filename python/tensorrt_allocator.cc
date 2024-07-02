#include "caching_allocator.h"
#include <tensorrt_allocator.h>

namespace mpool {

std::unordered_map<nvinfer1::IBuilder *, TensorRTAllocator *>
    TensorRTAllocator::builder_to_allocator;

TensorRTAllocator::TensorRTAllocator(std::vector<PyCachingAllocator> allocators)
    : allocators_(std::move(allocators)) {}

void *TensorRTAllocator::reallocate(void *const baseAddr, uint64_t alignment,
                                    uint64_t newSize) noexcept {
  CHECK_LE(alignment, 512); /* current alignment is fixed */
  int device;
  CUDA_CALL(cudaGetDevice(&device));
  auto &allocator = allocators_.at(device);
  std::unique_lock lock{mutex_};
  auto iter = mem_blocks_.find(static_cast<std::byte *>(baseAddr));
  if (iter == mem_blocks_.end()) {
    return nullptr;
  }
  auto *mem_block = iter->second;
  auto *resized_block =
      allocator->Realloc(mem_block, newSize, mem_block->stream, false);
  if (resized_block == nullptr) {
    return nullptr;
  }
  auto *resized_addr = allocator->GetBasePtr() + resized_block->addr_offset;
  return resized_addr;
}

void *TensorRTAllocator::allocateAsync(uint64_t const size,
                                       uint64_t const alignment,
                                       nvinfer1::AllocatorFlags const flags,
                                       cudaStream_t cuda_stream) noexcept {
  // CHECK_EQ(flags, 0);
  CHECK_LE(alignment, 512); /* current alignment is fixed */
  int device;
  CUDA_CALL(cudaGetDevice(&device));
  auto &allocator = allocators_.at(device);
  std::unique_lock lock{mutex_};
  auto *mem_block = allocator->Alloc(size, alignment, cuda_stream, CachingAllocator::ALLOC_TRY_EXPAND_VA);
  auto *addr = allocator->GetBasePtr() + mem_block->addr_offset;
  mem_blocks_.insert({addr, mem_block});
  return addr;
}

bool TensorRTAllocator::deallocateAsync(void *const memory,
                                        cudaStream_t cuda_stream) noexcept {
  std::unique_lock lock{mutex_};
  auto it = mem_blocks_.find(reinterpret_cast<std::byte *>(memory));
  if (it == mem_blocks_.end()) {
    return false;
  }
  auto *mem_block = it->second;
  if (mem_block->stream != cuda_stream) {
    return false;
  }
  auto &allocator = allocators_.at(mem_block->device_id);
  allocator->Free(mem_block);
  mem_blocks_.erase(it);
  return true;
}

nvinfer1::InterfaceInfo TensorRTAllocator::getInterfaceInfo() const noexcept {
  return {"TensoRT Allocator", VERSION_MAJOR, VERSION_MINOR};
}

} // namespace mpool