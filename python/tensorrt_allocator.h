#pragma once

#include <mutex>
#include <unordered_map>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <NvInfer.h>
#pragma GCC diagnostic pop

#include <caching_allocator.h>
#include <mem_block.h>
#include <py_wrap.hpp>

#include <glog/logging.h>

namespace mpool {
class TensorRTAllocator : public nvinfer1::IGpuAllocator {
private:
  std::vector<PyCachingAllocator> allocators_;
  std::unordered_map<std::byte *, MemBlock *> mem_blocks_;
  std::mutex mutex_;

public:
  TensorRTAllocator(std::vector<PyCachingAllocator> allocators)
      : allocators_(std::move(allocators)) {}
  TRT_DEPRECATED void *allocate(uint64_t const size, uint64_t const alignment,
                 nvinfer1::AllocatorFlags const flags) noexcept override {
    return allocateAsync(size, alignment, flags, 0);
  }

  void *reallocate(void *const baseAddr, uint64_t alignment,
                   uint64_t newSize) noexcept override {
    LOG(FATAL) << "NOT IMPL: reallocate.";
    return nullptr;
  }

  TRT_DEPRECATED bool deallocate(void *const memory) noexcept override {
    return deallocateAsync(memory, 0);
  }

  void *allocateAsync(uint64_t const size, uint64_t const alignment,
                      nvinfer1::AllocatorFlags const flags,
                      cudaStream_t cuda_stream) noexcept override {
    CHECK_EQ(flags, 0);
    CHECK_LT(alignment, 512); /* current alignment is fixed */
    int device;
    CUDA_CALL(cudaGetDevice(&device));
    auto &allocator = allocators_.at(device);
    std::unique_lock lock{mutex_};
    auto *mem_block = allocator->Alloc(size, cuda_stream, true);
    auto *addr = allocator->GetBasePtr() + mem_block->addr_offset;
    mem_blocks_.insert({addr, mem_block});
    return addr;
  }

  virtual bool deallocateAsync(void* const memory, cudaStream_t cuda_stream) noexcept override {
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

  nvinfer1::InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return {"TensoRT Allocator", VERSION_MAJOR, VERSION_MINOR};
    }
};
}; // namespace mpool