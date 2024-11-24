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

#include <mpool/logging_is_spdlog.h>

namespace mpool {
class TensorRTAllocator : public nvinfer1::IGpuAsyncAllocator {
private:
  std::vector<PyCachingAllocator> allocators_;
  std::unordered_map<std::byte *, MemBlock *> mem_blocks_;
  std::mutex mutex_;
  static std::unordered_map<nvinfer1::IBuilder *, TensorRTAllocator *>
      builder_to_allocator;

public:
  TensorRTAllocator(std::vector<PyCachingAllocator> allocators);

  static void SetIGPUAllocator(nvinfer1::IBuilder *builder,
                               std::vector<PyCachingAllocator> allocators) {
    auto allocator = new TensorRTAllocator(allocators);
    builder->setGpuAllocator(allocator);
    builder_to_allocator[builder] = allocator;
  }

  static void UnsetIGPUAllocator(nvinfer1::IBuilder *builder) {
    builder->setGpuAllocator(nullptr);
    if (auto iter = builder_to_allocator.find(builder);
        iter != builder_to_allocator.cend()) {
      delete iter->second;
      builder_to_allocator.erase(iter);
    }
  }

  void *reallocate(void *const baseAddr, uint64_t alignment,
                   uint64_t newSize) noexcept override;

  void *allocateAsync(uint64_t const size, uint64_t const alignment,
                      nvinfer1::AllocatorFlags const flags,
                      cudaStream_t cuda_stream) noexcept override;

  virtual bool deallocateAsync(void *const memory,
                               cudaStream_t cuda_stream) noexcept override;

  nvinfer1::InterfaceInfo getInterfaceInfo() const noexcept override;
};
}; // namespace mpool