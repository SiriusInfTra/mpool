#pragma once

#include <atomic>
#include <cstddef>
#include <unordered_map>

#include <caching_allocator.h>
#include <mem_block.h>
#include <py_wrap.hpp>
#include <shm.h>

#include <ATen/core/TensorBody.h>
#include <c10/core/Allocator.h>
#include <c10/core/Storage.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>

namespace mpool {

struct MemBlockExtraData {
  bool require_device_sync;
  bool require_event_sync;
  cudaEvent_t event;
  MemBlock *mem_block;
};
class TorchAllocator : public c10::cuda::CUDACachingAllocator::CUDAAllocator {
public:
  PyCachingAllocator _caching_allocator;
  // PyTorch torch/csrc/CudaIPCTypes.h
  // unofficial limit on the number of recorded blocking interprocess events
  const constexpr static int64_t CUDA_IPC_MAXIMUM_EVENTS_TO_USE = 1000;

private:
  std::unordered_map<std::byte *, MemBlock *> _mem_blocks;
  std::atomic<long> event_usage_counter;
  std::mutex lock_;
  bool initialized_ = false;

public:
  TorchAllocator(PyCachingAllocator caching_allocator)
      : _caching_allocator(std::move(caching_allocator)),
        event_usage_counter(0) {}

  c10::DataPtr allocate(size_t nbytes) const override;
  c10::DeleterFnPtr raw_deleter() const override;

  void *raw_alloc(size_t nbytes) override;
  void *raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override;
  void raw_delete(void *ptr) override;
  void init(int device_count) override;
  bool initialized() override;
  void emptyCache() override;
  void setMemoryFraction(double fraction, int device) override;
  void cacheInfo(int dev_id, size_t *largestBlock) override;
  void *getBaseAllocation(void *ptr, size_t *size) override;

  void recordStream(const c10::DataPtr &, at::cuda::CUDAStream stream) override;

  c10::cuda::CUDACachingAllocator::DeviceStats
  getDeviceStats(int device) override;
  void resetAccumulatedStats(int device) override;
  void resetPeakStats(int device) override;
  c10::cuda::CUDACachingAllocator::SnapshotInfo snapshot() override;
  void notifyCaptureBegin(int device, c10::cuda::CaptureId_t graph_id,
                          c10::cuda::MempoolId_t mempool_id) override;
  void notifyCaptureAboutToEnd(int device,
                               c10::cuda::CaptureId_t graph_id) override;
  void notifyCaptureEnded(int device, c10::cuda::CaptureId_t graph_id) override;
  void notifyCaptureDestroy(int device,
                            c10::cuda::MempoolId_t mempool_id) override;
  std::shared_ptr<void> getIpcDevPtr(std::string handle) override;
  void recordHistory(
      bool enabled,
      c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
      size_t alloc_trace_max_entries, bool alloc_trace_record_context) override;
  void attachOutOfMemoryObserver(
      c10::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) override;
  bool needsPoolSpecificPeerAccess() override;

  std::string name() override;

  c10::intrusive_ptr<c10::StorageImpl>
  ReceiveHandle(shm_ptr<MemBlock> handle, size_t storage_size,
                MemBlockExtraData *extra_data);
  shm_ptr<MemBlock> SendHandle(c10::StorageImpl *storage);

  bool IncerEventUsage() {
    if (event_usage_counter.load(std::memory_order_relaxed) >=
        CUDA_IPC_MAXIMUM_EVENTS_TO_USE) {
      return false;
    }
    event_usage_counter.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  void DecerEventUsage() {
    event_usage_counter.fetch_sub(0, std::memory_order_relaxed);
  }
};

void OverridePyTorchAllocator(PyCachingAllocator caching_allocator);
void ResetPyTorchAllocator();

TorchAllocator *GetTorchAllocator();
} // namespace mpool