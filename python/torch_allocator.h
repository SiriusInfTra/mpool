#pragma once

#include <mpool/mapping_region.h>
#include <mpool/caching_allocator.h>
#include <mpool/mem_block.h>
#include <mpool/shm.h>

#include <atomic>
#include <cstddef>

#include <py_wrap.hpp>
#include <ATen/core/TensorBody.h>
#include <c10/core/Allocator.h>
#include <c10/core/Storage.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>

namespace mpool {

struct MemBlockExtraData {
  MemBlock *mem_block;

  bool from_sharing;
  bool require_device_sync;
  bool require_event_sync;
  cudaEvent_t event;

  int event_count;
  std::vector<c10::cuda::CUDAStream> stream_set;
};

class TorchAllocator : public c10::cuda::CUDACachingAllocator::CUDAAllocator {
public:
  std::vector<PyCachingAllocator> caching_allocators_;
  // PyTorch torch/csrc/CudaIPCTypes.h
  // unofficial limit on the number of recorded blocking interprocess events
  const constexpr static int64_t CUDA_IPC_MAXIMUM_EVENTS_TO_USE = 1000;
  std::unordered_multimap<cudaStream_t,
                          std::pair<MemBlockExtraData *, cudaEvent_t>>
      _stream_events;

private:
  std::unordered_map<std::byte *, MemBlock *> _mem_blocks;

  std::atomic<long> event_usage_counter;
  std::mutex lock_;
  std::shared_ptr<OOMObserver> caching_allocator_oom_observer_;
  std::vector<c10::cuda::CUDACachingAllocator::OutOfMemoryObserver>
      torch_oom_observers_;

  bool initialized_ = false;

public:
  TorchAllocator(std::vector<PyCachingAllocator> caching_allocator)
      : caching_allocators_(std::move(caching_allocator)),
        event_usage_counter(0) {
    caching_allocator_oom_observer_ = std::make_shared<OOMObserver>(
        [&](int device_id, cudaStream_t cuda_stream, OOMReason reason) {

        });
    for (auto &caching_allocator : caching_allocators_) {
      caching_allocator->AddOOMObserver(caching_allocator_oom_observer_);
    }
  }

  void OnOutOfMemory(int device_id, size_t allocated_nbytes,
                     size_t device_total, size_t device_free) {
    for (auto &observer : torch_oom_observers_) {
      auto &caching_allocator = caching_allocators_.at(device_id);
      size_t allocated_nbytes = caching_allocator->belong.GetPagesNum();
      size_t device_total = caching_allocator->page_pool.config.pool_nbytes;
      size_t device_free = caching_allocator->GetDeviceFreeNbytes();
      observer(device_id, allocated_nbytes, device_total, device_free);
    }
  }

  ~TorchAllocator() {
    for (auto &caching_allocator : caching_allocators_) {
      caching_allocator->RemoveOOMObserver(caching_allocator_oom_observer_);
    }
  }

  virtual c10::DataPtr allocate(size_t nbytes) override;
  virtual c10::DeleterFnPtr raw_deleter() const override;
  virtual void copy_data(void* dest, const void* src, std::size_t count) const override;
  virtual void *raw_alloc(size_t nbytes) override;
  virtual void *raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override;
  virtual void raw_delete(void *ptr) override;
  virtual void init(int device_count) override;
  virtual bool initialized() override;
  virtual void emptyCache() override;
  virtual void setMemoryFraction(double fraction, c10::DeviceIndex device) override;
  virtual void cacheInfo(c10::DeviceIndex device, size_t *largestBlock) override;


  virtual std::shared_ptr<void> getIpcDevPtr(std::string handle) override;
  virtual c10::cuda::CUDACachingAllocator::ShareableHandle shareIpcHandle(void *ptr) override;
  virtual void *getBaseAllocation(void *ptr, size_t *size) override;
  virtual void enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access) override;

  virtual void beginAllocateToPool(c10::DeviceIndex device,
                                   c10::cuda::MempoolId_t mempool_id,
                                   std::function<bool(cudaStream_t)> filter) override;
  virtual void endAllocateToPool(c10::DeviceIndex device,
                                 c10::cuda::MempoolId_t mempool_id) override;
  virtual void releasePool(c10::DeviceIndex device, c10::cuda::MempoolId_t mempool_id) override;


  virtual void recordStream(const c10::DataPtr &data_ptr,
                    at::cuda::CUDAStream stream) override;
  virtual void recordHistory(
      bool enabled,
      c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      c10::cuda::CUDACachingAllocator::RecordContext when) override;

  virtual c10::cuda::CUDACachingAllocator::DeviceStats
  getDeviceStats(c10::DeviceIndex device) override;
  virtual void resetAccumulatedStats(c10::DeviceIndex device) override;
  virtual void resetPeakStats(c10::DeviceIndex device) override;
  virtual c10::cuda::CUDACachingAllocator::SnapshotInfo snapshot() override;
  virtual void attachOutOfMemoryObserver(
      c10::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) override;
  virtual cudaError_t memcpyAsync(
      void* dst,
      int dstDevice,
      const void* src,
      int srcDevice,
      size_t count,
      cudaStream_t stream,
      bool p2p_enabled) override;
  virtual std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState> getCheckpointState(
      c10::DeviceIndex device,
      c10::cuda::MempoolId_t id) override;
  virtual c10::cuda::CUDACachingAllocator::CheckpointDelta setCheckpointPoolState(
      c10::DeviceIndex device,
      std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState> pps) override;
  virtual void attachAllocatorTraceTracker(c10::cuda::CUDACachingAllocator::AllocatorTraceTracker tracker) override;
  std::string name() override;

  c10::intrusive_ptr<c10::StorageImpl>
  ReceiveHandle(int device_id, shm_ptr<MemBlock> handle, size_t storage_size,
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

  void ProcessEvent();

  void Dealloc(MemBlockExtraData *extra_data);
};

void OverridePyTorchAllocator(
    std::vector<PyCachingAllocator> caching_allocator);
void ResetPyTorchAllocator();

TorchAllocator *GetTorchAllocator();
} // namespace mpool