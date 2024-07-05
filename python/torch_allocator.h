#pragma once

#include "mapping_region.h"
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

  void recordStream(const c10::DataPtr &data_ptr,
                    at::cuda::CUDAStream stream) override;

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

  void Dealloc(MemBlockExtraData *extra_data) {
    if (extra_data->require_device_sync) {
      at::cuda::stream_synchronize(c10::cuda::getCurrentCUDAStream(0));
    }
    if (extra_data->require_event_sync) {
      /* ACC */ CUDA_CALL(cudaEventDestroy(extra_data->event));
      DecerEventUsage();
    }
    auto &caching_allocator =
        caching_allocators_.at(extra_data->mem_block->device_id);
    caching_allocator->Free(extra_data->mem_block);
    delete extra_data;
  }
};

void OverridePyTorchAllocator(
    std::vector<PyCachingAllocator> caching_allocator);
void ResetPyTorchAllocator();

TorchAllocator *GetTorchAllocator();
} // namespace mpool