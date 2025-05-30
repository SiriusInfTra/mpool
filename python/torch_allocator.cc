#include <Python.h>

#include <mpool/caching_allocator.h>
#include <mpool/mem_block.h>
#include <mpool/shm.h>
#include <torch_allocator.h>
#include <py_wrap.hpp>

#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAException.h>



#include <cstddef>
#include <c10/core/Storage.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/Exception.h>
#include <chrono>
#include <torch/csrc/Export.h>
#include <torch/csrc/Storage.h>
#include <torch/csrc/utils.h>
#include <utility>

namespace mpool {

c10::CachingDeviceAllocator::Stat &
GetStat(c10::CachingDeviceAllocator::StatArray &arr, bool is_small) {
  if (is_small) {
    return arr.at(static_cast<size_t>(
        c10::CachingDeviceAllocator::StatType::SMALL_POOL));
  } else {
    return arr.at(static_cast<size_t>(
        c10::CachingDeviceAllocator::StatType::LARGE_POOL));
  }
}

void SetStat(c10::CachingDeviceAllocator::Stat &stat,
             const Stat &caching_allocator_stat) {
  stat.current = caching_allocator_stat.current;
  stat.peak = caching_allocator_stat.peak;
  stat.allocated = caching_allocator_stat.allocated_free[0];
  stat.freed = caching_allocator_stat.allocated_free[1];
}

class Recorder {
private:
  std::chrono::steady_clock::time_point t0;
  std::string name;

public:
  Recorder(std::string name)
      : t0(std::chrono::steady_clock::now()), name(std::move(name)) {}
  ~Recorder() {
    auto t1 = std::chrono::steady_clock::now();
    auto dur =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    if (dur > 20) {
      LOG(WARNING) << name << " cost " << dur << "ms.";
    }
  }
};

static std::shared_ptr<TorchAllocator> torch_allocator_ = nullptr;

TorchAllocator *GetTorchAllocator() {
  TORCH_CHECK(torch_allocator_ != nullptr,
                  "TorchAllocator not initialized.");
  return torch_allocator_.get();
}

void OverridePyTorchAllocator(
    std::vector<PyCachingAllocator> caching_allocator) {
  if (auto *allocator = c10::cuda::CUDACachingAllocator::allocator.load();
      allocator != nullptr && allocator->initialized()) {
    TORCH_WARN_ONCE(
        "c10::cuda::CUDACachingAllocator::allocator is initialized!");
  }
  if (torch_allocator_ != nullptr) {
    TORCH_WARN_ONCE("already set torch allocator!");
  }
  torch_allocator_.reset(new TorchAllocator(std::move(caching_allocator)));
  c10::cuda::CUDACachingAllocator::allocator.store(torch_allocator_.get());
  TORCH_WARN("Successfully override pytorch default allocator.");
}

void ResetPyTorchAllocator() { torch_allocator_.reset(); }

void RawDeletePtr(void *ptr) {
  TORCH_CHECK(torch_allocator_ != nullptr, "Torch allocator is not set.");
  if (ptr == nullptr) {
    TORCH_WARN("ignore nullptr");
    return;
  }
  torch_allocator_->raw_delete(ptr);
}

void NullDeletePtr(void *null) {}

void BlockDeletePtr(void *_extra_data) {
  Recorder recorder("BlockDeletePtr");
  TORCH_CHECK_NOTNULL(torch_allocator_);
  TORCH_CHECK_NOTNULL(_extra_data);
  auto *extra_data = reinterpret_cast<MemBlockExtraData *>(_extra_data);
  if (!extra_data->stream_set.empty()) {
    auto streams = std::move(extra_data->stream_set);
    AT_ASSERT(extra_data->stream_set.empty());
    c10::DeviceGuard device_guard{c10::Device{
        c10::kCUDA,
        static_cast<c10::DeviceIndex>(extra_data->mem_block->device_id)}};
    cudaEvent_t event;
    // TODO EventPool
    /* ACC */ C10_CUDA_CHECK(
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    for (auto &&stream : streams) {
      /* ACC */ C10_CUDA_CHECK(cudaEventRecord(event, stream.stream()));
      extra_data->event_count++;
      torch_allocator_->_stream_events.insert(
          {stream.stream(), {extra_data, event}});
    }
  } else {
    torch_allocator_->Dealloc(extra_data);
  }
}

c10::DataPtr TorchAllocator::allocate(size_t nbytes) {
  Recorder recorder("allocate");
  auto cur_device = at::cuda::current_device();
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(cur_device);
  c10::DataPtr data_ptr;
  if (nbytes == 0) {
    data_ptr = {nullptr, nullptr, NullDeletePtr,
                c10::Device(c10::DeviceType::CUDA, cur_device)};
  } else {
    auto &caching_allocator =
        const_cast<PyCachingAllocator &>(caching_allocators_.at(cur_device));
    auto *block = caching_allocator->Alloc(nbytes, 512, stream, CachingAllocator::ALLOC_TRY_EXPAND_VA);
    void *addr = reinterpret_cast<void*>(reinterpret_cast<int64_t>(caching_allocator->GetBasePtr()) + block->addr_offset);
    auto *extra_data = new MemBlockExtraData{
        .mem_block = block, .from_sharing = false, .event_count = 0};
    data_ptr = {addr, extra_data, BlockDeletePtr,
                c10::Device(c10::DeviceType::CUDA, cur_device)};
  }
  return data_ptr;
}

c10::DeleterFnPtr TorchAllocator::raw_deleter() const { return &RawDeletePtr; }

void TorchAllocator::copy_data(void *dest, const void *src, std::size_t count) const {
  /* ACC */ C10_CUDA_CHECK(cudaMemcpy(dest, src, count, cudaMemcpyKind::cudaMemcpyDeviceToDevice));
}



void *TorchAllocator::raw_alloc(size_t nbytes) {
  int device;
  /* ACC */ C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device);
  return raw_alloc_with_stream(nbytes, stream);
}

void *TorchAllocator::raw_alloc_with_stream(size_t nbytes,
                                            cudaStream_t stream) {
  int device;
  /* ACC */ C10_CUDA_CHECK(cudaGetDevice(&device));
  auto &caching_allocator = caching_allocators_[device];
  auto *block = caching_allocator->Alloc(nbytes, 512, stream, CachingAllocator::ALLOC_TRY_EXPAND_VA);
  auto *addr = caching_allocator->GetBasePtr() + block->addr_offset;
  _mem_blocks.emplace(addr, block);
  // LOG(WARNING) << "raw_alloc_with_stream " << nbytes << " " << stream;
  return addr;
}

void TorchAllocator::raw_delete(void *ptr) {
  auto iter = _mem_blocks.find(reinterpret_cast<std::byte *>(ptr));
  TORCH_CHECK(iter != _mem_blocks.end(),
              "Trying to free a pointer not allocated here.");
  auto *block = iter->second;
  caching_allocators_[block->device_id]->Free(block);
  _mem_blocks.erase(iter);
  // LOG(WARNING) << "raw_delete " << ptr;
}

void TorchAllocator::init(int device_count) {
  /* current do nothing */
  initialized_ = true;
  // LOG(WARNING) << "init " << device_count;
}

bool TorchAllocator::initialized() {
  // LOG(WARNING) << "initialized " << initialized_;
  return initialized_;
}

void TorchAllocator::emptyCache() {
  ProcessEvent();
  for (auto &caching_allocator : caching_allocators_) {
    caching_allocator->EmptyCache();
  }
}

void TorchAllocator::setMemoryFraction(double fraction, c10::DeviceIndex device) {
  TORCH_CHECK_NOT_IMPLEMENTED(false, "setMemoryFraction");
  throw std::runtime_error("NOT IMPL");
}

void *TorchAllocator::getBaseAllocation(void *ptr, size_t *size) {
  TORCH_CHECK_NOT_IMPLEMENTED(false, "getBaseAllocation");
  throw std::runtime_error("NOT IMPL");
}

void TorchAllocator::enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access) {
  TORCH_CHECK_NOT_IMPLEMENTED(false, "enablePeerAccess");
  throw std::runtime_error("NOT IMPL");
}

void TorchAllocator::beginAllocateToPool(c10::DeviceIndex device,
                                   c10::cuda::MempoolId_t mempool_id,
                                   std::function<bool(cudaStream_t)> filter) {
  TORCH_CHECK_NOT_IMPLEMENTED(false, "beginAllocateToPool");
  throw std::runtime_error("NOT IMPL");
}

void TorchAllocator::endAllocateToPool(c10::DeviceIndex device,
                                       c10::cuda::MempoolId_t id) {
  TORCH_CHECK_NOT_IMPLEMENTED(false, "endAllocateToPool");
  throw std::runtime_error("NOT IMPL");
}

void TorchAllocator::releasePool(c10::DeviceIndex device,
                                 c10::cuda::MempoolId_t id) {
  TORCH_CHECK_NOT_IMPLEMENTED(false, "releasePool");
  throw std::runtime_error("NOT IMPL");
}

c10::cuda::CUDACachingAllocator::CheckpointDelta TorchAllocator::setCheckpointPoolState(c10::DeviceIndex device,
                                       std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState> pps) {
  TORCH_CHECK_NOT_IMPLEMENTED(false, "setCheckpointPoolState");
  throw std::runtime_error("NOT IMPL");
}

std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState>
 TorchAllocator::getCheckpointState(c10::DeviceIndex device,
                                        c10::cuda::MempoolId_t id) {
  TORCH_CHECK_NOT_IMPLEMENTED(false, "getCheckpointState");
  throw std::runtime_error("NOT IMPL");
}

void TorchAllocator::cacheInfo(c10::DeviceIndex dev_id, size_t *largestBlock) {
  TORCH_CHECK_NOT_IMPLEMENTED(false, "cacheInfo");
  throw std::runtime_error("NOT IMPL");
}

void TorchAllocator::recordStream(const c10::DataPtr &data_ptr,
                                  at::cuda::CUDAStream stream) {
  auto *extra_data =
      reinterpret_cast<MemBlockExtraData *>(data_ptr.get_context());
  if (extra_data->mem_block->stream == stream.stream() ||
      extra_data->from_sharing) {
    return;
  }
  // TODO set
  extra_data->stream_set.push_back(stream);
  extra_data->event_count++;
}


void TorchAllocator::ProcessEvent() {
  for (auto it = _stream_events.begin(); it != _stream_events.end();) {
    auto [extra_data, event] = it->second;
    cudaError_t err = C10_CUDA_ERROR_HANDLED(cudaEventQuery(event));
    if (err == cudaErrorNotReady) {
      cudaGetLastError();
      it++;
    } else if (err != cudaSuccess) {
      /* ACC */ C10_CUDA_CHECK(err);
    } else {
      /* ACC */ C10_CUDA_CHECK(cudaEventDestroy(event));
      extra_data->event_count--;
      if (extra_data->event_count == 0) {
        torch_allocator_->Dealloc(extra_data);
      }
      it = _stream_events.erase(it);
    }
  }
}

c10::cuda::CUDACachingAllocator::DeviceStats
TorchAllocator::getDeviceStats(c10::DeviceIndex device) {
  c10::cuda::CUDACachingAllocator::DeviceStats stats;
  auto &allocator = caching_allocators_.at(device);
  auto &caching_allocator_stats = allocator->GetStats();
  for (bool is_small : {false, true}) {
    auto &stat = GetStat(stats.allocated_bytes, is_small);
    SetStat(stat, caching_allocator_stats.mem_block_nbytes[is_small]);
  }
  for (bool is_small : {false, true}) {
    auto &stat = GetStat(stats.active, is_small);
    SetStat(stat, caching_allocator_stats.mem_block_count[is_small]);
  }
  for (bool is_small : {false, true}) {
    auto &stat = GetStat(stats.reserved_bytes, is_small);
    SetStat(stat, caching_allocator_stats.mem_block_count[is_small]);
    stat.allocated *= allocator->page_pool.config.page_nbytes;
    stat.freed *= allocator->page_pool.config.page_nbytes;
    stat.current *= allocator->page_pool.config.page_nbytes;
    stat.peak *= allocator->page_pool.config.page_nbytes;
  }
  return stats;
}

void TorchAllocator::resetAccumulatedStats(c10::DeviceIndex device) {
  TORCH_WARN_ONCE("resetAccumulatedStats has no effects.");
}

void TorchAllocator::resetPeakStats(c10::DeviceIndex device) {
  auto &allocator = caching_allocators_.at(device);
  allocator->ResetPeakStats();
}

c10::cuda::CUDACachingAllocator::SnapshotInfo TorchAllocator::snapshot() {
  c10::cuda::CUDACachingAllocator::SnapshotInfo snapshot_info;
  for (auto &caching_allocator : caching_allocators_) {
    auto [begin, end] = caching_allocator->GetAllBlocks();
    for (auto iter = begin; iter != end; ++iter) {
      auto *mem_block = *iter;
      snapshot_info.segments.push_back(
          c10::cuda::CUDACachingAllocator::SegmentInfo{
              .device = static_cast<c10::DeviceIndex>(mem_block->device_id),
              .address = reinterpret_cast<uintptr_t>(
                  caching_allocator->GetBasePtr() + mem_block->addr_offset),
              .total_size = static_cast<size_t>(mem_block->nbytes),
              .requested_size = static_cast<size_t>(mem_block->nbytes),
              .allocated_size = static_cast<size_t>(mem_block->nbytes),
              .active_size = static_cast<size_t>(mem_block->nbytes),
              .stream = mem_block->stream,
              .is_large = !mem_block->is_small,
          });
    }
  }
  throw std::runtime_error("NOT IMPL");
}

std::shared_ptr<void> TorchAllocator::getIpcDevPtr(std::string handle) {
  TORCH_CHECK_NOT_IMPLEMENTED(false, "getIpcDevPtr");
  throw std::runtime_error("NOT IMPL");
}

c10::cuda::CUDACachingAllocator::ShareableHandle TorchAllocator::shareIpcHandle(
    void *ptr) {
  TORCH_CHECK_NOT_IMPLEMENTED(false, "shareIpcHandle");
  throw std::runtime_error("NOT IMPL");
}

void TorchAllocator::recordHistory(
      bool enabled,
      c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      c10::cuda::CUDACachingAllocator::RecordContext when) {
  TORCH_CHECK_NOT_IMPLEMENTED(false, "recordHistory");
  throw std::runtime_error("NOT IMPL");
}

void TorchAllocator::attachOutOfMemoryObserver(
    c10::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) {
  TORCH_CHECK_NOT_IMPLEMENTED(false, "attachOutOfMemoryObserver");
  throw std::runtime_error("NOT IMPL");
}

void TorchAllocator::attachAllocatorTraceTracker(
    c10::cuda::CUDACachingAllocator::AllocatorTraceTracker tracker) {
  TORCH_CHECK_NOT_IMPLEMENTED(false, "attachAllocatorTraceTracker");
  throw std::runtime_error("NOT IMPL");
}

cudaError_t TorchAllocator::memcpyAsync(void *dst, int dstDevice, const void *src,
                                 int srcDevice, size_t count, cudaStream_t stream,
                                 bool p2p_enabled) {
  if (p2p_enabled || srcDevice == dstDevice) {
    return cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
  }
  return cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
}




std::string TorchAllocator::name() { return "TorchAllocator"; }

c10::intrusive_ptr<c10::StorageImpl>
TorchAllocator::ReceiveHandle(int device_id, shm_ptr<MemBlock> handle,
                              size_t storage_size,
                              MemBlockExtraData *extra_data) {
  auto &caching_allocator = caching_allocators_[device_id];
  auto *mem_block = caching_allocator->ReceiveMemBlock(handle);
  extra_data->mem_block = mem_block;
  TORCH_CHECK_LE(storage_size, mem_block->nbytes);

  auto cur_device = at::cuda::current_device();
  auto *addr = caching_allocator->GetBasePtr() + mem_block->addr_offset;
  c10::DataPtr data_ptr = {addr, extra_data, BlockDeletePtr,
                           c10::Device(c10::DeviceType::CUDA, cur_device)};
  auto base = c10::make_intrusive<at::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), storage_size, std::move(data_ptr),
      /*allocator=*/this,
      /*resizable=*/false);
  return base;
}
shm_ptr<MemBlock> TorchAllocator::SendHandle(c10::StorageImpl *storage) {
  auto &caching_allocator = caching_allocators_[storage->device().index()];
  auto *mem_block =
      reinterpret_cast<MemBlockExtraData *>(storage->data_ptr().get_context())
          ->mem_block;
  return caching_allocator->SendMemBlock(mem_block);
}

void TorchAllocator::Dealloc(MemBlockExtraData *extra_data) {
  if (extra_data->require_device_sync) {
    at::cuda::stream_synchronize(c10::cuda::getCurrentCUDAStream(0));
  }
  if (extra_data->require_event_sync) {
    /* ACC */ C10_CUDA_CHECK(cudaEventDestroy(extra_data->event));
    DecerEventUsage();
  }
  auto &caching_allocator =
      caching_allocators_.at(extra_data->mem_block->device_id);
  caching_allocator->Free(extra_data->mem_block);
  delete extra_data;
}
} // namespace mpool