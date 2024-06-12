#include <Python.h>
#include "torch_allocator.h"
#include "caching_allocator.h"
#include "mem_block.h"
#include <atomic>
#include <c10/core/Storage.h>
#include <c10/util/logging_is_not_google_glog.h>
#include <cstddef>
#include <memory>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/Exception.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/Storage.h>
#include <torch/csrc/utils.h>


namespace mpool {

static std::unique_ptr<TorchAllocator> torch_allocator_ = nullptr;

TorchAllocator *GetTorchAllocator() {
  THPUtils_assert(torch_allocator_ != nullptr, "TorchAllocator not initialized.");
  return torch_allocator_.get();
}

void OverridePyTorchAllocator(CachingAllocator *caching_allocator) {
  if (auto *allocator = c10::cuda::CUDACachingAllocator::allocator.load(); allocator != nullptr && allocator->initialized()) {
    TORCH_WARN_ONCE("c10::cuda::CUDACachingAllocator::allocator is initialized!");
  }
  if (torch_allocator_ != nullptr) {
    TORCH_WARN_ONCE("already set torch allocator!");
  }
  torch_allocator_.reset(new TorchAllocator(*caching_allocator));
  c10::cuda::CUDACachingAllocator::allocator.store(torch_allocator_.get());
}


void RawDeletePtr(void *ptr) {
  TORCH_CHECK(torch_allocator_ != nullptr, "Torch allocator is not set.");
  torch_allocator_->raw_delete(ptr);
}

void BlockDeletePtr(void *mem_block_ptr) {
  TORCH_CHECK(torch_allocator_ != nullptr, "Torch allocator is not set.");
  auto *block = reinterpret_cast<MemBlock*>(mem_block_ptr);
  torch_allocator_->_caching_allocator.Free(block);
}

c10::DataPtr TorchAllocator::allocate(size_t nbytes) const {
  auto cur_device = at::cuda::current_device();
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(cur_device);
  auto *block = _caching_allocator.Alloc(nbytes, stream);
  auto *addr = _caching_allocator.GetBasePtr() + block->addr_offset;
  c10::DataPtr data_ptr = {addr, block, BlockDeletePtr,
                           c10::Device(c10::DeviceType::CUDA, cur_device)};
  // LOG(WARNING) << "allocate " << nbytes;
  return data_ptr;
}

c10::DeleterFnPtr TorchAllocator::raw_deleter() const {
  return &RawDeletePtr;
}

void *TorchAllocator::raw_alloc(size_t nbytes) {
  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device);
  return raw_alloc_with_stream(nbytes, stream);
}

void *TorchAllocator::raw_alloc_with_stream(size_t nbytes,
                                            cudaStream_t stream) {
  auto *block = _caching_allocator.Alloc(nbytes, stream);
  auto *addr = _caching_allocator.GetBasePtr() + block->addr_offset;
  _mem_blocks.emplace(addr, block);
  // LOG(WARNING) << "raw_alloc_with_stream " << nbytes << " " << stream;
  return addr;
}

void TorchAllocator::raw_delete(void *ptr) {
  auto iter = _mem_blocks.find(reinterpret_cast<std::byte*>(ptr));
  TORCH_CHECK(iter != _mem_blocks.end(), "Trying to free a pointer not allocated here.");
  auto *block = iter->second;
  _caching_allocator.Free(block);
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
    _caching_allocator.EmptyCache();
}

void TorchAllocator::setMemoryFraction(double fraction, int device) {
  throw std::runtime_error("NOT IMPL");
}

void *TorchAllocator::getBaseAllocation(void *ptr, size_t *size) {
  TORCH_CHECK(_caching_allocator.GetBasePtr() <= ptr && ptr < _caching_allocator.GetEndPtr(), "Trying to get a pointer not allocated here.");
  *size = 0;
  return _caching_allocator.GetBasePtr();
}

void TorchAllocator::cacheInfo(int dev_id, size_t *largestBlock) {
  throw std::runtime_error("NOT IMPL");
}

void TorchAllocator::recordStream(const c10::DataPtr &,
                                  at::cuda::CUDAStream) {
  throw std::runtime_error("NOT IMPL");
}

c10::cuda::CUDACachingAllocator::DeviceStats
TorchAllocator::getDeviceStats(int device) {
  throw std::runtime_error("NOT IMPL");
}

void TorchAllocator::resetAccumulatedStats(int device) {
  throw std::runtime_error("NOT IMPL");
}

void TorchAllocator::resetPeakStats(int device) {
  throw std::runtime_error("NOT IMPL");
}

c10::cuda::CUDACachingAllocator::SnapshotInfo TorchAllocator::snapshot() {
  throw std::runtime_error("NOT IMPL");
}

void TorchAllocator::notifyCaptureBegin(int device,
                                        c10::cuda::CaptureId_t graph_id,
                                        c10::cuda::MempoolId_t mempool_id) {
  throw std::runtime_error("NOT IMPL");
}

void TorchAllocator::notifyCaptureAboutToEnd(int device,
                                             c10::cuda::CaptureId_t graph_id) {
  throw std::runtime_error("NOT IMPL");
}

void TorchAllocator::notifyCaptureEnded(int device,
                                        c10::cuda::CaptureId_t graph_id) {
  throw std::runtime_error("NOT IMPL");
}

void TorchAllocator::notifyCaptureDestroy(int device,
                                          c10::cuda::MempoolId_t mempool_id) {
  throw std::runtime_error("NOT IMPL");
}

std::shared_ptr<void> TorchAllocator::getIpcDevPtr(std::string handle) {
  throw std::runtime_error("NOT IMPL");
}

void TorchAllocator::recordHistory(
    bool enabled,
    c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
    size_t alloc_trace_max_entries, bool alloc_trace_record_context) {
  throw std::runtime_error("NOT IMPL");
}

void TorchAllocator::attachOutOfMemoryObserver(
    c10::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) {
  throw std::runtime_error("NOT IMPL");
}

bool TorchAllocator::needsPoolSpecificPeerAccess() {
  return false;
}

std::string TorchAllocator::name() { return "TorchAllocator"; }

c10::intrusive_ptr<c10::StorageImpl> TorchAllocator::ReceiveHandle(shm_handle<MemBlock> handle,
                                           size_t storage_size) {
  auto *mem_block = _caching_allocator.ReceiveMemBlock(handle);
  CHECK_LE(storage_size, mem_block->nbytes);

  auto cur_device = at::cuda::current_device();
  auto *addr = _caching_allocator.GetBasePtr() + mem_block->addr_offset;
  c10::DataPtr data_ptr = {addr, mem_block, BlockDeletePtr,
                           c10::Device(c10::DeviceType::CUDA, cur_device)};
  auto base = c10::make_intrusive<at::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), storage_size, std::move(data_ptr),
      /*allocator=*/this,
      /*resizable=*/false);
  return base;
}
shm_handle<MemBlock> TorchAllocator::SendHandle(c10::StorageImpl *storage) {
  auto *mem_block =
      reinterpret_cast<MemBlock *>(storage->data_ptr().get_context());
  return _caching_allocator.SendMemBlock(mem_block);
}
} // namespace mpool