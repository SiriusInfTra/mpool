#pragma once

#include <Python.h>


#include <cstring>

#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Storage.h>
#include <torch/csrc/utils.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_numbers.h>

#include <caching_allocator.h>
#include <mem_block.h>
#include <pages_pool.h>
#include <shm.h>
#include <torch_allocator.h>

#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace mpool {
static py::object THPStorage_newSharedCuda(py::object _unused, py::args args) {
  HANDLE_TH_ERRORS
  TORCH_WARN("THPStorage_newSharedCuda");
  TORCH_CHECK_EQ(args.size(), 8) << "tuple of 8 items expected.";
  long device = py::cast<long>(args[0]);
  auto handle = static_cast<shm_ptr<MemBlock>>(py::cast<bip_shm::handle_t>(args[1]));
  size_t storage_size = py::cast<size_t>(args[2]);

  std::string s_ipc_event_handle = py::cast<py::bytes>(args[6]);
  bool event_sync_required = py::cast<bool>(args[7]);
  cudaEvent_t event;
  if (event_sync_required) {
    cudaIpcEventHandle_t ipc_event_handle;
    TORCH_CHECK_EQ(s_ipc_event_handle.size(), sizeof(ipc_event_handle))
        << "Bad ipc event handle.";
    memcpy(&ipc_event_handle, s_ipc_event_handle.data(),
           sizeof(ipc_event_handle));

    C10_CUDA_CHECK(cudaIpcOpenEventHandle(&event, ipc_event_handle));
    C10_CUDA_CHECK(
        cudaStreamWaitEvent(c10::cuda::getCurrentCUDAStream(device), event, 0));
  }
  at::cuda::CUDAGuard device_guard(device);
  auto *extra_data = new MemBlockExtraData{
      .require_device_sync = true,
      .require_event_sync = event_sync_required,
      .event = event
  };
  auto storage_impl = GetTorchAllocator()->ReceiveHandle(handle, storage_size, extra_data);
  py::object storage = py::module_::import("torch").attr("UntypedStorage")(0);
  reinterpret_cast<THPStorage *>(storage.ptr())->cdata = storage_impl.release();
  return storage;
  END_HANDLE_TH_ERRORS_PYBIND
}

static py::tuple THPStorage_shareCuda(py::object *_self, py::args noargs) {
  HANDLE_TH_ERRORS
  TORCH_WARN("THPStorage_shareCuda");
  PyObject *py_object = _self->ptr();
  auto *self = reinterpret_cast<THPStorage *>(py_object);
  c10::StorageImpl *storage = self->cdata;

  auto *allocator = dynamic_cast<TorchAllocator *>(storage->allocator());
  TORCH_CHECK_NOTNULL(allocator);
  TORCH_CHECK_NOTNULL(storage->data());

  at::DeviceGuard device_guard(storage->device());
  auto _handle = allocator->SendHandle(storage);

  cudaEvent_t event;
  cudaIpcEventHandle_t event_handle;
  // FIXME cuda have a limit on cudaEventInterprocess
  // TODO Destroy cuda event
  C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming |
                                                      cudaEventInterprocess |
                                                      cudaEventBlockingSync));
  C10_CUDA_CHECK(cudaEventRecord(
      event, c10::cuda::getCurrentCUDAStream(storage->device().index())));
  C10_CUDA_CHECK(cudaIpcGetEventHandle(&event_handle, event));
  std::string s_ipc_event_handle(reinterpret_cast<const char *>(&event_handle),
                                 sizeof(event_handle));
  py::tuple tuple(8);
  tuple[0] = static_cast<long>(storage->device().index()); // storage_device
  tuple[1] = static_cast<bip_shm::handle_t>(_handle);      // storage_handle
  tuple[2] = storage->nbytes();                            // storage_size_bytes
  tuple[3] = 0;                             // storage_offset_bytes
  tuple[4] = 0;                             // ref_counter_handle
  tuple[5] = 0;                             // ref_counter_offset
  tuple[6] = py::bytes(s_ipc_event_handle); // event_handle
  tuple[7] = true;                         // event_sync_required
  return tuple;
  END_HANDLE_TH_ERRORS_PYBIND
}

inline void RegisterPyTorch(py::module &m) {
  m.def("override_pytorch_allocator", OverridePyTorchAllocator);
  m.def("reset_pytorch_allocator", ResetPyTorchAllocator);
  m.def("mpool_new_shared_cuda", THPStorage_newSharedCuda);
  m.def("mpool_share_cuda", THPStorage_shareCuda);
}
} // namespace mpool