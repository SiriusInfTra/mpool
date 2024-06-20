#pragma once

#include <Python.h>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Storage.h>
#include <torch/csrc/utils.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_numbers.h>

#include <mem_block.h>
#include <shm.h>
#include <torch_allocator.h>
#include <caching_allocator.h>
#include <pages_pool.h>

#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>


namespace mpool {
static py::object THPStorage_newSharedCuda(py::object _unused, py::args args) {
  HANDLE_TH_ERRORS
  TORCH_WARN("THPStorage_newSharedCuda");
  if (args.size() != 8) {
    TORCH_WARN("tuple of 8 items expected: ", args.size(), ".");
    return py::none();
  }
  long device = py::cast<long>(args[0]);
  auto handle = static_cast<shm_handle<MemBlock>>(py::cast<long>(args[1]));
  size_t storage_size = py::cast<size_t>(args[2]);
  at::cuda::CUDAGuard device_guard(device);

  auto storage_impl = GetTorchAllocator()->ReceiveHandle(handle, storage_size);
  py::object storage = py::module_::import("torch").attr("UntypedStorage")(0);
  reinterpret_cast<THPStorage*>(storage.ptr())->cdata = storage_impl.release();
  return storage;
  END_HANDLE_TH_ERRORS_PYBIND
}

static py::tuple THPStorage_shareCuda(py::object *_self, py::args noargs) {
  HANDLE_TH_ERRORS
  TORCH_WARN("THPStorage_shareCuda");
  PyObject *py_object = _self->ptr();
  auto *self = reinterpret_cast<THPStorage*>(py_object);
  c10::StorageImpl *storage = self->cdata;
  
  auto *allocator = dynamic_cast<TorchAllocator *>(storage->allocator());
  if (allocator == nullptr) {
    TORCH_WARN("Allocator should not be nullptr.");
    return py::none();
  } 
  if (storage->data() == nullptr) {
    TORCH_WARN("Storage should contains not-null data.");
    return py::none();
  }

  at::DeviceGuard device_guard(storage->device());
  auto _handle = allocator->SendHandle(storage);
  py::tuple tuple(8);
  tuple[0] = static_cast<long>(storage->device().index()); // storage_device
  tuple[1] = static_cast<size_t>(_handle); // storage_handle
  tuple[2] = storage->nbytes(); // storage_size_bytes
  tuple[3] = 0; // storage_offset_bytes
  tuple[4] = 0; // ref_counter_handle
  tuple[5] = 0; // ref_counter_offset
  tuple[6] = 0; // event_handle
  tuple[7] = false; // event_sync_required
  return tuple;
  END_HANDLE_TH_ERRORS_PYBIND
}

inline void RegisterPyTorch(py::module &m) {
  m.def("override_pytorch_allocator", OverridePyTorchAllocator);
  m.def("reset_pytorch_allocator", ResetPyTorchAllocator);
  m.def("mpool_new_shared_cuda", THPStorage_newSharedCuda);
  m.def("mpool_share_cuda", THPStorage_shareCuda);
}
}