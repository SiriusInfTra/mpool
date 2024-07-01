#pragma once

#include <Python.h>

#include <pybind11/pybind11.h>

#include <tensorrt_allocator.h>

namespace mpool {
inline void RegisterTensorRT(py::module &m) {
  m.def("set_igpu_allocator", [](nvinfer1::IBuilder *builder,
                                 std::vector<PyCachingAllocator> allocators) {
     TensorRTAllocator::SetIGPUAllocator(builder, allocators);
  });
  m.def("unset_igpu_allocator", [](nvinfer1::IBuilder *builder) {
     TensorRTAllocator::UnsetIGPUAllocator(builder);
  });
}

}