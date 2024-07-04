#include <pybind11/pybind11.h>

#include <py_export.hpp>
#ifdef ENABLE_TORCH
#include <pytorch.hpp>
#endif
#ifdef ENABLE_TENSORRT
#include <tensorrt.hpp>
#endif

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

using namespace mpool;


PYBIND11_MODULE(_C, m) {
  m.doc() = R"pbdoc(
        MemoryPool Python Binding
    )pbdoc";
  RegisterPagesPool(m);
  RegisterCachingAllocator(m);
  RegisterInstance(m);
#ifdef ENABLE_TORCH
  RegisterPyTorch(m);
#endif
#ifdef ENABLE_TENSORRT
  RegisterTensorRT(m);
#endif
#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}