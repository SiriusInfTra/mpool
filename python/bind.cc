#include <pybind11/pybind11.h>

#include <py_export.hpp>
#include <pytorch.hpp>
#include <tensorrt.hpp>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

using namespace mpool;


PYBIND11_MODULE(_C, m) {
  m.doc() = R"pbdoc(
        mpool pytorch bind
    )pbdoc";
  RegisterPagesPool(m);
  RegisterCachingAllocator(m);
  RegisterInstance(m);
  RegisterPyTorch(m);
  RegisterTensorRT(m);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}