#include "belong.h"
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <pages_pool.h>
#include <caching_allocator.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace mpool;


Belong GetBelong(PagesPool *page_pool, std::string name) {
    return page_pool->GetBelongRegistry().GetOrCreateBelong(name);
}

// CachingAllocator *CreateCachingAllocator(PagesPool *page_pool, CachingAllocatorConfig conf) {
//     return new CachingAllocator{

//     }
// }

PYBIND11_MODULE(_C, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: scikit_build_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    py::class_<PagesPoolConf>(m, "C_PagesPoolConf")
        .def(py::init<size_t, size_t, const std::string &, const std::string &, size_t>())
        .def_readwrite("page_nbytes", &PagesPoolConf::page_nbytes)
        .def_readwrite("pool_nbytes", &PagesPoolConf::pool_nbytes)
        .def_readwrite("shm_name", &PagesPoolConf::shm_name)
        .def_readwrite("log_prefix", &PagesPoolConf::log_prefix)
        .def_readwrite("shm_nbytes", &PagesPoolConf::shm_nbytes);
    py::class_<Belong>(m, "C_Belong");
    py::class_<PagesPool>(m, "C_PagesPool")
        .def(py::init<PagesPoolConf>());
        // .def("PagesView", &PagesPool::PagesView)
        // .def("Lock", &PagesPool::Lock)
        // .def("GetBelongRegistry", &PagesPool::GetBelongRegistry);
py::class_<CachingAllocatorConfig>(m, "C_CachingAllocatorConfig")
        .def(py::init<const std::string&, const std::string&, size_t, size_t, const std::string&, size_t, size_t>())
        .def_readwrite("log_prefix", &CachingAllocatorConfig::log_prefix)
        .def_readwrite("shm_name", &CachingAllocatorConfig::shm_name)
        .def_readwrite("shm_nbytes", &CachingAllocatorConfig::shm_nbytes)
        .def_readwrite("va_range_scale", &CachingAllocatorConfig::va_range_scale)
        .def_readwrite("belong_name", &CachingAllocatorConfig::belong_name)
        .def_readwrite("small_block_nbytes", &CachingAllocatorConfig::small_block_nbytes)
        .def_readwrite("align_nbytes", &CachingAllocatorConfig::align_nbytes);
        
    py::class_<SharedMemory>(m, "C_SharedMemory")
        .def(py::init<std::string, size_t>());
    m.def("GetBelong", &GetBelong);

    // py::class_<CachingAllocator>(m, "C_CachingAllocator")
    //     .def(py::init())
    // py::class_<BelongImpl, std::unique_ptr<Belong, py::nodelete>>(m, "BelongImpl")
    //             .def(py::init<>());
    // py::class_<PagesPool>(m, "PagesPool")
    //     .def(py::init<PagesPoolConf, bool>())
    //     .def("AllocConPages", &PagesPool::AllocConPages)
    //     .def("AllocDisPages", &PagesPool::AllocDisPages)
    //     .def("FreePages", &PagesPool::FreePages)
    //     .def("FreePages", &PagesPool::FreePages)
    //     .def("PagesView", &PagesPool::PagesView)
    //     .def("Lock", &PagesPool::Lock);

    // m.def("GetBelong", , const Extra &extra...)

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}