#include <Python.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <import.h>
#include <longobject.h>
#include <object.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Storage.h>
#include <torch/csrc/utils.h>

// #include <torch/csrc/utils/python_numbers.h>
#include "belong.h"
#include "mem_block.h"
#include "pages.h"
#include "shm.h"
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <memory>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "torch_allocator.h"
#include <caching_allocator.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <pages_pool.h>
#include <string>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_numbers.h>
#include <type_traits>
#include <unordered_map>
#include "py_wrap.hpp"

namespace py = pybind11;

namespace mpool {

inline void RegisterPagesPool(py::module &m) {
  py::class_<Belong>(m, "C_Belong");
  py::class_<bip::scoped_lock<bip::interprocess_mutex>>(m, "C_Lock");
  py::class_<PagesPoolConf>(m, "C_PagesPoolConf")
      .def(py::init<size_t, size_t, const std::string &, const std::string &,
                    size_t>(),
           py::arg("page_nbytes"), py::arg("pool_nbytes"), py::arg("shm_name"),
           py::arg("log_prefix"), py::arg("shm_nbytes"))
      .def_readwrite("page_nbytes", &PagesPoolConf::page_nbytes)
      .def_readwrite("pool_nbytes", &PagesPoolConf::pool_nbytes)
      .def_readwrite("shm_name", &PagesPoolConf::shm_name)
      .def_readwrite("log_prefix", &PagesPoolConf::log_prefix)
      .def_readwrite("shm_nbytes", &PagesPoolConf::shm_nbytes);
  py::class_<PyPagePool>(m, "C_PagesPool")
      .def(py::init([](PagesPoolConf conf) {
        return PyPagePool{new SharableObject<PagesPool>{
          conf.shm_name, conf.shm_nbytes, conf}};
      }), py::return_value_policy::move)
      .def("lock", [](PyPagePool &page_pool) {
             return page_pool->Lock();
           })
      .def("get_belong",
           [](PyPagePool &page_pool, const std::string &name) {
             return page_pool->GetBelong(name);
           })
      .def("alloc_con_pages",
           [](PyPagePool &self, Belong blg, num_t num_req,
              bip::scoped_lock<bip_mutex> &lock) {
             return self->AllocConPages(blg, num_req, lock);
           })
      .def("alloc_dis_pages",
           [](PyPagePool &self, Belong blg, num_t num_req,
              bip::scoped_lock<bip_mutex> &lock) {
             return self->AllocDisPages(blg, num_req, lock);
           })
      .def("free_pages",
           [](PyPagePool &self,
              const std::vector<index_t> &pages, Belong blg,
              bip::scoped_lock<bip_mutex> &lock) {
             self->FreePages(pages, blg, lock);
           })
      .attr("INSUFFICIENT_PAGE") = PagesPool::INSUFFICIENT_PAGE;
}


inline void RegisterCachingAllocator(py::module &m) {
  py::class_<CachingAllocatorConfig>(m, "C_CachingAllocatorConfig")
      .def(py::init<const std::string &, const std::string &, size_t, size_t,
                    const std::string &, size_t, size_t>(),
           py::arg("log_prefix"), py::arg("shm_name"), py::arg("shm_nbytes"),
           py::arg("va_range_scale"), py::arg("belong_name"),
           py::arg("small_block_nbytes"), py::arg("align_nbytes"))
      .def_readwrite("log_prefix", &CachingAllocatorConfig::log_prefix)
      .def_readwrite("shm_name", &CachingAllocatorConfig::shm_name)
      .def_readwrite("shm_nbytes", &CachingAllocatorConfig::shm_nbytes)
      .def_readwrite("va_range_scale", &CachingAllocatorConfig::va_range_scale)
      .def_readwrite("belong_name", &CachingAllocatorConfig::belong_name)
      .def_readwrite("small_block_nbytes",
                     &CachingAllocatorConfig::small_block_nbytes)
      .def_readwrite("align_nbytes", &CachingAllocatorConfig::align_nbytes);
  py::class_<MemBlock>(m, "C_MemBlock")
      .def_readonly("addr_offset", &MemBlock::addr_offset)
      .def_readonly("nbytes", &MemBlock::nbytes)
      .def_property_readonly("stream", [](const MemBlock *mem_block) {
        return reinterpret_cast<long>(mem_block->stream);
      });
  py::class_<PyCachingAllocator>(m, "C_CachingAllocator")
      .def(py::init([](PyPagePool pages_pool, CachingAllocatorConfig conf) {
        auto &pages_pool_ref = *pages_pool.GetReference().GetObject();
        auto *caching_allocator = new SharableObject<CachingAllocator>{conf.shm_name, conf.shm_nbytes, pages_pool_ref, conf};
        return PyCachingAllocator{pages_pool, caching_allocator};
      }), py::return_value_policy::move)
      .def(
          "alloc",
          [](PyCachingAllocator &self, size_t nbytes, long cuda_stream,
             bool try_expand_VA) {
            return self->Alloc(nbytes,
                                   reinterpret_cast<cudaStream_t>(cuda_stream),
                                   try_expand_VA);
          },
          py::return_value_policy::reference)
      .def("free", [](PyCachingAllocator &self, const MemBlock *block) {
        return self->Free(block);
      })
      .def_property_readonly(
          "base_ptr", [](PyCachingAllocator &caching_allocator) {
            return reinterpret_cast<long>(caching_allocator->GetBasePtr());
          });
}

inline void RegisterInstance(py::module &m) {
  class RAIIMemBlock {
  private:
    PyCachingAllocator &allocator_;
    const MemBlock *mem_block_;

  public:
    RAIIMemBlock(PyCachingAllocator &allocator, const MemBlock *mem_block)
        : allocator_(allocator), mem_block_(mem_block) {}
    ~RAIIMemBlock() { allocator_->Free(mem_block_); }
  };
  py::class_<RAIIMemBlock, std::shared_ptr<RAIIMemBlock>>(m, "C_RAIIMemBlock")
      .def(py::init<PyCachingAllocator&, const MemBlock *>(),
           py::arg("allocator"), py::arg("mem_block"));
}

} // namespace mpool
