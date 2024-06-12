#include <Python.h>
#include <c10/cuda/CUDAGuard.h>
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

namespace py = pybind11;

namespace mpool {

class Instance {
public:
  std::unordered_map<std::string, std::unique_ptr<SharableObject<PagesPool>>>
      pages_pool_instance;
  std::unordered_map<std::string,
                     std::unique_ptr<SharableObject<CachingAllocator>>>
      caching_allocator_instance;

  ~Instance() {
    caching_allocator_instance.clear();
    pages_pool_instance.clear();
  }
};

void RegisterPagesPool(py::module &m) {
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
  py::class_<PagesPool>(m, "C_PagesPool")
      .def("Lock", &PagesPool::Lock)
      .def("GetBelong", &PagesPool::GetBelong)
      .def("AllocConPages", &PagesPool::AllocConPages)
      .def("AllocDisPages", &PagesPool::AllocDisPages)
      .def("FreePages", &PagesPool::FreePages)
      .attr("INSUFFICIENT_PAGE") = PagesPool::INSUFFICIENT_PAGE;
}

static PyObject *THPStorage_newSharedCuda(PyObject *_unused, PyObject *args) {
  HANDLE_TH_ERRORS
  THPUtils_assert(PyTuple_GET_SIZE(args) == 3, "tuple of 3 items expected");
  PyObject *_device = PyTuple_GET_ITEM(args, 0);
  PyObject *_handle = PyTuple_GET_ITEM(args, 1); /* shm_handle<MemBlock> */
  PyObject *_size_bytes = PyTuple_GET_ITEM(args, 2);
  if (!(THPUtils_checkLong(_device) && THPUtils_checkLong(_handle) &&
        THPUtils_checkLong(_size_bytes))) {
    THPUtils_invalidArguments(
        args, nullptr, "_new_shared in CUDA mode", 1,
        "(int device, int handle, int storage_size_bytes)");
    return nullptr;
  }
  size_t storage_size =
      (size_t)THPUtils_unpackLong(_size_bytes) / sizeof(uint8_t);
  int64_t device = THPUtils_unpackLong(_device);
  shm_handle<MemBlock> m_handle{PyLong_AsLong(_handle)};
  at::cuda::CUDAGuard device_guard(device);

  auto storage = GetTorchAllocator()->ReceiveHandle(m_handle, storage_size);
  auto *torch_module = PyImport_ImportModule("torch");
  auto *THPStorageClass =
      PyObject_GetAttrString(torch_module, "UntypedStorage");
  PyTypeObject *type = (PyTypeObject *)THPStorageClass;
  PyObject *obj = type->tp_alloc(type, 0);
  if (obj) {
    ((THPStorage *)obj)->cdata = storage.release();
  }
  return obj;
  END_HANDLE_TH_ERRORS
}

static PyObject *THPStorage_shareCuda(PyObject *_self, PyObject *noargs) {
  HANDLE_TH_ERRORS
  auto self = (THPStorage *)_self;
  c10::StorageImpl *storage = self->cdata;

  auto *allocator = dynamic_cast<TorchAllocator*>(storage->allocator());
  THPUtils_assert(allocator != nullptr, "Allocator should not be nullptr."); 
  THPUtils_assert(storage->data() != nullptr, "Storage should contains not-null data."); 

  at::DeviceGuard device_guard(storage->device());
  
  auto _handle = allocator->SendHandle(storage);
  THPObjectPtr tuple(PyTuple_New(3));
  THPObjectPtr device(THPUtils_packInt32(storage->device().index()));
  THPObjectPtr handle(THPUtils_packInt64(_handle));
  THPObjectPtr size_bytes(THPUtils_packInt64(storage->nbytes()));

  if (!tuple || !device || !handle) {
    return nullptr;
  }
  PyTuple_SET_ITEM(tuple.get(), 0, device.release());
  PyTuple_SET_ITEM(tuple.get(), 1, handle.release());
  PyTuple_SET_ITEM(tuple.get(), 2, size_bytes.release());
  return tuple.release();
  END_HANDLE_TH_ERRORS
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
  py::class_<CachingAllocator>(m, "C_CachingAllocator")
      .def(
          "Alloc",
          [](CachingAllocator &allocator, size_t nbytes, long cuda_stream,
             bool try_expand_VA) {
            return allocator.Alloc(nbytes,
                                   reinterpret_cast<cudaStream_t>(cuda_stream),
                                   try_expand_VA);
          },
          py::return_value_policy::reference)
      .def("Free", &CachingAllocator::Free)
      .def_property_readonly(
          "base_ptr", [](const CachingAllocator *caching_allocator) {
            return reinterpret_cast<long>(caching_allocator->GetBasePtr());
          });
  m.def("override_pytorch_allocator", OverridePyTorchAllocator);
  m.def("mpool_new_shared_cuda", THPStorage_newSharedCuda);
  m.def("mpool_share_cuda", THPStorage_shareCuda);
}
static Instance ins;
inline void RegisterInstance(py::module &m) {

  m.def(
      "get_pages_pool",
      [](const std::string &name) {
        auto it = ins.pages_pool_instance.find(name);
        if (it != ins.pages_pool_instance.end()) {
          return it->second->GetObject();
        }
        return static_cast<PagesPool *>(nullptr);
      },
      py::return_value_policy::reference);
  m.def(
      "create_pages_pool",
      [](const std::string &name, const PagesPoolConf &conf) {
        auto it = ins.pages_pool_instance.find(name);
        CHECK(it == ins.pages_pool_instance.cend());
        auto *obj =
            new SharableObject<PagesPool>{conf.shm_name, conf.shm_nbytes, conf};
        ins.pages_pool_instance.insert(
            {name, std::unique_ptr<SharableObject<PagesPool>>{obj}});
        return obj->GetObject();
      },
      py::return_value_policy::reference);
  m.def(
      "delete_pages_pool",
      [](const std::string &name) {
        auto it = ins.pages_pool_instance.find(name);
        CHECK(it != ins.pages_pool_instance.cend());
        ins.pages_pool_instance.erase(it);
      },
      py::return_value_policy::reference);

  m.def(
      "get_caching_allocator",
      [](const std::string &name) {
        auto it = ins.caching_allocator_instance.find(name);
        if (it != ins.caching_allocator_instance.end()) {
          return it->second->GetObject();
        }
        return static_cast<CachingAllocator *>(nullptr);
      },
      py::return_value_policy::reference);
  m.def(
      "create_caching_allocator",
      [](const std::string &name, PagesPool &page_pool,
         const CachingAllocatorConfig &conf) {
        auto it = ins.caching_allocator_instance.find(name);
        CHECK(it == ins.caching_allocator_instance.cend());
        auto *obj = new SharableObject<CachingAllocator>{
            conf.shm_name, conf.shm_nbytes, page_pool, conf};
        ins.caching_allocator_instance.insert(
            {name, std::unique_ptr<SharableObject<CachingAllocator>>{obj}});
        return obj->GetObject();
      },
      py::return_value_policy::reference);
  m.def(
      "delete_caching_allocator",
      [](const std::string &name) {
        auto it = ins.caching_allocator_instance.find(name);
        CHECK(it != ins.caching_allocator_instance.cend());
        ins.caching_allocator_instance.erase(it);
      },
      py::return_value_policy::reference);
  class RAIIMemBlock {
  private:
    CachingAllocator *allocator_;
    const MemBlock *mem_block_;

  public:
    RAIIMemBlock(CachingAllocator *allocator, const MemBlock *mem_block)
        : allocator_(allocator), mem_block_(mem_block) {}
    ~RAIIMemBlock() { allocator_->Free(mem_block_); }
  };
  py::class_<RAIIMemBlock, std::shared_ptr<RAIIMemBlock>>(m, "C_RAIIMemBlock")
      .def(py::init<CachingAllocator *, const MemBlock *>(),
           py::arg("allocator"), py::arg("mem_block"));
}

} // namespace mpool
