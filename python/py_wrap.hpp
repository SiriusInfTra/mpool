#pragma once
#include <cstdio>
#include <cstdlib>
#include <memory>

#include <mpool/shm.h>
#include <mpool/pages_pool.h>
#include <mpool/caching_allocator.h>


namespace mpool {
class PyPagePool {
private:
  std::shared_ptr<SharableObject<PagesPool>> pages_pool_;

public:
  PyPagePool(const PyPagePool &other) : pages_pool_(other.pages_pool_) {
    // std::cout << "PyPagePool copy constructed, curr ref_cnt: "
    //           << pages_pool_.use_count() << std::endl;
    // std::flush(std::cout);
  }

  PyPagePool(PyPagePool &&other) : pages_pool_(std::move(other.pages_pool_)) {
    // std::cout << "PyPagePool move constructed, curr ref_cnt: "
    //           << pages_pool_.use_count() << std::endl;
    // std::flush(std::cout);
  }

  PyPagePool(SharableObject<PagesPool> *pages_pool) : pages_pool_(pages_pool) {
    // std::cout << "PyPagePool constructed, curr ref_cnt: "
    //           << pages_pool_.use_count() << std::endl;
    // std::flush(std::cout);
  }

  ~PyPagePool() {
    // std::cout << "PyPagePool destructed, curr ref_cnt: "
    //           << pages_pool_.use_count() << std::endl;
    // std::flush(std::cout);
  }

  PagesPool *operator->() { return pages_pool_->GetObject(); }

  const PagesPool *operator->() const { return pages_pool_->GetObject(); }

  SharableObject<PagesPool> &GetReference() { return *pages_pool_; }
};

class PyCachingAllocator {
private:
  PyPagePool pages_pool_;
  std::shared_ptr<SharableObject<CachingAllocator>> caching_allocator_;

public:
  PyCachingAllocator(PyPagePool pages_pool,
                     SharableObject<CachingAllocator> *caching_allocator)
      : pages_pool_(std::move(pages_pool)),
        caching_allocator_(caching_allocator) {
    // std::cout << "PyCachingAllocator constructed, curr ref_cnt: "
    //           << caching_allocator_.use_count() << std::endl;
    // std::flush(std::cout);
  }

  PyCachingAllocator(PyCachingAllocator &&other)
      : pages_pool_(std::move(other.pages_pool_)),
        caching_allocator_(std::move(other.caching_allocator_)) {
    // std::cout << "PyCachingAllocator move constructed, curr ref_cnt: "
    //           << caching_allocator_.use_count() << std::endl;
    // std::flush(std::cout);
  }

  PyCachingAllocator(const PyCachingAllocator &other)
      : pages_pool_(other.pages_pool_),
        caching_allocator_(other.caching_allocator_) {
    // std::cout << "PyCachingAllocator copy constructed, curr ref_cnt: "
    //           << caching_allocator_.use_count() << std::endl;
    // std::flush(std::cout);
  }

  ~PyCachingAllocator() {
    // std::cout << "PyCachingAllocator destructed, curr ref_cnt: "
    //           << caching_allocator_.use_count() << std::endl;
    // std::flush(std::cout);
  }
  CachingAllocator *operator->() { return caching_allocator_->GetObject(); }

  // const CachingAllocator *operator->() const {
  //   return caching_allocator_->GetObject();
  // }
};
} // namespace mpool