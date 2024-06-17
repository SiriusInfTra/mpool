#pragma once

#include "caching_allocator.h"
#include "shm.h"
#include <memory>
#include <pages_pool.h>

namespace mpool {
class PyPagePool {
private:
  std::shared_ptr<SharableObject<PagesPool>> pages_pool_;
public:
  PyPagePool(SharableObject<PagesPool> *pages_pool): pages_pool_(pages_pool) {}

  PagesPool *operator->() {
    return pages_pool_->GetObject();
  }

  const PagesPool *operator->() const {
    return pages_pool_->GetObject();
  }

  SharableObject<PagesPool> &GetReference() {
    return *pages_pool_;
  }
};

class PyCachingAllocator {
private:
  PyPagePool pages_pool_;
  std::shared_ptr<SharableObject<CachingAllocator>> caching_allocator_;
public:
  PyCachingAllocator(PyPagePool pages_pool, SharableObject<CachingAllocator> *caching_allocator):
    pages_pool_(std::move(pages_pool)), caching_allocator_(caching_allocator) {}
  CachingAllocator *operator->() {
    return caching_allocator_->GetObject();
  }

  // const CachingAllocator *operator->() const {
  //   return caching_allocator_->GetObject();
  // }
};
}