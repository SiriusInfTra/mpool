#pragma once

#include <vmm_allocator.h>

namespace mpool {
class CachingAllocator : public VMMAllocator {
public:
  CachingAllocator(SharedMemory &shared_memory, PagesPool &page_pool,
                   CachingAllocatorConfig config, bool first_init)
      : VMMAllocator(shared_memory, page_pool, config, first_init) {}
};
}