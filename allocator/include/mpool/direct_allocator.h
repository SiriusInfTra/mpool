#pragma once

#include <cstddef>

#include <mpool/allocator.h>
#include <mpool/mapping_region.h>
#include <mpool/mem_block.h>
#include <mpool/pages_pool.h>
#include <mpool/util.h>
#include <mpool/vmm_allocator.h>
namespace mpool {

using VMMAllocatorConfig = VMMAllocatorConfig;

class DirectAllocator : public VMMAllocator {
private:
  StaticMappingRegion mapping_region_;

  void DumpStateWithLock();

public:
  DirectAllocator(SharedMemory &shared_memory, PagesPool &page_pool,
                  VMMAllocatorConfig config, bool first_init);
  virtual ~DirectAllocator() = default;
  virtual MemBlock *Alloc(size_t request_nbytes, size_t alignment,
                          cudaStream_t cuda_stream, size_t flags) override;
  virtual MemBlock *Realloc(MemBlock *block, size_t nbytes, size_t alignment, 
                            cudaStream_t cuda_stream, size_t flags) override {
    return nullptr;
  }
  virtual void Free(const MemBlock *block, size_t flags) override;

  size_t AlignToPages(size_t nbytes) {
    return nbytes >= page_pool.config.page_nbytes 
           ? AlignNbytes(nbytes, page_pool.config.page_nbytes) 
           : AlignNbytes(nbytes, page_pool.config.page_nbytes / 2);
  }

  std::byte *GetBasePtr() const override {
    return mapping_region_.GetBasePtr();
  }

  void ReportOOM(cudaStream_t cuda_stream, OOMReason reason,
                 bool force_abort) override;
};
} // namespace mpool