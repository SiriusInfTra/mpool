#pragma once

#include <mpool/stream_context.h>
#include <mpool/vmm_allocator.h>

#include <boost/interprocess/sync/scoped_lock.hpp>

namespace mpool {
class CachingAllocator : public VMMAllocator {
private:
  DynamicMappingRegion mapping_region_;

  bip_unordered_map<cudaStream_t, shm_ptr<StreamContext>> &stream_context_map_;

  MemBlock *AllocWithContext(size_t nbytes, StreamContext &stream_context,
                             const bip::scoped_lock<bip_mutex> &lock);

  void FreeWithLock(MemBlock *block, bip::scoped_lock<bip::interprocess_mutex> &lock);

  MemBlock *AllocWithLock(size_t nbytes, cudaStream_t cuda_stream, bool try_expand_VA,
                   bip::scoped_lock<bip::interprocess_mutex> &lock);

  bool CheckStateInternal(const bip::scoped_lock<bip_mutex> &lock);

  void DumpStateWithLock();

  StreamContext &GetStreamContext(cudaStream_t cuda_stream,
                            const bip::scoped_lock<bip_mutex> &lock) override;

public:
  CachingAllocator(SharedMemory &shared_memory, PagesPool &page_pool,
                   VMMAllocatorConfig config, bool first_init);

  MemBlock *Alloc(size_t nbytes, size_t alignment, cudaStream_t cuda_stream,
                  size_t flags = 0) override;

  MemBlock *Realloc(MemBlock *block, size_t nbytes, size_t alignment, cudaStream_t cuda_stream,
                    size_t flags = 0) override;
  
  void Free(const MemBlock *block, size_t flags = 0) override;

  std::byte *GetBasePtr() const override {
    return mapping_region_.GetBasePtr();
  }

  MemBlock *ReceiveMemBlock(shm_ptr<MemBlock> handle);

  shm_ptr<MemBlock> SendMemBlock(MemBlock *mem_block);

  void EmptyCache();

  void ReportOOM(cudaStream_t cuda_stream, OOMReason reason, bool force_abort) override;
};
} // namespace mpool