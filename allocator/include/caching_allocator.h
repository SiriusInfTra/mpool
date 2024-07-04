#pragma once

#include <vmm_allocator.h>

namespace mpool {
class CachingAllocator : public VMMAllocator {
private:
  DynamicMappingRegion mapping_region_;
  ProcessLocalData process_local_;
  bip_unordered_map<cudaStream_t, shm_ptr<StreamContext>> &stream_context_map_;

  StreamContext &GetStreamContext(cudaStream_t cuda_stream,
                                  const bip::scoped_lock<bip_mutex> &lock);

  MemBlock *AllocWithContext(size_t nbytes, StreamContext &stream_context,
                             const bip::scoped_lock<bip_mutex> &lock);

  void FreeWithLock(MemBlock *block, bip::scoped_lock<bip::interprocess_mutex> &lock);

  MemBlock *AllocWithLock(size_t nbytes, cudaStream_t cuda_stream, bool try_expand_VA,
                   bip::scoped_lock<bip::interprocess_mutex> &lock);
  bool CheckStateInternal(const bip::scoped_lock<bip_mutex> &lock);

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

  void DumpState();

  void ReportOOM(cudaStream_t cuda_stream, OOMReason reason, bool force_abort) override;
};
} // namespace mpool