#pragma once

#include "pages_pool.h"
#include <belong.h>
#include <shm.h>
#include <cstddef>
#include <string>
#include <cuda_runtime_api.h>
#include <glog/logging.h>

namespace mpool {

struct CachingAllocatorConfig {
    std::string log_prefix;
    std::string shm_name;
    size_t shm_nbytes;
};

struct MemBlock {
    ptrdiff_t       addr_offset;
    size_t          nbytes;
    cudaStream_t    stream;

    bip_list<shm_handle<MemBlock>>::iterator                iter_all_block_list;
    bip_list<shm_handle<MemBlock>>::iterator                iter_stream_block_list;
    bip_multimap<size_t, shm_handle<MemBlock>>::iterator    iter_unalloc_block_list;
    bip_multimap<size_t, shm_handle<MemBlock>>::iterator    iter_avail_block_list;

};

class StreamContext {
    bip_list<shm_handle<MemBlock>>                stream_block_list;
    bip_multimap<size_t, shm_handle<MemBlock>>    unalloc_block_list;
    bip_multimap<size_t, shm_handle<MemBlock>>    avail_block_list;
};

class CachingAllocator {
private:
    CachingAllocatorConfig conf_;
    PagesPool &page_pool_;

    bip_list<shm_handle<MemBlock>>                all_block_list;
    
public:
    CachingAllocator(PagesPool &page_pool, Belong belong, CachingAllocatorConfig config): conf_(std::move(config)), page_pool_(page_pool) {
        LOG(INFO) << conf_.log_prefix << "Init Caching Allocator, belong = " << belong;
    };

    ~CachingAllocator() {}

    std::byte *Alloc(size_t nbytes, cudaStream_t stream) {
        bip::scoped_lock defer_lock{mempool_.GetMutex(), bip::defer_lock};


    }

    void Free(std::byte *ptr, cudaStream_t stream);

    void EmptyCache();

};

}