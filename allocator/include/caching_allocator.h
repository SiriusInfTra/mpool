#pragma once
#include <algorithm>
#include <util.h>
#include <belong.h>
#include <shm.h>
#include <pages.h>
#include <pages_pool.h>


#include <cstddef>
#include <iterator>
#include <string>

#include <boost/unordered_map.hpp>
#include <boost/container/list.hpp>
#include <boost/container/map.hpp>

#include <cuda_runtime_api.h>


namespace mpool {


const static constexpr bool VERBOSE = true;
const static constexpr bool CHECK_STATE = true;
const static constexpr bool MORE_CHECK_STATE = true;

struct CachingAllocatorConfig {
    std::string log_prefix;
    std::string shm_name;
    size_t shm_nbytes;
    size_t va_range_scale;
    Belong belong;
    size_t small_block_nbytes;
    size_t align_nbytes;
};

struct MemBlock {
    ptrdiff_t       addr_offset;
    size_t          nbytes;
    cudaStream_t    stream;
    unsigned        unalloc_pages;
    bool            is_free;
    bool            is_small;

    
    bip_list<shm_handle<MemBlock>>::iterator                iter_all_block_list;
    bip_list<shm_handle<MemBlock>>::iterator                iter_stream_block_list;
    bip_multimap<size_t, shm_handle<MemBlock>>::iterator    iter_free_block_list;

};

inline std::ostream& operator<<(std::ostream& out, const MemBlock& block) {
  out << "MemBlock {" << "addr_offset: " << block.addr_offset << ", "
      << "nbytes: " << block.nbytes << ", "
      << "stream: " << block.stream << ", "
      << "unalloc_pages: " << block.unalloc_pages << ", "
      << "is_free: " << (block.is_free ? "true" : "false") << ", "
      << "is_small: " << (block.is_small ? "true" : "false") << "}";
  return out;
}

inline std::ostream& operator<<(std::ostream& out, MemBlock *block) {
  if (block == nullptr) {
    out << "nullptr";
  } else {
    out << *block;
  }
  return out;
}

class MappingRegion {
public:
    const std::string log_prefix;
    const size_t mem_block_nbytes;
    const size_t mem_block_num;
    const size_t va_range_scale;
    const Belong belong;
private:
    SharedMemory &shared_memory_;
    std::byte *base_ptr_;
    std::vector<PhyPage *> mapping_pages_;
    PagesPool &page_pool_;

public:
    MappingRegion(SharedMemory &shared_memory, PagesPool &page_pool, Belong belong, std::string log_prefix, size_t va_range_scale);

    std::pair<index_t, index_t> MappingPageRange(ptrdiff_t addr_offset, size_t nbytes) const;

    index_t GetMappingPage(ptrdiff_t addr_offset) const {
        return addr_offset / mem_block_nbytes;
    }

    void EnsureBlockWithPage(MemBlock *block);

    unsigned GetUnallocPages(ptrdiff_t addr_offset, size_t nbytes);

    bool IsMappingIndexWithPage(index_t mapping_index) const {
        return mapping_pages_[mapping_index] != nullptr;
    }

    void UnMapPages(const std::vector<index_t> &release_pages) {
        std::vector<index_t> pages;
        for (index_t mapping_index : release_pages) {
            index_t page_index = mapping_pages_[mapping_index]->index;
            mapping_pages_[mapping_index] = nullptr;
            pages.push_back(page_index);
        }
        auto lock = page_pool_.Lock();
        page_pool_.FreePages(pages, belong, lock);

    }

        
};

class StreamBlockList {
private:
    cudaStream_t    current_stream_;
    MappingRegion   &mapping_region_;
    SharedMemory    &shared_memory_;
    
    bip_list<shm_handle<MemBlock>> &all_block_list_;
    bip_list<shm_handle<MemBlock>> stream_block_list_;

    size_t small_block_nbytes_;
public:
  StreamBlockList(cudaStream_t cuda_stream, SharedMemory &shared_memory,
                  MappingRegion &mapping_region,
                  bip_list<shm_handle<MemBlock>> &all_block_list,
                  size_t small_block_nbytes);
  StreamBlockList& operator=(const StreamBlockList&) = delete;
  StreamBlockList(const StreamBlockList&) = delete;

  MemBlock *CreateEntryExpandVA(size_t nbytes);

  MemBlock *GetPrevEntry(MemBlock *entry);

  MemBlock *GetNextEntry(MemBlock *entry);

  MemBlock *SplitEntry(MemBlock *origin_entry, size_t remain);

  MemBlock *MergeMemEntry(MemBlock *first_block, MemBlock *secound_block);

  void DumpMemBlockColumns(std::ostream &out);
  void DumpMemBlock(std::ostream &out, MemBlock *block);

  void DumpStreamBlockList(std::ostream &out = std::cout);

  void EmptyCache() {
    std::vector<index_t> release_pages;
    auto iter = stream_block_list_.begin();
    auto block = iter->ptr(shared_memory_);
    for (index_t mapping_index = 0; mapping_index < mapping_region_.mem_block_num; ++mapping_index) {
      if (mapping_region_.IsMappingIndexWithPage(mapping_index)) {
        continue;
      }
      /* 1. skip checked block by moving iter to the first block with mapping_index-th page. */
      while (mapping_region_.GetMappingPage(block->addr_offset) < mapping_index) {
        iter++;
        block = iter->ptr(shared_memory_);
      }
      CHECK_LT(mapping_region_.GetMappingPage(block->addr_offset + block->nbytes), mapping_index);
      /* 2. check whether mapping_index-th is free. */
      bool page_is_free = true;
      std::vector<MemBlock *> blocks_with_pages;
      do {
        blocks_with_pages.push_back(block);
        if (!block->is_free) { 
          page_is_free = false;
          break;
        }
        iter++;
        block = iter->ptr(shared_memory_);
      } while (mapping_region_.GetMappingPage(block->addr_offset + block->nbytes) < mapping_index + 1);
      /* 3. release page if free and update block's unalloc page. */
      if (page_is_free) {
        release_pages.push_back(mapping_index);
        for (auto block : blocks_with_pages) {
          block->unalloc_pages++;
        }
      }
    }
    mapping_region_.UnMapPages(release_pages);
  }

};

class StreamFreeList {
    static const constexpr size_t LARGE = false;
    static const constexpr size_t SMALL = true;
private:
    SharedMemory         &shared_memory_;
    cudaStream_t    current_stream_;

    MappingRegion   &mapping_region_;
    StreamBlockList &stream_block_list_;

    std::array<bip_multimap<size_t, shm_handle<MemBlock>>, 2>   free_block_list_;
public:
  StreamFreeList(SharedMemory &shared_memory, cudaStream_t cuda_stream,
                 MappingRegion &mapping_region,
                 StreamBlockList &stream_block_list);
  
  StreamFreeList& operator=(const StreamFreeList&) = delete;
  StreamFreeList(const StreamFreeList&) = delete;

  MemBlock *PopBlock(bool is_small, size_t nbytes, size_t find_optimal_retry);

  MemBlock *PopBlock(MemBlock *block);

  MemBlock *PushBlock(MemBlock *block);

  MemBlock *MaybeMergeAdj(MemBlock *entry);

  void DumpFreeBlockList(bool is_small, std::ostream &out = std::cout) {
    stream_block_list_.DumpMemBlockColumns(out);
    for (auto [nbytes, handle] : free_block_list_[is_small]) {
      auto *block = handle.ptr(shared_memory_);
      stream_block_list_.DumpMemBlock(out, block);
    }
  
  }

};

struct StreamContext {
  cudaStream_t    cuda_stream;
  StreamBlockList stream_block_list;
  StreamFreeList  stream_free_list;

  StreamContext(cudaStream_t cuda_stream, SharedMemory &shared_memory,
                MappingRegion &mapping_region,
                bip_list<shm_handle<MemBlock>> &all_block_list, size_t small_block_nbytes)
      : cuda_stream{cuda_stream},
        stream_block_list{cuda_stream, shared_memory, mapping_region,
                          all_block_list, small_block_nbytes},
        stream_free_list{shared_memory, cuda_stream, mapping_region,
                         stream_block_list} {}
  StreamContext& operator=(const StreamContext&) = delete;
  StreamContext(const StreamContext&) = delete;
};;

class CachingAllocator {
public:
    const CachingAllocatorConfig config;

    static bool RemoveShm(const CachingAllocatorConfig &config) {
        return bip::shared_memory_object::remove(config.shm_name.c_str());
    }
private:
    PagesPool       &page_pool_;
    SharedMemory    &shared_memory_;
    MappingRegion   mapping_region_;
 
    bip_list<shm_handle<MemBlock>> &all_block_list_;
    StreamContext   &global_stream_context_;



    bip_unordered_map<cudaStream_t, shm_handle<StreamContext>>  &stream_context_map_;

    StreamContext &GetStreamContext(cudaStream_t cuda_stream);

  public:

    CachingAllocator(SharedMemory &shared_memory, PagesPool &page_pool, CachingAllocatorConfig config, bool first_init);

    ~CachingAllocator();

    MemBlock *AllocWithContext(size_t nbytes, StreamContext &stream_context) {
      bool is_small = nbytes <= config.small_block_nbytes;
      auto *free_block =
          stream_context.stream_free_list.PopBlock(is_small, nbytes, 0);
      // LOG(INFO) << free_block << " is small " << is_small;
      if (free_block == nullptr && is_small) {
        free_block = stream_context.stream_free_list.PopBlock(false, config.small_block_nbytes, 0);
        if (free_block != nullptr) {
          free_block->is_small = true;
          free_block = stream_context.stream_free_list.PushBlock(free_block);
          free_block = stream_context.stream_free_list.PopBlock(true, nbytes, 0);
        }
      }
      return free_block;
    }

    MemBlock *Alloc(size_t nbytes, cudaStream_t cuda_stream, bool try_expand_VA = true);

    void Free(MemBlock *block);

    void EmptyCache(cudaStream_t cuda_stream);
};

}