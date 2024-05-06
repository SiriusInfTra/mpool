#pragma once
#include <algorithm>
#include <belong.h>
#include <pages.h>
#include <pages_pool.h>
#include <shm.h>
#include <util.h>

#include <cstddef>
#include <iterator>
#include <string>
#include <unordered_set>

#include <boost/container/list.hpp>
#include <boost/container/map.hpp>
#include <boost/unordered_map.hpp>

#include <cuda_runtime_api.h>

namespace mpool {

const static constexpr bool VERBOSE = true;
const static constexpr bool CHECK_STATE = true;
const static constexpr bool MORE_CHECK_STATE = true;
const static constexpr bool MORE_MORE_CHECK_STATE = true;

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
  ptrdiff_t addr_offset;
  size_t nbytes;
  cudaStream_t stream;
  int32_t unalloc_pages;
  bool is_free;
  bool is_small;

  bip_list<shm_handle<MemBlock>>::iterator iter_all_block_list;
  bip_list<shm_handle<MemBlock>>::iterator iter_stream_block_list;
  bip_multimap<size_t, shm_handle<MemBlock>>::iterator iter_free_block_list;
};

inline std::ostream &operator<<(std::ostream &out, const MemBlock &block) {
  out << "MemBlock {"
      << "addr_offset: " << block.addr_offset << ", "
      << "nbytes: " << block.nbytes << ", "
      << "stream: " << block.stream << ", "
      << "unalloc_pages: " << block.unalloc_pages << ", "
      << "is_free: " << (block.is_free ? "true" : "false") << ", "
      << "is_small: " << (block.is_small ? "true" : "false") << "}";
  return out;
}

inline std::ostream &operator<<(std::ostream &out, MemBlock *block) {
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
  std::vector<const PhyPage *> mapping_pages_;
  PagesPool &page_pool_;

public:
  MappingRegion(SharedMemory &shared_memory, PagesPool &page_pool,
                Belong belong, std::string log_prefix, size_t va_range_scale);

  std::pair<index_t, index_t> MappingPageRange(ptrdiff_t addr_offset,
                                               size_t nbytes) const;

  index_t GetMappingPage(ptrdiff_t addr_offset) const {
    return addr_offset / mem_block_nbytes;
  }

  std::vector<index_t> EnsureBlockWithPage(const MemBlock *block);

  int32_t GetUnallocPages(ptrdiff_t addr_offset, size_t nbytes);

  bool IsMappingIndexWithPage(index_t mapping_index) const {
    return mapping_pages_[mapping_index] != nullptr;
  }

  void UnMapPages(const std::vector<index_t> &release_pages) {
    /* release physical page to mem pool */
    {
      std::vector<index_t> pages;
      for (index_t mapping_index : release_pages) {
        index_t page_index = mapping_pages_[mapping_index]->index;
        mapping_pages_[mapping_index] = nullptr;

        pages.push_back(page_index);
      }
      auto lock = page_pool_.Lock();
      page_pool_.FreePages(pages, belong, lock);
    }

    /* unmap corresponding physical pages */
    auto iter = release_pages.cbegin();
    do {
      auto iter_dis = std::adjacent_find(iter, release_pages.cend(),
                                [](index_t a, index_t b) { return a + 1 != b; });
      if (iter != iter_dis) {
        /* unmap continuous physical pages */
        CU_CALL(cuMemUnmap(
              reinterpret_cast<CUdeviceptr>(base_ptr_ + *iter * mem_block_nbytes), 
              std::distance(iter, iter_dis) * mem_block_nbytes));
      }
      if (iter_dis != release_pages.cend()) {
        /* unmap discontinuous physical pages */
        CU_CALL(cuMemUnmap(
              reinterpret_cast<CUdeviceptr>(base_ptr_ + *iter_dis * mem_block_nbytes), 
              mem_block_nbytes));
        iter = std::next(iter_dis);
      } else {
        break;
      }
    } while (iter != release_pages.cend());
  }

  void EmptyCache(bip_list<shm_handle<MemBlock>> &block_list) {
    std::vector<index_t> release_pages;
    auto iter = block_list.begin();
    auto block = iter->ptr(shared_memory_);
    for (index_t mapping_index = 0; mapping_index < mapping_pages_.size();
         ++mapping_index) {
      if (mapping_pages_[mapping_index] == nullptr) {
        continue;
      }
      /* 1. skip checked block by moving iter to the first block with
       * mapping_index-th page. */
      while (GetMappingPage(block->addr_offset + block->nbytes - 1) < mapping_index) {
        iter++;
        block = iter->ptr(shared_memory_);
      }
      CHECK_LE(GetMappingPage(block->addr_offset), mapping_index);

      /* 2. check whether mapping_index-th is free. */
      bool page_is_free = true;
      std::vector<MemBlock *> blocks_with_pages;
      while (true) {
        blocks_with_pages.push_back(block);
        if (!block->is_free) {
          page_is_free = false;
          break;
        }
        if (GetMappingPage(block->addr_offset + block->nbytes - 1) >= mapping_index) {
          break;
        }
        iter++;
        block = iter->ptr(shared_memory_);
      }

      /* 3. release page if free and update block's unalloc page. */
      if (page_is_free) {
        release_pages.push_back(mapping_index);
        for (auto block : blocks_with_pages) {
          block->unalloc_pages++;
        }
      }
    }
    UnMapPages(release_pages);
  }

};

class StreamBlockList {
private:
  cudaStream_t current_stream_;
  MappingRegion &mapping_region_;
  SharedMemory &shared_memory_;

  bip_list<shm_handle<MemBlock>> &all_block_list_;
  bip_list<shm_handle<MemBlock>> stream_block_list_;

  size_t small_block_nbytes_;

public:
  StreamBlockList(cudaStream_t cuda_stream, SharedMemory &shared_memory,
                  MappingRegion &mapping_region,
                  bip_list<shm_handle<MemBlock>> &all_block_list,
                  size_t small_block_nbytes);
  StreamBlockList &operator=(const StreamBlockList &) = delete;
  StreamBlockList(const StreamBlockList &) = delete;

  MemBlock *CreateEntryExpandVA(size_t nbytes);

  MemBlock *GetPrevEntry(MemBlock *entry);

  MemBlock *GetNextEntry(MemBlock *entry);

  MemBlock *SplitEntry(MemBlock *origin_entry, size_t remain);

  MemBlock *MergeMemEntry(MemBlock *first_block, MemBlock *secound_block);

  std::pair<bip_list<shm_handle<MemBlock>>::const_iterator,
            bip_list<shm_handle<MemBlock>>::const_iterator>
  Iterators() const {
    return {stream_block_list_.begin(), stream_block_list_.end()};
  }

  void EnsureBlockWithPage(MemBlock *block);

  void DumpMemBlockColumns(std::ostream &out);
  void DumpMemBlock(std::ostream &out, MemBlock *block);

  void DumpStreamBlockList(std::ostream &out = std::cout);

  bool CheckState(bool check_global_block_list = false) {
    for (auto iter = stream_block_list_.cbegin();
         iter != stream_block_list_.cend(); ++iter) {
      auto *block = iter->ptr(shared_memory_);
      if (block->stream != current_stream_) {
        LOG(FATAL) << "block's stream is not current stream: " << block << ".";
        return false;
      }
      if (int32_t unalloc_pages = mapping_region_.GetUnallocPages(
              block->addr_offset, block->nbytes);
          block->unalloc_pages != unalloc_pages) {
        LOG(INFO) << mapping_region_.GetUnallocPages(block->addr_offset,
                                                     block->nbytes);
        LOG(FATAL) << "block's unalloc_pages is not match: " << block
                   << ", ground truth: " << unalloc_pages << ".";
        return false;
      }
      // if (!block->is_free && block->unalloc_pages != 0) {
      //   LOG(FATAL) << "block is free not but unalloc_pages is not 0: " <<
      //   block << "."; return false;
      // }
      if (auto next_iter = std::next(block->iter_stream_block_list);
          next_iter != stream_block_list_.cend()) {
        auto *next_block = next_iter->ptr(shared_memory_);
        if (next_block->addr_offset <
            block->addr_offset + static_cast<ptrdiff_t>(block->nbytes)) {
          LOG(FATAL) << "block's next block is not (potential) continuous: "
                     << block << " -> " << next_block << ".";
          return false;
        }
      }
    }
    if (check_global_block_list) {
      for (auto iter = all_block_list_.cbegin(); iter != all_block_list_.cend();
           ++iter) {
        auto *block = iter->ptr(shared_memory_);
        if (auto next_iter = std::next(block->iter_all_block_list);
            next_iter != all_block_list_.cend()) {
          auto *next_block = next_iter->ptr(shared_memory_);
          if (next_block->addr_offset !=
              block->addr_offset + static_cast<ptrdiff_t>(block->nbytes)) {
            LOG(FATAL) << "block's next block is not continuous: " << block
                       << " -> " << next_block << ".";
            return false;
          }
        }
      }
    }
    return true;
  }

  void EmptyCache() {
    std::vector<index_t> release_pages;
    auto iter = stream_block_list_.begin();
    auto block = iter->ptr(shared_memory_);
    for (index_t mapping_index = 0; mapping_index < mapping_region_.mem_block_num;
         ++mapping_index) {
      if (mapping_region_.IsMappingIndexWithPage(mapping_index)) {
        continue;
      }
      /* 1. skip checked block by moving iter to the first block with
       * mapping_index-th page. */
      while (mapping_region_.GetMappingPage(block->addr_offset) <
             mapping_index) {
        iter++;
        block = iter->ptr(shared_memory_);
      }
      CHECK_LT(
          mapping_region_.GetMappingPage(block->addr_offset + block->nbytes),
          mapping_index);
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
      } while (mapping_region_.GetMappingPage(
                   block->addr_offset + block->nbytes) < mapping_index + 1);
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
  SharedMemory &shared_memory_;
  cudaStream_t current_stream_;

  MappingRegion &mapping_region_;
  StreamBlockList &stream_block_list_;

  std::array<bip_multimap<size_t, shm_handle<MemBlock>>, 2> free_block_list_;

public:
  StreamFreeList(SharedMemory &shared_memory, cudaStream_t cuda_stream,
                 MappingRegion &mapping_region,
                 StreamBlockList &stream_block_list);

  StreamFreeList &operator=(const StreamFreeList &) = delete;
  StreamFreeList(const StreamFreeList &) = delete;

  MemBlock *PopBlock(bool is_small, size_t nbytes, size_t find_optimal_retry);

  MemBlock *PopBlock(MemBlock *block);

  MemBlock *PushBlock(MemBlock *block);

  MemBlock *MaybeMergeAdj(MemBlock *entry);

  bool CheckState() {
    std::unordered_set<MemBlock *> free_blocks;
    for (auto &freelist : free_block_list_) {
      for (auto [nbytes, handle] : freelist) {
        auto *block = handle.ptr(shared_memory_);
        if (block->is_free == false) {
          LOG(FATAL) << "block is not free but in freeelist: " << block << ".";
          return false;
        }
        if (block->stream != current_stream_) {
          LOG(FATAL) << "block's stream is not current stream: " << block
                     << ".";
          return false;
        }
        if (block->nbytes != nbytes) {
          LOG(FATAL) << "block's nbytes is not the nbytes in freelist: "
                     << block << ", nbytes in freelist:" << nbytes << ".";
          return false;
        }
        free_blocks.insert(block);
      }
    }
    for (auto [iter, end] = stream_block_list_.Iterators(); iter != end;
         ++iter) {
      auto *block = iter->ptr(shared_memory_);
      if (block->is_free == true && free_blocks.count(block) == 0) {
        LOG(FATAL) << "block is free but not in freelist: " << block << ".";
        return false;
      }
    }

    return true;
  }

  void DumpFreeBlockList(bool is_small, std::ostream &out = std::cout);
};

struct StreamContext {
  cudaStream_t cuda_stream;
  StreamBlockList stream_block_list;
  StreamFreeList stream_free_list;

  StreamContext(cudaStream_t cuda_stream, SharedMemory &shared_memory,
                MappingRegion &mapping_region,
                bip_list<shm_handle<MemBlock>> &all_block_list,
                size_t small_block_nbytes)
      : cuda_stream{cuda_stream},
        stream_block_list{cuda_stream, shared_memory, mapping_region,
                          all_block_list, small_block_nbytes},
        stream_free_list{shared_memory, cuda_stream, mapping_region,
                         stream_block_list} {}
  StreamContext &operator=(const StreamContext &) = delete;
  StreamContext(const StreamContext &) = delete;
};
;

class CachingAllocator {
public:
  const CachingAllocatorConfig config;

  static bool RemoveShm(const CachingAllocatorConfig &config) {
    return bip::shared_memory_object::remove(config.shm_name.c_str());
  }

private:
  PagesPool &page_pool_;
  SharedMemory &shared_memory_;
  MappingRegion mapping_region_;

  bip_list<shm_handle<MemBlock>> &all_block_list_;
  StreamContext &global_stream_context_;

  bip_unordered_map<cudaStream_t, shm_handle<StreamContext>>
      &stream_context_map_;

  StreamContext &GetStreamContext(cudaStream_t cuda_stream);

public:
  CachingAllocator(SharedMemory &shared_memory, PagesPool &page_pool,
                   CachingAllocatorConfig config, bool first_init);

  ~CachingAllocator();

  MemBlock *AllocWithContext(size_t nbytes, StreamContext &stream_context) {
    bool is_small = nbytes <= config.small_block_nbytes;
    CHECK(!MORE_CHECK_STATE || CheckState());
    auto *free_block =
        stream_context.stream_free_list.PopBlock(is_small, nbytes, 50);
    CHECK(!MORE_CHECK_STATE || CheckState());
    // LOG(INFO) << free_block << " is small " << is_small;
    if (free_block == nullptr && is_small) {
      free_block = stream_context.stream_free_list.PopBlock(
          false, config.small_block_nbytes, 50);
      if (free_block != nullptr) {
        free_block->is_small = true;
        free_block = stream_context.stream_free_list.PushBlock(free_block);
        free_block = stream_context.stream_free_list.PopBlock(true, nbytes, 0);
      }
    }
    CHECK(!MORE_CHECK_STATE || CheckState());
    return free_block;
  }

  MemBlock *Alloc(size_t nbytes, cudaStream_t cuda_stream,
                  bool try_expand_VA = true);

  void Free(MemBlock *block);

  void EmptyCache(cudaStream_t cuda_stream);

  bool CheckState() {
    bool ret = true;
    ret &= global_stream_context_.stream_block_list.CheckState(true);
    ret &= global_stream_context_.stream_free_list.CheckState();
    for (auto &&[cuda_stream, context] : stream_context_map_) {
      ret &= context.ptr(shared_memory_)->stream_block_list.CheckState();
      ret &= context.ptr(shared_memory_)->stream_free_list.CheckState();
    }
    return ret;
  }
};
}