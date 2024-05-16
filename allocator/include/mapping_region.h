#pragma once 

#include <cstddef>
#include <string>
#include <vector>

#include <shm.h>
#include <util.h>
#include <belong.h>
#include <pages_pool.h>
#include <mem_block.h>

namespace mpool {

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

  index_t GetMappingPage(ptrdiff_t addr_offset) const;

  void EnsureBlockWithPage(MemBlock *block,
                           bip_list<shm_handle<MemBlock>> &all_block_list);

  int32_t GetUnallocPages(ptrdiff_t addr_offset, size_t nbytes);

  void UnMapPages(const std::vector<index_t> &release_pages);

  void EmptyCache(bip_list<shm_handle<MemBlock>> &block_list);

  std::byte *GetBasePtr() const { return base_ptr_; }

  std::byte *GetEndPtr() const {
    return base_ptr_ + mem_block_nbytes * mapping_pages_.size();
  }
};

}