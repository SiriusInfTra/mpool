#pragma once

#include <shm.h>

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

#include <boost/container/list.hpp>
#include <boost/container/map.hpp>

namespace mpool {

struct MemBlock {
  ptrdiff_t addr_offset;
  size_t nbytes;
  cudaStream_t stream;
  int32_t unalloc_pages;
  bool is_free;
  bool is_small;
  int32_t ref_count;

  bip_list<shm_ptr<MemBlock>>::iterator iter_all_block_list;
  bip_list<shm_ptr<MemBlock>>::iterator iter_stream_block_list;
  bip_multimap<size_t, shm_ptr<MemBlock>>::iterator iter_free_block_list;
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

}