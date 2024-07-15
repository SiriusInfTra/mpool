#pragma once

#include <mpool/shm.h>

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
  int device_id;
  bool is_free;
  bool is_small;
  int32_t ref_count;


  // For quickly finding adjacent blocks.
  bip_list<shm_ptr<MemBlock>>::iterator iter_all_block_list;
  // For quickly finding prev/next block in the same stream.
  bip_list<shm_ptr<MemBlock>>::iterator iter_stream_block_list;
  // For quickly finding the block by address. 
  bip_map<ptrdiff_t, shm_ptr<MemBlock>>::iterator iter_all_block_map;
  // For quickly finding the block in freelist by nbytes.
  bip_multimap<size_t, shm_ptr<MemBlock>>::iterator iter_free_block_list;
};

inline std::ostream &operator<<(std::ostream &out, const MemBlock &block) {
  out << "MemBlock {"
      << "addr_offset: " << block.addr_offset << ", "
      << "nbytes: " << block.nbytes << ", "
      << "stream: " << block.stream << ", "
      << "unalloc_pages: " << block.unalloc_pages << ", "
      << "device_id: " << block.device_id << ", "
      << "is_free: " << (block.is_free ? "true" : "false") << ", "
      << "is_small: " << (block.is_small ? "true" : "false") << ", "
      << "ref_count" << block.ref_count <<  "}";
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