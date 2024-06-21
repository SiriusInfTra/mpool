#include "pages.h"
#include <stream_context.h>

namespace mpool {

StreamBlockList::StreamBlockList(int device_id, cudaStream_t cuda_stream,
                                 SharedMemory &shared_memory,
                                 size_t small_block_nbytes)
    : device_id(device_id), current_stream_(cuda_stream),
      stream_block_list_(shared_memory->get_segment_manager()),
      small_block_nbytes_(small_block_nbytes) {}

MemBlock *StreamBlockList::CreateEntryExpandVA(ProcessLocalData &local,
                                               size_t nbytes) {
  ptrdiff_t addr_offset;
  if (local.all_block_list_.empty()) {
    addr_offset = 0;
  } else {
    auto *last_block =
        std::prev(local.all_block_list_.cend())->ptr(local.shared_memory_);
    addr_offset = last_block->addr_offset + last_block->nbytes;
  }
  auto *block = new (local.shared_memory_->allocate(sizeof(MemBlock)))
      MemBlock{.addr_offset = addr_offset,
               .nbytes = nbytes,
               .stream = current_stream_,
               .unalloc_pages =
                   local.mapping_region_.GetUnallocPages(addr_offset, nbytes),
               .device_id = device_id,
               .is_free = false,
               .is_small = nbytes < small_block_nbytes_};
  shm_ptr handle{block, local.shared_memory_};
  block->iter_all_block_list =
      local.all_block_list_.insert(local.all_block_list_.cend(), handle);
  block->iter_stream_block_list =
      stream_block_list_.insert(stream_block_list_.cend(), handle);
  return block;
}

MemBlock *StreamBlockList::GetPrevEntry(ProcessLocalData &local,
                                        MemBlock *entry) {
  auto iter = entry->iter_all_block_list;
  if (iter == local.all_block_list_.cbegin()) {
    return nullptr;
  }
  return std::prev(iter)->ptr(local.shared_memory_);
}

MemBlock *StreamBlockList::GetNextEntry(ProcessLocalData &local,
                                        MemBlock *entry) {
  auto iter = std::next(entry->iter_all_block_list);
  if (iter == local.all_block_list_.cend()) {
    return nullptr;
  }
  return iter->ptr(local.shared_memory_);
}

MemBlock *StreamBlockList::SplitBlock(ProcessLocalData &local,
                                      MemBlock *origin_entry, size_t remain) {
  CHECK_GT(origin_entry->nbytes, remain);
  CHECK_EQ(origin_entry->ref_count, 0) << origin_entry;
  /* [origin: remain] [insert_after_entry: nbytes - remain] */
  auto *insert_after_entry =
      new (local.shared_memory_->allocate(sizeof(MemBlock)))
          MemBlock{.addr_offset = origin_entry->addr_offset +
                                  static_cast<ptrdiff_t>(remain),
                   .nbytes = origin_entry->nbytes - remain,
                   .stream = origin_entry->stream,
                   .device_id = origin_entry->device_id,
                   .is_free = origin_entry->is_free,
                   .is_small = origin_entry->is_small,
                   .ref_count = 0};
  shm_ptr insert_after_entry_handle{insert_after_entry, local.shared_memory_};
  insert_after_entry->iter_all_block_list = local.all_block_list_.insert(
      std::next(origin_entry->iter_all_block_list), insert_after_entry_handle);
  insert_after_entry->iter_stream_block_list =
      stream_block_list_.insert(std::next(origin_entry->iter_stream_block_list),
                                insert_after_entry_handle);

  origin_entry->nbytes = remain;

  if (origin_entry->unalloc_pages > 0) {
    insert_after_entry->unalloc_pages = local.mapping_region_.GetUnallocPages(
        insert_after_entry->addr_offset, insert_after_entry->nbytes);
    origin_entry->unalloc_pages = local.mapping_region_.GetUnallocPages(
        origin_entry->addr_offset, origin_entry->nbytes);
  }

  return insert_after_entry;
}

MemBlock *StreamBlockList::MergeMemEntry(ProcessLocalData &local,
                                         MemBlock *first_block,
                                         MemBlock *secound_block) {
  CHECK_EQ(first_block->addr_offset + first_block->nbytes,
           secound_block->addr_offset);
  CHECK_EQ(first_block->is_free, secound_block->is_free);
  CHECK_EQ(first_block->is_small, secound_block->is_small);
  CHECK_EQ(first_block->ref_count, 0);
  CHECK_EQ(secound_block->ref_count, 0);

  first_block->nbytes += secound_block->nbytes;
  if (first_block->unalloc_pages > 0 || secound_block->unalloc_pages > 0) {
    first_block->unalloc_pages = local.mapping_region_.GetUnallocPages(
        first_block->addr_offset, first_block->nbytes);
  }

  local.all_block_list_.erase(secound_block->iter_all_block_list);
  stream_block_list_.erase(secound_block->iter_stream_block_list);
  memset(secound_block, 63, sizeof(MemBlock));
  local.shared_memory_->deallocate(secound_block);

  return first_block;
}
StreamFreeList::StreamFreeList(int device_id, cudaStream_t cuda_stream,
                               SharedMemory &shared_memory,

                               StreamBlockList &stream_block_list)
    : device_id(device_id), current_stream_(cuda_stream),
      stream_block_list_{&stream_block_list, shared_memory},
      free_block_list_{bip_multimap<size_t, shm_ptr<MemBlock>>(
                           shared_memory->get_segment_manager()),
                       bip_multimap<size_t, shm_ptr<MemBlock>>(
                           shared_memory->get_segment_manager())} {}

MemBlock *StreamFreeList::PopBlock(ProcessLocalData &local, bool is_small,
                                   size_t nbytes, size_t find_optimal_retry) {
  auto &free_list = free_block_list_[is_small];
  auto iter = free_list.lower_bound(nbytes);
  if (iter == free_list.cend()) {
    return nullptr;
  }
  auto *block = iter->second.ptr(local.shared_memory_);
  if (block->unalloc_pages > 0 && find_optimal_retry > 0) {
    // try to minimize the number of unallocated pages
    auto *optimal_block = block;
    auto optimal_iter = iter;
    for (auto iter1 = iter; iter1 != free_list.cend() && find_optimal_retry > 0;
         ++iter1, --find_optimal_retry) {
      if (auto *block1 = iter1->second.ptr(local.shared_memory_);
          block1->unalloc_pages < optimal_block->unalloc_pages) {
        optimal_block = block1;
        optimal_iter = iter1;
      }
    }
    block = optimal_block;
    iter = optimal_iter;
  }
  // LOG(INFO)  << "is small " << block->is_small << "free_list " <<
  // free_list.size();
  block->is_free = false;
  free_list.erase(iter);
  if (block->nbytes > nbytes) {
    auto *split_block = stream_block_list_.ptr(local.shared_memory_)
                            ->SplitBlock(local, block, nbytes);
    PushBlock(local, split_block);
    // split_block->iter_free_block_list = free_list.insert(std::make_pair(
    // split_block->nbytes, shm_ptr{split_block, shared_memory_}));
  }
  CHECK_EQ(block->ref_count, 0);
  return block;
}

MemBlock *StreamFreeList::PopBlock(ProcessLocalData &local, MemBlock *block) {
  CHECK(block->is_free);
  free_block_list_[block->is_small].erase(block->iter_free_block_list);
  block->is_free = false;
  return block;
}

MemBlock *StreamFreeList::PushBlock(ProcessLocalData &local, MemBlock *block) {
  LOG_IF(INFO, VERBOSE_LEVEL >= 2) << "PushBlock " << *block;
  CHECK_GT(block->nbytes, 0) << block;
  auto &free_list = free_block_list_[block->is_small];
  block->is_free = true;
  auto *stream_block_list_ptr = stream_block_list_.ptr(local.shared_memory_);
  if (auto prev_block = stream_block_list_ptr->GetPrevEntry(local, block);
      prev_block && prev_block->is_free &&
      prev_block->is_small == block->is_small &&
      prev_block->unalloc_pages == 0 && prev_block->stream == current_stream_) {
    // LOG(INFO)  << "is small " << block->is_small << "free_list " <<
    // free_list.size();
    free_list.erase(prev_block->iter_free_block_list);
    block = stream_block_list_ptr->MergeMemEntry(local, prev_block, block);
  }
  if (auto next_block = stream_block_list_ptr->GetNextEntry(local, block);
      next_block && next_block->is_free &&
      next_block->is_small == block->is_small &&
      next_block->unalloc_pages == 0 && next_block->stream == current_stream_) {
    // LOG(INFO)  << "is small " << block->is_small << "free_list " <<
    // free_list.size();
    free_list.erase(next_block->iter_free_block_list);
    block = stream_block_list_ptr->MergeMemEntry(local, block, next_block);
  }

  block->iter_free_block_list = free_list.insert(
      std::make_pair(block->nbytes, shm_ptr{block, local.shared_memory_}));
  // LOG(INFO)  << "is small " << block->is_small << "free_list " <<
  // free_list.size();
  return block;
}

MemBlock *StreamFreeList::MaybeMergeAdj(ProcessLocalData &local,
                                        MemBlock *entry) {
  if (entry->unalloc_pages > 0) {
    return entry;
  }
  CHECK(entry->is_free) << entry;
  CHECK(entry->is_small) << entry;
  auto *stream_block_list_ptr = stream_block_list_.ptr(local.shared_memory_);
  auto *prev_block = stream_block_list_ptr->GetPrevEntry(local, entry);
  DCHECK(prev_block == nullptr || !prev_block->is_free ||
         !prev_block->is_small || prev_block->unalloc_pages > 0)
      << prev_block;
  auto *next_block = stream_block_list_ptr->GetNextEntry(local, entry);
  DCHECK(next_block == nullptr || !next_block->is_free ||
         !next_block->is_small || next_block->unalloc_pages > 0)
      << next_block;
  bool put_free_list_large = true;
  size_t total_nbytes = entry->nbytes;
  for (auto *adj_block : {prev_block, next_block}) {
    if (adj_block == nullptr) {
      continue;
    }
    put_free_list_large &= !adj_block->is_small;
    total_nbytes += adj_block->nbytes;
  }
  if (put_free_list_large && total_nbytes >= 2_MB) {
    entry = PopBlock(local, entry);
    entry->is_small = false;
    entry = PushBlock(local, entry);
  }
  return entry;
}

void StreamBlockList::DumpStreamBlockList(ProcessLocalData &local,
                                          std::ostream &out) {
  DumpMemBlockColumns(out);
  for (auto handle : stream_block_list_) {
    DumpMemBlock(local, out, handle.ptr(local.shared_memory_));
  }
}
void StreamBlockList::DumpMemBlockColumns(std::ostream &out) {
  out << "start,len,next,prev,unalloc_pages,is_free,is_small" << "\n";
}
void StreamBlockList::DumpMemBlock(ProcessLocalData &local, std::ostream &out,
                                   MemBlock *block) {
  auto *prev = GetPrevEntry(local, block);
  auto *next = GetNextEntry(local, block);
  out << block->addr_offset << "," << block->nbytes << ","
      << (next ? next->addr_offset : -1) << ","
      << (prev ? prev->addr_offset : -1) << "," << block->unalloc_pages << ","
      << block->is_free << "," << block->is_small << "\n";
}

void StreamFreeList::DumpFreeBlockList(ProcessLocalData &local, bool is_small,
                                       std::ostream &out) {
  auto *stream_block_list_ptr = stream_block_list_.ptr(local.shared_memory_);
  stream_block_list_ptr->DumpMemBlockColumns(out);
  for (auto [nbytes, handle] : free_block_list_[is_small]) {
    auto *block = handle.ptr(local.shared_memory_);
    stream_block_list_ptr->DumpMemBlock(local, out, block);
  }
}

bool StreamBlockList::CheckState(ProcessLocalData &local,
                                 bool check_global_block_list) {
  for (auto iter = stream_block_list_.cbegin();
       iter != stream_block_list_.cend(); ++iter) {
    auto *block = iter->ptr(local.shared_memory_);
    if (block->stream != current_stream_) {
      LOG(FATAL) << "block's stream is not current stream: " << block << ".";
      return false;
    }
    if (int32_t unalloc_pages = local.mapping_region_.GetUnallocPages(
            block->addr_offset, block->nbytes);
        block->unalloc_pages != unalloc_pages) {
      LOG(INFO) << local.mapping_region_.GetUnallocPages(block->addr_offset,
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
      auto *next_block = next_iter->ptr(local.shared_memory_);
      if (next_block->addr_offset <
          block->addr_offset + static_cast<ptrdiff_t>(block->nbytes)) {
        LOG(FATAL) << "block's next block is not (potential) continuous: "
                   << block << " -> " << next_block << ".";
        return false;
      }
    }
  }
  if (check_global_block_list) {
    for (auto iter = local.all_block_list_.cbegin(); iter != local.all_block_list_.cend();
         ++iter) {
      auto *block = iter->ptr(local.shared_memory_);
      if (auto next_iter = std::next(block->iter_all_block_list);
          next_iter != local.all_block_list_.cend()) {
        auto *next_block = next_iter->ptr(local.shared_memory_);
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
bool StreamFreeList::CheckState(ProcessLocalData &local) {
  std::unordered_set<MemBlock *> free_blocks;
  for (auto &freelist : free_block_list_) {
    for (auto [nbytes, handle] : freelist) {
      auto *block = handle.ptr(local.shared_memory_);
      if (block->is_free == false) {
        LOG(FATAL) << "block is not free but in freeelist: " << block << ".";
        return false;
      }
      if (block->stream != current_stream_) {
        LOG(FATAL) << "block's stream is not current stream: " << block << ".";
        return false;
      }
      if (block->nbytes != nbytes) {
        LOG(FATAL) << "block's nbytes is not the nbytes in freelist: " << block
                   << ", nbytes in freelist:" << nbytes << ".";
        return false;
      }
      free_blocks.insert(block);
    }
  }
  auto *stream_block_list_ptr = stream_block_list_.ptr(local.shared_memory_);
  for (auto [iter, end] = stream_block_list_ptr->Iterators(); iter != end;
       ++iter) {
    auto *block = iter->ptr(local.shared_memory_);
    if (block->is_free == true && free_blocks.count(block) == 0) {
      LOG(FATAL) << "block is free but not in freelist: " << block << ".";
      return false;
    }
  }

  return true;
}
void StreamContext::MoveFreeBlockTo(ProcessLocalData &local,
                                    StreamContext &other_context) {
  /* 1. fast return */
  if (other_context.stream_block_list.stream_block_list_.empty()) {
    return;
  }

  /* 2. make sure this->stream_block_list.back().addr_offset >
   * other->stream_block_list.back().addr_offset */
  MemBlock last_mem_block{.addr_offset = std::numeric_limits<ptrdiff_t>::max(),
                          .nbytes = 0};
  this->stream_block_list.stream_block_list_.insert(
      this->stream_block_list.stream_block_list_.cend(),
      shm_ptr{&last_mem_block, local.shared_memory_});

  /* merge two stream_*/
  auto iter = this->stream_block_list.stream_block_list_.begin();
  auto *block = iter->ptr(local.shared_memory_);
  for (auto iter_other =
           other_context.stream_block_list.stream_block_list_.begin();
       iter_other !=
       other_context.stream_block_list.stream_block_list_.cend();) {
    auto *block_other = iter_other->ptr(local.shared_memory_);
    if (block_other->is_free) {
      block_other->stream = cuda_stream;
      while (block->addr_offset < block_other->addr_offset) {
        ++iter;
        block = iter->ptr(local.shared_memory_);
      }
      block_other->iter_stream_block_list =
          this->stream_block_list.stream_block_list_.insert(iter, *iter_other);
      other_context.stream_free_list.PopBlock(local, block);
      this->stream_free_list.PushBlock(local, block);
      iter_other = this->stream_block_list.stream_block_list_.erase(
          block_other->iter_stream_block_list);
    } else {
      ++iter_other;
    }
  }

  /* remove last fake memory block */
  CHECK_EQ(this->stream_block_list.stream_block_list_.back().ptr(
               local.shared_memory_),
           &last_mem_block);
  this->stream_block_list.stream_block_list_.erase(
      std::prev(this->stream_block_list.stream_block_list_.cend()));
}
std::pair<bip_list<shm_ptr<MemBlock>>::const_iterator,
          bip_list<shm_ptr<MemBlock>>::const_iterator>
StreamBlockList::Iterators() const {
  return {stream_block_list_.begin(), stream_block_list_.end()};
}
} // namespace mpool