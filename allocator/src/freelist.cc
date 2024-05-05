#include <cstddef>
#include <freelist2.h>

#include <glog/logging.h>
namespace mpool {

unsigned EntryList::GetUnallocPages(ptrdiff_t addr_offset, size_t nbytes) {
    index_t index_begin = addr_offset / MEM_BLOCK_NBYTES;
    index_t index_end =
        (addr_offset + nbytes + MEM_BLOCK_NBYTES - 1) / MEM_BLOCK_NBYTES;
    unsigned unalloc_pages = 0;
    for (index_t index = index_begin; index < index_end; ++index) {
        if (mapped_mem_list_[index] == nullptr) {
            unalloc_pages++;
        }
    }
    return unalloc_pages;
}

EntryList::EntryList(
    size_t page_nbytes, 
    const std::string &log_prefix, 
    Belong belong,
    bip_shm &shm,
    const std::vector<const PhyPage*> &ref_mapped_pages
): MEM_BLOCK_NBYTES(page_nbytes), shared_memory_(shm), policy_(belong), log_prefix_(log_prefix),
 mapped_mem_list_(ref_mapped_pages) {}

void EntryList::Init() {
    {
    std::string name = "EL_entry_list_" + policy_.GetName();
    entry_list_ = shared_memory_.find_or_construct<bip_list<shm_handle<MemEntry>>>(name.c_str())
        (shared_memory_.get_segment_manager());
    }
    {
    std::string name = "EL_entry_by_addr_" + policy_.GetName();
    entry_by_addr_ = shared_memory_.find_or_construct<bip_map<ptrdiff_t, shm_handle<MemEntry>>>(name.c_str())
        (shared_memory_.get_segment_manager());
    }
}

void EntryList::LinkNewEntry(MemEntry *entry) {
    if (entry_list_->empty()) {
        CHECK_EQ(entry->addr_offset, 0);
    } else {
        auto *last_entry = entry_list_->back().ptr(shared_memory_);
        CHECK_EQ(entry->addr_offset, last_entry->addr_offset + last_entry->nbytes);
    }

    bool succ;
    std::tie(entry->pos_entrytable, succ) = entry_by_addr_->insert(std::make_pair(entry->addr_offset, shm_handle{entry, shared_memory_}));
    CHECK(succ);
    entry->pos_entrylist = entry_list_->insert(entry_list_->cend(), shm_handle{entry, shared_memory_});
}

MemEntry *EntryList::GetEntry(std::ptrdiff_t addr_offset) {
  auto iter = entry_by_addr_->find(addr_offset);
  if (iter == entry_by_addr_->cend()) {
    return nullptr;
  }
  auto *entry = iter->second.ptr(shared_memory_);
  CHECK(iter == entry->pos_entrytable); 
  return entry;
}

MemEntry *EntryList::GetEntryWithAddr(ptrdiff_t addr_offset) {
    auto iter = entry_by_addr_->lower_bound(addr_offset);
    MemEntry *entry;
    if (iter == entry_by_addr_->cend()) {
        entry = std::prev(entry_list_->cend())->ptr(shared_memory_);
    } else if (entry = iter->second.ptr(shared_memory_); entry->addr_offset > addr_offset) {
        entry = GetPrevEntry(entry);
    }
    return entry;
}

MemEntry *EntryList::IterateRange(ptrdiff_t addr_offset, size_t nbytes, std::function<MemEntry *(MemEntry *)> func) {
    auto *entry = GetEntryWithAddr(addr_offset);
    CHECK(entry != nullptr);
    CHECK_LE(entry->addr_offset, addr_offset);
    while (true) {
        entry = func(entry);
        if (!entry) {
            break;
        }
        if (entry->addr_offset + entry->nbytes >= addr_offset + nbytes) {
            break;
        }
        entry = GetNextEntry(entry);
        if (!entry) {
            break;
        }
    }
    return entry;
}

MemEntry *EntryList::GetPrevEntry(MemEntry *entry) {
    auto iter = entry->pos_entrylist;
    if (iter == entry_list_->cbegin()) {
        return nullptr;
    }
    return std::prev(iter)->ptr(shared_memory_);
}

MemEntry *EntryList::GetNextEntry(MemEntry *entry) {
    auto iter = std::next(entry->pos_entrylist);
    if (iter == entry_list_->cend()) {
        return nullptr;
    }
    return iter->ptr(shared_memory_);
}

MemEntry *EntryList::SplitEntry(MemEntry *origin_entry, size_t remain) {
    CHECK_GT(origin_entry->nbytes, remain);
    auto *entry_split = reinterpret_cast<MemEntry *>(shared_memory_.allocate(sizeof(MemEntry)));
    shm_handle handle_split{entry_split, shared_memory_};
    auto pos_entrylist = entry_list_->insert(std::next(origin_entry->pos_entrylist), handle_split);
    auto [pos_entrytable, insert_succ] = entry_by_addr_->insert(std::make_pair(entry_split->addr_offset, handle_split));
    CHECK(insert_succ);

    /* [origin: remain] [split: nbytes - remain] */
    ptrdiff_t addr_offset = origin_entry->addr_offset + static_cast<ptrdiff_t>(remain);
    size_t nbytes = origin_entry->nbytes - remain;
    new (entry_split) MemEntry {
        .addr_offset = addr_offset,
        .nbytes = nbytes,
        .unalloc_pages = origin_entry->unalloc_pages == 0 ? 0 : GetUnallocPages(addr_offset, nbytes),
        .is_free = origin_entry->is_free,
        .is_small = origin_entry->is_small,
        .pos_entrylist = pos_entrylist,
        .pos_entrytable = pos_entrytable
    };

    origin_entry->nbytes = remain;
    origin_entry->unalloc_pages = origin_entry->unalloc_pages == 0 ? 0 : 
        GetUnallocPages(origin_entry->addr_offset, origin_entry->nbytes);

    return entry_split;
}

MemEntry *EntryList::MergeMemEntry(MemEntry *first_entry, MemEntry *secound_entry) {
    CHECK_EQ(first_entry->addr_offset + first_entry->nbytes, secound_entry->addr_offset);
    CHECK_EQ(first_entry->is_free, secound_entry->is_free);
    CHECK_EQ(first_entry->is_small, secound_entry->is_small);

    first_entry->nbytes += secound_entry->nbytes;
    if (first_entry->unalloc_pages != 0 || secound_entry->unalloc_pages != 0) {
        first_entry->unalloc_pages = 
            GetUnallocPages(first_entry->unalloc_pages, secound_entry->unalloc_pages);
    }

    entry_list_->erase(secound_entry->pos_entrylist);
    entry_by_addr_->erase(secound_entry->pos_entrytable);
    memset(secound_entry, 63, sizeof(MemEntry));
    shared_memory_.deallocate(secound_entry);

    return first_entry;
}

void EntryList::DumpMemEntryColumns(std::ostream &out) {
    out << "start,len,next,prev,unalloc_pages,is_free,is_train"
        << "\n";
}
void EntryList::DumpMemEntry(std::ostream &out, MemEntry *entry) {
    auto *prev = GetPrevEntry(entry);
    auto *next = GetNextEntry(entry);
    out << entry->addr_offset << "," << entry->nbytes << ","
        << (next ? next->addr_offset : -1) << ","
        << (prev ? prev->addr_offset : -1) << "," << entry->unalloc_pages << ","
        << entry->is_free << "," << entry->is_small << "\n";
}

void EntryList::DumpMemEntryList(std::ostream &out) {
    DumpMemEntryColumns(out);
    for (auto handle : *entry_list_) {
        auto *entry = handle.ptr(shared_memory_);
        DumpMemEntry(out, entry);
    }
    out << std::flush;
}
bool EntryList::CheckState() {
    if constexpr (DUMP_BEFORE_CHECK) {
        LOG(INFO) << log_prefix_ << "Dump entry_list.";
        DumpMemEntryList(std::cerr);
    }
    for (auto handle : *entry_list_) {
        auto *entry = handle.ptr(shared_memory_);
        if (auto *prev = GetPrevEntry(entry); prev) {
            CHECK_EQ(prev->addr_offset + prev->nbytes, entry->addr_offset);
        }
        if (auto *next = GetNextEntry(entry); next) {
            CHECK_EQ(entry->addr_offset + entry->nbytes, next->addr_offset);
        }
    }
    return true;
}
FreeList::FreeList(
    size_t MEM_BLOCK_NBYTES, 
    bip_shm &shared_memory, 
    EntryList &list_index, 
    bool is_small, 
    const std::string &log_prefix, 
    Belong policy, 
    size_t small_block_nbytes
) : MEM_BLOCK_NBYTES(MEM_BLOCK_NBYTES), shared_memory_(shared_memory), log_prefix_(log_prefix), 
        list_index_(list_index), is_small_(is_small), policy_(policy),  small_block_nbytes_(small_block_nbytes) {}

MemEntry *FreeList::PopFreeEntry(size_t nbytes, bool do_split, size_t require_allocated) {
    auto iter = entry_by_nbytes_->lower_bound(nbytes);
    if (iter == entry_by_nbytes_->cend()) {
        return nullptr;
    }

    auto *free_entry = iter->second.ptr(shared_memory_);
    if (free_entry->unalloc_pages > 0 && require_allocated > 0) {
        auto min_unalloc_iter = iter;
        auto min_unalloc_num = free_entry->unalloc_pages;
        auto min_unalloc_entry = free_entry;
        do {
            ++iter;
            --require_allocated;
            free_entry = iter->second.ptr(shared_memory_);
            if (free_entry->unalloc_pages < min_unalloc_num) {
                min_unalloc_iter = iter;
                min_unalloc_num = free_entry->unalloc_pages;
                min_unalloc_entry = free_entry;
            }
        } while (min_unalloc_num > 0 && require_allocated > 0 && iter != entry_by_nbytes_->cend());
        free_entry = min_unalloc_entry;
        iter = min_unalloc_iter;
    }

    entry_by_nbytes_->erase(free_entry->pos_freelist);
    free_entry->is_free = false;

    CHECK_GE(free_entry->nbytes, nbytes);
    if (do_split && free_entry->nbytes > nbytes) {
        auto split_entry = list_index_.SplitEntry(free_entry, nbytes);
        split_entry->is_free = false;
        PushFreeEntry(split_entry);
    }
    // if (policy_ == Belong::kInfer && !is_small_) {
    //     CHECK_EQ(free_entry->addr_offset % MEM_BLOCK_NBYTES, 0);
    //     CHECK_EQ(free_entry->nbytes % MEM_BLOCK_NBYTES, 0);
    // }
    CHECK_EQ(free_entry->is_small, is_small_);
    return free_entry;
}

MemEntry *FreeList::PopFreeEntry(MemEntry *free_entry) {
    CHECK_EQ(free_entry->is_small, is_small_);
    CHECK(free_entry->is_free);
    entry_by_nbytes_->erase(free_entry->pos_freelist);
    free_entry->is_free = false;
    CHECK_EQ(free_entry->is_small, is_small_);
    return free_entry;
}

MemEntry* FreeList::PushFreeEntry(MemEntry *entry) {
    CHECK_EQ(entry->is_free, false);
    CHECK_EQ(entry->is_small, is_small_);
    // if (policy_ == Belong::kInfer && !is_small_) {
    //     CHECK_EQ(entry->addr_offset % MEM_BLOCK_NBYTES, 0);
    //     CHECK_EQ(entry->nbytes % MEM_BLOCK_NBYTES, 0);
    // }
    entry->is_free = true;
    if (auto prev_entry = list_index_.GetPrevEntry(entry); prev_entry 
        && prev_entry->is_free 
        && prev_entry->is_small == entry->is_small
    ) {
        entry_by_nbytes_->erase(prev_entry->pos_freelist);
        entry = list_index_.MergeMemEntry(prev_entry, entry);
    }
    if (auto next_entry = list_index_.GetNextEntry(entry); next_entry 
        && next_entry->is_free 
        && next_entry->is_small == entry->is_small
    ) {
        entry_by_nbytes_->erase(next_entry->pos_freelist);
        entry = list_index_.MergeMemEntry(entry, next_entry);
    }

    entry->pos_freelist =
        entry_by_nbytes_->insert(std::make_pair(entry->nbytes, shm_handle{entry, shared_memory_}));
    // if (!is_small_) {
    //   CHECK_GE(entry->nbytes, small_block_nbytes_);
    // }
    // CHECK(!alloc_conf::STRICT_CHECK_STATE || CheckState());
    return entry;
}

void FreeList::DumpFreeList(std::ostream &out) {
    list_index_.DumpMemEntryColumns(out);
    for (auto &&[nbytes, shm_handle] : *entry_by_nbytes_) {
        auto *entry = shm_handle.ptr(shared_memory_);
        list_index_.DumpMemEntry(out, entry);
    }
    out << std::flush;
}

bool FreeList::CheckState() {
    if constexpr (DUMP_BEFORE_CHECK) {
    LOG(INFO) << log_prefix_ << "Dump free_list "<< (is_small_ ? "small" : "large") << ".";
        DumpFreeList(std::cerr);
    }
    for (auto &&[nbytes, shm_handle] : *entry_by_nbytes_) {
        auto *entry = shm_handle.ptr(shared_memory_);
        CHECK_EQ(entry->is_free, true);
        CHECK_EQ(entry->nbytes, nbytes);
        CHECK_EQ(entry->is_small, is_small_);
        PopFreeEntry(entry);
        entry = PushFreeEntry(entry);
        CHECK_EQ(entry->nbytes, nbytes) << entry;
    }
    return true;
}


} // namespace mpool