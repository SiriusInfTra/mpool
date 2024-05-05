#pragma once
#include <cstddef>


#include <pages.h>
#include <shm.h>
#include <belong.h>

#include <boost/container/list.hpp>
#include <boost/container/map.hpp>

namespace mpool {

const constexpr bool DUMP_BEFORE_CHECK = false;
const constexpr bool ENABLE_CHECK_OP = false;
const constexpr bool ENABLE_CHECK_U_OP = false;


struct MemEntry {
    ptrdiff_t    addr_offset;
    size_t       nbytes;
    unsigned     unalloc_pages;
    
    bool    is_free;
    bool    is_small;

    bip_list<shm_handle<MemEntry>>::iterator                pos_entrylist;
    bip_map<ptrdiff_t, shm_handle<MemEntry>>::iterator      pos_entrytable;
    bip_multimap<size_t, shm_handle<MemEntry>>::iterator    pos_freelist;
};

inline std::string ToString(const MemEntry *entry) {
    if (entry == nullptr) { return "nullptr"; }
    std::stringstream ss;
    ss << "{addr_offset=" << entry->addr_offset 
        << ", nbytes=" << entry->nbytes
        << ", unalloc_pages=" << entry->unalloc_pages
        << ", is_free=" << entry->is_free
        << ", is_small=" << entry->is_small << "}";
    return ss.str();
}



inline std::ostream & operator<<(std::ostream &os, const MemEntry *entry)    {
    os << ToString(entry);
    return os;
}

class EntryList {
private:
    const size_t MEM_BLOCK_NBYTES;
    bip_shm &shared_memory_;
    const Belong policy_;
    const std::string &log_prefix_;
    const std::vector<const PhyPage*> &mapped_mem_list_;


    bip_list<shm_handle<MemEntry>>              *entry_list_;
    bip_map<ptrdiff_t, shm_handle<MemEntry>>    *entry_by_addr_;

    unsigned GetUnallocPages(ptrdiff_t addr_offset, size_t nbytes);

  public:
    EntryList(size_t page_nbytes, const std::string &log_prefix, Belong belong, bip_shm &shm, 
                                            const std::vector<const PhyPage*> &ref_mapped_pages);

    void Init();

    void LinkNewEntry(MemEntry *entry);

    MemEntry *GetEntry(std::ptrdiff_t addr_offset);

    MemEntry *GetEntryWithAddr(ptrdiff_t addr_offset);

    MemEntry *IterateRange(ptrdiff_t addr_offset, size_t nbytes, std::function<MemEntry *(MemEntry *)> func);

    MemEntry *GetPrevEntry(MemEntry *entry);

    MemEntry *GetNextEntry(MemEntry *entry);

    MemEntry *SplitEntry(MemEntry *origin_entry, size_t remain);

    MemEntry *MergeMemEntry(MemEntry *first_entry, MemEntry *secound_entry);

    void DumpMemEntryColumns(std::ostream &out);

    void DumpMemEntry(std::ostream &out, MemEntry *entry);

    void DumpMemEntryList(std::ostream &out);

    bool CheckState();
};

class FreeList {
private:
    const size_t MEM_BLOCK_NBYTES;
    bip_shm &shared_memory_;
    const std::string &log_prefix_;
    EntryList& list_index_;
    const bool is_small_;
    const Belong policy_;
    const size_t small_block_nbytes_;
    bip_multimap<size_t, shm_handle<MemEntry>> *entry_by_nbytes_;
public:
    FreeList(size_t MEM_BLOCK_NBYTES, bip_shm &shared_memory, EntryList &list_index, bool is_small, const std::string &log_prefix,
            Belong policy, size_t small_block_nbytes);

    MemEntry *PopFreeEntry(size_t nbytes, bool do_split = true, size_t require_allocated = 0);

    MemEntry *PopFreeEntry(MemEntry *free_entry);

    MemEntry* PushFreeEntry(MemEntry *entry);

    void DumpFreeList(std::ostream &out);

    bool CheckState();
};

}