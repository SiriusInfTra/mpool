#pragma once

#include <string>
#include <vector>

#include <mpool/util.h>
#include <mpool/shm.h>
#include <mpool/pages.h>
#include <mpool/belong.h>
#include <mpool/pages_list.h>
#include <mpool/cuda_handle.h>

#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/container/map.hpp>

#include <cuda.h>

namespace mpool {

class PagesPool {
public:
    static const constexpr index_t INSUFFICIENT_PAGE = FreeList::INVALID_POS;
    const PagesPoolConf config;
    const size_t page_num;
private:
    SharedMemory  &shared_memory_;

    std::vector<PhyPage>    phy_pages;
    
    FreeList                free_list_;
    PageHandleTransfer      handle_transfer_;
    BelongRegistry          belong_registery_;

    void Initialize(bool first_init);
public:
    static bool RemoveShm(const PagesPoolConf &config) {
        return bip::shared_memory_object::remove(config.shm_name.c_str());
    }
    PagesPool(SharedMemory &shared_memory, PagesPoolConf conf, bool first_init);

    ~PagesPool();

    BelongRegistry &GetBelongRegistry();

    

    Belong GetBelong(const std::string &name) {
        return belong_registery_.GetOrCreateBelong(name);
    }

    PhyPage* RetainPage(index_t index, Belong blg) {
        return nullptr; /*TODO */
    }
    
    index_t AllocConPages(Belong blg, num_t num_req, bip::scoped_lock<bip_mutex> &lock);

    index_t AllocConPagesAlign(Belong blg, num_t num_req, size_t align,
                               bip::scoped_lock<bip_mutex> &lock);

    std::vector<index_t> AllocDisPages(Belong blg, num_t num_req, bip::scoped_lock<bip_mutex> &lock);

    void FreePages(const std::vector<index_t> &pages, Belong blg, bip::scoped_lock<bip_mutex> &lock);

    // void FreePages(index_t index_begin, num_t pages_len, Belong blg, bip::scoped_lock<bip_mutex> &lock);

    const std::vector<PhyPage>& PagesView() const {
        return phy_pages;
    }

    bip::scoped_lock<bip_mutex> Lock() {
        return bip::scoped_lock<bip_mutex>{shared_memory_.GetMutex()};
    }

    void PrintStats();
};


}

