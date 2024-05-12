#pragma once
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <memory>
#include <string>
#include <vector>
#include <memory>

#include <util.h>
#include <shm.h>
#include <pages.h>
#include <belong.h>
#include <pages_list.h>

#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/container/map.hpp>

#include <cuda.h>

namespace mpool {

class PagesPool {
public:
    static const constexpr index_t INSUFFICIENT_PAGE = FreeList::INVALID_POS;
    const PagesPoolConf config;
private:
    bip_shm  shared_memory_;

    bip_mutex     *mutex_;
    int                         *ref_count_;
    bool                        self_master;

    std::vector<PhyPage>    phy_pages;
    
    FreeList                free_list_;
    HandleTransfer          handle_transfer_;
    BelongRegistry          belong_registery_;
public:
    static bool RemoveShm(const PagesPoolConf &config) {
        return bip::shared_memory_object::remove(config.shm_name.c_str());
    }
    PagesPool(PagesPoolConf conf, bool force_cleanup = false);

    ~PagesPool();

    BelongRegistry &GetBelongRegistry();

    index_t AllocConPages(Belong blg, num_t num_req, bip::scoped_lock<bip_mutex> &lock);

    std::vector<index_t> AllocDisPages(Belong blg, num_t num_req, bip::scoped_lock<bip_mutex> &lock);

    void FreePages(const std::vector<index_t> &pages, Belong blg, bip::scoped_lock<bip_mutex> &lock);

    void FreePages(index_t index_begin, num_t pages_len, Belong blg, bip::scoped_lock<bip_mutex> &lock);

    const std::vector<PhyPage>& PagesView() const {
        return phy_pages;
    }

    bip::scoped_lock<bip_mutex> Lock() {
        return bip::scoped_lock<bip_mutex>{*mutex_};
    }

};


}

