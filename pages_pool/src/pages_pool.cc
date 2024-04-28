#include "belong.h"
#include "freelist.h"
#include "pages.h"
#include "shm.h"
#include "util.h"
#include <algorithm>
#include <atomic>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <cinttypes>
#include <cstddef>
#include <iterator>
#include <limits>
#include <pages_pool.h>
#include <glog/logging.h>

#include <boost/container/map.hpp>
#include <utility>
#include <vector>
#include <numeric>

namespace mpool {


PagesPool::PagesPool(PagesPoolConf conf, bool force_cleanup): config(std::move(conf)), shared_memory_(),
    free_list_(shared_memory_), handle_transfer_(shared_memory_, config, phy_pages),
    belong_registery_(shared_memory_)
{
    if (force_cleanup) {
        bip::shared_memory_object::remove(config.shm_name.c_str());
    }
    shared_memory_ = bip_shm{
        bip::open_or_create, 
        config.shm_name.c_str(),
        config.shm_nbytes
    };
    CU_CALL(cuInit(0));
    auto atomic_init = [&] {
        ref_count_ = shared_memory_.find_or_construct<int>("ME_ref_count")();
        mutex_ = shared_memory_.find_or_construct<bip_mutex>("ME_mutex")();
        belong_registery_.Init();
    };
    shared_memory_.atomic_func(atomic_init);
    {
        bip::scoped_lock locker(*mutex_);
        self_master = (*ref_count_)++ == 0;
        if (self_master) {
            LOG(INFO) << config.log_prefix << "Init master";
            handle_transfer_.InitMaster(belong_registery_.GetFreeBelong());
        } else {
            LOG(INFO) << config.log_prefix << "Init slave";
            handle_transfer_.InitSlave(belong_registery_.GetFreeBelong());
        }
        free_list_.Init(config.POOL_NBYTES / config.PAGE_NBYTES);
    }
    LOG(INFO) << config.log_prefix;
}
index_t PagesPool::AllocConPages(Belong blg, num_t num_req, bip::scoped_lock<bip_mutex> &lock) {
    DLOG(INFO) << "AllocConPages: blg=" << blg << ", num_req=" << num_req << "."; 
    CHECK(lock.owns());
    size_t index_begin = free_list_.FindBestFit(num_req);
    if (index_begin == FreeList::INVALID_POS) {
        return PagesPool::INSUFFICIENT_PAGE;
    }
    free_list_.ClaimPages(index_begin, num_req);
    for (size_t k = 0; k < num_req; ++k) {
        auto &phy_page = phy_pages[index_begin + k];
        CHECK_EQ(*phy_page.belong, belong_registery_.GetFreeBelong());
        *phy_page.belong = blg;
    }
    blg.impl_->pages_num.fetch_add(num_req, std::memory_order_relaxed);
    return index_begin;
}
std::vector<index_t> PagesPool::AllocDisPages(Belong blg, num_t num_req, bip::scoped_lock<bip_mutex> &lock) {
    DLOG(INFO) << "AllocDisPages: blg=" << blg << ", num_req=" << num_req << "."; 
    CHECK(lock.owns());
    std::vector<index_t> ret;
    ret.reserve(num_req);
    for (size_t k = 0; k < num_req; ++k) {
        index_t index = free_list_.FindBestFit(1);
        if (index == FreeList::INVALID_POS) { break; }
        ret.push_back(index);
    }
    blg.impl_->pages_num.fetch_add(ret.size(), std::memory_order_relaxed);
    return ret;
}

void PagesPool::FreePages(const std::vector<index_t> &pages, Belong blg, bip::scoped_lock<bip_mutex> &lock) {
    CHECK(lock.owns());
    for (index_t index: pages) {
        auto &page = phy_pages[index];
        CHECK_EQ(*page.belong, blg);
        *page.belong = belong_registery_.GetFreeBelong();
        free_list_.ReleasePages(index, 1);
    }
    blg.impl_->pages_num.fetch_sub(pages.size(), std::memory_order_relaxed);
}

void PagesPool::FreePages(index_t index_begin, num_t pages_len, Belong blg, bip::scoped_lock<bip_mutex> &lock) {
    CHECK(lock.owns());
    for (size_t index = index_begin; index < index_begin + pages_len; ++index) {
        auto &page = phy_pages[index];
        CHECK_EQ(*page.belong, blg);
        *page.belong = belong_registery_.GetFreeBelong();
    }
    free_list_.ReleasePages(index_begin, pages_len);
    blg.impl_->pages_num.fetch_sub(pages_len, std::memory_order_relaxed);
}
PagesPool::~PagesPool() {
    if (self_master) {
        auto getRefCount = [&] {
          bip::scoped_lock locker(*mutex_);
          return *ref_count_;
        };
        int ref_count;
        while ((ref_count = getRefCount()) > 1) {
            LOG(INFO) << "[mempool] master wait slave shutdown, ref_count = "
                      << ref_count << ".";
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        handle_transfer_.ReleaseMaster();
        bip::shared_memory_object::remove(config.shm_name.c_str());
        LOG(INFO) << "[mempool] free master.";
    } else {
        bip::scoped_lock locker(*mutex_);
        --(*ref_count_);
        LOG(INFO) << "[mempool] free slave.";
    }
}

BelongRegistry &PagesPool::GetBelongRegistry() { return belong_registery_; }

}; // namespace mpool