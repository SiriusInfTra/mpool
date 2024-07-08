#include <cstddef>
#include <atomic>
#include <utility>
#include <vector>

#include <mpool/belong.h>
#include <mpool/pages.h>
#include <mpool/pages_list.h>
#include <mpool/shm.h>
#include <mpool/util.h>
#include <mpool/pages_pool.h>

#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/container/map.hpp>

#include <glog/logging.h>


namespace mpool {

std::ostream &operator<<(std::ostream &out, std::vector<index_t> vec) {
  out << "{";
  for (size_t k = 0; k < vec.size(); ++k) {
    if (k != 0) {
      out << ",";
    }
    out << vec[k];
  }
  out << "}";
  return out;
}

PagesPool::PagesPool(SharedMemory &shared_memory, PagesPoolConf conf,
                     bool first_init)
    : config(std::move(conf)), shared_memory_(shared_memory),
      free_list_(shared_memory_),
      handle_transfer_(shared_memory_, phy_pages, config),
      belong_registery_(shared_memory_) {
  CU_CALL(cuInit(0));
  belong_registery_.Init(config.pool_nbytes / config.page_nbytes);
  if (first_init) {
    LOG(INFO) << config.log_prefix << "Init PagesPool[Master].";
    handle_transfer_.InitMaster(belong_registery_.GetFreeBelong());
  } else {
    LOG(INFO) << config.log_prefix << "Init PagesPool[Mirror]";
    handle_transfer_.InitMirror();
  }

  free_list_.Init(config.pool_nbytes / config.page_nbytes);
  LOG(INFO) << config.log_prefix << "Init PagesPool OK.";
}
index_t PagesPool::AllocConPages(Belong blg, num_t num_req,
                                 bip::scoped_lock<bip_mutex> &lock) {
  LOG_IF(INFO, VERBOSE_LEVEL >= 2)
      << "AllocConPages: blg=" << blg << ", num_req=" << num_req << ".";
  CHECK(lock.owns());
  size_t index_begin = free_list_.FindBestFit(num_req);
  if (index_begin == FreeList::INVALID_POS) {
    return PagesPool::INSUFFICIENT_PAGE;
  }
  free_list_.ClaimPages(index_begin, num_req);
  for (size_t k = 0; k < num_req; ++k) {
    auto &phy_page = phy_pages[index_begin + k];
    CHECK_EQ(belong_registery_.GetFreeBelong(),
             belong_registery_.GetBelong(*phy_page.belong));
    *phy_page.belong = blg;
  }
  blg.Get()->pages_num.fetch_add(num_req, std::memory_order_relaxed);
  belong_registery_.GetFreeBelong().Get()->pages_num.fetch_sub(num_req, std::memory_order_relaxed);
  return index_begin;
}
std::vector<index_t>
PagesPool::AllocDisPages(Belong blg, num_t num_req,
                         bip::scoped_lock<bip_mutex> &lock) {
  LOG_IF(INFO, VERBOSE_LEVEL >= 2)
      << "AllocDisPages: blg=" << blg << ", num_req=" << num_req << ".";
  CHECK(lock.owns());
  std::vector<index_t> ret;
  ret.reserve(num_req);
  for (size_t k = 0; k < num_req; ++k) {
    index_t index = free_list_.FindBestFit(1);
    if (index == FreeList::INVALID_POS) {
      break;
    }
    auto &phy_page = phy_pages[index];
    free_list_.ClaimPages(index);
    CHECK_EQ(belong_registery_.GetBelong(*phy_page.belong),
             belong_registery_.GetFreeBelong());
    *phy_page.belong = blg;
    ret.push_back(index);
  }
  blg.Get()->pages_num.fetch_add(ret.size(), std::memory_order_relaxed);
  belong_registery_.GetFreeBelong().Get()->pages_num.fetch_sub(ret.size(), std::memory_order_relaxed);
  return ret;
}

void PagesPool::FreePages(const std::vector<index_t> &pages, Belong blg,
                          bip::scoped_lock<bip_mutex> &lock) {
  LOG_IF(INFO, VERBOSE_LEVEL >= 2)
      << "FreePages: pages=" << pages << ", blg=" << blg;
  CHECK(lock.owns());
  for (index_t index : pages) {
    auto &page = phy_pages[index];
    CHECK_EQ(belong_registery_.GetBelong(*page.belong), blg);
    *page.belong = belong_registery_.GetFreeBelong();
    free_list_.ReleasePages(index, 1);
  }
  blg.Get()->pages_num.fetch_sub(pages.size(), std::memory_order_relaxed);
  belong_registery_.GetFreeBelong().Get()->pages_num.fetch_add(pages.size(), std::memory_order_relaxed);
}

PagesPool::~PagesPool() {
  LOG(INFO) << config.log_prefix << "Release PagesPool";
}

BelongRegistry &PagesPool::GetBelongRegistry() { return belong_registery_; }

void PagesPool::PrintStats() {
  for (auto belong : belong_registery_.GetBelongs()) {
    LOG(INFO) << "~~~~~~~~~~ Belong " << belong.GetName() <<" ~~~~~~~~~~";
    LOG(INFO) << "pages_num: " << belong.GetPagesNum();
    LOG(INFO) << "allocated_nbytes: " << belong.GetAllocatedNbytes();
    // LOG(INFO) << ""
  }

}
}; // namespace mpool