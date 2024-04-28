#pragma once

#include <ostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <memory>
#include <atomic>

#include <belong.h>
#include <shm.h>

#include <cuda.h>

#include <boost/interprocess/interprocess_fwd.hpp>

namespace mpool {
namespace bip = boost::interprocess;

struct PagesPoolConf {
        size_t PAGE_NBYTES;
        size_t POOL_NBYTES;
        std::string shm_name;
        std::string log_prefix;
        size_t shm_nbytes;
};

struct PhyPage {
        const size_t index;
        const CUmemGenericAllocationHandle cu_handle;
        Belong * const belong;
};

inline std::ostream& operator<<(std::ostream& out, const PhyPage &page)    {
        out << page.index;
        return out;
}

inline std::ostream& operator<<(std::ostream& out, const PhyPage *page)    {
        out << (page == nullptr ? "nullptr" : std::to_string(page->index));
        return out;
}

inline std::ostream& operator<<(std::ostream& out, const std::vector<const PhyPage *> pages) {
    out << "{len=" << pages.size() << "|";
    for (size_t k = 0; k < pages.size(); ++k) {
        if (k != 0) { out << ","; }
        out << pages[k]->index;
    }
    out << "}";
    return out;
}

// void ChangeBelong()


class HandleTransfer {
private:
    // typically, there is a limit on the maximum number of transferred FD
    // include/net/scm.h SCM_MAX_FD 253
    static const constexpr size_t TRANSFER_CHUNK_SIZE = 128;

    bip_shm    &shared_memory_;
    bip_mutex         *request_mutex_;
    bip::interprocess_condition *request_cond_;
    bip_mutex         *ready_mutex_;
    bip::interprocess_condition *ready_cond_;
    Belong                                            *shm_belong_list_;

    std::string                 master_name_;
    std::string                 slave_name_;
    std::vector<PhyPage> &phy_mem_list_;
    size_t                            phy_pages_num_;
    size_t                            phy_pages_nbytes_;

    /* master only */
    std::atomic<bool>            vmm_export_running_;
    std::unique_ptr<std::thread> vmm_export_thread_;

    void InitShm(Belong kFree) {
        request_mutex_ = shared_memory_.find_or_construct<bip_mutex>("HT_request_mutex_")();
        request_cond_ = shared_memory_.find_or_construct<bip::interprocess_condition>("HT_request_cond")();
        ready_mutex_ = shared_memory_.find_or_construct<bip_mutex>("HT_ready_mutex_")();
        ready_cond_ = shared_memory_.find_or_construct<bip::interprocess_condition>("HT_ready_cond")();
        shm_belong_list_ = shared_memory_.find_or_construct<Belong>("HT_belong_list")[phy_pages_num_](kFree);
    }

    void SendHandles(int fd_list[], size_t len, bip::scoped_lock<bip_mutex> &ready_lock);

    void ReceiveHandle(int fd_list[], size_t len);

    void ExportWorker();

public:
    HandleTransfer(bip_shm &shm, const PagesPoolConf &conf, std::vector<PhyPage> &ref_phy_pages);
    void InitMaster(Belong kFree);
    void InitSlave(Belong kFree);
    void ReleaseMaster();
};

}