#include <algorithm>
#include <chrono>
#include <functional>
#include <random>
#include <thread>
#include <unordered_set>
#include <unistd.h>

#include <mpool/shm.h>
#include <mpool/pages.h>
#include <mpool/pages_pool.h>
#include <mpool/util.h>

#include <glog/logging.h>
using namespace mpool;

class Recorder {
private:
    std::string name;
    size_t rep;
    std::chrono::steady_clock::time_point t0;
public:
    Recorder(std::string name, size_t rep = 1): name(std::move(name)), rep(rep), t0(std::chrono::steady_clock::now()) {}
    ~Recorder() {
        auto t1 = std::chrono::steady_clock::now();
        std::cout << name << " costs " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / rep
            << " us." << std::endl;
    }
};

void run(const PagesPoolConf &conf, const std::string &name, int seed) {
    SharableObject<PagesPool> page_pool{conf.shm_name, conf.shm_nbytes, conf};

    std::mt19937 rng{static_cast<unsigned long>(seed)};
    auto belong = page_pool->GetBelongRegistry().GetOrCreateBelong(name);
    std::unordered_set<index_t> own_pages;

    for (size_t k = 0; k < 1000; ++k) {
        DLOG(INFO) << "k = " << k << " usage = " << belong.GetPagesNum() << ".";
        num_t num_req = std::uniform_int_distribution<num_t>(1, 1_GB / conf.page_nbytes)(rng);
        double req_con = std::uniform_real_distribution<double>(0, 1)(rng);
        size_t s0 = own_pages.size();
        auto lock = page_pool->Lock();
        if (req_con > -20.9) {
            index_t index_begin;
            {
                Recorder recorder{"AllocConPages"};
                index_begin = page_pool->AllocConPages(belong, num_req, lock);
            }
            DLOG(INFO) << index_begin;
            if (index_begin != PagesPool::INSUFFICIENT_PAGE) {
                for (index_t k = 0; k < num_req; ++k) {
                    auto [_, succ] = own_pages.insert(index_begin + k);
                    CHECK(succ);
                }
            }
        } else {
            auto pages = page_pool->AllocDisPages(belong, num_req, lock);
            for (index_t index : pages) {
                auto [_, succ] = own_pages.insert(index);
                CHECK(succ);
            }
        }
        size_t s1 = own_pages.size();
        if (s0 == s1) {
            // alloc fail
            std::vector<index_t> vv(own_pages.cbegin(), own_pages.cend());
            std::shuffle(vv.begin(), vv.end(), rng);
            {
                Recorder recorder{"FreePages"};
                page_pool->FreePages(vv, belong, lock);
            }
            own_pages.clear();
            DLOG(INFO) << "OOM clean.";
        }
        lock.unlock();
        std::this_thread::sleep_for(
            std::chrono::milliseconds(std::uniform_int_distribution<int32_t long>(10, 50)(rng)));
        CHECK_EQ(belong.GetPagesNum(), own_pages.size());
        
    }
}
int main() {
    PagesPoolConf conf{
        .page_nbytes = 32_MB,
        .pool_nbytes = 12_GB,
        .shm_name = "mempool_wyk",
        .log_prefix = "mempool",
        .shm_nbytes = 1_GB,
    };
    PagesPool::RemoveShm(conf);
    std::thread kTrain{run, std::ref(conf), "train", 42};   
    std::thread kInfer{run, std::ref(conf), "infer", 43};
    kTrain.join();
    kInfer.join();
}