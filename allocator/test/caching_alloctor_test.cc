
#include "shm.h"
#include "util.h"
#include <chrono>
#include <memory>
#include <pages_pool.h>
#include <caching_allocator.h>
#include <unordered_set>
#include <random>
#include <thread>

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
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
        if (dur > std::chrono::milliseconds(1)) {
            std::cout << name << " costs " << std::chrono::duration_cast<std::chrono::microseconds>(dur).count() / rep
                << " us." << std::endl;
        }

    }
};


class WrapMemBlock: public std::enable_shared_from_this<WrapMemBlock> {
private:
    CachingAllocator &allocator;
    size_t nbytes;
    MemBlock *mem_block;
    static size_t total_nbytes;
public:

    WrapMemBlock(size_t nbytes, CachingAllocator &allocator): allocator(allocator), nbytes(nbytes) {
        mem_block = allocator.Alloc(nbytes, 0);
        LOG(INFO) << "Allocated " << nbytes << " bytes at " << mem_block << ".";
        total_nbytes += nbytes;
    }
    ~WrapMemBlock() {
        allocator.Free(mem_block);
        LOG(INFO) << "Freed " << nbytes << " bytes at " << mem_block << ".";
        total_nbytes -= nbytes;
    }

    static size_t GetTotalNBytes() {
        return total_nbytes;
    }
};

size_t WrapMemBlock::total_nbytes = 0;

void run(const PagesPoolConf &config, const std::string &name, int seed) {
    PagesPool page_pool{config, false};
    CachingAllocatorConfig caching_allocator_config{
        .log_prefix = "CA ",
        .shm_name = "ca_wyk",
        .shm_nbytes = 1_GB,
        .va_range_scale = 4,
        .belong_name = name,
        .small_block_nbytes = 2_MB,
        .align_nbytes = 512_B
    };
    CachingAllocator::RemoveShm(caching_allocator_config);
    SharedMemory shared_memory{caching_allocator_config.shm_name, caching_allocator_config.shm_nbytes};
    CachingAllocator caching_allocator{shared_memory, page_pool, caching_allocator_config, true};

    std::mt19937 rng{static_cast<unsigned long>(seed)};
    
    std::vector<std::shared_ptr<WrapMemBlock>> own_pages;

    for (size_t k = 0; k < 1000; ++k) {
        // DLOG(INFO) << "k = " << k << " usage = " << ByteDisplay(WrapMemBlock::GetTotalNBytes()) << " | " << caching_allocator_config.belong.GetPagesNum() << ".";
        size_t nbytes = std::uniform_int_distribution<num_t>(1, 1_GB)(rng);
        // auto lock = page_pool.Lock();
        if (WrapMemBlock::GetTotalNBytes() + nbytes > 11_GB) {
            std::shuffle(own_pages.begin(), own_pages.end(), rng);
            own_pages.clear();
            caching_allocator.EmptyCache(0);
        } else {
            own_pages.push_back(std::make_shared<WrapMemBlock>(nbytes, caching_allocator));
        }
        std::this_thread::sleep_for(
            std::chrono::milliseconds(std::uniform_int_distribution<unsigned long>(10, 50)(rng)));
        
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
    run(conf, "test", 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
}