#pragma once

#include <atomic>
#include <cstddef>
#include <iostream>
#include <ostream>
#include <string>

#include <shm.h>
#include <util.h>

#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_sharable_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

namespace mpool {
struct BelongImpl {
    const index_t      index;
    const std::string  name;
    std::atomic<num_t> pages_num;
};

class Belong {
    friend class PagesPool;
private:
    BelongImpl *impl_;
public:
    Belong(BelongImpl *impl): impl_(impl) {}

    num_t GetPagesNum() const {
        return impl_->pages_num.load(std::memory_order_relaxed);
    }

    index_t GetIndex() const {
        return impl_->index;
    }

    const std::string &GetName() const {
        return impl_->name;
    }

    bool operator==(const Belong& rhs) const {
        auto ptr = dynamic_cast<const Belong*>(&rhs); 
        if (ptr != nullptr) {
            return impl_->index == ptr->impl_->index;
        } 
        return false;
    }

    friend std::ostream& operator<<(std::ostream& out, const Belong &belong)  {
        out << belong.impl_->name;
        return out;
    }
};


class BelongRegistry {
private:
    bip_shm &shm_;
    bip_vector<shm_handle<BelongImpl>> *registered_belongs;
    bip::interprocess_sharable_mutex *mutex;
    BelongImpl *kFreeBelong;

    BelongImpl *CreateBelong(size_t index, std::string name) {
        auto *belong_impl = shm_.allocate(sizeof(BelongImpl));
        return new (belong_impl) BelongImpl{index, name, 0};
    }

public:
    BelongRegistry(bip_shm &shm): shm_(shm) {}

    void Init() {  
        registered_belongs = shm_.find_or_construct<bip_vector<shm_handle<BelongImpl>>>
            ("BR_registered_belongs")(shm_.get_segment_manager());
        mutex = shm_.find_or_construct<bip::interprocess_sharable_mutex>("BR_mutex")();
        bip::scoped_lock lock{*mutex};
        if (registered_belongs->empty()) {
            kFreeBelong = CreateBelong(0, "FREE");
            registered_belongs->emplace_back(kFreeBelong, shm_);
        } else {
            kFreeBelong = registered_belongs->front().ptr(shm_);
        }
    }

    Belong GetOrCreateBelong(std::string name) {
        bip::scoped_lock lock{*mutex};
        for (auto handle : *registered_belongs) {
            auto *belong = handle.ptr(shm_);
            if (belong->name == name) {
                return belong;
            }
        }
        auto *belong = CreateBelong(registered_belongs->size(), name);
        registered_belongs->emplace_back(belong, shm_);
        return belong;
    }

    Belong GetFreeBelong() const {
        return kFreeBelong;
    }
};
}