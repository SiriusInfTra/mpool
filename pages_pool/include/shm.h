#pragma once

#include <boost/container/container_fwd.hpp>
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>

namespace mpool {

namespace bip = boost::interprocess;

using bip_shm = bip::managed_shared_memory;
using bip_mutex = bip::interprocess_mutex;

class bip_shm2 {
private:
    bip::managed_shared_memory shared_memory;
public:
    bip_mutex *mutex;

    bip_shm2(std::string name, size_t nbytes): shared_memory(bip::open_or_create, name.c_str(), nbytes) {
        auto init_func = [&](){
            mutex = shared_memory.find_or_construct<bip_mutex>("mutex")();
        };
        shared_memory.atomic_func(init_func);
        
    }

    bip::managed_shared_memory& operator->() {
        return shared_memory;
    }

    bip_mutex &GetMutex() {
        return *mutex;
    }
};

template<typename T>
class shm_handle {
private:
    bip_shm::handle_t handle_;
public:
    shm_handle(T *t, bip_shm &shm): handle_(shm.get_handle_from_address(t)) {}

    T *ptr(bip_shm &shm) const { 
        return reinterpret_cast<T*>(shm.get_address_from_handle(handle_));
    }
};

template<typename Key, typename Value>
using bip_map = boost::container::map<
    Key, 
    Value,
    std::less<Key>, bip::allocator<std::pair<const Key, Value>, bip_shm::segment_manager>>;

template<typename Key, typename Value>
using bip_multimap = boost::container::multimap<
    Key, 
    Value,
    std::less<Key>,
    bip::allocator<std::pair<const Key, Value>, bip_shm::segment_manager>>;

template<typename Type>
using bip_vector = boost::container::vector<
    Type,
    bip::allocator<Type, bip_shm::segment_manager>>;

template<typename Type>
using bip_list = boost::container::list<
    Type,
    bip::allocator<Type, bip_shm::segment_manager>>;

}
