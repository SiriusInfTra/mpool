#pragma once

#include "util.h"
#include <boost/container/container_fwd.hpp>
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/unordered/unordered_map_fwd.hpp>
#include <functional>
#include <glog/logging.h>

namespace mpool {

namespace bip = boost::interprocess;

using bip_shm = bip::managed_shared_memory;

template <class Mutex> using bip_lock = bip::scoped_lock<Mutex>;

using bip_mutex = bip::interprocess_mutex;
using bip_cond = bip::interprocess_condition;

class SharedMemory {
public:
  const std::string name;

private:
  bip::managed_shared_memory shared_memory;

  bip_mutex *mutex_;
  int *ref_count_;

  bool is_init_ = false;
  bool is_deinit_ = false;

public:
  SharedMemory(std::string name, size_t nbytes)
      : name(name), shared_memory(bip::open_or_create, name.c_str(), nbytes) {
    auto init_func = [&]() {
      mutex_ = shared_memory.find_or_construct<bip_mutex>("mutex")();
      ref_count_ = shared_memory.find_or_construct<int>("ref_count")(0);
    };
    shared_memory.atomic_func(init_func);
  }

  bip::managed_shared_memory *operator->() { return &shared_memory; }

  void OnInit(const std::function<void(bool first_init)> &init_func) {
    CHECK(!is_init_);
    bip::scoped_lock lock{*mutex_};
    bool first_init = (*ref_count_)++ == 0;
    init_func(first_init);
  }

  void OnDeinit(const std::function<void(bool last_deinit)> &deinit_func) {
    CHECK(!is_deinit_);
    bip::scoped_lock lock{*mutex_};
    bool last_deinit = --(*ref_count_) == 0;
    deinit_func(last_deinit);
    if (last_deinit) {
      lock.unlock();
      bip::shared_memory_object::remove(name.c_str());
    }
    is_deinit_ = true;
  }

  bip_mutex &GetMutex() { return *mutex_; }

  ~SharedMemory() { CHECK(is_deinit_); }
};

template <typename T> class 
SharableObject {
private:
  SharedMemory shared_memory_;
  T *object_;

public:
  
  template<typename ... Args>
  SharableObject(
      std::string name, size_t nbytes,
      Args && ... args)
      : shared_memory_(name, nbytes) {
    shared_memory_.OnInit([&](bool first_init) {
      object_ = new T(shared_memory_, args..., first_init);
    });
  }

  T *operator->() { return object_; }

  T *GetObject() { return object_; }

  operator T*() { return object_; }

  ~SharableObject() {
    shared_memory_.OnDeinit(
        [&](bool last_deinit) { delete object_; object_ = nullptr; });
  }
};

template <typename T> class shm_handle {
private:
  bip_shm::handle_t handle_;

public:
  shm_handle() : handle_(reinterpret_cast<bip_shm::handle_t>(nullptr)) {}
  shm_handle(bip_shm::handle_t handle) : handle_(handle) {}
  shm_handle(T *t, bip_shm &shm) : handle_(shm.get_handle_from_address(t)) {}
  shm_handle(T *t, SharedMemory &shm)
      : handle_(shm->get_handle_from_address(t)) {}

  T *ptr(bip_shm &shm) const {
    return reinterpret_cast<T *>(shm.get_address_from_handle(handle_));
  }
  T *ptr(SharedMemory &shm) const {
    return reinterpret_cast<T *>(shm->get_address_from_handle(handle_));
  }

  operator bip_shm::handle_t() const {
    return handle_;
  }

  bool operator==(const shm_handle<T>& rhs) const {
    return this->handle_ == rhs.handle_;
  }
};

template <typename Key, typename Value>
using bip_map = boost::container::map<
    Key, Value, std::less<Key>,
    bip::allocator<std::pair<const Key, Value>, bip_shm::segment_manager>>;

template <typename Key, typename Value>
using bip_multimap = boost::container::multimap<
    Key, Value, std::less<Key>,
    bip::allocator<std::pair<const Key, Value>, bip_shm::segment_manager>>;
template <typename Key, typename Value>
using bip_unordered_map = boost::unordered_map<
    Key, Value, boost::hash<Key>, std::equal_to<Key>,
    bip::allocator<std::pair<const Key, Value>, bip_shm::segment_manager>>;

template <typename Type>
using bip_vector =
    boost::container::vector<Type,
                             bip::allocator<Type, bip_shm::segment_manager>>;

template <typename Type>
using bip_list =
    boost::container::list<Type,
                           bip::allocator<Type, bip_shm::segment_manager>>;

} // namespace mpool
