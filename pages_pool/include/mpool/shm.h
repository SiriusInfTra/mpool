#pragma once

#include <boost/container/container_fwd.hpp>
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/unordered/unordered_map_fwd.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <functional>
#include <memory>

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
      : name(name), shared_memory(bip::open_or_create, name.c_str(), nbytes),
        is_init_(false), is_deinit_(false) {
    auto init_func = [&]() {
      mutex_ = shared_memory.find_or_construct<bip_mutex>("mutex")();
      ref_count_ = shared_memory.find_or_construct<int>("ref_count")(0);
    };
    shared_memory.atomic_func(init_func);
  }

  bip::managed_shared_memory *operator->() { return &shared_memory; }

  void OnInit(const std::function<void(bool first_init)> &init_func);

  void OnDeinit(const std::function<void(bool last_deinit)> &deinit_func);

  bip_mutex &GetMutex() { return *mutex_; }

  ~SharedMemory();
};

template <typename T> class 
SharableObject: public std::enable_shared_from_this<SharableObject<T>> {
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

  const T *operator->() const { return object_; }

  T *operator->() { return object_; }

  T *GetObject() { return object_; }

  operator T*() { return object_; }

  ~SharableObject() {
    shared_memory_.OnDeinit(
        [&](bool last_deinit) { delete object_; object_ = nullptr; });
  }
};

template <typename T> class shm_ptr {
private:
  bip_shm::handle_t handle_;

public:
  shm_ptr() : handle_(reinterpret_cast<bip_shm::handle_t>(nullptr)) {}
  shm_ptr(bip_shm::handle_t handle) : handle_(handle) {}
  shm_ptr(T *t, bip_shm &shm) : handle_(shm.get_handle_from_address(t)) {}
  shm_ptr(T *t, SharedMemory &shm)
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

  bool operator==(const shm_ptr<T>& rhs) const {
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

template<typename Type>
class bip_list_iterator {
private:
  bip_shm &shm_;
  typename bip_list<shm_ptr<Type>>::iterator iter_;
public:
  using iterator_category = std::bidirectional_iterator_tag;
  using value_type = Type;
  using difference_type = std::ptrdiff_t;
  using pointer = Type*;
  using reference = Type&;

  bip_list_iterator(typename bip_list<shm_ptr<Type>>::iterator iter, bip_shm &shm)
    : shm_(shm), iter_(iter) {}
  bip_list_iterator(typename bip_list<shm_ptr<Type>>::iterator iter, SharedMemory &shm)
    : bip_list_iterator(iter, *shm.operator->()) {}

  bip_list_iterator &operator--() {
    iter_--;
    return *this;
  }

  bip_list_iterator &operator++() {
    iter_++;
    return *this;
  }

  pointer operator*() const { return iter_->ptr(shm_); }

  friend bool operator==(const bip_list_iterator& a, const bip_list_iterator& b) {
      return a.iter_ == b.iter_;
  }

  friend bool operator!=(const bip_list_iterator& a, const bip_list_iterator& b) {
      return a.iter_ != b.iter_;
  }

};

using bip_string = bip::basic_string<char, std::char_traits<char>, 
                                     bip::allocator<char, bip_shm::segment_manager>>;

} // namespace mpool
