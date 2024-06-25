#pragma once

#include <atomic>
#include <cstddef>
#include <iostream>
#include <ostream>
#include <string>

#include <shm.h>
#include <string_view>
#include <util.h>

#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_sharable_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

namespace mpool {
struct BelongImpl {
  const index_t index;
  const bip_string name;
  std::atomic<num_t> pages_num;
  std::atomic<size_t> allocated_nbytes;
  std::atomic<size_t> peek_allocated_nbytes;

  friend std::ostream &operator<<(std::ostream &out, const BelongImpl &impl) {
    out << impl.name;
    return out;
  }
};

class Belong {
  friend class PagesPool;

private:
  SharedMemory &shared_memory_;
  shm_ptr<BelongImpl> handle_;

public:
  Belong(shm_ptr<BelongImpl> handle, SharedMemory &shared_memory)
      : shared_memory_(shared_memory), handle_(handle) {}
  Belong(BelongImpl *impl, SharedMemory &shared_memory)
      : shared_memory_(shared_memory), handle_{impl, shared_memory} {}

  BelongImpl *Get() { return handle_.ptr(shared_memory_); }

  const BelongImpl *Get() const { return handle_.ptr(shared_memory_); }

  num_t GetPagesNum() const {
    return Get()->pages_num.load(std::memory_order_relaxed);
  }

  size_t GetAllocatedNbytes() const {
    return Get()->allocated_nbytes.load(std::memory_order_relaxed);
  }

  index_t GetIndex() const { return Get()->index; }

  std::string_view GetName() const { 
    return {Get()->name.c_str(), Get()->name.size()};
  }

  bool operator==(const Belong &rhs) const {
    return this->handle_ == rhs.handle_;
  }

  bool operator==(const shm_ptr<BelongImpl> &rhs) const {
    return this->handle_ == rhs;
  }

  operator shm_ptr<BelongImpl>() const { return handle_; }

  friend std::ostream &operator<<(std::ostream &out, const Belong &belong) {
    out << belong.Get()->name;
    return out;
  }
};

class BelongRegistry {
private:
  SharedMemory &shared_memory_;
  bip_vector<shm_ptr<BelongImpl>> *registered_belongs;
  bip::interprocess_sharable_mutex *mutex;
  BelongImpl *kFreeBelong;

  BelongImpl *CreateBelong(size_t index, std::string name) {
    auto *belong_impl = shared_memory_->allocate(sizeof(BelongImpl));
    bip_string shared_memory_string = {name.c_str(), name.length(), shared_memory_->get_segment_manager()};
    return new (belong_impl) BelongImpl{index, shared_memory_string, 0};
  }

public:
  BelongRegistry(SharedMemory &shared_memory) : shared_memory_(shared_memory) {}

  void Init(size_t total_pages) {
    registered_belongs =
        shared_memory_->find_or_construct<bip_vector<shm_ptr<BelongImpl>>>(
            "BR_registered_belongs")(shared_memory_->get_segment_manager());
    mutex = shared_memory_->find_or_construct<bip::interprocess_sharable_mutex>(
        "BR_mutex")();
    bip::scoped_lock lock{*mutex};
    if (registered_belongs->empty()) {
      kFreeBelong = CreateBelong(0, "FREE");
      kFreeBelong->pages_num.store(total_pages, std::memory_order_relaxed);
      registered_belongs->emplace_back(kFreeBelong, shared_memory_);
    } else {
      kFreeBelong = registered_belongs->front().ptr(shared_memory_);
    }
  }

  Belong GetOrCreateBelong(const std::string &name) {
    bip::scoped_lock lock{*mutex};
    for (auto handle : *registered_belongs) {
      auto *belong = handle.ptr(shared_memory_);
      if (std::string_view(belong->name.c_str(), belong->name.length()) == name) {
        return {belong, shared_memory_};
      }
    }
    auto *belong = CreateBelong(registered_belongs->size(), name);
    registered_belongs->emplace_back(belong, shared_memory_);
    return {belong, shared_memory_};
  }

  Belong GetBelong(shm_ptr<BelongImpl> handle) {
    return {handle, shared_memory_};
  }

  Belong GetFreeBelong() const { return {kFreeBelong, shared_memory_}; }
};
} // namespace mpool