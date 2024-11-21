#include <mpool/shm.h>

#include <glog/logging.h>

namespace mpool {

void SharedMemory::OnInit(const std::function<void(bool first_init)> &init_func) {
  CHECK(!is_init_);
  bip::scoped_lock lock{*mutex_};
  bool first_init = (*ref_count_)++ == 0;
  init_func(first_init);
}

void SharedMemory::OnDeinit(
    const std::function<void(bool last_deinit)> &deinit_func) {
  CHECK(!is_deinit_);
  bip::scoped_lock lock{*mutex_};
  bool last_deinit = --(*ref_count_) == 0;
  deinit_func(last_deinit);
  if (last_deinit) {
    lock.unlock();
    // bip::shared_memory_object::remove(name.c_str());
  }
  is_deinit_ = true;
}

SharedMemory::SharedMemory(std::string name, size_t nbytes)
    : name(name), shared_memory(bip::open_or_create, name.c_str(), nbytes),
      is_init_(false), is_deinit_(false) {
  auto init_func = [&]() {
    mutex_ = shared_memory.find_or_construct<bip_mutex>("mutex")();
    ref_count_ = shared_memory.find_or_construct<int>("ref_count")(0);
  };
  shared_memory.atomic_func(init_func);
}

SharedMemory::~SharedMemory() { CHECK(is_deinit_); }


} // namespace mpool