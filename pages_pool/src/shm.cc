#include <mpool/shm.h>

#include <mpool/logging_is_spdlog.h>

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

SharedMemory::~SharedMemory() { CHECK(is_deinit_); }


} // namespace mpool