#pragma once
#include <cstddef>


namespace mpool {

struct Stat {
  size_t sum = 0;
  size_t cnt = 0;

  void AddBlock(size_t nbytes) {
    sum += nbytes;
    cnt += 1;
  }

  void RemoveBlock(size_t nbytes) {
    sum -= nbytes;
    cnt -= 1;
  }
};

struct CachingAllocatorStats {
  Stat allocated;
  Stat free;
};

}