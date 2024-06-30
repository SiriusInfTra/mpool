#pragma once
#include "mem_block.h"
#include <array>
#include <cstddef>
#include <ostream>
#include <random>


namespace mpool {

struct Stat {
  int64_t sum = 0;
  int64_t cnt = 0;

  void AddBlock(size_t nbytes) {
    sum += nbytes;
    cnt += 1;
  }

  void RemoveBlock(size_t nbytes) {
    sum -= nbytes;
    cnt -= 1;
    CHECK_GE(cnt, 0);
    CHECK_GE(sum, 0);
  }

  void MergeBlock() {
    cnt -= 1;
  }

  void SplitBlock() {
    cnt += 1;
  }



};

inline std::ostream &operator<<(std::ostream &out, const Stat &stat) {
  out << "{sum: " << stat.sum << ", cnt: " << stat.cnt << "}";
  return out;
}

inline bool operator==(const Stat &lhs, const Stat &rhs) {
  return lhs.sum == rhs.sum && lhs.cnt == rhs.cnt;
}

using StatArray = std::array<Stat, 2>;

struct CachingAllocatorStats {
  StatArray allocated;
  StatArray free;
};

}