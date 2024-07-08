#pragma once
#include <mpool/mem_block.h>

#include <algorithm>
#include <array>
#include <ostream>

namespace mpool {

struct Stat {
  int64_t current = 0;
  int64_t peak = 0;
  int64_t allocated_free[2] = {0, 0};
};

inline std::ostream &operator<<(std::ostream &out, const Stat &stat) {
  out << "{current=" << stat.current << ", peak=" << stat.peak
      << ", allocated=" << stat.allocated_free[0] << ", freed=" << stat.allocated_free[1] << "}";
  return out;
}

inline bool operator==(const Stat &lhs, const Stat &rhs) {
  return lhs.current == rhs.current && lhs.peak == rhs.peak &&
         lhs.allocated_free[0] == rhs.allocated_free[0] &&
         lhs.allocated_free[1] == rhs.allocated_free[1];
}

using StatArray = std::array<Stat, 2>;

struct CachingAllocatorStats {
  StatArray mem_block_nbytes;
  StatArray mem_block_count;

  void SetBlockFree(MemBlock *block, bool is_free) {
    mem_block_nbytes.at(block->is_small).allocated_free[block->is_free] -=
        block->nbytes;
    mem_block_count.at(block->is_small).allocated_free[block->is_free] -= 1;

    block->is_free = is_free;
    mem_block_nbytes.at(block->is_small).allocated_free[block->is_free] +=
        block->nbytes;
    mem_block_count.at(block->is_small).allocated_free[block->is_free] += 1;
  }

  void SetBlockIsSmall(MemBlock *block, bool is_small) {
    mem_block_nbytes.at(block->is_small).current -= block->nbytes;
    mem_block_nbytes.at(block->is_small).allocated_free[block->is_free] -=
        block->nbytes;

    mem_block_count.at(block->is_small).current -= 1;
    mem_block_count.at(block->is_small).allocated_free[block->is_free] -= 1;


    block->is_small = is_small;
    mem_block_nbytes.at(block->is_small).current += block->nbytes;
    mem_block_nbytes.at(block->is_small).allocated_free[block->is_free] +=
        block->nbytes;
    mem_block_nbytes.at(block->is_small).peak =
        std::max(mem_block_nbytes.at(block->is_small).current,
                 mem_block_nbytes.at(block->is_small).peak);

    mem_block_count.at(block->is_small).current += 1;
    mem_block_count.at(block->is_small).allocated_free[block->is_free] += 1;
    mem_block_count.at(block->is_small).peak =
        std::max(mem_block_count.at(block->is_small).current,
                 mem_block_count.at(block->is_small).peak);
  }

  void CreateBlock(MemBlock *block) {
    mem_block_nbytes.at(block->is_small).current += block->nbytes;
    mem_block_nbytes.at(block->is_small).allocated_free[block->is_free] +=
        block->nbytes;
    mem_block_nbytes.at(block->is_small).peak =
        std::max(mem_block_nbytes.at(block->is_small).current,
                 mem_block_nbytes.at(block->is_small).peak);
    
    mem_block_count.at(block->is_small).current += 1;
    mem_block_count.at(block->is_small).allocated_free[block->is_free] += 1;
    mem_block_count.at(block->is_small).peak =
        std::max(mem_block_count.at(block->is_small).current,
                 mem_block_count.at(block->is_small).peak);
  }

  void MergeBlock(MemBlock *block) {
    mem_block_count.at(block->is_small).current -= 1;
    mem_block_count.at(block->is_small).allocated_free[block->is_free] -= 1;
  }

  void SplitBlock(MemBlock *block) {
    mem_block_count.at(block->is_small).current += 1;
    mem_block_count.at(block->is_small).allocated_free[block->is_free] += 1;
    mem_block_count.at(block->is_small).peak =
        std::max(mem_block_count.at(block->is_small).current,
                 mem_block_count.at(block->is_small).peak);
  }

  

  void ResetPeakStats() {
    for (bool is_small : {false, true}) {
      mem_block_count[is_small].peak = mem_block_count[is_small].current;
      mem_block_nbytes[is_small].peak = mem_block_nbytes[is_small].current;
    }
  }
};

} // namespace mpool