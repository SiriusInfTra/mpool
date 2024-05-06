#pragma once
#include <algorithm>
#include <util.h>
#include <shm.h>

#include <boost/container/container_fwd.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/container/vector.hpp>

#include <glog/logging.h>

namespace mpool {
class FreeList {
public:
    static const constexpr num_t INVALID_LEN = std::numeric_limits<size_t>::max();
    static const constexpr index_t INVALID_POS = std::numeric_limits<size_t>::max();
private:
    bip_vector<int> *alloc_bitmap_;
    bip_shm &shm_;
public:
    FreeList(bip_shm &shm): shm_(shm) {}

    index_t FindBestFit(num_t request_len) {
        size_t best_fit_len = INVALID_LEN;
        size_t best_fit_pos = 0;

        size_t curr_pos = 0;
        size_t curr_len = 0;

        size_t bitset_len = alloc_bitmap_->size();
        auto &alloc_bitmap = *alloc_bitmap_;

        for (size_t k = 0; k < bitset_len; ++k) {
            if (alloc_bitmap[k] == false) { /* curr page is available */
                curr_len++;
            } else { /* curr page is not available */
                if (curr_len == request_len) {
                    // exactly match, return
                    return curr_pos;
                } else if (curr_len >= request_len && curr_len < best_fit_len) {
                    best_fit_len = curr_len;
                    best_fit_pos = curr_pos;
                }
                curr_len = 0;
                curr_pos = k + 1;
            }
        }
        if (curr_len >= request_len && curr_len < best_fit_len) {
            best_fit_len = curr_len;
            best_fit_pos = curr_pos;
        }
        return best_fit_len < INVALID_LEN ? best_fit_pos : INVALID_POS;
    }

    void ClaimPages(index_t index, num_t len) {
        std::fill(alloc_bitmap_->begin() + index, alloc_bitmap_->begin() + index + len, true);
    }

    void ClaimPages(index_t index) {
        (*alloc_bitmap_)[index] = static_cast<int>(true);
    }

    void ReleasePages(index_t index, num_t len) {
        if (len == 1) {
            (*alloc_bitmap_)[index] = false;
        } else {
            std::fill(alloc_bitmap_->begin() + index, alloc_bitmap_->begin() + index + len, false);
        }
   
    }

    void Init(num_t pages_nums) {
        alloc_bitmap_ = shm_.find_or_construct<bip_vector<int>>("FL_alloc_bitmap_")(shm_.get_segment_manager());
        if (alloc_bitmap_->empty()) {
            alloc_bitmap_->resize(pages_nums);
            std::fill(alloc_bitmap_->begin(), alloc_bitmap_->end(), false);
        }
    }
    
};
}