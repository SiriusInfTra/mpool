
#include <cstddef>
#include <vector>
#include <iostream>
#include <algorithm>

using index_t = int;
int main() {
    std::vector<ptrdiff_t> free_mapping_index = {-1, 1, 2, 3, 4, 5, 8};
    std::vector<std::pair<ptrdiff_t, size_t>> free_mapping_ranges;
    auto iter = free_mapping_index.cbegin();
    while (true) {
        auto iter_dis = std::adjacent_find(iter, free_mapping_index.cend(), [](index_t a, index_t b) { return a + 1 != b; });
        std::cout << *iter_dis << std::endl;
        if (iter_dis == free_mapping_index.cend()) {
            free_mapping_ranges.emplace_back(*iter, std::distance(iter, iter_dis));
            break;
        } else {
            free_mapping_ranges.emplace_back(*iter, std::distance(iter, iter_dis) + 1);
            iter = std::next(iter_dis);
        }
    }
    for (auto && range : free_mapping_ranges) {
        std::cout << "<" << range.first << ", " << range.second << "> ";
    }
    std::cout << std::endl;
}