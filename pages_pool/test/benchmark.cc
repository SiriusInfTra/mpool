#include <boost/dynamic_bitset.hpp>
#include <chrono>
#include <cstddef>
#include <limits>
#include <random>
#include <map>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

class Recorder {
private:
    std::string name;
    size_t rep;
    std::chrono::steady_clock::time_point t0;
public:
    Recorder(std::string name, size_t rep): name(std::move(name)), rep(rep), t0(std::chrono::steady_clock::now()) {}
    ~Recorder() {
        auto t1 = std::chrono::steady_clock::now();
        std::cout << name << " costs " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / rep
            << " us." << std::endl;
    }
};

template<typename T>
void RandomInit(T &container, size_t len, int32_t long seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dist(0, 1);
    for (size_t k = 0; k < len; ++k) {
        container[k] = dist(gen);
    }
}


template<typename T>
    size_t FindBestFit(T &bitset, size_t n) {
    size_t best_fit_len = std::numeric_limits<size_t>::max();
    size_t best_fit_pos = 0;
    size_t curr_pos = 0;
    size_t curr_len = 0;

    size_t bitset_len = bitset.size();
    for (size_t k=0;k< bitset_len; ++k){
        if (bitset[k]) {/* curr page is available */
            curr_len++;
        } else {/* curr page is not available */
            if (curr_len == n){
                //exactly match, return
                return curr_pos;
            } else if (curr_len >n && curr_len< best_fit_len) {
                best_fit_len = curr_len;
                best_fit_pos = curr_pos;
            }
            curr_len = 0;
            curr_pos = k + 1;
        }
    }

    return best_fit_pos;
}


size_t FindBestFit(std::multimap<size_t, size_t> &index, size_t n) {
    auto iter = index.lower_bound(n);
    if (iter == index.cend()) {
        return std::numeric_limits<size_t>::max();
    } else {
        size_t len = iter->first;
        size_t pos = iter->second;
        index.erase(iter);
        index.insert(std::make_pair(len, pos));
    }
    return iter->second;
}

template<typename T>
std::multimap<size_t, size_t> BuildIndex(const T &bitset) {
    std::multimap<size_t, size_t> index;
    size_t curr_pos = 0;
    size_t curr_len = 0;

    size_t bitset_len = bitset.size();

    for (size_t k = 0; k < bitset_len; ++k) {
        if (bitset[k]) { /* curr page is available */
            curr_len++;
        } else { /* curr page is not available */
            if (curr_len > 0) {
                index.insert(std::make_pair(curr_len, curr_pos));
            }
            curr_len = 0;
            curr_pos = k + 1;
        }
    }
    if (curr_len > 0) {
        index.insert(std::make_pair(curr_len, curr_pos));
    }

    return index;
}



const constexpr size_t LEN = 400;
const constexpr size_t REP = 100000;
int main() {
    {    
        // boost::dynamic_bitset<> x(LEN);
        std::vector<int> x(LEN);
        std::vector<size_t> pos(REP);
        RandomInit(x, LEN, 42);
        {
            Recorder recorder{"dynamic_bitset", REP / 100};
            for (size_t k = 0; k < REP; ++k) {
                pos[k] = FindBestFit(x, 8);
            }
        }
        std::cout << pos[0] << std::endl;
    }
    {
        boost::dynamic_bitset<> x(LEN);
        RandomInit(x, LEN, 42);
        auto index = BuildIndex(x);
        std::vector<size_t> pos(REP);
        {
            Recorder recorder{"dynamic_bitset", REP / 100};
            for (size_t k = 0; k < REP; ++k) {
                pos[k] = FindBestFit(x, 8);

            }
        }
        std::cout << pos[0] << std::endl;
    }

}