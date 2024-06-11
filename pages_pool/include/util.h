#pragma once

#include <cstddef>
#include <iomanip>
#include <limits>
#include <string>

#define CUDA_CALL(func) do { \
  auto error = func; \
  if (error != cudaSuccess) { \
    LOG(FATAL) << #func << " " << cudaGetErrorString(error); \
    exit(EXIT_FAILURE); \
  } \
  } while (0)

#define CU_CALL(func) \
  do { \
    auto err = func; \
    if (err != CUDA_SUCCESS) { \
      const char* pstr = nullptr; \
      cuGetErrorString(err, &pstr); \
      LOG(FATAL) << #func << ": " << pstr; \
      exit(EXIT_FAILURE); \
    } \
  } while (0);

namespace mpool {

using nbytes_t = size_t;
using num_t = size_t;
using index_t = size_t;
const constexpr index_t INVALID_INDEX = std::numeric_limits<index_t>::max();

inline std::string ByteDisplay(size_t nbytes) {
  std::stringstream ss;
  ss << std::fixed << std::setprecision(2)
     << static_cast<double>(nbytes) / 1024 / 1024 << "MB (" << nbytes << " Bytes)";
  return ss.str();
}

constexpr size_t operator ""_B(unsigned long long n) {
  return static_cast<size_t>(n);
}

constexpr size_t operator ""_B(long double n) {
  return static_cast<size_t>(n);
}

constexpr size_t operator ""_KB(unsigned long long n) {
  return static_cast<size_t>(n) * 1024;
}

constexpr size_t operator ""_KB(long double n) {
  return static_cast<size_t>(n * 1024);
}

constexpr size_t operator ""_MB(unsigned long long n) {
  return static_cast<size_t>(n) * 1024 * 1024;
}

constexpr size_t operator ""_MB(long double n) {
  return static_cast<size_t>(n * 1024 * 1024);
}

constexpr size_t operator ""_GB(unsigned long long n) {
  return static_cast<size_t>(n) * 1024 * 1024 * 1024;
}

constexpr size_t operator ""_GB(long double n) {
  return static_cast<size_t>(n * 1024 * 1024 * 1024);
}

}