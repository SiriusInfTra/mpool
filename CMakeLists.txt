cmake_minimum_required(VERSION 3.16)
project(mpool LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -Wreturn-type -Wno-missing-field-initializers -Wno-class-memaccess")

find_package(CUDAToolkit 11 REQUIRED)
find_package(Threads REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 70)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)


add_subdirectory(pages_pool)
add_subdirectory(allocator)
add_executable(mpool main.cc)
target_link_libraries(mpool PRIVATE pages_pool)
include(cmake/glog.cmake)
