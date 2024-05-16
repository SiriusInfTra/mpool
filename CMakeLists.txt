cmake_minimum_required(VERSION 3.18)
project(mpool LANGUAGES CXX C CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -Wreturn-type -Wno-missing-field-initializers -Wno-class-memaccess -Wno-unused-parameter")

find_package(CUDAToolkit 11 REQUIRED)
find_package(Threads REQUIRED)
find_package(Boost)
set(CMAKE_CUDA_ARCHITECTURES 70)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)


add_subdirectory(pages_pool)
add_subdirectory(allocator)
add_subdirectory(python)
add_executable(mpool main.cc)
target_link_libraries(mpool PRIVATE pages_pool)

# include(/home/wyk/miniforge3/envs/dev2/lib/python3.10/site-packages/cython_cmake/cmake/FindCython.cmake)
# include(/home/wyk/miniforge3/envs/dev2/lib/python3.10/site-packages/cython_cmake/cmake/UseCython.cmake)
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}:$ENV{HOME}/miniforge3/envs/dev2/lib/python3.10/site-packages/cython_cmake/cmake)
include(cmake/glog.cmake)

