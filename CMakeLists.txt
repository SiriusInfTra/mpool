cmake_minimum_required(VERSION 3.15)
project(mpool VERSION 1.0.0 LANGUAGES CXX C CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)


option(CUDA_TOOLKIT_VERSION "CUDA Toolkit Version" 11)

option(ENABLE_PYTHON_BINDING "build python extension" ON)
option(ENABLE_TORCH "build PyTorch support" ON)
option(ENABLE_TENSORRT "build TensorRT support" OFF)
option(ENABLE_CUPY "build CuPy support" OFF)
option(ENABLE_NUMBA "build Numba support" OFF)
option(MPOOL_CHECK_LEVEL "mpool check state level (0 ~ 3)" 0)
option(MPOOL_VERBOSE_LEVEL "mpool version level (0 ~ 3)" 0)

set(CONFIG_TEMPLATE ${PROJECT_SOURCE_DIR}/config.cmake.in)
set(CONFIG_FILE ${PROJECT_BINARY_DIR}/config.cmake)
if(NOT EXISTS ${CONFIG_FILE}) 
    configure_file(${CONFIG_TEMPLATE} ${CONFIG_FILE} @ONLY)
else()
  message(STATUS "File ${CONFIG_FILE} already exists. Skipping configure_file.")
endif()
# configure_file(${CONFIG_TEMPLATE} ${CONFIG_FILE} @ONLY)
include(${CONFIG_FILE})

find_package(CUDAToolkit ${CUDA_TOOLKIT_VERSION} REQUIRED)
find_package(Threads REQUIRED)
find_package(Boost REQUIRED)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -Wreturn-type -Wno-missing-field-initializers -Wno-class-memaccess -Wno-unused-parameter")


# include(FetchContent)
# set(FETCHCONTENT_QUIET OFF)

add_subdirectory(pages_pool)
add_subdirectory(allocator)
if(ENABLE_PYTHON_BINDING)
  message("Python Binding is enabled.")
  add_subdirectory(python)
else()
  message("Python Binding is not enabled.")
endif()

include(cmake/glog.cmake)
