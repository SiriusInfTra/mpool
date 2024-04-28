set(ALLOCATOR_SRC
    src/freelist.cc
)

add_library(allocator ${ALLOCATOR_SRC})
include(${PROJECT_SOURCE_DIR}/cmake/glog.cmake)
target_link_libraries(allocator PUBLIC CUDA::cudart CUDA::cuda_driver Threads::Threads rt glog::glog pages_pool)
target_include_directories(allocator PUBLIC include)

add_executable(async_malloc test/async_malloc.cu)
target_link_libraries(async_malloc allocator)
