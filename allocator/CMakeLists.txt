set(ALLOCATOR_SRC
    src/caching_alloctor.cc
    src/direct_allocator.cc
    src/vmm_allocator.cc
    src/mapping_region.cc
    src/stream_context.cc

)

add_library(allocator SHARED ${ALLOCATOR_SRC})
include(${PROJECT_SOURCE_DIR}/cmake/glog.cmake)
target_link_libraries(allocator PUBLIC CUDA::cudart CUDA::cuda_driver Threads::Threads rt glog::glog pages_pool)
target_include_directories(allocator PUBLIC include)
# add_executable(async_malloc test/async_malloc.cu)
# target_link_libraries(async_malloc allocator)

# add_executable(caching_allocator_test test/caching_allocator_test.cc)
# target_link_libraries(caching_allocator_test allocator)
# add_executable(direct_allocator_test test/direct_allocator_test.cc)
# target_link_libraries(direct_allocator_test allocator)