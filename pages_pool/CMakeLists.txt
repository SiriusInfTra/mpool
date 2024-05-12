set(PAGES_POOL_SRC
    src/pages_pool.cc
    src/pages.cc
    src/pages_list.cc
    src/belong.cc
)
set(PAGES_POOL_TEST
    test/simple_test.cc
)


add_library(pages_pool SHARED ${PAGES_POOL_SRC})
include(${PROJECT_SOURCE_DIR}/cmake/glog.cmake)
target_link_libraries(pages_pool PUBLIC CUDA::cudart CUDA::cuda_driver Threads::Threads rt glog::glog)
target_include_directories(pages_pool PUBLIC include)

add_executable(pages_pool_test ${PAGES_POOL_TEST})
target_link_libraries(pages_pool_test PRIVATE pages_pool)

add_executable(benchmark test/benchmark.cc)
target_link_libraries(benchmark PRIVATE pages_pool)