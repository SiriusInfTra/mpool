set(PAGES_POOL_SRC
    src/pages_pool.cc
    src/pages.cc
    src/pages_list.cc
    src/belong.cc
    src/shm.cc
    src/cuda_handle.cc
)
set(PAGES_POOL_TEST
    test/simple_test.cc
)


add_library(pages_pool SHARED ${PAGES_POOL_SRC})
target_compile_definitions(pages_pool PUBLIC MPOOL_CHECK_LEVEL=${MPOOL_CHECK_LEVEL})
target_compile_definitions(pages_pool PUBLIC MPOOL_VERBOSE_LEVEL=${MPOOL_VERBOSE_LEVEL})


include(${PROJECT_SOURCE_DIR}/cmake/glog.cmake)
target_link_libraries(pages_pool PUBLIC 
    CUDA::cudart 
    CUDA::cuda_driver 
    Threads::Threads 
    glog::glog
    spdlog::spdlog
    rt 
    ${Boost_LIBRARIES}
)
target_include_directories(pages_pool PUBLIC include ${Boost_INCLUDE_DIRS})

# add_executable(pages_pool_test ${PAGES_POOL_TEST})
# target_link_libraries(pages_pool_test PRIVATE pages_pool)

add_executable(benchmark test/benchmark.cc)
target_link_libraries(benchmark PRIVATE pages_pool)