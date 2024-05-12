include(../cmake/glog.cmake)
set (MPOOL_LINKED_LIBRARY pages_pool allocator CUDA::cudart CUDA::cuda_driver)
foreach(lib ${MPOOL_LINKED_LIBRARY})
    get_target_property(LIB_INCLUDE_DIRS ${lib} INTERFACE_INCLUDE_DIRECTORIES)
    include_directories(${LIB_INCLUDE_DIRS})
endforeach()


add_cython_target(_C  PY3 CXX pages_pool.pxd)
add_library(_C  MODULE ${_C})
python_extension_module(_C)

# target_link_directories(_C  PUBLIC PY_LINK_LIBRARIES)

install(TARGETS _C LIBRARY DESTINATION pympool)