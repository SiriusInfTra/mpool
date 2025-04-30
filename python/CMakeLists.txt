cmake_minimum_required(VERSION 3.15)
project(py_mpool)
set(CMAKE_CXX_STANDARDC 17)
set(CMAKE_C_STANDARD 11)

find_package(Torch REQUIRED)
find_package(PythonLibs 3 REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

set(C_MODULE_SOURCES bind.cc)
set(C_MODULE_LIBRARIES pybind11::headers pages_pool allocator)
set(C_MODULE_HEADERS .)
set(C_MODULE_DEFINITION)

if (ENABLE_TORCH)
  message("PyTorch build is enabled.")

  set(torch_python_lib)
  foreach(torch_lib IN ITEMS ${TORCH_LIBRARIES})
    if(torch_lib MATCHES ".*torch/lib/libc10.so")
      string(REPLACE "torch/lib/libc10.so" "torch/lib/libtorch_python.so" torch_python_lib ${torch_lib})
    endif()
  endforeach()

  if (NOT torch_python_lib)
    message(FATAL_ERROR "torch_python_lib not found")
  endif()

  list(APPEND C_MODULE_SOURCES torch_allocator.cc)
  list(APPEND C_MODULE_LIBRARIES ${TORCH_LIBRARIES})
  list(APPEND C_MODULE_LIBRARIES ${torch_python_lib})
  list(APPEND C_MODULE_DEFINITION ENABLE_TORCH)
else()
  message("PyTorch build is disabled.")
endif()
if (ENABLE_TENSORRT)
  list(APPEND C_MODULE_SOURCES tensorrt_allocator.cc)
  list(APPEND C_MODULE_HEADERS ${TENSORRT_DIR}/include)
  list(APPEND C_MODULE_DEFINITION ENABLE_TENSORRT)
endif()


pybind11_add_module(_C MODULE ${C_MODULE_SOURCES})
target_link_libraries(_C PRIVATE ${C_MODULE_LIBRARIES})

target_compile_definitions(_C PRIVATE VERSION_INFO=${CMAKE_PROJECT_VERSION})
target_compile_definitions(_C PRIVATE VERSION_MAJOR=${CMAKE_PROJECT_VERSION_MAJOR})
target_compile_definitions(_C PRIVATE VERSION_MINOR=${CMAKE_PROJECT_VERSION_MINOR})
target_compile_definitions(_C PRIVATE ${C_MODULE_DEFINITION})
target_include_directories(_C PRIVATE ${C_MODULE_HEADERS})
get_directory_property(compile_defs COMPILE_DEFINITIONS)
message("COMPILE_DEFINITIONS: ${compile_defs}")

# 对于特定目标
get_target_property(target_defs _C COMPILE_DEFINITIONS)
message("Target compile definitions: ${target_defs}")


set_target_properties(_C PROPERTIES INSTALL_RPATH "$ORIGIN")
install(TARGETS _C DESTINATION mpool)
install(TARGETS pages_pool DESTINATION mpool)
install(TARGETS allocator DESTINATION mpool)
