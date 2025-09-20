cmake_minimum_required(VERSION 3.13)

project(tvm_relax_cpp_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(3rdparty/tvm/3rdparty/tvm-ffi tvm_ffi)

add_executable(main main.cc)

target_include_directories(main PUBLIC
  ${CMAKE_SOURCE_DIR}/3rdparty/tvm/include
  ${CMAKE_SOURCE_DIR}/3rdparty/tvm/3rdparty/dlpack/include
  ${CMAKE_SOURCE_DIR}/3rdparty/tvm/3rdparty/dmlc-core/include
)

target_link_libraries(main
  ${CMAKE_SOURCE_DIR}/3rdparty/tvm/build/libtvm_runtime.so
)

target_link_libraries(main tvm_ffi_shared)
