cmake_minimum_required(VERSION 3.5.1)

project(n-queens)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

include_directories(${CUDA_INCLUDE_DIRS})

cuda_add_executable(n-queens n-queens.cc n-queens.cu)

target_link_libraries(
    n-queens
    ${CUDA_LIBRARIES})
