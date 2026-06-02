cmake_minimum_required(VERSION 3.18)
project(lw_kernel_cuda)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "-O3 ${CMAKE_CUDA_FLAGS}")

# Detect CUDA architecture (relative to project root)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cuda_code/cmake/cuda.cmake)

# Header-only include library
add_library(lw_kernel_headers INTERFACE)
target_include_directories(lw_kernel_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Collect all CUDA sources into an OBJECT library for incremental compilation
file(GLOB_RECURSE CUDA_SRCS "lw_kernel_cuda/csrc/cuda/*.cu")
add_library(lw_kernel_ops OBJECT ${CUDA_SRCS})
target_include_directories(lw_kernel_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/lw_kernel_cuda/csrc
)
target_link_libraries(lw_kernel_ops PRIVATE lw_kernel_headers)

# Collect all test sources
file(GLOB_RECURSE TEST_SRCS "tests/*.cu" "tests/*.cpp")
if(TEST_SRCS)
    add_executable(run_tests ${TEST_SRCS})
    target_include_directories(run_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(run_tests PRIVATE lw_kernel_headers)
endif()

# Collect all benchmark sources
file(GLOB_RECURSE BENCH_SRCS "benchmarks/*.cu" "benchmarks/*.cpp")
if(BENCH_SRCS)
    add_executable(run_benchmarks ${BENCH_SRCS})
    target_include_directories(run_benchmarks PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(run_benchmarks PRIVATE lw_kernel_headers)
endif()