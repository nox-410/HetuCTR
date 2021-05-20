cmake_minimum_required(VERSION 3.18)

project(HetuGPUTable C CXX CUDA)

enable_testing()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/)

find_package(CUDAToolkit REQUIRED)

# set(NCCL_ROOT)
find_package(NCCL REQUIRED 2.7...)

include(FetchContent)

FetchContent_Declare(thrust URL https://github.com/NVIDIA/thrust/archive/refs/tags/1.12.0.tar.gz)
message(STATUS "Preparing thrust ...")
FetchContent_MakeAvailable(thrust)

FetchContent_Declare(cub URL https://github.com/NVlabs/cub/archive/refs/tags/1.12.0.tar.gz)
message(STATUS "Preparing cub ...")
FetchContent_MakeAvailable(cub)

find_package(pybind11 2.6... CONFIG)

add_subdirectory(${PROJECT_SOURCE_DIR}/csrc)
add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
