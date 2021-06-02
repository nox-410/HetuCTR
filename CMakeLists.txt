cmake_minimum_required(VERSION 3.18)

project(hetuCTR C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_CXX_FLAGS "-O3 -Wall")
set(CMAKE_CUDA_FLAGS "-O3 -w")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)

set(CUDAToolkit_ROOT /usr/local/cuda)
set(CMAKE_CUDA_COMPILER ${CUDAToolkit_ROOT}/bin/nvcc)
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

set(NCCL_ROOT $ENV{CONDA_PREFIX})
find_package(NCCL 2.8 REQUIRED)

include(FetchContent)

FetchContent_Declare(thrust URL https://github.com/NVIDIA/thrust/archive/refs/tags/1.12.0.tar.gz)
message(STATUS "Preparing thrust ...")
FetchContent_MakeAvailable(thrust)

FetchContent_Declare(cub URL https://github.com/NVlabs/cub/archive/refs/tags/1.12.0.tar.gz)
message(STATUS "Preparing cub ...")
FetchContent_MakeAvailable(cub)

FetchContent_Declare(GKlib URL https://github.com/KarypisLab/GKlib/archive/refs/heads/master.zip)
message(STATUS "Preparing metis-GKlib ...")
FetchContent_MakeAvailable(GKlib)

FetchContent_Declare(metis64 URL https://github.com/KarypisLab/METIS/archive/refs/heads/master.zip)
message(STATUS "Preparing metis ...")
FetchContent_GetProperties(metis64)
if(NOT metis64_POPULATED)
    FetchContent_Populate(metis64)
    add_subdirectory(${metis64_SOURCE_DIR}/libmetis ${metis64_BINARY_DIR})
    target_link_libraries(metis GKlib)
    target_include_directories(metis PRIVATE ${GKlib_SOURCE_DIR})
    target_include_directories(metis PUBLIC ${metis64_SOURCE_DIR}/include)
    target_compile_definitions(metis PUBLIC -DIDXTYPEWIDTH=64 -DREALTYPEWIDTH=32)
    target_compile_options(metis PRIVATE -fPIC)
endif()

find_package(MPI)

FetchContent_Declare(parmetis URL https://github.com/KarypisLab/ParMETIS/archive/refs/heads/main.zip)
message(STATUS "Preparing parmetis ...")
FetchContent_GetProperties(parmetis)
if(NOT parmetis_POPULATED)
    FetchContent_Populate(parmetis)
    add_subdirectory(${parmetis_SOURCE_DIR}/libparmetis ${parmetis_BINARY_DIR})
    target_link_libraries(parmetis GKlib ${MPI_CXX_LIBRARIES})
    target_include_directories(parmetis PRIVATE ${GKlib_SOURCE_DIR} ${metis64_SOURCE_DIR}/include)
    target_include_directories(parmetis PUBLIC ${parmetis_SOURCE_DIR}/include ${MPI_CXX_INCLUDE_DIRS})
    target_compile_definitions(parmetis PUBLIC -DIDXTYPEWIDTH=64 -DREALTYPEWIDTH=32)
    target_compile_options(parmetis PRIVATE -fPIC)
endif()

find_package(ZMQ 4 REQUIRED)
find_package(pybind11 2.6 CONFIG)

add_subdirectory(${PROJECT_SOURCE_DIR}/csrc)

enable_testing()
