# Download metis parmetis and dependencies
FetchContent_Declare(GKlib URL https://github.com/KarypisLab/GKlib/archive/refs/heads/master.zip)
message(STATUS "Preparing metis-GKlib ...")
FetchContent_MakeAvailable(GKlib)

FetchContent_Declare(metis URL https://github.com/KarypisLab/METIS/archive/refs/heads/master.zip)
message(STATUS "Preparing metis ...")
FetchContent_GetProperties(metis)
if(NOT metis_POPULATED)
    FetchContent_Populate(metis)
    add_subdirectory(${metis_SOURCE_DIR}/libmetis ${metis_BINARY_DIR})
    target_link_libraries(metis GKlib)
    target_include_directories(metis PRIVATE ${GKlib_SOURCE_DIR})
    target_include_directories(metis PUBLIC ${metis_SOURCE_DIR}/include)
    target_compile_definitions(metis PUBLIC -DIDXTYPEWIDTH=64 -DREALTYPEWIDTH=32)
    target_compile_options(metis PUBLIC -fPIC)
endif()

find_package(MPI REQUIRED)

FetchContent_Declare(parmetis URL https://github.com/KarypisLab/ParMETIS/archive/refs/heads/main.zip)
message(STATUS "Preparing parmetis ...")
FetchContent_GetProperties(parmetis)
if(NOT parmetis_POPULATED)
    FetchContent_Populate(parmetis)
    add_subdirectory(${parmetis_SOURCE_DIR}/libparmetis ${parmetis_BINARY_DIR})
    target_link_libraries(parmetis metis GKlib ${MPI_CXX_LIBRARIES})
    target_include_directories(parmetis PRIVATE ${GKlib_SOURCE_DIR})
    target_include_directories(parmetis PUBLIC ${parmetis_SOURCE_DIR}/include ${metis_SOURCE_DIR}/include ${MPI_CXX_INCLUDE_DIRS})
endif()

aux_source_directory(. partition_src)

pybind11_add_module(hetuCTR_partition ${partition_src})

target_link_libraries(hetuCTR_partition PRIVATE parmetis)
