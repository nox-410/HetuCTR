aux_source_directory(hash_table hash_table_src)
aux_source_directory(rendezvous rendezvous_src)
aux_source_directory(pybind pybind_src)

pybind11_add_module(hetu_gpu_table ${pybind_src} ${hash_table_src} ${rendezvous_src})

target_include_directories(hetu_gpu_table
    PRIVATE
    .
    ${cub_SOURCE_DIR}
    ${thrust_SOURCE_DIR}
    ${NCCL_INCLUDE_DIRS}
    ${ZMQ_INCLUDE_DIRS}
)

target_link_libraries(hetu_gpu_table
    PRIVATE
    CUDA::cudart
    ${NCCL_LIBRARIES}
    ${ZMQ_LIBRARIES}
)

# this library is for c++ unit test, so no pybind11
add_library(hetu_gpu_table_cpp SHARED ${hash_table_src} ${rendezvous_src})

target_include_directories(hetu_gpu_table_cpp
    PUBLIC
    .
    ${cub_SOURCE_DIR}
    ${thrust_SOURCE_DIR}
    ${NCCL_INCLUDE_DIRS}
    ${ZMQ_INCLUDE_DIRS}
)

target_link_libraries(hetu_gpu_table_cpp
    PUBLIC
    CUDA::cudart
    ${NCCL_LIBRARIES}
    ${ZMQ_LIBRARIES}
)
