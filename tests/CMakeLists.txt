# add_executable(prefix prefix.cu)
# message(STATUS ${cub_SOURCE_DIR})
# target_include_directories(prefix PRIVATE ${cub_SOURCE_DIR} ${thrust_SOURCE_DIR})
# target_link_libraries(prefix CUDA::cudart)

# add_executable(prefix prefix.cu)
# target_link_libraries(prefix CUDA::cudart Thrust)

# add_executable(example example.cu)
# target_link_libraries(example CUDA::cudart Thrust)

# add_executable(reduce reduce.cu)
# target_link_libraries(reduce CUDA::cudart Thrust)

# add_executable(query cuda-samples/Samples/deviceQuery/deviceQuery.cpp)
# target_include_directories(query PUBLIC cuda-samples/Common)
# target_link_libraries(query CUDA::cudart Thrust)

# add_executable(cumap cumap.cu nv_hashtable.cu)
# target_link_libraries(cumap CUDA::cudart Thrust)

# add_executable(all2all all2all.cu)
# target_include_directories(all2all PUBLIC ${NCCL_INCLUDE_DIRS} "~/anaconda3/envs/geo/include")
# target_link_libraries(all2all CUDA::cudart ${NCCL_LIBRARIES} "~/anaconda3/envs/geo/lib/libmpi.so")
