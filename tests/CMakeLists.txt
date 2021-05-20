add_executable(all2all all2all.cu)
target_link_libraries(all2all PUBLIC hetu_gpu_table_cpp)

add_executable(init init.cu)
target_link_libraries(init PUBLIC hetu_gpu_table_cpp)
