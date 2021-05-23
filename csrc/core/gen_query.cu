#include "hetu_gpu_table.h"
#include "common/helper_cuda.h"

using namespace hetu;

__global__ void LookUpVersion(version_t* dst, const version_t* src, const index_t* offset, const int len) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < len) {
    index_t idx = offset[id];
    if (idx >= 0) dst[id] = src[idx];
    else dst[id] = kInvalidVersion;
  }
}

void HetuGPUTable::generateQuery() {
  // generate local version for each embedding lookup
  LookUpVersion<<<DIM_GRID(cur_batch_.unique_size), DIM_BLOCK, 0, stream_main_>>>(
    d_query_version_, d_version_, cur_batch_.d_offset, cur_batch_.unique_size);
  // Copy index to query buffer
  checkCudaErrors(cudaMemcpyAsync(
    d_query_idx_, cur_batch_.d_unique_idx, cur_batch_.unique_size * sizeof(index_t), cudaMemcpyDeviceToDevice, stream_main_));
}
