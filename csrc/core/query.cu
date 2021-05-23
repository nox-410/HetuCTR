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

__global__ void LookUpVersion(HetuGPUTable *tbl) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < tbl->cur_batch_.unique_size) {
    index_t idx = tbl->cur_batch_.d_offset[id];
    if (idx >= 0) tbl->d_query_version_[0][id] = tbl->d_version_[idx];
    else tbl->d_query_version_[0][id] = kInvalidVersion;
  }
}

void HetuGPUTable::generateQuery() {
  // generate local version for each embedding lookup
  LookUpVersion<<<DIM_GRID(cur_batch_.unique_size), DIM_BLOCK, 0, stream_main_>>>(this);
  // Copy index to query buffer
  checkCudaErrors(cudaMemcpyAsync(
    d_query_idx_[0], cur_batch_.d_unique_idx, cur_batch_.unique_size * sizeof(index_t), cudaMemcpyDeviceToDevice, stream_main_));
}

__global__ void computeReturnOutdated(HetuGPUTable *g, size_t len) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < len) {
    g->d_embedding_[id] = 0;
  }
}

void HetuGPUTable::handleQuery() {
  size_t num_rcvd = 0;
  for (int i = 0; i < nrank_; i++) num_rcvd += cur_batch_.u_shape_exchanged[i];
  INFO(num_rcvd, " received embedding index to handle.");
  computeReturnOutdated<<<1, 256, 0, stream_main_>>>(this, 256);
}
