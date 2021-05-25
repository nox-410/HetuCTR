#include "hetu_gpu_table.h"
#include "common/helper_cuda.h"

#include <cub/cub.cuh>

using namespace hetu;

__global__ void writeBackUpdateLocalKernel(HetuGPUTable *tbl, size_t len) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < len) {
    index_t embedding_idx = tbl->d_query_idx_[0][id];
    auto iter = tbl->table_->find(embedding_idx);
    if (iter != tbl->table_->end()) {
      index_t mem_offset = iter->second;
      tbl->d_version_[mem_offset] = tbl->d_return_version_[1][id];
      for (int i = 0; i < tbl->kEmbeddingWidth; i++) {
        tbl->d_embedding_[tbl->kEmbeddingWidth * mem_offset + i] = tbl->d_return_val_[1][tbl->kEmbeddingWidth * id + i];
      }
    }
  }
}

__global__ void writeBackTargetKernel(HetuGPUTable *tbl, embed_t *dst) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  size_t len = tbl->cur_batch_.batch_size;
  if (id < len) {
    index_t mapped_idx = tbl->cur_batch_.d_idx_map[id];
    embed_t *val;
    index_t mem_offset = tbl->cur_batch_.d_offset[mapped_idx];
    if (mem_offset == kInvalidIndex) {
      index_t ret_offset = tbl->d_return_outdated_[0][mapped_idx];
      val = &tbl->d_return_val_[1][tbl->kEmbeddingWidth * ret_offset];
    } else {
      val = &tbl->d_embedding_[tbl->kEmbeddingWidth * mem_offset];
    }
    for (int i = 0 ; i < tbl->kEmbeddingWidth; i++) {
      dst[tbl->kEmbeddingWidth * id + i] = val[i];
    }
  }
}

void HetuGPUTable::writeBack(embed_t *dst) {
  size_t num_rcvd = 0;
  for (int i = 0; i < (int)nrank_; i++)
    num_rcvd += cur_batch_.u_shape_exchanged[i];
  // Compute the prefix sum for return_outdated
  checkCudaErrors(cub::DeviceScan::ExclusiveSum(d_temp_, temp_bytes_,
    d_return_outdated_[1], d_return_outdated_[0], cur_batch_.unique_size, stream_main_));

  // Select index that need to be updated into d_query_idx[0]
  checkCudaErrors(cub::DeviceSelect::Flagged(d_temp_, temp_bytes_,
    d_query_idx_[0], d_return_outdated_[1], d_query_idx_[0],
    &cur_batch_.u_shape_exchanged[nrank_], cur_batch_.unique_size, stream_main_));

  // Update received value into local storage
  writeBackUpdateLocalKernel<<<DIM_GRID(num_rcvd), DIM_BLOCK, 0, stream_main_>>>(this, num_rcvd);
  writeBackTargetKernel<<<DIM_GRID(cur_batch_.batch_size), DIM_BLOCK, 0, stream_main_>>>(this, dst);
  checkCudaErrors(cudaStreamSynchronize(stream_main_));
}
