#include "hetu_gpu_table.h"
#include "common/helper_cuda.h"
#include <cub/cub.cuh>

namespace hetuCTR {

__global__ void computeReturnOutdated(HetuGPUTable *tbl, size_t len) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < len) {
    version_t local_version = tbl->d_query_version_[1][id];
    index_t embedding_idx = tbl->d_query_idx_[1][id];
    auto iter = tbl->table_->find(embedding_idx);

    assert(tbl->d_root_[embedding_idx] == tbl->rank_);
    assert(iter != tbl->table_->end());

    version_t global_version = tbl->d_version_[iter->second];
    if (local_version == kInvalidVersion || local_version + tbl->pull_bound_ <= global_version)
      tbl->d_return_outdated_[0][id] = 1;
    else tbl->d_return_outdated_[0][id] = 0;
  }
}

__global__ void write_return_value_kernel(HetuGPUTable *tbl) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  size_t len = *(tbl->d_shape_);
  if (id < len) {
    version_t local_version = tbl->d_query_version_[1][id];
    index_t embedding_idx = tbl->d_update_prefix_[id];
    auto iter = tbl->table_->find(embedding_idx);

    assert(tbl->d_root_[embedding_idx] == tbl->rank_);
    assert(iter != tbl->table_->end());
    index_t offset = iter->second;

    version_t global_version = tbl->d_version_[offset];
    tbl->d_return_version_[0][id] = global_version;
    for (int i = 0; i < tbl->kEmbeddingWidth; i++)
      tbl->d_return_val_[0][tbl->kEmbeddingWidth * id + i] = tbl->d_embedding_[tbl->kEmbeddingWidth * offset + i];
  }
}

void HetuGPUTable::handleQuery() {
  INFO(all2all_received_, " received embedding index to handle.");
  computeReturnOutdated<<<DIM_GRID(all2all_received_), DIM_BLOCK, 0, stream_main_>>>(d_this, all2all_received_);

  all2allReturnOutdated();

  checkCudaErrors(cub::DeviceScan::ExclusiveSum(d_temp_, temp_bytes_,
    cur_batch_.u_shape_exchanged, cur_batch_.u_shape_exchanged, nrank_ + 1, stream_main_));

  checkCudaErrors(cub::DeviceSegmentedReduce::Sum(d_temp_, temp_bytes_,
    d_return_outdated_[0], cur_batch_.u_shape, nrank_,
    cur_batch_.u_shape_exchanged, cur_batch_.u_shape_exchanged + 1, stream_main_));

  // exchange return value shape and copy them to host
  all2allExchangeShape(cur_batch_.u_shape, cur_batch_.u_shape_exchanged);
  checkCudaErrors(cudaMemcpyAsync(cur_batch_.h_shape, cur_batch_.u_shape,
    sizeof(size_t) * (nrank_ + 1), cudaMemcpyDeviceToHost, stream_main_));
  checkCudaErrors(cudaMemcpyAsync(cur_batch_.h_shape_exchanged, cur_batch_.u_shape_exchanged,
    sizeof(size_t) * (nrank_ + 1), cudaMemcpyDeviceToHost, stream_main_));

  // select index that requires update into d_update_prefix_
  // total number stored in d_shape_
  checkCudaErrors(cub::DeviceSelect::Flagged(d_temp_, temp_bytes_,
    d_query_idx_[1], d_return_outdated_[0], d_update_prefix_, d_shape_, all2all_received_, stream_main_));

  write_return_value_kernel<<<DIM_GRID(all2all_received_), DIM_BLOCK, 0, stream_main_>>>(d_this);

  checkCudaErrors(cudaStreamSynchronize(stream_main_));

  all2allReturnValue();
}

__global__ void table_update_remote_kernel(HetuGPUTable *tbl, size_t start, size_t len) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < len) {
    size_t width = tbl->kEmbeddingWidth;
    id += start;
    index_t embedding_idx = tbl->d_query_gradient_idx_[1][id];
    auto iter = tbl->table_->find(embedding_idx);

    assert(tbl->d_root_[embedding_idx] == tbl->rank_);
    assert(iter != tbl->table_->end());
    index_t offset = iter->second;

    tbl->d_version_[offset] += tbl->d_query_updates_[1][id];
    for (int i = 0; i < tbl->kEmbeddingWidth; i++)
      tbl->d_embedding_[offset * width + i] += tbl->d_query_val_[1][id * width + i];
  }
}

void HetuGPUTable::handleGradient() {
  size_t offset = 0;
  for (int i = 0 ; i < nrank_; i++) {
    size_t shape = prev_batch_.h_shape_exchanged[i];
    table_update_remote_kernel<<<DIM_GRID(shape), DIM_BLOCK, 0, stream_main_>>>(d_this, offset, shape);
    offset += shape;
  }
}

} // namespace hetuCTR
