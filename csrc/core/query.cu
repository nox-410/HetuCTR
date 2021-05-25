#include "hetu_gpu_table.h"
#include "common/helper_cuda.h"
#include <cub/cub.cuh>

using namespace hetu;

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

__global__ void writeReturnValue(HetuGPUTable *tbl) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  size_t len = tbl->cur_batch_.u_shape[tbl->nrank_];
  if (id < len) {
    version_t local_version = tbl->d_query_version_[1][id];
    index_t embedding_idx = tbl->d_query_idx_[1][id];
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
  size_t num_rcvd = 0;
  for (int i = 0; i < nrank_; i++) num_rcvd += cur_batch_.u_shape_exchanged[i];
  INFO(num_rcvd, " received embedding index to handle.");
  computeReturnOutdated<<<DIM_GRID(num_rcvd), DIM_BLOCK, 0, stream_main_>>>(this, num_rcvd);

  all2allReturnOutdated();

  checkCudaErrors(cub::DeviceScan::ExclusiveSum(d_temp_, temp_bytes_,
    cur_batch_.u_shape_exchanged, cur_batch_.u_shape, nrank_ + 1, stream_main_));

  checkCudaErrors(cub::DeviceSegmentedReduce::Sum(d_temp_, temp_bytes_,
    d_return_outdated_[0], cur_batch_.u_shape, nrank_, cur_batch_.u_shape, cur_batch_.u_shape + 1, stream_main_));

  all2allExchangeShape(cur_batch_.u_shape, cur_batch_.u_shape_exchanged);

  // select index that requires update into d_query_idx_[0]
  // total number stored in cur_batch_.u_shape[nrank_]
  checkCudaErrors(cub::DeviceSelect::Flagged(d_temp_, temp_bytes_,
    d_query_idx_[1], d_return_outdated_[0], d_query_idx_[1], &cur_batch_.u_shape[nrank_], num_rcvd, stream_main_));

  writeReturnValue<<<DIM_GRID(num_rcvd), DIM_BLOCK, 0, stream_main_>>>(this);

  checkCudaErrors(cudaStreamSynchronize(stream_main_));

  all2allReturnValue();
}
