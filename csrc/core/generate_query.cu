#include "hetu_gpu_table.h"
#include "common/helper_cuda.h"
#include <cub/cub.cuh>

using namespace hetu;

// aggregate all the gradients into storage
__global__ void table_update_kernel(HetuGPUTable *tbl, embed_t *grad) {
  const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t width = tbl->kEmbeddingWidth;
  if (id >= tbl->prev_batch_.unique_size) return;

  bool need_update = tbl->d_need_update_[id];
  index_t offset = tbl->prev_batch_.d_offset[id];

  int update_type;
  if (offset == kInvalidIndex)
    update_type = 0; // not storaged
  else if (offset >= tbl->kNonLocalStorageMax)
    update_type = 1; // storaged, local
  else if (need_update)
    update_type = 2; // storaged, non-local, update
  else update_type = 3; // storaged, non-local, no update

  index_t query_idx = tbl->d_update_prefix_[id];

  embed_t *dest_grad = tbl->d_gradient_ + offset * width;
  embed_t *dest_query = &tbl->d_query_val_[0][query_idx * width];
  embed_t *dest = tbl->d_embedding_ + offset * width;

  index_t l = tbl->prev_batch_.d_run_length[id], r = tbl->prev_batch_.d_run_length[id + 1];

  if (need_update) {
    version_t update_count = r - l;
    if (update_type == 2) {
      update_count = tbl->d_updates_[offset];
      tbl->d_updates_[offset] = 0;
    }
    tbl->d_query_gradient_idx_[0][query_idx] = tbl->prev_batch_.d_unique_idx[id];
    tbl->d_query_updates_[0][query_idx] = update_count;
  }

  for (size_t i = 0; i < width; i++) {
    embed_t sum = 0;
    for (index_t j = l; j < r; j++) {
      index_t grad_offset = tbl->prev_batch_.d_sorted_arg[j];
      sum += grad[grad_offset * width + i];
    }
    if (update_type != 0) dest[i] += sum;
    else dest_query[i] = sum;

    if (update_type >= 2) dest_grad[i] += sum;
    if (update_type == 2) {
      dest_query[i] = dest_grad[i];
      dest_grad[i] = 0;
    }
  }
}

void HetuGPUTable::generateGradient(embed_t *grad) {
  size_t num_unique = prev_batch_.unique_size;

  table_update_kernel<<<DIM_GRID(num_unique), DIM_BLOCK, 0, stream_main_>>>(this, grad);
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
