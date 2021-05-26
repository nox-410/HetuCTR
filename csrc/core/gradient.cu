#include "hetu_gpu_table.h"
#include "common/helper_cuda.h"
#include <cub/cub.cuh>

using namespace hetu;

__global__ void decide_update_kernel(HetuGPUTable *tbl) {
  const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < tbl->prev_batch_.unique_size) {
    version_t update_new = tbl->prev_batch_.d_run_length[id + 1] - tbl->prev_batch_.d_run_length[id];
    index_t offset = tbl->prev_batch_.d_offset[id];
    if (tbl->prev_batch_.d_root[id] == tbl->rank_) {
      tbl->d_need_update_[id] = 0;
      tbl->d_version_[offset] += update_new;
    } else if (offset == kInvalidIndex) {
      tbl->d_need_update_[id] = 1;
    } else {
      // assert(offset < tbl->kNonLocalStorageMax);
      version_t update_local = tbl->d_updates_[offset];
      tbl->d_need_update_[id] = update_local + update_new <= tbl->push_bound_ ? 0 : 1;
      tbl->d_updates_[offset] += update_new;
    }
    if (tbl->d_need_update_[id])
      atomicAdd(&tbl->prev_batch_.u_shape[tbl->prev_batch_.d_root[id]], 1);
  }
}

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

__global__ void write_gradient_shape_kernel(HetuGPUTable *tbl) {
  const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = tbl->prev_batch_.unique_size;
  if (id >= n) return;
}

void HetuGPUTable::generateGradient(embed_t *grad) {
  memset(prev_batch_.u_shape, 0 , nrank_ * sizeof(size_t));
  size_t num_unique = prev_batch_.unique_size;
  decide_update_kernel<<<DIM_GRID(num_unique), DIM_BLOCK, 0, stream_main_>>>(this);

  checkCudaErrors(cub::DeviceScan::ExclusiveSum(d_temp_, temp_bytes_,
    d_need_update_, d_update_prefix_, num_unique, stream_main_));

  table_update_kernel<<<DIM_GRID(num_unique), DIM_BLOCK, 0, stream_main_>>>(this, grad);

  all2allExchangeShape(prev_batch_.u_shape, prev_batch_.u_shape_exchanged);

  checkCudaErrors(cudaStreamSynchronize(stream_main_));

  // std::cout << (int)rank_ << " ";
  // for (int i = 0 ; i < nrank_; i++)
  //   std::cout << prev_batch_.u_shape[i] << " ";
  // std::cout << std::endl;
  // std::cout << (int)rank_ << " ";
  // for (int i = 0 ; i < nrank_; i++)
  //   std::cout << prev_batch_.u_shape_exchanged[i] << " ";
  // std::cout << std::endl;
}
