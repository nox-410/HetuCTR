#include "hetu_gpu_table.h"

#include <cmath>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "common/helper_cuda.h"

namespace hetu {

// This computes keys as <root_id, embedding_id>
__global__ void generateSortkeys(HetuGPUTable *tbl) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < tbl->cur_batch_.batch_size) {
    index_t embedding_idx = tbl->cur_batch_.d_idx[id];
    assert(embedding_idx < tbl->kEmbeddingIDMax);
    worker_t r = tbl->d_root_[embedding_idx];
    tbl->cur_batch_.d_idx_map[id] = embedding_idx + tbl->kEmbeddingIDMax * r;
    tbl->cur_batch_.d_offset[id] = id;
  }
}

__global__ void block_cvt_offset_to_shape_kernel(size_t *dst) {
  size_t id = threadIdx.x;
  size_t n = blockDim.x;
  extern __shared__ size_t shm[];
  size_t val = dst[id];
  shm[id] = val;
  __syncthreads();
  size_t val_nxt = id == n - 1 ? val : shm[id + 1];
  assert(val_nxt >= val);
  dst[id] = val_nxt - val;
}

__global__ void writeSortedIndex(HetuGPUTable *tbl) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < tbl->cur_batch_.batch_size) {
    index_t arg = tbl->cur_batch_.d_sorted_arg[id];
    index_t embedding_idx = tbl->cur_batch_.d_idx[arg];
    tbl->cur_batch_.d_offset[id] = embedding_idx;
  }
}

// This will compute cur_batch_.d_idx_map
// cur_batch_.d_root cur_batch_.u_shape
__global__ void computeBatch(HetuGPUTable *tbl) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = tbl->cur_batch_.unique_size;
  if (id < n) {
    index_t uid = tbl->cur_batch_.d_unique_idx[id];
    int r = tbl->d_root_[uid], r_prev;
    tbl->cur_batch_.d_root[id] = r;
    auto iter = tbl->table_->find(uid);
    if (iter == tbl->table_->end()) {
      tbl->cur_batch_.d_offset[id] = kInvalidIndex;
    } else {
      tbl->cur_batch_.d_offset[id] = iter->second;
    }
    if (id == 0) r_prev = -1;
    else r_prev = tbl->d_root_[tbl->cur_batch_.d_unique_idx[id - 1]];
    for (int i = r_prev + 1; i <= r; i++) {
      tbl->cur_batch_.u_shape[i] = id;
    }
    if (id == n - 1) {
      for (int i = r + 1; i <= tbl->nrank_; i++) {
        tbl->cur_batch_.u_shape[i] = n;
      }
    }

    // This computes where we can find the unique index from the original index
    index_t idx_start, idx_end;
    idx_start = tbl->cur_batch_.d_run_length[id];
    idx_end = tbl->cur_batch_.d_run_length[id + 1];
    for (index_t i = idx_start; i < idx_end; i++) {
      index_t arg = tbl->cur_batch_.d_sorted_arg[i];
      tbl->cur_batch_.d_idx_map[arg] = id;
    }
  }
}

void HetuGPUTable::preprocessIndex(index_t *data, size_t batch_size) {
  // Copy batch embedding index data into Device
  checkCudaErrors(cudaMemcpyAsync(
    cur_batch_.d_idx, data, sizeof(index_t) * batch_size, cudaMemcpyHostToDevice, stream_main_));

  // use unused memory here to store temp sort keys
  generateSortkeys<<<DIM_GRID(batch_size), DIM_BLOCK, 0, stream_main_>>>(this);
  // we don't need to sort all the bits when using radix sort.
  // using end_bit smaller than 64 can yield corresponding performance improvement
  int end_bit = std::ceil(std::log2(kEmbeddingIDMax * nrank_));
  // store temp unused temp result in d_offset
  checkCudaErrors(cub::DeviceRadixSort::SortPairs(
    d_temp_, temp_bytes_, cur_batch_.d_idx_map, cur_batch_.d_unique_idx, cur_batch_.d_offset, cur_batch_.d_sorted_arg,
    batch_size, 0, end_bit, stream_main_));

  // After argsort write value to d_offset (temp, modify in next step)
  writeSortedIndex<<<DIM_GRID(batch_size), DIM_BLOCK, 0, stream_main_>>>(this);

  // perform unique operation, store total number of unique embedding items;
  checkCudaErrors(cub::DeviceRunLengthEncode::Encode(
    d_temp_, temp_bytes_, cur_batch_.d_offset, cur_batch_.d_unique_idx, cur_batch_.d_run_length,
    &cur_batch_.unique_size, batch_size, stream_main_));

  // Store the predix sum of length, this will be used in gradient reduction
  // although we should compute [0, unique_size), but we don't want to sync here
  checkCudaErrors(cub::DeviceScan::ExclusiveSum(d_temp_, temp_bytes_,
    cur_batch_.d_run_length, cur_batch_.d_run_length, cur_batch_.batch_size + 1, stream_main_));

  // Computes other preprocess data
  computeBatch<<<DIM_GRID(cur_batch_.batch_size), DIM_BLOCK, 0, stream_main_>>>(this);

  // convert offset to shape
  block_cvt_offset_to_shape_kernel<<<1, (int)nrank_ + 1,
    sizeof(size_t) * ((int)nrank_ + 1), stream_main_>>>(cur_batch_.u_shape);

  // exchange shape with other workers
  all2allExchangeShape(cur_batch_.u_shape, cur_batch_.u_shape_exchanged);

  // std::cout << cur_batch_.batch_size << " " << cur_batch_.unique_size << std::endl;

  // std::vector<index_t> h(batch_size + 1);
  // checkCudaErrors(cudaMemcpy(h.data(), cur_batch_.d_run_length, (batch_size + 1) * 8, cudaMemcpyDeviceToHost));
  // if (rank_ == 0)
  // for (int  i = 0 ; i <= batch_size; i++) {
  //   std::cout << h[i] << std::endl;
  // }
  // std::cout << "rank " << rank_ << ":";
  // for (worker_t i = 0; i < nrank_; i++) {
  //   std::cout << cur_batch_.u_shape[i] << " ";
  // }
  // std::cout << std::endl;
}

// figure out all gradients to push
// 1. compute d_need_update_ as 0 or 1
// 2. update d_version_ (stored and root=self)
// 3. update d_updates_ (stored and root!=self)
//
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

void HetuGPUTable::preprocessGradient() {
  if (prev_batch_.batch_size == 0) return;
  checkCudaErrors(cudaMemsetAsync(prev_batch_.u_shape, 0, nrank_ * sizeof(size_t), stream_main_));
  size_t num_unique = prev_batch_.unique_size;
  decide_update_kernel<<<DIM_GRID(num_unique), DIM_BLOCK, 0, stream_main_>>>(this);

  // d_update_prefix_[i] stores which index maps to the gradient communication slot i
  checkCudaErrors(cub::DeviceScan::ExclusiveSum(d_temp_, temp_bytes_,
    d_need_update_, d_update_prefix_, num_unique, stream_main_));

  all2allExchangeShape(prev_batch_.u_shape, prev_batch_.u_shape_exchanged);
}

} // namespace hetu
