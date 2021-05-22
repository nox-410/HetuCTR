#include "preprocess.h"
#include "hetu_gpu_table.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "common/helper_cuda.h"

#include <thrust/binary_search.h>

namespace hetu {

void createPreprocessData(PreprocessData &pdata, size_t batch_size, size_t nrank) {
  assert(batch_size > 0);
  pdata.embed_root_shape.resize(nrank, 0);
  pdata.batch_size = batch_size;
  pdata.allocate_size = batch_size;
  checkCudaErrors(cudaMalloc(
    &pdata.d_idx, sizeof(index_t) * batch_size));
  checkCudaErrors(cudaMalloc(
    &pdata.d_unique_idx, sizeof(index_t) * batch_size));
  checkCudaErrors(cudaMalloc(
    &pdata.d_idx_map, sizeof(index_t) * batch_size));
  checkCudaErrors(cudaMalloc(
    &pdata.d_offset, sizeof(index_t) * batch_size));
  checkCudaErrors(cudaMalloc(
    &pdata.d_root, sizeof(worker_t) * batch_size));
  checkCudaErrors(cudaMalloc(
    &pdata.d_embed_root_shape, sizeof(size_t) * nrank));
}

void freePreprocessData(PreprocessData &pdata) {
  checkCudaErrors(cudaFree(pdata.d_idx));
  checkCudaErrors(cudaFree(pdata.d_unique_idx));
  checkCudaErrors(cudaFree(pdata.d_idx_map));
  checkCudaErrors(cudaFree(pdata.d_offset));
  checkCudaErrors(cudaFree(pdata.d_root));
  checkCudaErrors(cudaFree(pdata.d_embed_root_shape));
}

__global__ void generateSortkeys(index_t *dst, const index_t *d_index,
  const worker_t *d_root, const size_t n, const size_t idx_max) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n) {
    worker_t r = d_root[d_index[id]];
    assert(d_index[id] < idx_max);
    dst[id] = d_index[id] + idx_max * r;
  }
}

__global__ void computeIndexRootShape(worker_t *d_root_dst, size_t* d_offset,
  const index_t *d_uid, const worker_t *d_root, const size_t n, const worker_t nrank) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n) {
    int r = d_root[d_uid[id]], r_prev;
    d_root_dst[id] = r;
    if (id == 0) r_prev = -1;
    else r_prev = d_root[d_uid[id - 1]];
    for (int i = r_prev + 1; i <= r; i++) {
      d_offset[i] = id;
    }
    if (id == n - 1) {
      for (int i = r + 1; i < nrank; i++) {
        d_offset[i] = n;
      }
    }
  }
}

__device__ index_t lowerBound(const index_t *data, const size_t len, index_t target) {
  index_t start = 0, last = len;
	while (start < last) {
		index_t mid = (start + last) / 2;
		if (data[mid] >= target) last = mid;
		else start = mid + 1;
	}
	return start;
}

__global__ void parallelLowerBound(index_t *dst, const index_t *d_uid, const index_t *target,
  const size_t len, const size_t n) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n) {
    index_t val = target[id];
    dst[id] = lowerBound(d_uid, len, val);
  }
}

void HetuGPUTable::preprocessIndex(unsigned long data_ptr, size_t batch_size) {
  index_t *data = (index_t *)data_ptr;
  std::swap(cur_batch_, prev_batch_);
  if (batch_size > batch_size_reserved_) {
    allocateAuxillaryMemory(batch_size);
  }
  if (batch_size > cur_batch_.allocate_size) {
    INFO("ReAllocate cuda memory for batch ", cur_batch_.batch_size, "->" , batch_size);
    freePreprocessData(cur_batch_);
    createPreprocessData(cur_batch_, batch_size, nrank_);
  }
  cur_batch_.batch_size = batch_size;
  // Copy batch embedding index data into Device
  checkCudaErrors(cudaMemcpyAsync(
    cur_batch_.d_idx, data, sizeof(index_t) * batch_size, cudaMemcpyHostToDevice, stream_main_));

  // use unused memory here to store temp sort keys
  generateSortkeys<<<DIM_GRID(batch_size), DIM_BLOCK, 0, stream_main_>>>(
    cur_batch_.d_idx_map, cur_batch_.d_idx, d_root_, batch_size, kEmbeddingIDMax);

  checkCudaErrors(cub::DeviceRadixSort::SortPairs(
    d_temp_, temp_bytes_, cur_batch_.d_idx_map, cur_batch_.d_idx_map, cur_batch_.d_idx, cur_batch_.d_unique_idx,
    batch_size, 0, sizeof(index_t) * 8, stream_main_));

  // perform unique operation, store total number of unique embedding items;
  // we don't really need the count of each run, so we put it in offset and will overwrite it later
  checkCudaErrors(cub::DeviceRunLengthEncode::Encode(
    d_temp_, temp_bytes_, cur_batch_.d_unique_idx, cur_batch_.d_unique_idx, cur_batch_.d_offset, cur_batch_.d_idx_map,
    batch_size, stream_main_));

  // Take the number of unique embedding index to host
  checkCudaErrors(cudaMemcpyAsync(
    &cur_batch_.unique_size, cur_batch_.d_idx_map, sizeof(index_t), cudaMemcpyDeviceToHost, stream_main_));

  // We should only lookup index in range [0, unique_size)
  checkCudaErrors(cudaStreamSynchronize(stream_main_));
  // std::cout << cur_batch_.batch_size << " " << cur_batch_.unique_size << std::endl;
  hash_table_.get(cur_batch_.d_unique_idx, cur_batch_.d_offset, cur_batch_.unique_size, stream_main_);
  computeIndexRootShape<<<DIM_GRID(cur_batch_.unique_size), DIM_BLOCK, 0, stream_main_>>>(
    cur_batch_.d_root, cur_batch_.d_embed_root_shape , cur_batch_.d_unique_idx, d_root_, cur_batch_.unique_size, nrank_);
  parallelLowerBound<<<DIM_GRID(batch_size), DIM_BLOCK, 0, stream_main_>>>(
    cur_batch_.d_idx_map, cur_batch_.d_unique_idx, cur_batch_.d_idx, cur_batch_.unique_size, batch_size);

  checkCudaErrors(cudaMemcpyAsync(cur_batch_.embed_root_shape.data(), cur_batch_.d_embed_root_shape,
    sizeof(size_t) * nrank_, cudaMemcpyDeviceToHost, stream_main_));

  checkCudaErrors(cudaStreamSynchronize(stream_main_));
  // for (worker_t i = 0; i < nrank_; i++) {
  //   std::cout << cur_batch_.embed_root_shape[i] << " ";
  // }
  // std::cout << std::endl;
}

} // namespace hetu
