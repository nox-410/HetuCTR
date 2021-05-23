#include "preprocess.h"
#include "hetu_gpu_table.h"

#include <cmath>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "common/helper_cuda.h"

namespace hetu {

void createPreprocessData(PreprocessData &pdata, size_t batch_size, size_t nrank) {
  assert(batch_size > 0);
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
  checkCudaErrors(cudaMallocManaged(
    &pdata.u_shape, sizeof(size_t) * (nrank + 1)));
  checkCudaErrors(cudaMallocManaged(
    &pdata.u_shape_exchanged, sizeof(size_t) * (nrank + 1)));
}

void freePreprocessData(PreprocessData &pdata) {
  checkCudaErrors(cudaFree(pdata.d_idx));
  checkCudaErrors(cudaFree(pdata.d_unique_idx));
  checkCudaErrors(cudaFree(pdata.d_idx_map));
  checkCudaErrors(cudaFree(pdata.d_offset));
  checkCudaErrors(cudaFree(pdata.d_root));
  checkCudaErrors(cudaFree(pdata.u_shape));
  checkCudaErrors(cudaFree(pdata.u_shape_exchanged));
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
  // we don't need to sort all the bits when using radix sort.
  // using end_bit smaller than 64 can yield corresponding performance improvement
  int end_bit = std::ceil(std::log2(kEmbeddingIDMax * nrank_));
  checkCudaErrors(cub::DeviceRadixSort::SortPairs(
    d_temp_, temp_bytes_, cur_batch_.d_idx_map, cur_batch_.d_idx_map, cur_batch_.d_idx, cur_batch_.d_unique_idx,
    batch_size, 0, end_bit, stream_main_));

  // perform unique operation, store total number of unique embedding items;
  // we don't really need the count of each run, so we put it in offset and will overwrite it later
  checkCudaErrors(cub::DeviceRunLengthEncode::Encode(
    d_temp_, temp_bytes_, cur_batch_.d_unique_idx, cur_batch_.d_unique_idx, cur_batch_.d_offset,
    &cur_batch_.u_shape[nrank_], batch_size, stream_main_));

  // Take the number of unique embedding index to host
  checkCudaErrors(cudaStreamSynchronize(stream_main_));
  cur_batch_.unique_size = cur_batch_.u_shape[nrank_];

  // std::cout << cur_batch_.batch_size << " " << cur_batch_.unique_size << std::endl;

  // We should only lookup index in range [0, unique_size)
  hash_table_.get(cur_batch_.d_unique_idx, cur_batch_.d_offset, cur_batch_.unique_size, stream_main_);

  // This computes how many embedding belongs to each worker
  computeIndexRootShape<<<DIM_GRID(cur_batch_.unique_size), DIM_BLOCK, 0, stream_main_>>>(
    cur_batch_.d_root, cur_batch_.u_shape , cur_batch_.d_unique_idx, d_root_, cur_batch_.unique_size, nrank_);

  // This computes where we can find the unique index from the original index
  parallelLowerBound<<<DIM_GRID(batch_size), DIM_BLOCK, 0, stream_main_>>>(
    cur_batch_.d_idx_map, cur_batch_.d_unique_idx, cur_batch_.d_idx, cur_batch_.unique_size, batch_size);

  // convert offset to shape
  checkCudaErrors(cudaStreamSynchronize(stream_main_));
  for (int i = 0 ;i < nrank_; i++)
     cur_batch_.u_shape[i] = cur_batch_.u_shape[i + 1] - cur_batch_.u_shape[i];

  // exchange shape with other workers
  all2allExchangeShape(cur_batch_.u_shape, cur_batch_.u_shape_exchanged);

  cudaStreamSynchronize(stream_main_);

  // std::vector<index_t> h(batch_size);
  // checkCudaErrors(cudaMemcpy(h.data(), cur_batch_.d_offset, batch_size * 8, cudaMemcpyDeviceToHost));
  // if (rank_ == 0)
  // for (int  i = 0 ; i < batch_size; i++) {
  //   std::cout << h[i] << std::endl;
  // }
  // std::cout << "rank " << rank_ << ":";
  // for (worker_t i = 0; i < nrank_; i++) {
  //   std::cout << cur_batch_.u_shape[i] << " ";
  // }
  // std::cout << std::endl;
}

} // namespace hetu
