#include "preprocess.h"
#include "hetu_gpu_table.h"

#include <cuda_runtime.h>
#include "common/helper_cuda.h"

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
}

void freePreprocessData(PreprocessData &pdata) {
  checkCudaErrors(cudaFree(pdata.d_idx));
  checkCudaErrors(cudaFree(pdata.d_unique_idx));
  checkCudaErrors(cudaFree(pdata.d_idx_map));
  checkCudaErrors(cudaFree(pdata.d_offset));
  checkCudaErrors(cudaFree(pdata.d_root));
}

void HetuGPUTable::preprocessIndex(index_t *data, size_t batch_size) {
  std::swap(cur_batch_, prev_batch_);
  if (batch_size > cur_batch_.allocate_size) {
    INFO("ReAllocate cuda memory for batch ", cur_batch_.batch_size, "->" , batch_size);
    freePreprocessData(cur_batch_);
    createPreprocessData(cur_batch_, batch_size, nrank_);
  }
  cur_batch_.batch_size = batch_size;
  checkCudaErrors(cudaMemcpyAsync(
    cur_batch_.d_idx, data, sizeof(index_t) * batch_size, cudaMemcpyDefault, stream_main_));
  return;
}

} // namespace hetu
