#include "hetu_gpu_table.h"

#include "common/helper_cuda.h"

using namespace hetuCTR;

void HetuTable::pushPull(unsigned long grad, unsigned long dst) {
  checkCudaErrors(cudaSetDevice(device_id_));
  generateGradient((embed_t*)grad);

  generateQuery();

  all2allExchangeQuery();

  handleGradient();

  handleQuery();

  checkCudaErrors(cudaStreamSynchronize(stream_main_));

  all2allReturnValue();

  writeBack((embed_t*)dst);

  checkCudaErrors(cudaStreamSynchronize(stream_main_));
  return;
}

void HetuTable::preprocess(unsigned long data_ptr, size_t batch_size) {
  checkCudaErrors(cudaSetDevice(device_id_));
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

  // sync data with this pointer on device
  checkCudaErrors(cudaMemcpyAsync(
    d_this, this, sizeof(HetuTable), cudaMemcpyHostToDevice, stream_main_));

  preprocessIndex((index_t *)(data_ptr), batch_size);

  preprocessGradient();

  checkCudaErrors(cudaStreamSynchronize(stream_main_));
}
