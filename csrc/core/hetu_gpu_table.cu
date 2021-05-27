#include "hetu_gpu_table.h"

#include "common/helper_cuda.h"

using namespace hetuCTR;

void HetuTable::pushPull(unsigned long grad, unsigned long dst) {
  generateGradient((embed_t*)grad);

  generateQuery();

  all2allExchangeQuery();

  handleGradient();

  handleQuery();

  writeBack((embed_t*)dst);
  return;
}

void HetuTable::preprocess(unsigned long data_ptr, size_t batch_size) {
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

  checkCudaErrors(cudaMemcpyAsync(
    d_this, this, sizeof(HetuTable), cudaMemcpyHostToDevice, stream_main_));

  preprocessIndex((index_t *)(data_ptr), batch_size);

  preprocessGradient();

  checkCudaErrors(cudaStreamSynchronize(stream_main_));
}
