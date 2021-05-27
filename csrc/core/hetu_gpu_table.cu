#include "hetu_gpu_table.h"

#include "common/helper_cuda.h"

using namespace hetu;

void HetuGPUTable::pushPull(unsigned long grad, unsigned long dst) {
  generateGradient((embed_t*)grad);

  all2allGradient();

  generateQuery();

  all2allExchangeQuery();

  handleGradient();

  handleQuery();

  writeBack((embed_t*)dst);
  return;
}

void HetuGPUTable::preprocess(unsigned long data_ptr, size_t batch_size) {
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

  preprocessIndex((index_t *)(data_ptr), batch_size);

  preprocessGradient();

  checkCudaErrors(cudaStreamSynchronize(stream_main_));
}
