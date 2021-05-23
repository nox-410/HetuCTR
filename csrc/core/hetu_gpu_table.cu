#include "hetu_gpu_table.h"

#include "common/helper_cuda.h"

using namespace hetu;

void HetuGPUTable::pushPull(unsigned long grad, unsigned long dst) {
  generateQuery();

  all2allExchangeQuery();

  handleQuery();
  checkCudaErrors(cudaStreamSynchronize(stream_main_));
  return;
}