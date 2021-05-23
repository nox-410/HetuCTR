#include "hetu_gpu_table.h"

#include "common/helper_cuda.h"

using namespace hetu;

void HetuGPUTable::pushPull(unsigned long grad, unsigned long dst) {
  generateQuery();
  // Compute shape for query items
  all2allExchangeQuery();
  checkCudaErrors(cudaStreamSynchronize(stream_main_));
  return;
}
