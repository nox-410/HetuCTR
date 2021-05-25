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
