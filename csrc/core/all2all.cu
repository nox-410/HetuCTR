#include "hetu_gpu_table.h"
#include "common/helper_cuda.h"

using namespace hetu;

void HetuGPUTable::all2allExchangeShape(const size_t *shape, size_t *shape_out) {
  ncclGroupStart();
  for (int i = 0; i < (int)nrank_; i++) {
    ncclSend(shape + i, 1, ncclUint64, i, communicator_, stream_main_);
    ncclRecv(shape_out + i, 1, ncclUint64, i, communicator_, stream_main_);
  }
  ncclGroupEnd();
}
