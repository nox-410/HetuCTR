#include "hetu_gpu_table.h"
#include "common/helper_cuda.h"

using namespace hetu;

void HetuGPUTable::all2allExchangeShape(const size_t *shape, size_t *shape_out) {
  assert(shape != shape_out);
  checkCudaErrors(ncclGroupStart());
  for (int i = 0; i < (int)nrank_; i++) {
    checkCudaErrors(ncclSend(
      shape + i, 1, ncclUint64, i, communicator_, stream_main_));
    checkCudaErrors(ncclRecv(
      shape_out + i, 1, ncclUint64, i, communicator_, stream_main_));
  }
  checkCudaErrors(ncclGroupEnd());
}

void HetuGPUTable::all2allExchangeQuery() {
  checkCudaErrors(ncclGroupStart());
  size_t snd_offset = 0, rcvd_offset = 0;
  for (int i = 0; i < (int)nrank_; i++) {
    checkCudaErrors(ncclSend(
      d_query_idx_[0] + snd_offset, cur_batch_.u_shape[i], ncclInt64, i, communicator_, stream_main_));
    checkCudaErrors(ncclRecv(
      d_query_idx_[1] + rcvd_offset, cur_batch_.u_shape_exchanged[i], ncclInt64, i, communicator_, stream_main_));
    snd_offset += cur_batch_.u_shape[i];
    rcvd_offset += cur_batch_.u_shape_exchanged[i];
  }
  checkCudaErrors(ncclGroupEnd());
  checkCudaErrors(ncclGroupStart());
  snd_offset = 0, rcvd_offset = 0;
  for (int i = 0; i < (int)nrank_; i++) {
    checkCudaErrors(ncclSend(
      d_query_version_[0] + snd_offset, cur_batch_.u_shape[i], ncclInt64, i, communicator_, stream_main_));
    checkCudaErrors(ncclRecv(
      d_query_version_[1] + rcvd_offset, cur_batch_.u_shape_exchanged[i], ncclInt64, i, communicator_, stream_main_));
    snd_offset += cur_batch_.u_shape[i];
    rcvd_offset += cur_batch_.u_shape_exchanged[i];
  }
  checkCudaErrors(ncclGroupEnd());
}
