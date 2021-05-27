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
  all2all_received_ = rcvd_offset;

  // gradient part, using prev_batch
  checkCudaErrors(ncclGroupStart());
  snd_offset = 0, rcvd_offset = 0;
  for (int i = 0; i < (int)nrank_; i++) {
    checkCudaErrors(ncclSend(
      d_query_gradient_idx_[0] + snd_offset, prev_batch_.u_shape[i], ncclInt64, i, communicator_, stream_main_));
    checkCudaErrors(ncclRecv(
      d_query_gradient_idx_[1] + rcvd_offset, prev_batch_.u_shape_exchanged[i], ncclInt64, i, communicator_, stream_main_));
    snd_offset += prev_batch_.u_shape[i];
    rcvd_offset += prev_batch_.u_shape_exchanged[i];
  }
  checkCudaErrors(ncclGroupEnd());

  checkCudaErrors(ncclGroupStart());
  snd_offset = 0, rcvd_offset = 0;
  for (int i = 0; i < (int)nrank_; i++) {
    checkCudaErrors(ncclSend(
      d_query_updates_[0] + snd_offset, prev_batch_.u_shape[i], ncclInt64, i, communicator_, stream_main_));
    checkCudaErrors(ncclRecv(
      d_query_updates_[1] + rcvd_offset, prev_batch_.u_shape_exchanged[i], ncclInt64, i, communicator_, stream_main_));
    snd_offset += prev_batch_.u_shape[i];
    rcvd_offset += prev_batch_.u_shape_exchanged[i];
  }
  checkCudaErrors(ncclGroupEnd());

  checkCudaErrors(ncclGroupStart());
  snd_offset = 0, rcvd_offset = 0;
  for (int i = 0; i < (int)nrank_; i++) {
    checkCudaErrors(ncclSend(
      d_query_val_[0] + snd_offset * kEmbeddingWidth, prev_batch_.u_shape[i] * kEmbeddingWidth,
      ncclFloat32, i, communicator_, stream_main_));
    checkCudaErrors(ncclRecv(
      d_query_val_[1] + rcvd_offset * kEmbeddingWidth, prev_batch_.u_shape_exchanged[i] * kEmbeddingWidth,
      ncclFloat32, i, communicator_, stream_main_));
    snd_offset += prev_batch_.u_shape[i];
    rcvd_offset += prev_batch_.u_shape_exchanged[i];
  }
  checkCudaErrors(ncclGroupEnd());
  INFO("Total gradient update receive/push = ", rcvd_offset, "/", snd_offset);
}

void HetuGPUTable::all2allReturnOutdated() {
  checkCudaErrors(ncclGroupStart());
  size_t snd_offset = 0, rcvd_offset = 0;
  for (int i = 0; i < (int)nrank_; i++) {
    checkCudaErrors(ncclSend(
      d_return_outdated_[0] + snd_offset, cur_batch_.u_shape_exchanged[i], ncclInt64, i, communicator_, stream_main_));
    checkCudaErrors(ncclRecv(
      d_return_outdated_[1] + rcvd_offset, cur_batch_.u_shape[i], ncclInt64, i, communicator_, stream_main_));
    snd_offset += cur_batch_.u_shape_exchanged[i];
    rcvd_offset += cur_batch_.u_shape[i];
  }
  checkCudaErrors(ncclGroupEnd());
}

void HetuGPUTable::all2allReturnValue() {
  checkCudaErrors(ncclGroupStart());
  size_t snd_offset = 0, rcvd_offset = 0;
  for (int i = 0; i < (int)nrank_; i++) {
    checkCudaErrors(ncclSend(
      d_return_version_[0] + snd_offset, cur_batch_.u_shape[i], ncclInt64, i, communicator_, stream_main_));
    checkCudaErrors(ncclRecv(
      d_return_version_[1] + rcvd_offset, cur_batch_.u_shape_exchanged[i], ncclInt64, i, communicator_, stream_main_));
    snd_offset += cur_batch_.u_shape[i];
    rcvd_offset += cur_batch_.u_shape_exchanged[i];
  }
  checkCudaErrors(ncclGroupEnd());

  checkCudaErrors(ncclGroupStart());
  snd_offset = 0, rcvd_offset = 0;
  for (int i = 0; i < (int)nrank_; i++) {
    checkCudaErrors(ncclSend(
      d_return_val_[0] + snd_offset * kEmbeddingWidth, cur_batch_.u_shape[i] * kEmbeddingWidth,
      ncclFloat32, i, communicator_, stream_main_));
    checkCudaErrors(ncclRecv(
      d_return_val_[1] + rcvd_offset * kEmbeddingWidth, cur_batch_.u_shape_exchanged[i] * kEmbeddingWidth,
      ncclFloat32, i, communicator_, stream_main_));
    snd_offset += cur_batch_.u_shape[i];
    rcvd_offset += cur_batch_.u_shape_exchanged[i];
  }
  checkCudaErrors(ncclGroupEnd());
  all2all_received_ = rcvd_offset;
  INFO("Total embedding fetching serve/query = ", rcvd_offset, "/", snd_offset);
}
