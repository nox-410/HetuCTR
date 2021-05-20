#include "hetu_gpu_table.h"
#include "utils/rendezvous.h"
#include "common/helper_cuda.h"

using namespace hetu;

void HetuGPUTable::initializeNCCL(const std::string &ip, const int port) {
  checkCudaErrors(cudaSetDevice(rank_));
  checkCudaErrors(cudaStreamCreate(&stream_main_));
  checkCudaErrors(cudaStreamCreate(&stream_sub_));
  TCPRendezvous tcp(rank_, nrank_, ip, port);
  ncclUniqueId uid;
  if (rank_ == 0) {
    checkCudaErrors(ncclGetUniqueId(&uid));
  }
  tcp.broadcast(&uid, sizeof(uid));
  checkCudaErrors(ncclCommInitRank(&communicator_, nrank_, uid, rank_));
}

HetuGPUTable::HetuGPUTable(
  worker_t rank,
  worker_t nrank,
  worker_t device_id,
  std::string ip,
  int port,
  size_t embedding_length,
  size_t embedding_width,
  version_t pull_bound,
  version_t push_bound,
  SArray<worker_t> root_id_arr,
  SArray<index_t> storage_id_arr
) :
  rank_(rank),
  nrank_(nrank),
  device_id_(device_id),
  kEmbeddingIDMax(embedding_length),
  kEmbeddingWidth(embedding_width),
  kStorageMax(storage_id_arr.size()),
  pull_bound_(pull_bound),
  push_bound_(push_bound),
  hash_table_(kStorageMax, 0)
{
  initializeNCCL(ip, port);
}

HetuGPUTable::~HetuGPUTable() {
}
