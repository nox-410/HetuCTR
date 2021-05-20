#include "hetu_gpu_table.h"
#include "rendezvous/rendezvous.h"
#include "common/helper_cuda.h"

using namespace hetu;

void HetuGPUTable::initializeNCCL(const std::string &ip, const int port) {
  TCPRendezvous tcp(rank_, nrank_, ip, port);
  ncclUniqueId uid;
  if (rank_ == 0) {
    checkCudaErrors(ncclGetUniqueId(&uid));
  }
  tcp.broadcast(&uid, sizeof(uid));
  checkCudaErrors(cudaSetDevice(rank_));
  checkCudaErrors(cudaStreamCreate(&stream_main_));
  checkCudaErrors(cudaStreamCreate(&stream_sub_));
  checkCudaErrors(ncclCommInitRank(&communicator_, nrank_, uid, rank_));
}
