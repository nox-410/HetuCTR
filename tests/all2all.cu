#include "rendezvous/rendezvous.h"

#include <nccl.h>
#include <iostream>
#include <chrono>
#include <memory>

using namespace std::chrono;

int main(int argc, char* argv[]) {
    ncclUniqueId id;
    int myRank = atoi(argv[1]);
    int nRanks = atoi(argv[2]);
    auto tcp = std::make_shared<hetu::TCPRendezvous>(myRank, nRanks, "127.0.0.1", 23254);
    if (myRank == 0) ncclGetUniqueId(&id);
    tcp->broadcast(&id, sizeof(id));
    tcp.reset();
    ncclComm_t comm;
    cudaStream_t stream;
    cudaSetDevice(myRank);
    cudaStreamCreate(&stream);
    ncclCommInitRank(&comm, nRanks, id, myRank);
    float *sendbuff, *recvbuff;
    int N = 1024;
    cudaMalloc(&sendbuff, N * sizeof(float));
    cudaMalloc(&recvbuff, N * sizeof(float) * nRanks);
    for (int i = 0; i < 5; i++) {
        auto t1 = system_clock::now();
        ncclGroupStart();
        for (int r=0; r<nRanks; r++) {
            ncclSend(sendbuff, N, ncclFloat32, r, comm, stream);
            ncclRecv(recvbuff + r * N, N, ncclFloat32, r, comm, stream);
        }
        ncclGroupEnd();
        cudaStreamSynchronize(stream);
        auto t2 = system_clock::now();
        std::cout << (t2-t1).count() / 1e6 << std::endl;
    }
    ncclCommDestroy(comm);
    cudaStreamDestroy(stream);
}
