#include "hetu_gpu_table.h"
#include "utils/rendezvous.h"
#include "common/helper_cuda.h"

#include <chrono>
#include <thrust/partition.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>

#include <cub/cub.cuh>

using namespace hetu;

void HetuGPUTable::initializeNCCL(const std::string &ip, const int port) {
  checkCudaErrors(cudaSetDevice(device_id_));
  checkCudaErrors(cudaStreamCreate(&stream_main_));
  checkCudaErrors(cudaStreamCreate(&stream_sub_));
  INFO("Start TCP rendezvous at ", ip, ":", port);
  TCPRendezvous tcp(rank_, nrank_, ip, port);
  ncclUniqueId uid;
  if (rank_ == 0) {
    checkCudaErrors(ncclGetUniqueId(&uid));
  }
  tcp.broadcast(&uid, sizeof(uid));
  INFO("NCCL Connection built successfully");
  checkCudaErrors(ncclCommInitRank(&communicator_, nrank_, uid, rank_));
}

struct _PartitionPrediate {
  const int rank;
  const worker_t *d_root;

  _PartitionPrediate(int _rank, worker_t *_d_root) : rank(_rank), d_root(_d_root) {}

  __device__ bool operator()(index_t idx) const { return d_root[idx]!=rank; }
};

void HetuGPUTable::initializeTable(SArray<worker_t> root_id_arr, SArray<index_t> storage_id_arr) {
  // copy root id array, this indicates which worker holds an embedding.
  checkCudaErrors(cudaMalloc(
    &d_root_, sizeof(worker_t) * kEmbeddingIDMax));
  checkCudaErrors(cudaMemcpy(
    d_root_, root_id_arr.data(), sizeof(worker_t) * kEmbeddingIDMax, cudaMemcpyHostToDevice));

  // Prepare keys and values for HashTable
  // key : reordered storage index, non-local embedding first
  // value : memory offset from 0 to kStorageMax
  thrust::device_vector<index_t> key(kStorageMax), value(kStorageMax);
  thrust::sequence(value.begin(), value.end());
  checkCudaErrors(cudaMemcpy(
    key.data().get(), storage_id_arr.data(), sizeof(index_t) * kStorageMax, cudaMemcpyHostToDevice));
  // reorder key with Predicate
  auto partition_point = thrust::stable_partition(key.begin(), key.end(), _PartitionPrediate(rank_, d_root_));
  hash_table_.insert(key.data().get(), value.data().get(), kStorageMax, stream_main_);

  // We now know how many non-local embeddings we have, allocate gradients and updates memory for them
  // Do not allocate gradients and updates for local embeddings.
  kNonLocalStorageMax = partition_point - key.begin();
  checkCudaErrors(cudaMalloc(
    &d_updates_, sizeof(version_t) * kNonLocalStorageMax));
  checkCudaErrors(cudaMalloc(
    &d_version_, sizeof(version_t) * kStorageMax));
  checkCudaErrors(cudaMalloc(
    &d_embedding_, sizeof(embed_t) * kStorageMax * kEmbeddingWidth));
  checkCudaErrors(cudaMalloc(
    &d_gradient_, sizeof(embed_t) * kNonLocalStorageMax * kEmbeddingWidth));

  // Set Gradients and Updates to zero
  checkCudaErrors(cudaMemset(
    d_gradient_, 0, sizeof(embed_t) * kNonLocalStorageMax * kEmbeddingWidth));
  checkCudaErrors(cudaMemset(
    d_updates_, 0, sizeof(version_t) * kNonLocalStorageMax));

  // Initialize version, set local version to 1, set non-local version to invalid
  auto v_ptr = thrust::device_ptr<version_t>(d_version_);
  thrust::fill(v_ptr, v_ptr + kNonLocalStorageMax, kInvalidVersion);
  thrust::fill(v_ptr + kNonLocalStorageMax, v_ptr + kStorageMax, 1);
  checkCudaErrors(cudaStreamSynchronize(stream_main_));
}

HetuGPUTable::HetuGPUTable(
  const worker_t rank,
  const worker_t nrank,
  const worker_t device_id,
  const std::string &ip,
  const int port,
  const size_t embedding_length,
  const size_t embedding_width,
  const version_t pull_bound,
  const version_t push_bound,
  SArray<worker_t> root_id_arr,
  SArray<index_t> storage_id_arr,
  const Initializer &init,
  const int verbose
) :
  rank_(rank),
  nrank_(nrank),
  device_id_(device_id),
  kEmbeddingIDMax(embedding_length),
  kEmbeddingWidth(embedding_width),
  kStorageMax(storage_id_arr.size()),
  pull_bound_(pull_bound),
  push_bound_(push_bound),
  hash_table_(kStorageMax, 0),
  verbose_(verbose)
{
  // Check device id
  int num_gpus = 0;
  checkCudaErrors(cudaGetDeviceCount(&num_gpus));
  CHECK(device_id < num_gpus) << "InCorrect device number.";

  // Setup NCCL
  initializeNCCL(ip, port);

  // Setup Embedding Table
  initializeTable(root_id_arr, storage_id_arr);

  // Initialize Embedding Table with initializer
  unsigned int seed = 0;
  seed = std::chrono::system_clock::now().time_since_epoch().count();
  initialize(d_embedding_, kEmbeddingIDMax * kEmbeddingWidth, init, false, seed);
  INFO("Table Init Successfully");

  // Initialize preprocess data and auxillary memory
  createPreprocessData(cur_batch_, batch_size_reserved_, nrank_);
  createPreprocessData(prev_batch_, batch_size_reserved_, nrank_);
  allocateAuxillaryMemory(batch_size_reserved_);
}

void HetuGPUTable::allocateAuxillaryMemory(size_t batch_size) {
  freeAuxillaryMemory();
  size_t temp_bytes_max = 1, temp_bytes;

  // Check how much temp memory cub want to use
  index_t *ptr;
  checkCudaErrors(cub::DeviceRadixSort::SortPairs(
    nullptr, temp_bytes, ptr, ptr, ptr, ptr, batch_size));
  temp_bytes_max = std::max(temp_bytes, temp_bytes_max);
  checkCudaErrors(cub::DeviceRunLengthEncode::Encode(
    nullptr, temp_bytes, ptr, ptr, ptr, ptr, batch_size));
  temp_bytes_max = std::max(temp_bytes, temp_bytes_max);

  INFO("Allocate Temp Memory --- ", temp_bytes_max, " bytes");
  checkCudaErrors(cudaMalloc(&d_temp_, temp_bytes_max));
  batch_size_reserved_ = batch_size;
  temp_bytes_ = temp_bytes_max;

  // We need to allocate nrank * batch_size so that it will be enough for all-to-all query
  size_t batch_limit = batch_size * nrank_;
  checkCudaErrors(cudaMalloc(
    &d_query_idx_, batch_limit * sizeof(index_t)));
  checkCudaErrors(cudaMalloc(
    &d_query_gradient_idx_, batch_limit * sizeof(index_t)));
  checkCudaErrors(cudaMalloc(
    &d_query_version_, batch_limit * sizeof(version_t)));
  checkCudaErrors(cudaMalloc(
    &d_query_updates_, batch_limit * sizeof(version_t)));
  checkCudaErrors(cudaMalloc(
    &d_query_val_, batch_limit * sizeof(embed_t) * kEmbeddingWidth));
}

void HetuGPUTable::freeAuxillaryMemory() {
  checkCudaErrors(cudaFree(d_temp_));
  checkCudaErrors(cudaFree(d_query_idx_));
  checkCudaErrors(cudaFree(d_query_gradient_idx_));
  checkCudaErrors(cudaFree(d_query_version_));
  checkCudaErrors(cudaFree(d_query_updates_));
  checkCudaErrors(cudaFree(d_query_val_));
}

HetuGPUTable::~HetuGPUTable() {
  checkCudaErrors(ncclCommDestroy(communicator_));
  checkCudaErrors(cudaStreamDestroy(stream_main_));
  checkCudaErrors(cudaStreamDestroy(stream_sub_));
  checkCudaErrors(cudaFree(d_embedding_));
  checkCudaErrors(cudaFree(d_version_));
  checkCudaErrors(cudaFree(d_gradient_));
  checkCudaErrors(cudaFree(d_updates_));
  checkCudaErrors(cudaFree(d_root_));
  freePreprocessData(cur_batch_);
  freePreprocessData(prev_batch_);
  freeAuxillaryMemory();
}
