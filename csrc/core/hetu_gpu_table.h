#pragma once

#include "types.h"
#include "common/sarray.h"
#include "common/logging.h"

#include <cuda_runtime.h>
#include <nccl.h>

#include "hash_table/nv_hashtable.hpp"

using HugeCTR::HashTable;

namespace hetu {

/**
 * @brief Distributed GPU Table for embedding-based training
 *
 */
class HetuGPUTable {
private:
  const worker_t rank_;
  const worker_t nrank_;
  const worker_t device_id_;

  const size_t kEmbeddingIDMax;
  const size_t kEmbeddingWidth;
  const size_t kStorageMax;
  size_t kNonLocalStorageMax;

  const version_t pull_bound_, push_bound_;

  cudaStream_t stream_main_, stream_sub_;
  ncclComm_t communicator_;

  HashTable<index_t, index_t> hash_table_;

  /**
   * @brief Initialize cuda and nccl communicator
   *
   * @param ip IPv4 address to setup collective communication
   * @param port IPv4 port
   */
  void initializeNCCL(const std::string &ip, const int port);
public:
  HetuGPUTable(
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
  );
  HetuGPUTable(const HetuGPUTable &) = delete;
  HetuGPUTable& operator=(const HetuGPUTable&) = delete;
  ~HetuGPUTable();
};

} // namespace hetu
