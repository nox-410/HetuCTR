#pragma once

#include "types.h"
#include "common/sarray.h"
#include "common/logging.h"
#include "utils/initializer.h"

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

  embed_t * d_embedding_;
  embed_t * d_gradient_;
  version_t * d_updates_;
  version_t * d_version_;
  worker_t * d_root_;

  int verbose_;
  /**
   * @brief Initialize cuda and nccl communicator
   *
   * @param ip IPv4 address to setup collective communication
   * @param port IPv4 port
   */
  void initializeNCCL(const std::string &ip, const int port);
  void initializeTable(SArray<worker_t> root_id_arr, SArray<index_t> storage_id_arr);

  template <class T> int __printarg(T t) { std::cout << t; return 0; }
  template<class ...Args>
  inline void INFO(Args ...args) {
    if (verbose_ >= 1) {
      std::cout << "HetuGPUTable rank " << (int)rank_ << ": ";
      std::initializer_list<int>({__printarg(args)...});
      std::cout << std::endl;
    }
  }

public:
  HetuGPUTable(
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
  );
  HetuGPUTable(const HetuGPUTable &) = delete;
  HetuGPUTable& operator=(const HetuGPUTable&) = delete;
  std::string debugString();
  std::string debugStringFull();
  ~HetuGPUTable();
};

} // namespace hetu
