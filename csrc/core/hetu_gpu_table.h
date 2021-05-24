#pragma once

#include "types.h"
#include "preprocess.h"
#include "common/sarray.h"
#include "common/logging.h"
#include "utils/initializer.h"

#include <cuda_runtime.h>
#include <nccl.h>

#include "cudf/managed.cuh"
#include "cudf/concurrent_unordered_map.cuh"

namespace hetu {

/**
 * @brief Distributed GPU Table for embedding-based training
 *
 */
class HetuGPUTable : public managed {
public:
  const worker_t rank_;
  const worker_t nrank_;
  const worker_t device_id_;

  const size_t kEmbeddingIDMax;
  const size_t kEmbeddingWidth;
  const size_t kStorageMax;
  size_t kNonLocalStorageMax;

  // maxinum size of a batch fed,
  // when the table received a larger batch, it will have to reallocate memory
  size_t batch_size_reserved_ = 1;

  const version_t pull_bound_, push_bound_;

  cudaStream_t stream_main_, stream_sub_;
  ncclComm_t communicator_;

  embed_t * d_embedding_;
  embed_t * d_gradient_;
  version_t * d_updates_;
  version_t * d_version_;
  worker_t * d_root_;

  // temp memory used in some cuda-based algorithm
  void * d_temp_ = nullptr;
  size_t temp_bytes_ = 0;

  // query buffer, dual buffer for send and receive
  version_t * d_query_version_[2] = {};
  version_t * d_query_updates_[2] = {};
  index_t * d_query_idx_[2] = {};
  index_t * d_query_gradient_idx_[2] = {};
  embed_t * d_query_val_[2] = {};

  index_t * d_return_outdated_[2] = {};
  embed_t * d_return_val_[2] = {};
  version_t * d_return_version_[2] = {};

  PreprocessData cur_batch_, prev_batch_;
  concurrent_unordered_map<index_t, index_t, kInvalidIndex> *table_;

  int verbose_;
  /**
   * @brief Initialize cuda and nccl communicator
   *
   * @param ip IPv4 address to setup collective communication
   * @param port IPv4 port
   */
  void initializeNCCL(const std::string &ip, const int port);
  void initializeTable(SArray<worker_t> root_id_arr, SArray<index_t> storage_id_arr);
  void allocateAuxillaryMemory(size_t batch_size);
  void freeAuxillaryMemory();

  void generateQuery();
  void handleQuery();
  void writeBack(embed_t *dst);
  void all2allExchangeShape(const size_t *shape, size_t *shape_out);
  void all2allExchangeQuery();
  void all2allReturnOutdated();
  void all2allReturnValue();

  template <class T> int __printarg(T t) { std::cout << t; return 0; }
  template<class ...Args>
  inline void INFO(Args ...args) {
    if (verbose_ >= 1) {
      std::cout << "HetuGPUTable rank " << (int)rank_ << ": ";
      std::initializer_list<int>({__printarg(args)...});
      std::cout << std::endl;
    }
  }

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
  /**
   * @brief preprocess next batch index
   *
   * @param data_ptr an address holding index
   * @param len the length of index array
   */
  void preprocessIndex(unsigned long data_ptr, size_t batch_size);

  /**
   * @brief Update embedding Table with the gradients and then fetch embedding value to dst
   *
   * @param grad points to gradients array
   * @param dst where embedding are written to
   */
  void pushPull(unsigned long grad, unsigned long dst);
  std::string debugString();
  std::string debugStringFull();
  ~HetuGPUTable();
};

} // namespace hetu
