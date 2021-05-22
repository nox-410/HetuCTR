#pragma once

#include "types.h"
#include <vector>

namespace hetu {

struct PreprocessData {
  size_t batch_size = 0;
  size_t allocate_size = 0;
  size_t unique_size = 0;
  index_t *d_idx = nullptr;
  index_t *d_unique_idx = nullptr;
  index_t *d_idx_map = nullptr;
  worker_t *d_root = nullptr;
  index_t *d_offset = nullptr;
  std::vector<size_t> embed_root_shape;
};

void createPreprocessData(PreprocessData &pdata, size_t batch_size, size_t nrank);

void freePreprocessData(PreprocessData &pdata);

} // namespace hetu
