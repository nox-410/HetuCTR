#include "hetu_gpu_table.h"

using namespace hetuCTR;

std::string HetuGPUTable::debugString() {
  char buffer[1024];
  sprintf(buffer, "<HetuGPUTable root/store/all=%ld/%ld/%ld>",
    kStorageMax - kNonLocalStorageMax, kStorageMax, kEmbeddingIDMax);
  std::string str = buffer;
  return str;
}

std::string HetuGPUTable::debugStringFull() {
  char buffer[1024];
  sprintf(buffer,
"<HetuGPUTable root/store/all=%ld/%ld/%ld \
rank=%d nrank=%d device_id=%d \
width=%ld push/pull=%lld/%lld>",
    kStorageMax - kNonLocalStorageMax, kStorageMax, kEmbeddingIDMax,
    rank_, nrank_, device_id_,
    kEmbeddingWidth, push_bound_, pull_bound_);
  std::string str = buffer;
  return str;
}
