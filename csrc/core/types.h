#pragma once

#include <limits>

extern "C" {
#include <sys/types.h>
}

namespace hetu {

/// types of the embedding item
typedef float embed_t;

/// types of version number used in hetu
/// must be signed
typedef long long version_t;

/// types used in memory offset and feature input
typedef long long index_t;

/// Default invalid index usd in hash map
const index_t kInvalidIndex = std::numeric_limits<index_t>::max();

/// This version is smaller than any other version, so it will be regarded as outdated
const version_t kInvalidVersion = std::numeric_limits<version_t>::min();

/// types of worker id
typedef unsigned char worker_t;

} // namespace hetu
