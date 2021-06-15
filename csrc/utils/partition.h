#pragma once

#include "core/types.h"
#include "pybind/pybind.h"

namespace hetuCTR {

struct PartitionStruct;

std::unique_ptr<PartitionStruct> partition(const py::array_t<int>& _input_data, int n_part);

void pybindPartition(py::module &m);

} // namespace hetuCTR
