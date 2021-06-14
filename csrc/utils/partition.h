#pragma once

#include "core/types.h"
#include "pybind/pybind.h"

namespace hetuCTR {

py::tuple partition(const py::array_t<int>& _input_data, int n_part);

} // namespace hetuCTR
