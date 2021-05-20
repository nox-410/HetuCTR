#include "pybind.h"

#include "core/hetu_gpu_table.h"

using namespace hetu;

PYBIND11_MODULE(hetu_gpu_table, m) {
  m.doc() = "hetu GPU table C++/CUDA Implementation"; // module docstring

  py::class_<HetuGPUTable, std::unique_ptr<HetuGPUTable>>(m, "HetuGPUTable", py::module_local());

} // PYBIND11_MODULE
