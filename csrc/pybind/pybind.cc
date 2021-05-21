#include "pybind.h"

#include "core/hetu_gpu_table.h"
#include "utils/initializer.h"

using namespace hetu;

static std::unique_ptr<HetuGPUTable> makeHetuGPUTable(
  const worker_t rank,
  const worker_t nrank,
  const worker_t device_id,
  const std::string &ip,
  const int port,
  const size_t embedding_length,
  const size_t embedding_width,
  const version_t pull_bound,
  const version_t push_bound,
  py::array_t<worker_t> root_id_arr,
  py::array_t<index_t> storage_id_arr,
  const Initializer &init)
{
  SArray<worker_t> root_id_arr_shared(root_id_arr.mutable_data(), root_id_arr.size());
  SArray<index_t> storage_id_arr_shared(storage_id_arr.mutable_data(), storage_id_arr.size());
  return std::make_unique<HetuGPUTable>(
    rank, nrank, device_id,
    ip, port,
    embedding_length, embedding_width,
    pull_bound, push_bound,
    root_id_arr_shared, storage_id_arr_shared,
    init);
}

static std::unique_ptr<Initializer> makeInitializer(InitType type, float param_a, float param_b) {
  return std::make_unique<Initializer>(type, param_a, param_b);
}

PYBIND11_MODULE(hetu_gpu_table, m) {
  m.doc() = "hetu GPU table C++/CUDA Implementation"; // module docstring

  py::enum_<InitType>(m, "InitType", py::module_local())
    .value("Zero", InitType::kZero)
    .value("Normal", InitType::kNormal)
    .value("TruncatedNormal", InitType::kTruncatedNormal)
    .value("Uniform", InitType::kUniform);

  py::class_<Initializer, std::unique_ptr<Initializer>>(m, "Initializer", py::module_local())
    .def(py::init(&makeInitializer));

  py::class_<HetuGPUTable, std::unique_ptr<HetuGPUTable>>(m, "HetuGPUTable", py::module_local())
    .def(py::init(&makeHetuGPUTable),
      py::arg("rank"), py::arg("nrank"), py::arg("device_id"),
      py::arg("ip"), py::arg("port"),
      py::arg("length"), py::arg("width"),
      py::arg("pull_bound"), py::arg("push_bound"),
      py::arg("root_arr"), py::arg("storage_arr"),
      py::arg("init"));

} // PYBIND11_MODULE
