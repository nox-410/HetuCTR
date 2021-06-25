#include "pybind/pybind.h"
#include "partition.h"

namespace hetuCTR {

PYBIND11_MODULE(hetuCTR_partition, m) {
  m.doc() = "hetuCTR graph partition C++ implementation"; // module docstring
  py::class_<PartitionStruct>(m, "_PartitionStruct", py::module_local())
    .def("refine_data", &PartitionStruct::refineData)
    .def("refine_embed", &PartitionStruct::refineEmbed)
    .def("print_balance", &PartitionStruct::printBalance)
    .def("cost_model", &PartitionStruct::costModel)
    .def("get_priority", &PartitionStruct::getPriority)
    .def("get_result", [](PartitionStruct &func) {
      return py::make_tuple(bind::vec_nocp(func.res_data_), bind::vec_nocp(func.res_embed_));
    });
  m.def("partition", partition);

} // PYBIND11_MODULE

} // namespace hetuCTR
