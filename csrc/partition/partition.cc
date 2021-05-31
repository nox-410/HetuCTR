#include <metis.h>
#include "../pybind/pybind.h"

static_assert(sizeof(idx_t) == 8, "metis 64bit version should be used");

namespace hetuCTR {

py::array_t<idx_t>
bipartiteGraphPartition(py::array_t<idx_t> _indptr, py::array_t<idx_t> _indices, idx_t offset, idx_t nparts) {
  PYTHON_CHECK_ARRAY(_indptr);
  PYTHON_CHECK_ARRAY(_indices);

  auto indptr = _indptr.mutable_data();
  auto indices = _indices.mutable_data();

  auto partition_function = METIS_PartGraphKway;
  if (nparts > 8) {
    partition_function = METIS_PartGraphRecursive;
  }

  idx_t num_nodes = _indptr.size() - 1;
  idx_t ncon = 2;
  std::vector<idx_t> vwgt(ncon * num_nodes);

  for (idx_t i = 0; i < num_nodes; i++) {
    vwgt[i * ncon] = (idx_t)(i < offset);
    vwgt[i * ncon + 1] = (idx_t)(i >= offset);
  }
  idx_t edge_cut;

  py::array_t<idx_t> partition_result(num_nodes);

  int info = partition_function(
    &num_nodes, /* number of nodes */
    &ncon,
    indptr,
    indices,
    vwgt.data(),    /* weight of nodes */
    NULL,    /* The size of the vertices for computing the total communication volume */
    NULL,    /* weight of edges */
    &nparts, /* num parts */
    NULL,    /* the desired weight for each partition and constraint */
    NULL,    /* an array of size ncon that specifies the allowed load imbalance tolerance for each constraint */
    NULL,    /* options */
    &edge_cut,  /* store number of edge cut */
    partition_result.mutable_data() /* store partition result */
  );
  switch (info) {
  case METIS_OK:
    break;
  case METIS_ERROR_INPUT:
    printf("Metis error input");
    break;
  case METIS_ERROR_MEMORY:
    printf("Metis error memory");
    break;
  case METIS_ERROR:
  default:
    printf("Metis error");
    break;
  }
  assert(info == METIS_OK);
  py::print("Edge Cut : ", edge_cut);

  return partition_result;
}

} // namespace hetuCTR

PYBIND11_MODULE(hetuCTR_partition, m) {
  m.doc() = "hetuCTR data partition C++ implementation"; // module docstring

  m.def("partition", hetuCTR::bipartiteGraphPartition);

} // PYBIND11_MODULE
