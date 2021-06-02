#include <parmetis.h>
#include <mpi.h>
#include <iostream>
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

  auto ub_vec = std::vector<real_t>(ncon, 1.05);

  int info = partition_function(
    &num_nodes, /* number of nodes */
    &ncon,
    indptr,
    indices,
    vwgt.data(),    /* weight of nodes */
    nullptr,    /* The size of the vertices for computing the total communication volume */
    nullptr,    /* weight of edges */
    &nparts, /* num parts */
    nullptr,    /* the desired weight for each partition and constraint */
    ub_vec.data(),    /* an array of size ncon that specifies the allowed load imbalance tolerance for each constraint */
    nullptr,    /* options */
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

#define MPICHECK(cmd) do{                         \
      int e=cmd;                                  \
      if(e!= MPI_SUCCESS) {                       \
      printf("Failed: MPI error %s:%d '%d'\n",    \
      __FILE__,__LINE__, e);                      \
      exit(1);                                    \
      }                                           \
}while(0)

py::tuple mpi_init() {
  MPICHECK(MPI_Init(NULL, NULL));
  int rank, nrank;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nrank));
  return py::make_tuple(rank, nrank);
}

void mpi_finalize() {
  MPICHECK(MPI_Finalize());
}

py::array_t<idx_t>
parallel_partition(py::array_t<idx_t> _dist, py::array_t<idx_t> _indptr, py::array_t<idx_t> _indices,
  idx_t offset, idx_t nparts) {
  PYTHON_CHECK_ARRAY(_indptr);
  PYTHON_CHECK_ARRAY(_indices);

  auto indptr = _indptr.mutable_data();
  auto indices = _indices.mutable_data();
  auto dist = _dist.mutable_data();

  int rank;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  idx_t worker_offset = dist[rank];

  idx_t num_nodes = _indptr.size() - 1;
  idx_t ncon = 2;
  std::vector<idx_t> vwgt(ncon * num_nodes);

  for (idx_t i = 0; i < num_nodes; i++) {
    vwgt[i * ncon] = (idx_t)(worker_offset + i < offset);
    vwgt[i * ncon + 1] = (idx_t)(worker_offset + i >= offset);
  }
  idx_t edge_cut;

  py::array_t<idx_t> partition_result(num_nodes);

  idx_t wgtflag = 2, numflag = 0;

  MPI_Comm comm = MPI_COMM_WORLD;

  auto ub_vec = std::vector<real_t>(ncon, 1.05);

  auto tpwgts = std::vector<real_t>(ncon * nparts, 1.0f / nparts);

  auto options = std::vector<idx_t>(1, 0);

  int info = ParMETIS_V3_PartKway(
    dist,         /* distribution of nodes */
    indptr,
    indices,
    vwgt.data(),  /* weight of nodes */
    nullptr,      /* weight of edges */
    &wgtflag,
    &numflag,
    &ncon,
    &nparts,     /* num parts */
    tpwgts.data(),     /* the desired weight for each partition and constraint */
    ub_vec.data(),     /* an array of size ncon that specifies the allowed load imbalance tolerance for each constraint */
    options.data(),     /* options */
    &edge_cut,   /* store number of edge cut */
    partition_result.mutable_data(),  /* store partition result */
    &comm
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
  m.def("parallel_partition", hetuCTR::parallel_partition);
  m.def("init", hetuCTR::mpi_init);
  m.def("finalize", hetuCTR::mpi_finalize);

} // PYBIND11_MODULE
