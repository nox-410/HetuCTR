import numpy as np
import scipy.sparse as sp
import time
import argparse
import os.path as osp
import hetuCTR_partition

def run_parts(rank, nparts):
    idx_max = 33762577
    item_max = 41256556
    id_per_item = 26
    total = idx_max + item_max

    data = np.load("data/input_part{}.npz".format(rank))
    indptr = data["indptr"]
    indices = data["indices"]
    dist = data["dist"]

    # dist = np.arange(nrank + 1)
    # indptr = [0, nrank - 1]
    # indices = list(range(rank)) + list(range(rank+1, nrank))
    # indices = np.array(indices)
    # nparts = 8
    # idx_max = 16

    result = hetuCTR_partition.parallel_partition(dist, indptr, indices, idx_max, nparts)

    np.save("data/output_part{}.npy".format(rank), result)

if __name__ == '__main__':
    rank, nrank = hetuCTR_partition.init()
    print(rank, nrank)
    run_parts(rank, 8)
    hetuCTR_partition.finalize()

