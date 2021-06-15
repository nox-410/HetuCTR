import numpy as np
import scipy.sparse as sp
import time
import argparse
import os.path as osp

import hetuCTR

def load_criteo_data():
    path = osp.dirname(__file__)
    # fname = osp.join(path, "../.dataset/criteo/train_sparse_feats.npy")
    # fname = osp.join(path, "../.dataset/avazu/sparse.npy")
    fname = "small_data.npy"
    assert osp.exists(fname)
    data = np.load(fname)
    if not data.data.c_contiguous:
        data = np.ascontiguousarray(data)
    return data

def direct_partition(data, nparts):
    start = time.time()
    partition = hetuCTR.partition(data, nparts)

    print(partition.cost_model())
    for i in range(10):
        partition.refine_data()
        partition.refine_embed()
        partition.print_balance()
        print("Refine %d:" % i, partition.cost_model())
    item_partition, idx_partition = partition.get_result()
    end = time.time()
    print("Time : ", end-start)

    np.save("embed_partition.npy", idx_partition)
    np.save("data_partition.npy", item_partition)


if __name__ == '__main__':
    data = load_criteo_data()
    direct_partition(data, 8)

