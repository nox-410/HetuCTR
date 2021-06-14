import numpy as np
import scipy.sparse as sp
import time
import argparse
import os.path as osp

import hetuCTR

def load_criteo_data():
    path = osp.dirname(__file__)
    fname = osp.join(path, "../.dataset/criteo/train_sparse_feats.npy")
    # fname = "small_data.npy"
    assert osp.exists(fname)
    data = np.load(fname)
    return data

def direct_partition(data, nparts):
    start = time.time()
    item_partition, idx_partition = hetuCTR.partition(data, nparts)
    end = time.time()
    print("Time : ", end-start)

    np.save("embed_partition.npy", idx_partition)
    np.save("data_partition.npy", item_partition)


if __name__ == '__main__':
    data = load_criteo_data()
    direct_partition(data, 8)

