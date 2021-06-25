import numpy as np
import scipy.sparse as sp
import time
import argparse
import os.path as osp

def load_criteo_data():
    path = osp.dirname(__file__)
    fname = osp.join(path, "../.dataset/criteo/train_sparse_feats.npy")
    assert osp.exists(fname)
    data = np.load(fname)
    return data

def small(data, sz):
    idx_max = int(data.max()) + 1
    item_max = data.shape[0]
    id_per_item = data.shape[1]
    new_value = np.arange(idx_max)
    data = data[0:sz]
    idxs = np.unique(data)
    new_value[idxs] = np.arange(len(idxs))
    data = new_value[data]
    np.save("small_data.npy", data)


if __name__ == '__main__':
    data = load_criteo_data()
    small(data, 50000)

