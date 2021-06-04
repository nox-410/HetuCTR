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

def direct_partition(data, nparts):
    idx_max = int(data.max()) + 1 # 33762577
    item_max = data.shape[0] # 41256556
    id_per_item = data.shape[1] # 26

    data_partition = np.random.randint(0, nparts, item_max)
    # counts = np.zeros(shape=[nparts, idx_max])
    counts = np.random.rand(nparts, idx_max)
    for i in range(nparts):
        print(i)
        data_part = data[np.where(data_partition==i)]
        (idxs, cnts) = np.unique(data_part, return_counts=True)
        counts[i][idxs] += cnts
    argmax = np.argmax(counts, axis=0)
    np.save("data_partition.npy", data_partition)
    np.save("embedding_partition.npy", argmax)




if __name__ == '__main__':
    data = load_criteo_data()
    direct_partition(data, 8)

