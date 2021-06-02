import numpy as np
import scipy.sparse as sp
import time
import argparse
import os.path as osp
import hetuCTR_partition

def load_criteo_data():
    path = osp.dirname(__file__)
    fname = osp.join(path, "../.dataset/criteo/train_sparse_feats.npy")
    assert osp.exists(fname)
    data = np.load(fname)
    return data

def nproc_part(data, nproc):
    idx_max = int(data.max()) + 1 # 33762577
    item_max = data.shape[0] # 41256556
    id_per_item = data.shape[1] # 26
    total = idx_max + item_max
    coo_0 = np.repeat(np.arange(idx_max, total), id_per_item)
    coo_1 = data.flatten()

    coo_0, coo_1 = np.concatenate([coo_0, coo_1]), np.concatenate([coo_1, coo_0])

    sp_mat = sp.coo_matrix((np.ones(len(coo_0)), (coo_0, coo_1)), shape=(total, total))
    sp_mat = sp_mat.tocsr()

    dist = np.linspace(0, total, nproc + 1, dtype=np.int32)

    for i in range(nproc):
        indptr = sp_mat.indptr[dist[i] : dist[i + 1] + 1]
        indices = sp_mat.indices[indptr[0] : indptr[-1]]
        indptr = indptr - indptr[0]
        assert(len(indices) == indptr[-1])
        np.savez(file="./data/input_part{}.npz".format(i), indptr=indptr, indices=indices, dist=dist)

if __name__ == '__main__':
    data = load_criteo_data()
    nproc_part(data, 32)

