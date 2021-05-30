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

def partition(data, nrank):
    idx_max = data.max()
    item_max = data.shape[0]
    id_per_item = data.shape[1]
    total = idx_max + item_max
    coo_0 = np.repeat(np.arange(idx_max, total), id_per_item)
    coo_1 = data.flatten()

    sp_mat = sp.coo_matrix((np.ones(len(coo_0)), (coo_0, coo_1)), shape=(total, total))
    sp_mat = sp_mat + sp_mat.transpose()
    sp_mat.tocsr()
    print(sp_mat.indptr, sp_mat.indices)
    print(sp_mat.indptr.dtype, sp_mat.indices.dtype)
    print(sp_mat.indptr.shape, sp_mat.indices.shape)
    print(total)
    result = hetuCTR_partition.partition(sp_mat.indptr, sp_mat.indices, idx_max, nrank)
    idx_partition = result[:idx_max]
    item_partition = result[idx_max:]
    np.save("embed_partition.npy", idx_partition)
    np.save("data_partition.npy", item_partition)


if __name__ == '__main__':
    data = load_criteo_data()
    partition(data, 8)

