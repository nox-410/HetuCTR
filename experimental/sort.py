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

if __name__=='__main__':
    nrank = 8
    data = load_criteo_data()
    idx_max = int(data.max()) + 1
    item_max = data.shape[0]

    embed_partition = np.random.randint(nrank, size=[idx_max])
    data_partition =  np.random.randint(nrank, size=[item_max])

    total_frequency = np.zeros(idx_max)
    (idxs, cnts) = np.unique(data, return_counts=True)
    total_frequency[idxs] = cnts

    i = 3
    data_part = data[np.where(data_partition==i)]
    part_frequency = np.zeros(idx_max)
    (idxs, cnts) = np.unique(data_part, return_counts=True)
    part_frequency[idxs] = cnts
    other_frequency = total_frequency - part_frequency
    other_frequency[torch.where(other_frequency) == 0] = 1e-3 # avoid zero devision
    priority = np.pow(part_frequency, 2) / other_frequency
    sorted_embed = np.argsort(priority)
    print(sorted_embed)
