import torch
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
    data = torch.Tensor(data).cuda()
    idx_max = int(data.max()) + 1
    item_max = data.shape[0]

    embed_partition = torch.randint(nrank, size=[idx_max]).cuda()
    data_partition = torch.randint(nrank, size=[item_max]).cuda()

    total_frequency = torch.zeros(idx_max).cuda()
    (idxs, cnts) = torch.unique(data, return_counts=True)
    total_frequency[idxs.to(torch.long)] = cnts.to(torch.float)
    total_frequency += 1e-3 # avoid zero devision

    i = 3
    data_part = data[torch.where(data_partition==i)]
    part_frequency = torch.zeros(idx_max).cuda()
    (idxs, cnts) = torch.unique(data_part, return_counts=True)
    part_frequency[idxs.to(torch.long)] = cnts.to(torch.float)
    other_frequency = total_frequency - part_frequency
    priority = torch.pow(part_frequency, 2) / other_frequency
    sorted_embed = torch.argsort(priority, descending=True)
    print(sorted_embed)
    np.save("a.npy", sorted_embed.cpu().numpy())
    np.save("f.npy", total_frequency.cpu().numpy())
    np.save("p.npy", part_frequency.cpu().numpy())
