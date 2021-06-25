import numpy as np
import scipy.sparse as sp
import time
import argparse
import os.path as osp

import hetuCTR_partition

def load_criteo_data():
    path = osp.dirname(__file__)
    fname = osp.join(path, "../.dataset/criteo/train_sparse_feats.npy")
    # fname = osp.join(path, "../.dataset/avazu/sparse.npy")
    fname = osp.join(osp.dirname(__file__), "small_data.npy")
    assert osp.exists(fname)
    data = np.load(fname)
    if not data.data.c_contiguous:
        data = np.ascontiguousarray(data)
    return data

def direct_partition(data, nparts, batch_size, rerun):
    start = time.time()
    partition = hetuCTR_partition.partition(data, nparts, batch_size)

    print(partition.cost_model())
    for i in range(rerun):
        partition.refine_data()
        partition.refine_embed()
        partition.print_balance()
        print("Refine %d:" % i, partition.cost_model())
    item_partition, idx_partition = partition.get_result()
    print("Partition Time : ", time.time()-start)
    start = time.time()

    arr_dict = {"embed_partition" : idx_partition, "data_partition" : item_partition}
    priority = partition.get_priority()
    for i in range(nparts):
        idxs = np.where(idx_partition==i)[0]
        priority[i][idxs] = -1 # remove embedding that has been stored
        arr = np.argsort(priority[i])[len(idxs):][ : : -1]
        arr_dict[str(i)] = arr
    print("Sort priority Time : ", time.time()-start)

    np.savez("partition_{}.npz".format(nparts), **arr_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrank", "-n" , type=int, default=8)
    parser.add_argument("--batch_size", "-b" , type=int, default=8192)
    parser.add_argument("--rerun", "-r" , type=int, default=10)
    args = parser.parse_args()
    data = load_criteo_data()
    direct_partition(data, args.nrank, args.batch_size, args.rerun)
