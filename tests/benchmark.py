import hetuCTR
import numpy as np
import multiprocessing
import argparse
import time
import os.path as osp
import torch
import torch.distributed as dist

nrank = 8
length = 33762577

ip = "127.0.0.1"
port = 23456
np.random.seed(0)
root_arr = np.random.randint(nrank, size=length)

# data_id = np.load("data_partition.npy")

def get_sample_data(data, bs):
   limit = data.shape[0]
   idx = np.random.randint(limit, size=bs)
   sampled_data = data[idx]
   return sampled_data.flatten()

def load_criteo_data():
    path = osp.dirname(__file__)
    fname = osp.join(path, "../.dataset/criteo/train_sparse_feats.npy")
    assert osp.exists(fname)
    data = np.load(fname).astype(np.int64)
    return data

def torch_sync_data(*args):
    # all-reduce train stats
    t = torch.tensor(args, dtype=torch.float64, device='cuda')
    dist.barrier()  # synchronizes all processes
    dist.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return t

def worker(rank, batch_size, width):
    # global data
    # data = data[np.where(data_id==rank)]
    item_size = batch_size * 26
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:12323", world_size=nrank, rank=rank)
    dest = torch.zeros([item_size, width]).cuda()
    grad = torch.rand([item_size, width]).cuda()
    init = hetuCTR.Initializer(hetuCTR.InitType.Normal, 0 , 0.1)
    storage_arr = np.where(root_arr == rank)[0]
    table = hetuCTR.HetuTable(
        rank=rank, nrank=nrank, device_id=rank, ip=ip, port=port,
        pull_bound = 0, push_bound = 0, init=init,
        length = length, width = width,
        root_arr = root_arr, storage_arr = storage_arr, verbose=0
    )
    np.random.seed(rank)
    time_total = 0
    time_preprocess = 0
    for i in range(1000):
        embed_id = get_sample_data(data, batch_size)
        dist.barrier()
        start = time.time()
        table.preprocess(embed_id.ctypes.data, embed_id.shape[0])
        time_preprocess += time.time() - start
        table.push_pull(grad.data_ptr(), dest.data_ptr())
        time_total += time.time() - start
    time_preprocess, time_total = torch_sync_data(time_preprocess, time_total)
    time_preprocess, time_total = time_preprocess / nrank, time_total / nrank
    if rank==0:
        print("batch_size {} : embed_dim : {} time : {:.3f} ({:.3f})".format(batch_size, width, time_total, time_preprocess))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", default=-1, type=int)
    parser.add_argument("--batch_size", default=2048, type=int)
    parser.add_argument("--embed_dim", default=128, type=int)
    args = parser.parse_args()
    data = load_criteo_data()
    i = args.i
    if i == -1:
        for i in range(nrank):
            proc = multiprocessing.Process(target=worker, args=[i, args.batch_size, args.embed_dim])
            proc.start()
    else:
        worker(i, args.batch_size, args.embed_dim)
