import hetuCTR
import numpy as np
import multiprocessing
import argparse
import time

nrank = 4
ip = "127.0.0.1"
port = 23456
width = 128
length = 1024
batch_size = 1024
np.random.seed(0)
root_arr = np.random.randint(nrank, size=length)

def worker(rank):
    import torch
    torch.cuda.set_device(rank)
    dest = torch.zeros([batch_size, width]).cuda()
    grad = torch.zeros([batch_size, width]).cuda()
    init = hetuCTR.Initializer(hetuCTR.InitType.Normal, 0 , 1)
    storage_arr = np.where(root_arr == rank)[0]
    table = hetuCTR.HetuTable(
        rank=rank, nrank=nrank, device_id=rank, ip=ip, port=port,
        pull_bound = 0, push_bound = 0, init=init,
        length = length, width = width,
        root_arr = root_arr, storage_arr = storage_arr, verbose=1
    )
    print(root_arr[:10])
    embed_id = np.array(range(10), dtype=np.int64)
    table.preprocess(embed_id.ctypes.data, embed_id.shape[0])
    table.preprocess(embed_id.ctypes.data, embed_id.shape[0])
    table.push_pull(grad.data_ptr(), dest.data_ptr())
    start = time.time()
    for i in range(1000):
        embed_id = np.random.randint(length, size=batch_size, dtype=np.int64)
        table.preprocess(embed_id.ctypes.data, embed_id.shape[0])
        table.push_pull(grad.data_ptr(), dest.data_ptr())
    print(time.time()-start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", default=-1, type=int)
    args = parser.parse_args()
    i = args.i
    if i == -1:
        for i in range(nrank):
            proc = multiprocessing.Process(target=worker, args=[i])
            proc.start()
    else:
        worker(i)
