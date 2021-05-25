import hetu_gpu_table
import numpy as np
import multiprocessing

nrank = 4
ip = "127.0.0.1"
port = 23456
width = 128
length = 1024
batch_size = 1024
np.random.seed(0)
root_arr = np.random.randint(nrank, size=length)

def checkvalue(k, v):
    n = len(k)
    v = v[:n, 0]
    kv = {}
    for i in range(n):
        if k[i] in kv.keys():
            assert kv[k[i]] == v[i]
        else:
            assert v[i] not in kv.values()
            kv[k[i]] = v[i]

def worker(rank):
    import torch
    torch.cuda.set_device(rank)
    dest = torch.zeros([batch_size, width]).cuda()
    init = hetu_gpu_table.Initializer(hetu_gpu_table.InitType.Normal, 0 , 1)
    storage_arr = np.where(root_arr <= rank)[0]
    table = hetu_gpu_table.HetuGPUTable(
        rank=rank, nrank=nrank, device_id=rank, ip=ip, port=port,
        pull_bound = 10, push_bound = 10, init=init,
        length = length, width = width,
        root_arr = root_arr, storage_arr = storage_arr, verbose=1
    )
    for i in range(10):
        embed_id = np.random.randint(length, size=batch_size, dtype=np.int64)
        table.preprocess(embed_id.ctypes.data, embed_id.shape[0])
        table.push_pull(0, dest.data_ptr())
        checkvalue(embed_id, dest.cpu())

if __name__ == '__main__':
    for i in range(nrank):
        proc = multiprocessing.Process(target=worker, args=[i])
        proc.start()
