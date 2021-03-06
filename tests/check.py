import hetuCTR
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
    grad = torch.rand([batch_size, width]).cuda()
    init = hetuCTR.Initializer(hetuCTR.InitType.Normal, 0 , 1)
    storage_arr = np.where(root_arr == rank)[0]
    table = hetuCTR.HetuTable(
        rank=rank, nrank=nrank, device_id=rank, ip=ip, port=port,
        pull_bound = 10, push_bound = 10, init=init, learning_rate=1,
        length = length, width = width,
        root_arr = root_arr, storage_arr = storage_arr, verbose=1
    )

    embed_id = np.random.randint(length, size=batch_size, dtype=np.int64)
    table.preprocess(embed_id.ctypes.data, 0)
    for i in range(5):
        embed_id = np.random.randint(length, size=batch_size, dtype=np.int64)
        table.preprocess(embed_id.ctypes.data, embed_id.shape[0])
        table.push_pull(grad.data_ptr(), dest.data_ptr())
        checkvalue(embed_id, dest.cpu())
        print(dest.cpu()[:, 0].sum())

if __name__ == '__main__':
    for i in range(nrank):
        proc = multiprocessing.Process(target=worker, args=[i])
        proc.start()
