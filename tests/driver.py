import hetu_gpu_table
import numpy as np
import multiprocessing

nrank = 4
ip = "127.0.0.1"
port = 23456
width = 128
length = 1000
root_arr = np.random.randint(nrank, size=1000)

def worker(rank):
    init = hetu_gpu_table.Initializer(hetu_gpu_table.InitType.Normal, 0 , 0.1)
    storage_arr = np.where(root_arr == rank)[0]
    storage_arr = np.where(root_arr <= rank)[0]
    table = hetu_gpu_table.HetuGPUTable(
        rank=rank, nrank=nrank, device_id=rank, ip=ip, port=port,
        pull_bound = 10, push_bound = 10, init=init,
        length = length, width = width,
        root_arr = root_arr, storage_arr = storage_arr, verbose=1
    )
    embed_id = np.array([1,2, 3, 1, 2], dtype=np.int64)
    table.preprocess(embed_id.ctypes.data, embed_id.shape[0])
    print(table.debug())


if __name__ == '__main__':
    for i in range(nrank):
        proc = multiprocessing.Process(target=worker, args=[i])
        proc.start()
