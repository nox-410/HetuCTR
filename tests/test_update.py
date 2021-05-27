import hetuCTR
import numpy as np
import multiprocessing

import torch
import torch.distributed as dist

nrank = 4
ip = "127.0.0.1"
port = 23456
width = 128
length = 1024
batch_size = 1024
np.random.seed(0)
root_arr = np.random.randint(nrank, size=length)

def test_one(rank):
    dest = torch.zeros([batch_size, width]).cuda()
    grad = torch.ones([batch_size, width]).cuda()
    init = hetuCTR.Initializer(hetuCTR.InitType.Zero, 0 , 0)
    storage_arr = np.where(root_arr <= rank)[0]
    table = hetuCTR.HetuTable(
        rank=rank, nrank=nrank, device_id=rank, ip=ip, port=port,
        pull_bound = 0, push_bound = 0, init=init,
        length = length, width = width,
        root_arr = root_arr, storage_arr = storage_arr, verbose=0
    )
    embed_id = np.random.randint(length, size=0, dtype=np.int64)
    table.preprocess(embed_id.ctypes.data, 0)
    update_store = np.zeros(length)
    np.random.seed(rank)
    for i in range(50):
        for id in embed_id:
            update_store[id]+=1.0
        embed_id = np.random.randint(length, size=batch_size, dtype=np.int64)
        table.preprocess(embed_id.ctypes.data, embed_id.shape[0])
        table.push_pull(grad.data_ptr(), dest.data_ptr())

    update_store = torch.Tensor(update_store).cuda()
    dist.all_reduce(update_store)
    val = dest[:len(embed_id), 0]
    # print(update_store[embed_id], val)
    assert torch.all(val == update_store[embed_id])

def test_two(rank):
    dest = torch.zeros([batch_size, width]).cuda()
    init = hetuCTR.Initializer(hetuCTR.InitType.Zero, 0 , 0)
    storage_arr = np.where(root_arr <= rank)[0]
    table = hetuCTR.HetuTable(
        rank=rank, nrank=nrank, device_id=rank, ip=ip, port=port,
        pull_bound = 0, push_bound = 0, init=init,
        length = length, width = width,
        root_arr = root_arr, storage_arr = storage_arr, verbose=0
    )
    embed_id = np.random.randint(length, size=0, dtype=np.int64)
    table.preprocess(embed_id.ctypes.data, 0)
    update_store = np.zeros(length)
    np.random.seed(rank)
    for i in range(50):
        grad_cpu = torch.rand([batch_size, width])
        grad = grad_cpu.cuda()
        for grad_id, id in enumerate(embed_id):
            update_store[id]+=grad_cpu[grad_id, 0]
        embed_id = np.random.randint(length, size=batch_size, dtype=np.int64)
        table.preprocess(embed_id.ctypes.data, embed_id.shape[0])
        table.push_pull(grad.data_ptr(), dest.data_ptr())

    update_store = torch.Tensor(update_store).cuda()
    dist.all_reduce(update_store)
    val = dest[:len(embed_id), 0]
    assert( torch.mean(torch.abs(val - update_store[embed_id])) < 1e-4 )

def worker(rank):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:12323", world_size=nrank, rank=rank)

    test_one(rank)
    if rank == 0:
        print("Update test 1 passed")

    test_two(rank)
    if rank == 0:
        print("Update test 2 passed")

if __name__ == '__main__':
    for i in range(nrank):
        proc = multiprocessing.Process(target=worker, args=[i])
        proc.start()
