import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet152
from torchvision.datasets import FakeData
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import mpool



page_conf = mpool.C_PagesPoolConf(
    page_nbytes=32 * 1024 * 1024,
    pool_nbytes=12 * 1024 * 1024 * 1024,
    shm_name='test',
    log_prefix='mpool ',
    shm_nbytes=1 * 1024 * 1024 * 1024,
)
page_pool = mpool.C_PagesPool(page_conf)
belong = page_pool.get_belong('test')
lock = page_pool.lock()
pages = page_pool.alloc_dis_pages(belong, 10, lock)
page_pool.free_pages(pages, belong, lock)
print(pages)
del lock


caching_allocator_conf = mpool.C_CachingAllocatorConfig(
    align_nbytes=512,
    belong_name='test',
    log_prefix='',
    shm_name='test_caching_allocator',
    shm_nbytes=1 * 1024 * 1024 * 1024,
    small_block_nbytes=1024,
    va_range_scale=1,
)
caching_allocator = mpool.C_CachingAllocator(page_pool, caching_allocator_conf)
# block = caching_allocator.Alloc(1024, 0)
# caching_allocator.Free(block)
mpool.override_pytorch_allocator(caching_allocator)
mpool.override_ipc(caching_allocator)

try:
    x = torch.zeros(1).cuda()
    print(x)
except Exception as e:
    print(e)
print("CUDA is available")

if __name__ == "__main__":
    import torch
import torch.multiprocessing as mp
import time

def tensor_operations(rank, size, return_dict):
    """
    每个进程执行的Tensor运算函数
    """
    torch.manual_seed(rank)
    start_time = time.time()

    # # 创建随机张量
    # a = torch.randn(1000, 1000).cuda()
    # b = torch.randn(1000, 1000).cuda()

    # # 执行一些张量运算
    # for _ in range(100):
    #     c = torch.mm(a, b)
    #     d = c.sum()

    end_time = time.time()
    elapsed_time = end_time - start_time

    # # 将结果存储在共享字典中
    # return_dict[rank] = elapsed_time
    print(f"进程 {rank} 完成，耗时 {elapsed_time:.4f} 秒")

def main():
    # 设置进程数
    num_processes = 4

    # 创建一个共享字典来存储每个进程的结果
    manager = mp.Manager()
    return_dict = manager.dict()

    # 创建进程
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=tensor_operations, args=(rank, num_processes, return_dict))
        p.start()
        processes.append(p)

    # 等待所有进程完成
    for p in processes:
        p.join()

    # 打印所有进程的运算时间
    for rank in range(num_processes):
        print(f"进程 {rank} 的运算时间: {return_dict[rank]:.4f} 秒")

if __name__ == "__main__":
    # 需要在Windows上运行多进程代码时添加这行
    mp.set_start_method('spawn')
    main()

