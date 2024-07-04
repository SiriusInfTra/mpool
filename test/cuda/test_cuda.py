import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet152
from torchvision.datasets import FakeData
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import mpool


def use_shared_tensor():
    page_conf = mpool.C_PagesPoolConf(
        page_nbytes=32 * 1024 * 1024,
        pool_nbytes=12 * 1024 * 1024 * 1024,
        shm_name='test',
        log_prefix='mpool ',
        shm_nbytes=1 * 1024 * 1024 * 1024,
    )
    page_pool = mpool.C_PagesPool(page_conf)
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
    block = caching_allocator.alloc(1024, 0, True)
    caching_allocator.free(block)
    mpool.override_pytorch_allocator(caching_allocator)
    mpool.override_ipc(caching_allocator)
    del caching_allocator
    del page_pool

def reset_shared_tensor():
    mpool.reset_pytorch_allocator()

use_shared_tensor()


try:
    torch.zeros(1).cuda()
except Exception as e:
    print(e)

print("CUDA is available")