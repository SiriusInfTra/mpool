import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet152
from torchvision.datasets import FakeData
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import mpool


page_conf = mpool.PagesPoolConf(
    page_nbytes=32 * 1024 * 1024,
    pool_nbytes=12 * 1024 * 1024 * 1024,
    shm_name='test',
    log_prefix='mpool ',
    shm_nbytes=1 * 1024 * 1024 * 1024,
)
page_pool = mpool.PagesPool(page_conf)
belong = page_pool.get_belong('test')
with page_pool.with_lock() as lock:
    pages = page_pool.alloc_dis_pages(belong, 10, lock)
    page_pool.free_pages(pages, belong, lock)
    print(pages)


caching_allocator_conf = mpool.CachingAllocatorConfig(
    align_nbytes=512,
    belong_name='test',
    log_prefix='',
    shm_name='test_caching_allocator',
    shm_nbytes=1 * 1024 * 1024 * 1024,
    small_block_nbytes=1024,
    va_range_scale=1,
)
caching_allocator = mpool.CachingAllocator(page_pool, caching_allocator_conf)
# block = caching_allocator.Alloc(1024, 0)
# caching_allocator.Free(block)

caching_allocator.RegisterAsPyTorchAllocator()


try:
    torch.zeros(1).cuda()
except Exception as e:
    print(e)

print("CUDA is available")