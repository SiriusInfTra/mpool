from __future__ import annotations
from threading import local
from ._C import *
# from . import cupy_integrate
# from . import numba_integrate

import os


def override_torch_allocator(name: str, nbytes_per_device: int):
    import torch
    assert torch.cuda.is_available()
    caching_allocator_list = []
    username = os.getlogin()
    process_id = os.getpid()
    for device_id in range(torch.cuda.device_count()):
        page_pool_config = C_PagesPoolConf(
            device_id = device_id,
            log_prefix= '[PagesPool] ',
            page_nbytes = 32 * 1024 * 1024, 
            pool_nbytes = nbytes_per_device,
            shm_name = f'{name}_page_pool_{device_id}',
            shm_nbytes = 16 * 1024 * 1024 # 16MB
        )
        page_pool = C_PagesPool(page_pool_config)
        caching_allocator_config = C_CachingAllocatorConfig(
            align_nbytes = 512,
            belong_name = f'torch',
            log_prefix = '[CachingAllocator] ',
            shm_name = f'{name}_caching_allocator_{device_id}',
            shm_nbytes = 128 * 1024 * 1024, # 128MB
            small_block_nbytes = 1024,
            va_range_scale = 1
        )
        caching_allocator = C_CachingAllocator(page_pool, caching_allocator_config)
        caching_allocator_list.append(caching_allocator)
    _C._override_torch_allocator(caching_allocator_list)
    def share_cuda(self):
        return _C._share_cuda(self)
    def new_shared_cuda(*args):
        return _C._new_shared_cuda(None, *args)
    torch.UntypedStorage._share_cuda_ = share_cuda
    torch.UntypedStorage._new_shared_cuda = new_shared_cuda


