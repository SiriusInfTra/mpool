from __future__ import annotations
import contextlib
from threading import local
import contextvars
import torch
from ._C import *

import dataclasses

@dataclasses.dataclass
class PagesPoolConf:
    page_nbytes: int
    pool_nbytes: int
    shm_name: str
    log_prefix: str
    shm_nbytes: int

class PagesPoolLock:
    def __init__(self, pages_pool: PagesPool):
        self._pages_pool = pages_pool
    
    def __enter__(self) -> PagesPoolLock:
        self._lock = self._pages_pool._lock()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del self._lock
  
class PagesPool:
    def __init__(self, conf: PagesPoolConf):
        self._conf = C_PagesPoolConf(conf.page_nbytes, conf.pool_nbytes, conf.shm_name, conf.log_prefix, conf.shm_nbytes)
        self._page_pool = C_PagesPool(self._conf)

    def with_lock(self):
        return PagesPoolLock(self)
        
    def _lock(self) -> C_Lock:
        return self._page_pool.Lock()
    
    def get_belong(self, name: str) -> Belong:
        return Belong(self._page_pool.GetBelong(name))
    
    def alloc_dis_pages(self, belong: Belong, n: int, lock: PagesPoolLock) -> list[int]:
        return self._page_pool.AllocDisPages(belong._belong, n, lock._lock)
        
    def alloc_con_pages(self, belong: Belong, n: int, lock: PagesPoolLock) -> range:
        begin: int = self._page_pool.AllocConPages(belong._belong, n, lock._lock)
        if begin != C_PagesPool.INSUFFICIENT_PAGE:
            return range(begin, begin + n)
        else:
            return range(0, 0)
    
    def free_pages(self, pages: list[int], belong: Belong, lock: PagesPoolLock):
        self._page_pool.FreePages(pages, belong._belong, lock._lock)
        

class Belong:
    def __init__(self, belong: C_Belong):
        self._belong = belong

@dataclasses.dataclass
class CachingAllocatorConfig:
    log_prefix: str
    shm_name: str
    shm_nbytes: int
    va_range_scale: int
    belong_name: str
    small_block_nbytes: int
    align_nbytes: int

class CachingAllocator:
    def __init__(self, pages_pool: PagesPool, conf: CachingAllocatorConfig) -> None:
        self._conf = C_CachingAllocatorConfig(conf.log_prefix, conf.shm_name, conf.shm_nbytes, conf.va_range_scale, conf.belong_name, conf.small_block_nbytes, conf.align_nbytes)
        self._shm = C_SharedMemory(conf.shm_name, conf.shm_nbytes)
        self._caching_allocator = C_CachingAllocator(self._shm, pages_pool._page_pool, self._conf)
    
    def Alloc(self, nbytes: int, cuda_stream: int) -> C_MemBlock:
        return self._caching_allocator.Alloc(nbytes, cuda_stream, True)
    
    def Free(self, block: C_MemBlock):
        self._caching_allocator.Free(block)
        
    def RegisterAsPyTorchAllocator(self):
        self._caching_allocator.RegisterAsPyTorchAllocator()