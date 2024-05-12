from __future__ import annotations

from ._C import *

import dataclasses

@dataclasses.dataclass
class PagesPoolConf:
    page_nbytes: int
    pool_nbytes: int
    shm_name: str
    log_prefix: str
    shm_nbytes: int

class PagesPool:
    def __init__(self, conf: PagesPoolConf):
        self._conf = C_PagesPoolConf(conf.page_nbytes, conf.pool_nbytes, conf.shm_name, conf.log_prefix, conf.shm_nbytes)
        self._page_pool = C_PagesPool(self._conf)

    def get(self, key: int) -> bytes:
        pass

    def set(self, key: int, value: bytes) -> None:
        pass

    def del_(self, key: int) -> None:
        pass

    def __del__(self):
        pass

class Belong:
    def __init__(self, pages_pool: PagesPool, name: str):
        pass

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
    def __init__(self, conf: CachingAllocatorConfig) -> None:
        self._conf = C_CachingAllocatorConfig(conf.log_prefix, conf.shm_name, conf.shm_nbytes, conf.va_range_scale, conf.belong_name, conf.small_block_nbytes, conf.align_nbytes)
        self._shm = C_SharedMemory(conf.shm_name, conf.shm_nbytes)
        self._caching_allocator = C_CachingAllocator(self._conf)