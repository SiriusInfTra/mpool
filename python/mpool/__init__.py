from __future__ import annotations
from threading import local
from ._C import *
from . import cupy_integrate
from . import numba_integrate



def override_ipc(caching_alloactor: C_CachingAllocator):
    import torch
    def _share_cuda(self):
        return mpool_share_cuda(self)
    def _new_shared_cuda(*args):
        return mpool_new_shared_cuda(None, *args)
    torch.UntypedStorage._share_cuda_ = _share_cuda
    torch.UntypedStorage._new_shared_cuda = _new_shared_cuda
