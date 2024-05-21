import numba
from numba import cuda

import numpy as np
import mpool
from mpool import numba_integrate

@cuda.jit
def test_kernel(d_arr):
    idx = cuda.grid(1)
    if idx < d_arr.size:
        d_arr[idx] *= 2

pages_pool_conf = mpool.C_PagesPoolConf(
    page_nbytes=32 * 1024 * 1024,
    pool_nbytes=int(13.5 * 1024 * 1024 * 1024),
    shm_name='test',
    log_prefix='mpool ',
    shm_nbytes=1 * 1024 * 1024 * 1024,
)

caching_allocator_conf = mpool.C_CachingAllocatorConfig(
    align_nbytes=512,
    belong_name='test',
    log_prefix='',
    shm_name='test_caching_allocator',
    shm_nbytes=1 * 1024 * 1024 * 1024,
    small_block_nbytes=1024,
    va_range_scale=4,
)

pages_pool = mpool.create_pages_pool('default_pagepool', pages_pool_conf)
caching_allocator = mpool.create_caching_allocator('default_allocator', pages_pool, caching_allocator_conf)

numba_integrate.override_numba_allocator(caching_allocator)


size = 12 * 1024 * 1024 * 1024 / 4
arr = np.arange(size, dtype=np.float32)
d_arr = cuda.to_device(arr)

# 定义CUDA网格和块大小
threads_per_block = 256
blocks_per_grid = (arr.size + (threads_per_block - 1)) // threads_per_block

# 启动CUDA内核
test_kernel[blocks_per_grid, threads_per_block](d_arr)


result_array = d_arr.copy_to_host()

expected = arr * 2
assert np.allclose(result_array, expected), "result not match"

print("ok")