import time
import cupy
import mpool
import mpool.cupy_integrate
import mpool.cupy_mpool

mempool = cupy.get_default_memory_pool()
x = cupy.zeros(1)
mempool.free_all_blocks()
print("Alloc native OK!")

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

mpool.cupy_integrate.override_cupy_allocator(caching_allocator)

cupy.zeros(int(12 * 1024 * 1024 * 1024 / 8))
print("Alloc MPOOL OK!")