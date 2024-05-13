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
block = caching_allocator.Alloc(1024, 0)
caching_allocator.Free(block)