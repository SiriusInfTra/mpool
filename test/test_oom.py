import mpool
import os
import random

DEVICE_ID = 0

os.environ['CUDA_VISIBLE_DEVICES'] = str(DEVICE_ID)

page_pool_config = mpool.C_PagesPoolConf(
    device_id = DEVICE_ID,
    log_prefix= '[PagesPool] ',
    page_nbytes = 32 * 1024 * 1024, 
    pool_nbytes = 12 * 1024 * 1024 * 1024, # 12GB
    shm_name = f'test_oom_page_pool',
    shm_nbytes = 16 * 1024 * 1024 # 16MB
)
page_pool = mpool.C_PagesPool(page_pool_config)
caching_allocator_config = mpool.C_CachingAllocatorConfig(
    align_nbytes = 512,
    belong_name = f'torch',
    log_prefix = '[CachingAllocator] ',
    shm_name = f'test_oom_caching_allocator',
    shm_nbytes = 128 * 1024 * 1024, # 128MB
    small_block_nbytes = 1024,
    va_range_scale = 1
)
caching_allocator = mpool.C_CachingAllocator(page_pool, caching_allocator_config)


while True:
    caching_allocator.alloc(random.randint(1, 1024 * 1024 * 1024), 123, True)


