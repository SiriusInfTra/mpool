import mpool

page_conf = mpool.PagesPoolConf(
    page_nbytes=32 * 1024 * 1024,
    pool_nbytes=12 * 1024 * 1024 * 1024,
    shm_name='test',
    log_prefix='mpool ',
    shm_nbytes=1 * 1024 * 1024 * 1024,
)
page_pool = mpool.PagesPool(page_conf)