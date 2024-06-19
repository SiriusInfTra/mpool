import time
import mpool

import torch
import torch.multiprocessing as mp

def use_shared_tensor():
    page_conf = mpool.C_PagesPoolConf(
        page_nbytes=32 * 1024 * 1024,
        pool_nbytes=12 * 1024 * 1024 * 1024,
        shm_name='test',
        log_prefix='mpool ',
        shm_nbytes=1 * 1024 * 1024 * 1024,
    )
    page_pool = mpool.C_PagesPool(page_conf)
    caching_allocator_conf = mpool.C_CachingAllocatorConfig(
        align_nbytes=512,
        belong_name='test',
        log_prefix='',
        shm_name='test_caching_allocator',
        shm_nbytes=1 * 1024 * 1024 * 1024,
        small_block_nbytes=1024,
        va_range_scale=1,
    )
    caching_allocator = mpool.C_CachingAllocator(page_pool, caching_allocator_conf)
    block = caching_allocator.alloc(1024, 0, True)
    caching_allocator.free(block)
    mpool.override_pytorch_allocator(caching_allocator)
    mpool.override_ipc(caching_allocator)
    del caching_allocator
    del page_pool

def reset_shared_tensor():
    mpool.reset_pytorch_allocator()

def producer(q: mp.Queue, lock: mp.Lock):
    try:
        print("Launch producer1")
        # time.sleep(3)
        use_shared_tensor()
        # time.sleep(5)
        print("Launch producer2")
        # lock.acquire()
        tensor = torch.zeros(3, 3).cuda()
        print(f"Shared tensor before modified: {tensor}")
        q.put(tensor)
        # lock.release()
        print("Launch producer ok")
        time.sleep(20)
        reset_shared_tensor()
        print("Launch producer exit")
    except Exception as e:
        print(e)


def consumer(q: mp.Queue, lock: mp.Lock):
    try:
        print("Launch consumer1")
        time.sleep(3)
        use_shared_tensor()
        print("Launch consumer2")
        # lock.acquire()
        # tensor = 1
        tensor = q.get()
        tensor += 1
        print(f"Worker modified tensor: {tensor}")
        # lock.release()
        print("Launch consumer ok")
        reset_shared_tensor()
        print("Launch consumer exit")
    except Exception as e:
        print(e)

def wait_exit(p: mp.Process, name: str):
    p.join()
    print(f'is alive: {p.is_alive()}')
    print(f'thread {name} exit with: {p.exitcode}.')

def main():
    # 创建一个管道
    q = mp.Queue()
    lock = mp.Lock()
    # 创建并启动子进程
    pp = mp.Process(target=producer, args=(q, lock))
    pp.start()

    pc = mp.Process(target=consumer, args=(q, lock))
    pc.start()
    
    wait_exit(pp, 'producer')
    wait_exit(pc, 'comsumer')
    

if __name__ == "__main__":
    mp.set_start_method('fork')
    main()
