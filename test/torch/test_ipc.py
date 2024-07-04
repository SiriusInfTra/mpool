import time
import mpool

import torch
import torch.multiprocessing as mp

def use_shared_tensor():
    mpool.override_torch_allocator('test_ipc', 12 * 1024 * 1024 * 1024)

def reset_shared_tensor():
    return
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
        time.sleep(10)
        print(f"Shared tensor after modified: {tensor}")
        reset_shared_tensor()
        print("Launch producer exit")
    except Exception as e:
        print(e)


def consumer(q: mp.Queue, lock: mp.Lock):
    try:
        print("Launch consumer1")
        # time.sleep(3)
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
    mp.set_start_method('spawn')
    main()
