import mpool
import os
import random
import torch

DEVICE_ID = 0

os.environ['CUDA_VISIBLE_DEVICES'] = str(DEVICE_ID)


mpool.override_torch_allocator('test_oom', 12 * 1024 * 1024 * 1024)


while True:
    x1 = torch.zeros(9 * 1024 * 1024 * 1024 // 4, device='cuda', dtype=torch.float32 )
    x2 = torch.zeros(9 * 1024 * 1024 * 1024 // 4, device='cuda', dtype=torch.float32 )


