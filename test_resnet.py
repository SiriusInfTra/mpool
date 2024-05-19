import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet152
from torchvision.datasets import FakeData
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import mpool

# torch.zeros(1).cuda()
# print("CUDA is available")

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
    page_pool.free_pages(pages, belong, lock)
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

caching_allocator.RegisterAsPyTorchAllocator()
# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建一个未经预训练的 ResNet152 模型
model = resnet152(pretrained=False)
model.to(device)

# 创建假数据
# 假设输入数据为3通道图像，图像大小为224x224，有10个类别
dataset = FakeData(size=100, image_size=(3, 224, 224), num_classes=10, transform=ToTensor())
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}')

print("Training complete")
