import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet152
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import time

import mpool
mpool.override_torch_allocator('test_resnet_152', int(14.5 * 1024 * 1024 * 1024))

# 超参数
batch_size = 72
epochs = 10
learning_rate = 0.001

# 使用FakeData生成器
transform = ToTensor()
train_dataset = FakeData(transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化ResNet-152模型
model = resnet152()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 使用GPU进行训练（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练循环
for epoch in range(epochs):
    start_time = time.time()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    end_time = time.time()
    print(f"Epoch {epoch+1}/{epochs} finished in {end_time - start_time:.2f} seconds")

print("Training complete.")
