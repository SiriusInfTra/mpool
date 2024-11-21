import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, RandomCrop, ToTensor
from torchvision.models import swin_b
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import time
import numpy as np
import mpool

mpool.override_torch_allocator("__TEST", 12 * 1024**3)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_gpu_memory_usage():
    """获取当前GPU显存使用情况"""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024 / 1024  # MB
    return 0

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, args):
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    end = time.time()
    train_pbar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{args.num_epochs}] Training')
    
    for inputs, labels in train_pbar:
        data_time.update(time.time() - end)
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        acc = (outputs.argmax(1) == labels).float().mean()
        train_loss.update(loss.item(), inputs.size(0))
        train_acc.update(acc.item(), inputs.size(0))
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        images_per_sec = args.batch_size / batch_time.val
        gpu_memory = get_gpu_memory_usage()
        
        train_pbar.set_postfix({
            'loss': f'{train_loss.avg:.4f}',
            'acc': f'{train_acc.avg:.4f}',
            'img/s': f'{images_per_sec:.1f}',
            'GPU mem': f'{gpu_memory:.1f}MB'
        })
    
    return train_loss.avg, train_acc.avg, images_per_sec

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    
    val_pbar = tqdm(val_loader, desc='Validation')
    with torch.no_grad():
        for inputs, labels in val_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            acc = (outputs.argmax(1) == labels).float().mean()
            val_loss.update(loss.item(), inputs.size(0))
            val_acc.update(acc.item(), inputs.size(0))
            
            val_pbar.set_postfix({
                'loss': f'{val_loss.avg:.4f}',
                'acc': f'{val_acc.avg:.4f}'
            })
    
    return val_loss.avg, val_acc.avg

def main(args):
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据预处理
    transform = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 数据加载
    train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=args.num_workers, 
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers, 
                           pin_memory=True)
    
    # 模型初始化
    model = swin_b(pretrained=True)
    model.head = nn.Linear(model.head.in_features, 100)
    model = model.to(device)
    
    if torch.cuda.device_count() > 1 and args.multi_gpu:
        model = nn.DataParallel(model)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 打印初始GPU显存占用
    print(f"Initial GPU memory usage: {get_gpu_memory_usage():.1f}MB")
    
    best_acc = 0.0
    for epoch in range(args.num_epochs):
        # 训练阶段
        train_loss, train_acc, throughput = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args)
        
        # 验证阶段
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        # 打印epoch总结
        print(f"\nEpoch [{epoch+1}/{args.num_epochs}] Summary:")
        print(f"Training Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"GPU Memory Usage: {get_gpu_memory_usage():.1f}MB")
        print(f"Average Throughput: {throughput:.1f} images/sec")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, 'best_model.pth')
            print(f"New best model saved with accuracy: {best_acc:.4f}")
        
        print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwinTransformer Finetuning on CIFAR100")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    main(parser.parse_args())
