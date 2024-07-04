import os
import torch
import torch.distributed as dist
import torch.multiprocessing.spawn
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    import mpool
    mpool.override_torch_allocator('test_dist_train', 12 * 1024 * 1024 * 1024)


def cleanup():
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.fc(x)

def main(rank, world_size):
    setup(rank, world_size)

    # Create model and move it to GPU with id rank
    model = SimpleModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Create dataset and dataloader
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(2):
        sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(rank), target.to(rank)

            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0 and rank == 0:
                print(f"Epoch [{epoch}/{2}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

    cleanup()

if __name__ == "__main__":
    world_size = 4
    torch.multiprocessing.start_processes(main, args=(world_size,), nprocs=world_size, join=True, start_method='spawn')
