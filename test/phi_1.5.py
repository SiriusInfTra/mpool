import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.optim import AdamW, SGD
from tqdm import tqdm
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def setup(rank, world_size):
    # torch.set_default_device('cuda')+
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    # import mpool
    # mpool.override_torch_allocator('test_dist_train', int(14.5 * 1024 * 1024 * 1024))

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    # Load dataset
    dataset = load_dataset("./model/wikitext", "wikitext-2-raw-v1", split='train')
    tokenizer = AutoTokenizer.from_pretrained("./model/phi-1.5")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    
    def tokenize_function(examples):
        return tokenizer(examples['text'], return_tensors='pt', truncation=True, padding='max_length', max_length=512)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    train_sampler = DistributedSampler(tokenized_datasets, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(tokenized_datasets, sampler=train_sampler, batch_size=2, pin_memory=True, collate_fn=data_collator)

    model = AutoModelForCausalLM.from_pretrained("./model/phi-1.5", torch_dtype=torch.dtyp, trust_remote_code=True)
    model.to(rank)
    model = DDP(model, device_ids=[rank])
    model = FSDP(model.module)

    optimizer = SGD(model.parameters(), lr=5e-5, momentum=0.0)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)
    )

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            batch = {k: v.to(rank) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    cleanup()

if __name__ == '__main__':
    world_size = 4  # Number of GPUs
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
