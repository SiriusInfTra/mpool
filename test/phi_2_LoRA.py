import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

model_name_or_path = "./model/gpt2-xl"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, model_max_length=1024)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model.resize_token_embeddings(len(tokenizer))
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding='max_length', return_tensors="pt", max_length=128)

train_dataset = load_dataset("./model/wikitext", "wikitext-2-raw-v1", split='train')
train_dataset = train_dataset.map(tokenize_function, batched=True)
print(train_dataset[1])

lora_config = LoraConfig(
    r=8,  # 低秩矩阵的秩
    lora_alpha=32,  # LORA的alpha值
    target_modules=["c_attn"],  # 目标模块
    lora_dropout=0.1,  # Dropout几率
    bias="none",  # 是否在LORA层中使用偏置
    task_type=TaskType.CAUSAL_LM,  # 任务类型
)

model = get_peft_model(model, lora_config)
print(model)

from transformers import Trainer, TrainingArguments
model.train()
# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)



# 创建Trainer对象
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # 你的训练数据集
)

# 开始训练
trainer.train()
print(train_dataset[1753399])
