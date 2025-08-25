
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import json

print("🚀 WORKING 250-STEP TRAINING")

# Load smaller model that works
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("gpt2")

# Simple dataset
texts = [f"WCS Analysis {i}: Call center performance shows {i} calls with strategic recommendations." for i in range(100)]

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)

dataset = Dataset.from_dict({"text": texts})
dataset = dataset.map(tokenize, batched=True)

# Training args
args = TrainingArguments(
    output_dir="./wcs-gpt2-250",
    max_steps=250,
    per_device_train_batch_size=2,
    logging_steps=10,
    save_steps=50,
    learning_rate=5e-4,
    disable_tqdm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

print("Starting 250-step training...")
trainer.train()
print("✅ Training completed!")
trainer.save_model("./wcs-gpt2-250")
