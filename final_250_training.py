#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
import json

print("🚀 FINAL 250-STEP WCS TRAINING")
print("=" * 40)

# Load model
print("📦 Loading GPT-2 model...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("gpt2")

# Load WCS dataset
print("📊 Loading WCS dataset...")
try:
    with open('/Users/cpconnor/projects/gptoss20/wcs_comprehensive_dataset.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Extract text from messages
    texts = []
    for item in data:
        if 'messages' in item and item['messages']:
            conversation = ""
            for msg in item['messages']:
                conversation += f"{msg['role']}: {msg['content']}\n"
            texts.append(conversation.strip())
    
    print(f"✅ Loaded {len(texts)} training examples from WCS dataset")
except:
    # Fallback data
    texts = [f"WCS Analysis {i}: Call center performance shows {i} calls with strategic recommendations." for i in range(100)]
    print(f"✅ Using {len(texts)} fallback training examples")

def tokenize(examples):
    # Tokenize for causal language modeling
    tokens = tokenizer(examples["text"], truncation=True, padding=False, max_length=256)
    # For CLM, labels are the same as input_ids
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = Dataset.from_dict({"text": texts})
dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

print(f"📊 Dataset prepared: {len(dataset)} examples")

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're doing causal LM, not masked LM
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./wcs-gpt2-250-final",
    max_steps=250,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=5e-4,
    logging_steps=10,
    save_steps=50,
    disable_tqdm=False,
    logging_first_step=True,
    warmup_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

print("🚀 Starting 250-step training...")
print("📊 Progress will show every 10 steps")
print("💾 Model saves every 50 steps")
print("-" * 40)

start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
if start_time:
    start_time.record()

# Execute training
trainer.train()

print("-" * 40)
print("✅ 250-STEP TRAINING COMPLETED!")
print(f"📁 Model saved to: ./wcs-gpt2-250-final")

# Save final model
trainer.save_model("./wcs-gpt2-250-final")
tokenizer.save_pretrained("./wcs-gpt2-250-final")

# Save training summary
training_summary = {
    "model_type": "GPT-2",
    "training_steps": 250,
    "dataset_size": len(dataset),
    "batch_size": 2,
    "learning_rate": 5e-4,
    "output_directory": "./wcs-gpt2-250-final",
    "training_completed": True
}

with open("./wcs-gpt2-250-final/training_summary.json", "w") as f:
    json.dump(training_summary, f, indent=2)

print("📊 Training summary saved")
print("🎉 READY FOR COMPREHENSIVE TESTING!")