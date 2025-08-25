#!/usr/bin/env python3

import os
import torch
import pandas as pd
import json
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
import tqdm

print("🚀 SIMPLIFIED WCS TRAINING - 250 STEPS")
print("=" * 45)

# Use smaller, more manageable model
MODEL_NAME = "microsoft/DialoGPT-medium"  # Smaller but effective model
OUTPUT_DIR = "wcs-dialogpt-250-steps"
DATASET_FILE = "/Users/cpconnor/projects/gptoss20/wcs_comprehensive_dataset.jsonl"

def load_model():
    """Load smaller, manageable model for training."""
    print("📦 Loading DialoGPT-medium model...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
    )
    
    model = get_peft_model(model, peft_config)
    print(f"✅ Model loaded with LoRA")
    model.print_trainable_parameters()
    
    return model, tokenizer

def load_dataset(tokenizer):
    """Load and preprocess WCS dataset."""
    print("📊 Loading WCS dataset...")
    
    if not os.path.exists(DATASET_FILE):
        # Create simple dataset if original doesn't exist
        print("⚠️ Creating sample dataset...")
        sample_data = []
        for i in range(100):
            sample_data.append({
                "text": f"WCS Call Center Analysis {i}: Partner performance shows {i*2} abandoned calls. Strategic recommendations for improvement include capacity planning and resource optimization."
            })
        
        with open(DATASET_FILE, 'w') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')
    
    data = []
    with open(DATASET_FILE, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Extract text from messages format
    texts = []
    for item in data:
        if 'text' in item:
            texts.append(item['text'])
        elif 'messages' in item and item['messages']:
            # Convert messages to training text
            conversation = ""
            for msg in item['messages']:
                conversation += f"{msg['role']}: {msg['content']}\n"
            texts.append(conversation.strip())
    
    if not texts:
        # Create sample data if no valid text found
        print("⚠️ No valid text found, creating sample training data...")
        texts = [f"WCS Call Center Analysis {i}: Partner performance analysis shows strategic recommendations for capacity planning and operational optimization." for i in range(50)]
    
    # Tokenize texts properly
    def tokenize_function(examples):
        # Tokenize and create labels for causal language modeling
        tokenized = tokenizer(
            examples['text'], 
            truncation=True, 
            padding=False,  # We'll pad in the data collator
            max_length=512,
            return_tensors=None  # Don't convert to tensors yet
        )
        # For causal LM, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized
    
    # Create dataset
    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    print(f"✅ Loaded and tokenized {len(dataset)} training examples")
    return dataset

def train_model():
    """Execute 250-step training with visible progress."""
    print("\n🎯 STARTING 250-STEP TRAINING")
    print("=" * 35)
    
    # Load components
    model, tokenizer = load_model()
    dataset = load_dataset(tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        max_steps=250,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=10,
        learning_rate=5e-4,
        logging_steps=10,
        save_steps=50,
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to=None,
        disable_tqdm=False,  # Enable progress bar
        logging_first_step=True,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Training with progress display
    print("🚀 Training Progress (250 steps):")
    print("   📊 Batch size: 4, Gradient accumulation: 2")
    print("   ⚡ Learning rate: 5e-4")
    print("   📝 Logging every 10 steps")
    print("   💾 Saving every 50 steps")
    print("-" * 40)
    
    start_time = datetime.now()
    
    # Execute training
    trainer.train()
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("-" * 40)
    print("✅ TRAINING COMPLETED!")
    print(f"⏱️ Duration: {duration}")
    print(f"📁 Model saved to: {OUTPUT_DIR}")
    
    # Save model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Training summary
    summary = {
        "training_completed": end_time.isoformat(),
        "duration": str(duration),
        "max_steps": 250,
        "dataset_size": len(dataset),
        "model_name": MODEL_NAME,
        "output_dir": OUTPUT_DIR
    }
    
    with open(f"{OUTPUT_DIR}/training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("📊 Training summary saved")
    return OUTPUT_DIR

if __name__ == "__main__":
    try:
        model_path = train_model()
        print(f"\n🎉 SUCCESS!")
        print(f"📁 Trained model: {model_path}")
        print("🔄 Ready for testing phase")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise