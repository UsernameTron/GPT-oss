#!/usr/bin/env python3

import os
import torch
import pandas as pd
import glob
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel, 
    AutoTokenizer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType
import json
from datetime import datetime

print("🚀 EXTENDED WCS MODEL TRAINING - 250 STEPS")
print("=" * 50)

# Configuration
MODEL_NAME = "openai/gpt-oss-20b"
OUTPUT_DIR = "gpt-oss-20b-wcs-extended-250"
TRAINING_DATA_DIR = "/Users/cpconnor/projects/gptoss20/Training Data/WCS 123123-010624"
DATASET_FILE = "/Users/cpconnor/projects/gptoss20/wcs_comprehensive_dataset.jsonl"

def load_and_prepare_model():
    """Load model with LoRA configuration for extended training."""
    print("📦 Loading GPT-OSS-20B model...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # LoRA Configuration for extended training
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj", "c_fc"]
    )
    
    model = get_peft_model(model, peft_config)
    print(f"✅ Model loaded with LoRA adapter")
    print(f"📊 Trainable parameters: {model.num_parameters()}")
    
    return model, tokenizer

def load_comprehensive_dataset():
    """Load the comprehensive WCS dataset."""
    print("📊 Loading comprehensive WCS dataset...")
    
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"Dataset file not found: {DATASET_FILE}")
    
    data = []
    with open(DATASET_FILE, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"✅ Loaded {len(data)} training examples")
    return Dataset.from_list(data)

def setup_extended_trainer(model, tokenizer, dataset):
    """Set up trainer for extended 250-step training."""
    print("🔧 Setting up extended trainer configuration...")
    
    # Training arguments for extended training with real-time logging
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        max_steps=250,  # Extended from 88 steps
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,  # More frequent logging
        save_steps=25,    # More frequent saves
        eval_steps=25,
        save_strategy="steps",
        evaluation_strategy="no",
        load_best_model_at_end=False,
        report_to=None,
        remove_unused_columns=False,
        disable_tqdm=False,  # Enable progress bar
        log_level="info",
        logging_first_step=True,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=1024,
    )
    
    print("✅ Extended trainer configured for 250 steps")
    return trainer

def execute_extended_training():
    """Execute the extended training process."""
    print("\n🎯 STARTING EXTENDED TRAINING PROCESS")
    print("=" * 40)
    
    # Load model and tokenizer
    model, tokenizer = load_and_prepare_model()
    
    # Load dataset
    dataset = load_comprehensive_dataset()
    
    # Setup trainer
    trainer = setup_extended_trainer(model, tokenizer, dataset)
    
    # Execute training with progress tracking
    print("🚀 Starting extended training (250 steps)...")
    print("📊 Training Progress:")
    print("   ✅ Steps will be logged every 5 steps")
    print("   💾 Model checkpoints saved every 25 steps")
    print("   🎯 Target: 250 total training steps")
    print("-" * 50)
    
    start_time = datetime.now()
    
    # Train with progress bar visible
    trainer.train()
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("-" * 50)
    print("🎯 TRAINING METRICS:")
    print(f"   📈 Total Steps Completed: 250")
    print(f"   ⏱️ Training Duration: {duration}")
    print(f"   🔢 Dataset Size: {len(trainer.train_dataset)} examples")
    
    print(f"✅ Extended training completed!")
    print(f"⏱️ Training duration: {duration}")
    print(f"📁 Model saved to: {OUTPUT_DIR}")
    
    # Save final model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save training metadata
    metadata = {
        "training_completed": end_time.isoformat(),
        "training_duration": str(duration),
        "max_steps": 250,
        "dataset_examples": len(dataset),
        "model_output_dir": OUTPUT_DIR,
        "extended_training": True
    }
    
    with open(f"{OUTPUT_DIR}/extended_training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📊 Training metadata saved")
    
    return OUTPUT_DIR

if __name__ == "__main__":
    try:
        model_path = execute_extended_training()
        print(f"\n🎉 EXTENDED TRAINING SUCCESS!")
        print(f"📁 Extended model available at: {model_path}")
        print("🔄 Ready for comprehensive testing across all WCS files")
        
    except Exception as e:
        print(f"❌ Extended training failed: {str(e)}")
        raise