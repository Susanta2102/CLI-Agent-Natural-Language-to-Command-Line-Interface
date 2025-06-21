#!/usr/bin/env python3
"""
Fine-tuning Script for CLI Assistant Model
Uses LoRA/QLoRA to fine-tune TinyLlama on command-line Q&A data
"""

import os
import json
import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLIModelTrainer:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # LoRA configuration
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,  # rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            use_cache=False  # Disable for training
        )
        
        # Apply LoRA
        self.peft_model = get_peft_model(self.model, self.lora_config)
        self.peft_model.print_trainable_parameters()
        
        logger.info("Model and tokenizer setup completed")
    
    def load_and_process_data(self, data_path="data/processed_training_data.json"):
        """Load and tokenize training data"""
        logger.info(f"Loading training data from {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Format data for instruction tuning
        formatted_data = []
        for item in data:
            # Create instruction format similar to Alpaca
            if item['input'].strip():
                prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
            else:
                prompt = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
            
            formatted_data.append({"text": prompt})
        
        logger.info(f"Formatted {len(formatted_data)} training examples")
        
        # Create dataset
        dataset = Dataset.from_list(formatted_data)
        
        # Tokenize
        def tokenize_function(examples):
            # Tokenize the text
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        logger.info("Data tokenization completed")
        return tokenized_dataset
    
    def train(self, dataset, output_dir="training/adapters"):
        """Fine-tune the model with LoRA"""
        logger.info("Starting training...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            max_steps=500,  # Limit steps for quick training
            learning_rate=1e-4,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=None,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
            dataloader_pin_memory=False,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            bf16=False,  # Use fp16 instead for broader compatibility
            fp16=torch.cuda.is_available(),
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train the model
        trainer.train()
        
        # Save the adapter
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Training completed. Adapter saved to {output_dir}")
        
        return trainer

def main():
    """Main training function"""
    # Create training directory
    os.makedirs("training", exist_ok=True)
    
    # Initialize trainer
    trainer = CLIModelTrainer()
    
    # Setup model and tokenizer
    trainer.setup_model_and_tokenizer()
    
    # Load and process data
    dataset = trainer.load_and_process_data()
    
    # Train the model
    trained_model = trainer.train(dataset)
    
    logger.info("Training pipeline completed successfully!")
    
    # Save training log
    training_log = {
        "model_name": trainer.model_name,
        "dataset_size": len(dataset),
        "lora_config": {
            "r": trainer.lora_config.r,
            "lora_alpha": trainer.lora_config.lora_alpha,
            "lora_dropout": trainer.lora_config.lora_dropout,
            "target_modules": trainer.lora_config.target_modules
        },
        "training_completed": True
    }
    
    with open("training/training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

if __name__ == "__main__":
    main()