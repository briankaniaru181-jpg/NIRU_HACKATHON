# Swahili Gemma 1B Conversational Fine-tuning
# Stage 2: Parameter-Efficient Conversational Fine-tuning with LoRA
# Optimized for Kaggle P100 GPU

# Install required packages
!pip install -q -U transformers datasets peft bitsandbytes accelerate trl

import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import gc
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
MODEL_NAME = "CraneAILabs/swahili-gemma-1b"
DATASET_PATH = "/kaggle/input/datasets/briangreenheart/merged-shuffled-jsonl/merged_shuffled.jsonl"  # Update this path
OUTPUT_DIR = "/kaggle/working/swahili-gemma-finetuned"
MAX_SEQ_LENGTH = 512
TRAIN_SPLIT = 0.9

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Verify GPU (should be P100)
print("=" * 60)
print("GPU Information:")
print("=" * 60)
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print("=" * 60 + "\n")

# Clear cache
gc.collect()
torch.cuda.empty_cache()

# ============================
# 1. LOAD TOKENIZER
# ============================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ============================
# 2. LOAD AND PREPARE DATASET
# ============================
print("Loading dataset...")
dataset = load_dataset('json', data_files=DATASET_PATH, split='train')

print(f"Total examples: {len(dataset)}")
print(f"Sample data: {dataset[0]}")

# Split into train and validation (90/10)
dataset = dataset.train_test_split(train_size=TRAIN_SPLIT, seed=42)
train_dataset = dataset['train']
eval_dataset = dataset['test']

print(f"Training examples: {len(train_dataset)}")
print(f"Validation examples: {len(eval_dataset)}")

# ============================
# 3. FORMATTING FUNCTION
# ============================
def format_conversation(example):
    """
    Format conversations with Gemma-specific tokens.
    Your data format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    messages = example.get('messages', [])
    
    formatted_text = ""
    for message in messages:
        role = message.get('role', '')
        content = message.get('content', '')
        
        if role == 'user':
            formatted_text += f"<start_of_turn>user\n{content}<end_of_turn>\n"
        elif role == 'assistant':
            formatted_text += f"<start_of_turn>model\n{content}<end_of_turn>\n"
    
    return {"text": formatted_text}

# Apply formatting
print("\nFormatting dataset...")
train_dataset = train_dataset.map(format_conversation, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(format_conversation, remove_columns=eval_dataset.column_names)

# Verify formatting
print("\nSample formatted conversation:")
print("-" * 60)
print(train_dataset[0]['text'])
print("-" * 60 + "\n")

# ============================
# 4. TOKENIZATION FUNCTION
# ============================
def tokenize_function(examples):
    """Tokenize the text data"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        return_tensors=None
    )

# Tokenize datasets
print("Tokenizing datasets...")
train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing training data"
)

eval_dataset = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=eval_dataset.column_names,
    desc="Tokenizing validation data"
)

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

print(f"Tokenized training dataset: {train_dataset}")
print(f"Tokenized eval dataset: {eval_dataset}")

# ============================
# 5. QUANTIZATION CONFIG (4-bit NF4)
# ============================
print("\nConfiguring 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # Double quantization
    bnb_4bit_quant_type="nf4",  # NF4 quantization
    bnb_4bit_compute_dtype=torch.float16  # FP16 for computation
)

# ============================
# 6. LOAD MODEL
# ============================
print("Loading base model with quantization...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False  # Required for gradient checkpointing

# ============================
# 7. LORA CONFIGURATION
# ============================
print("Configuring LoRA...")
peft_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=16,  # Alpha
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Target modules
    lora_dropout=0.05,  # Dropout
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

# Print trainable parameters
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}%"
    )

print_trainable_parameters(model)

# ============================
# 8. DATA COLLATOR
# ============================
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're doing causal LM, not masked LM
)

# ============================
# 9. CUSTOM CALLBACK FOR TRACKING
# ============================
class MetricsCallback(TrainerCallback):
    """Custom callback to track and save metrics"""
    
    def __init__(self):
        self.metrics_history = []
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation"""
        if metrics is not None:
            epoch = int(state.epoch) if state.epoch is not None else 0
            
            # Calculate perplexity from eval_loss
            eval_loss = metrics.get('eval_loss', 0)
            perplexity = np.exp(eval_loss)
            
            # Get training loss (average of recent steps)
            train_loss = state.log_history[-2].get('loss', 0) if len(state.log_history) > 1 else 0
            
            metric_entry = {
                'epoch': epoch,
                'train_loss': round(train_loss, 3),
                'eval_loss': round(eval_loss, 3),
                'perplexity': round(perplexity, 2)
            }
            
            self.metrics_history.append(metric_entry)
            
            # Print metrics in a formatted table
            print(f"\n{'='*60}")
            print(f"Epoch {epoch} Metrics:")
            print(f"{'='*60}")
            print(f"Training Loss:    {train_loss:.3f}")
            print(f"Validation Loss:  {eval_loss:.3f}")
            print(f"Perplexity:       {perplexity:.2f}")
            print(f"{'='*60}\n")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training"""
        self.save_metrics_table()
        self.plot_metrics()
    
    def save_metrics_table(self):
        """Save metrics as a formatted table"""
        if not self.metrics_history:
            return
        
        df = pd.DataFrame(self.metrics_history)
        
        # Save as CSV
        csv_path = os.path.join(OUTPUT_DIR, "training_metrics.csv")
        df.to_csv(csv_path, index=False)
        
        # Print final table
        print("\n" + "="*60)
        print("FINAL TRAINING METRICS")
        print("="*60)
        print(df.to_string(index=False))
        print("="*60 + "\n")
        
        # Save as formatted text
        txt_path = os.path.join(OUTPUT_DIR, "training_metrics.txt")
        with open(txt_path, 'w') as f:
            f.write("Epoch | Training Loss | Validation Loss | Perplexity\n")
            f.write("-" * 60 + "\n")
            for metric in self.metrics_history:
                f.write(f"{metric['epoch']:5d} | {metric['train_loss']:13.3f} | "
                       f"{metric['eval_loss']:15.3f} | {metric['perplexity']:10.2f}\n")
        
        print(f" Metrics saved to: {csv_path}")
        print(f" Metrics saved to: {txt_path}")
    
    def plot_metrics(self):
        """Plot training and validation metrics"""
        if not self.metrics_history:
            return
        
        df = pd.DataFrame(self.metrics_history)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Training and Validation Loss
        axes[0].plot(df['epoch'], df['train_loss'], marker='o', label='Training Loss', linewidth=2)
        axes[0].plot(df['epoch'], df['eval_loss'], marker='s', label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Perplexity
        axes[1].plot(df['epoch'], df['perplexity'], marker='D', color='green', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Perplexity', fontsize=12)
        axes[1].set_title('Validation Perplexity', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "training_metrics.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f" Training plot saved to: {plot_path}")
        plt.show()

# Initialize callback
metrics_callback = MetricsCallback()

# ============================
# 10. TRAINING ARGUMENTS
# ============================
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=15,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_grad_norm=1.0,
    fp16=True,
    optim="paged_adamw_8bit",
    logging_steps=10,
    logging_first_step=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=2,
    dataloader_pin_memory=True
)

# ============================
# 11. TRAINER SETUP
# ============================
print("Initializing trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3),
        metrics_callback
    ]
)

# ============================
# 12. TRAIN MODEL
# ============================
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60 + "\n")

trainer.train()

# ============================
# 13. SAVE MODEL
# ============================
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60 + "\n")

trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save merged model (LoRA + base)
print("Merging LoRA adapters with base model...")
merged_model = model.merge_and_unload()
merged_model.save_pretrained(f"{OUTPUT_DIR}/merged_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/merged_model")

print(f"\n Training complete!")
print(f" Model saved to: {OUTPUT_DIR}")
print(f" Merged model saved to: {OUTPUT_DIR}/merged_model")

# ============================
# 14. INFERENCE TEST
# ============================
print("\n" + "="*60)
print("TESTING FINE-TUNED MODEL")
print("="*60 + "\n")

# Clear memory before inference
del trainer
del model
gc.collect()
torch.cuda.empty_cache()

# Load the merged model for inference
inference_model = AutoModelForCausalLM.from_pretrained(
    f"{OUTPUT_DIR}/merged_model",
    device_map="auto",
    torch_dtype=torch.float16
)

# Test prompts in Swahili
test_prompts = [
    "<start_of_turn>user\nAndika udogo wa sentensi: 'Watoto wanacheza uwanjani.'<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nJe, Dar es Salaam iko wapi?<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nNiambie kuhusu Tanzania.<end_of_turn>\n<start_of_turn>model\n",
]

for i, test_prompt in enumerate(test_prompts, 1):
    print(f"Test {i}:")
    print("-" * 60)
    inputs = tokenizer(test_prompt, return_tensors="pt").to(inference_model.device)
    outputs = inference_model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"Response:\n{response}\n")
    print("=" * 60 + "\n")

print("="*60)
print("SCRIPT COMPLETED SUCCESSFULLY!")
print("="*60)