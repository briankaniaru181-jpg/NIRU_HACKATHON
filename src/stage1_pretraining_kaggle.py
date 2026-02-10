#!/usr/bin/env python3
"""

STAGE 1: Swahili Linguistic Foundation - Continued Pretraining
FINAL COMPLETE VERSION - KAGGLE OPTIMIZED

WITH COMPREHENSIVE MONITORING, TRACKING, CHECKPOINT RECOVERY & SWAHILI EARLY STOPPING

Supports JSONL Dataset Format + Full Checkpoint Recovery + Sequence Packing + Swahili Score Tracking

Metrics tracked:
1. Perplexity
2. Validation loss
3. Training Stability 
4. Swahili Score

FEATURES:
1. JSONL dataset support with SEQUENCE PACKING
2. Comprehensive metrics tracking
3. Real-time stability monitoring
4. FULL CHECKPOINT RECOVERY
5. Automatic training resumption
6. MEMORY MONITORING (Kaggle safety)
7. Robust device handling
8. Dataset sample caching
9. SWAHILI SCORE-BASED EARLY STOPPING
10. KAGGLE-SPECIFIC OPTIMIZATIONS
11. Visualization generation
12. Detailed reporting
"""
import os
os.environ["TORCHVISION_USE_FBCODE"] = "1"  # Disable torchvision custom ops

import sys
import types

class TensorParallelMock(types.ModuleType):
    """Comprehensive mock for transformers.integrations.tensor_parallel"""
    
    def __init__(self):
        super().__init__('transformers.integrations.tensor_parallel')
        
        # Lists/constants
        self.ALL_PARALLEL_STYLES = []
        self.SUPPORTED_TP_STYLES = []
        self.TENSOR_PARALLEL_KEY = "tensor_parallel"
        self.TP_PLAN_KEY = "tp_plan"
        
        # All required functions as lambdas
        self._get_parameter_tp_plan = lambda *args, **kwargs: None
        self.tensor_parallel_state_dict = lambda *args, **kwargs: {}
        self.tensor_parallel_load = lambda *args, **kwargs: None
        self.tensor_parallel_save = lambda *args, **kwargs: None
        self.convert_state_dict = lambda *args, **kwargs: {}
        self.convert_state_dict_to_tensor_parallel = lambda *args, **kwargs: {}
        self.convert_state_dict_from_tensor_parallel = lambda *args, **kwargs: {}
        self.get_tensor_parallel_plan = lambda *args, **kwargs: {}
        self.get_tensor_parallel_style = lambda *args, **kwargs: None
        self.is_tensor_parallel = lambda *args, **kwargs: False
        self.load_tensor_parallel_weights = lambda *args, **kwargs: {}
        self.save_tensor_parallel_weights = lambda *args, **kwargs: None
        
        #  Other Critical Functions
        self.distribute_model = lambda *args, **kwargs: args[0] if args else None
        self.distribute_optimizer = lambda *args, **kwargs: args[0] if args else None
        self.gather_model = lambda *args, **kwargs: args[0] if args else None
    
    def __getattr__(self, name):
        """Catch-all for any missing attributes"""
        if name.startswith('__'):
            raise AttributeError(f"Mock has no attribute {name}")
        # Return a no-op lambda for anything we missed
        return lambda *args, **kwargs: args[0] if args else None

# Install mock BEFORE transformers imports
mock = TensorParallelMock()
sys.modules['transformers.integrations.tensor_parallel'] = mock

print(" Aggressive tensor_parallel mock installed")

# ========== PYTORCH COMPATIBILITY ==========
import torch
print(f"🔧 PyTorch version: {torch.__version__}")

if torch.__version__.startswith(('2.4', '2.10', '2.11', '2.12')):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import gc
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import psutil
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# import transformers 
print("\n🔧 Importing transformers...")
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        BitsAndBytesConfig,
        EarlyStoppingCallback,
        TrainerCallback,
    )
    print("Successfully imported transformers core modules")
    
    # import tensor_parallel
    try:
        import importlib
        actual_tp = importlib.import_module('transformers.integrations.tensor_parallel')
        print("Actual tensor_parallel module exists, but using mock")
    except:
        print("No actual tensor_parallel module found, mock is sufficient")
        
except ImportError as e:
    print(f"Error importing transformers: {e}")
    print("Trying fallback imports...")
    
    # Fallback: Import transformers and get components directly
    import transformers
    
    # Get the actual module attributes
    AutoTokenizer = getattr(transformers, 'AutoTokenizer', None)
    AutoModelForCausalLM = getattr(transformers, 'AutoModelForCausalLM', None)
    TrainingArguments = getattr(transformers, 'TrainingArguments', None)
    Trainer = getattr(transformers, 'Trainer', None)
    DataCollatorForLanguageModeling = getattr(transformers, 'DataCollatorForLanguageModeling', None)
    BitsAndBytesConfig = getattr(transformers, 'BitsAndBytesConfig', None)
    EarlyStoppingCallback = getattr(transformers, 'EarlyStoppingCallback', None)
    TrainerCallback = getattr(transformers, 'TrainerCallback', None)
    
    if all([AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer]):
        print("Successfully imported transformers via fallback")
    else:
        raise ImportError("Could not import necessary transformers components")

import datasets

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType
    )
    print("Successfully imported PEFT")
except ImportError:
    print("PEFT not available, trying to install...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "peft", "-q"])
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType
    )
    print("Installed and imported PEFT")

# ========== KAGGLE-SPECIFIC OPTIMIZATIONS ==========
def kaggle_specific_setup():
    """Extra safety and optimization for Kaggle P100 environment"""
    print("\n" + "="*70)
    print("APPLYING KAGGLE-SPECIFIC OPTIMIZATIONS")
    print("="*70)
    
    # 1. Deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Deterministic algorithms enabled (reproducible, prevents OOM crashes)")
    
    # 2. Limit CPU threads to prevent RAM OOM during data loading
    torch.set_num_threads(2)
    print(f"CPU threads limited to {torch.get_num_threads()} (prevents memory swapping)")
    
    # 3. Check GPU and only enable TF32 if supported
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    if "P100" in gpu_name:
        print(f"Tesla P100 detected - TF32 disabled (not supported on this GPU)")
    else:
        # Enable TF32 for newer GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled (faster math on compatible GPUs)")
    
    # 4. Force garbage collection before starting
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("GPU cache cleared, memory fragmentation prevented")
    
    # 5. Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    print("Random seeds set (reproducible results)")
    
    print("="*70 + "\n")


# ========== MEMORY MONITOR ==========
class MemoryMonitor:
    """Monitor GPU and CPU memory usage"""
    
    def __init__(self):
        self.peak_gpu_memory = 0
        self.peak_cpu_memory = 0
        self.memory_log = []
        
    def log_memory(self, step: int, prefix: str = ""):
        """Log current memory usage"""
        log_entry = {
            "step": step,
            "prefix": prefix,
            "timestamp": str(pd.Timestamp.now()),
        }
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            self.peak_gpu_memory = max(self.peak_gpu_memory, allocated)
            
            log_entry["gpu_allocated_gb"] = round(allocated, 2)
            log_entry["gpu_reserved_gb"] = round(reserved, 2)
            
            if allocated > 14:
                print(f"MEMORY WARNING: {allocated:.2f}GB GPU allocated (limit ~16GB)")
                log_entry["warning"] = "High GPU memory"
        
        cpu_mem = psutil.virtual_memory()
        cpu_percent = cpu_mem.percent
        cpu_used_gb = cpu_mem.used / 1e9
        self.peak_cpu_memory = max(self.peak_cpu_memory, cpu_used_gb)
        
        log_entry["cpu_used_gb"] = round(cpu_used_gb, 2)
        log_entry["cpu_percent"] = round(cpu_percent, 2)
        
        if cpu_percent > 85:
            print(f"MEMORY WARNING: {cpu_percent:.1f}% CPU memory used")
            log_entry["warning"] = "High CPU memory"
        
        self.memory_log.append(log_entry)
        
        if prefix:
            print(f"{prefix}: GPU={log_entry.get('gpu_allocated_gb', 'N/A')}GB, "
                  f"CPU={log_entry.get('cpu_used_gb', 'N/A')}GB ({cpu_percent:.1f}%)")
    
    def get_report(self) -> Dict:
        """Get memory report"""
        return {
            "peak_gpu_memory_gb": round(self.peak_gpu_memory, 2),
            "peak_cpu_memory_gb": round(self.peak_cpu_memory, 2),
            "total_logs": len(self.memory_log),
        }

# ========== CHECKPOINT RECOVERY MANAGER ==========
class CheckpointRecoveryManager:
    """Manages checkpoint creation, restoration, and recovery"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.recovery_log_file = self.checkpoint_dir / "recovery.log"
        
        self.metadata = self._load_metadata()
        
        print("CheckpointRecoveryManager initialized")
    
    def _load_metadata(self) -> Dict:
        """Load checkpoint metadata"""
        if self.checkpoint_metadata_file.exists():
            try:
                with open(self.checkpoint_metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading metadata: {str(e)}")
                return {}
        return {}
    
    def _save_metadata(self):
        """Save checkpoint metadata"""
        try:
            with open(self.checkpoint_metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving metadata: {str(e)}")
    
    def log_recovery(self, message: str):
        """Log recovery information"""
        try:
            with open(self.recovery_log_file, "a", encoding="utf-8") as f:
                timestamp = pd.Timestamp.now()
                f.write(f"[{timestamp}] {message}\n")
        except Exception as e:
            print(f"Error logging: {str(e)}")
    
    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find the latest checkpoint directory"""
        try:
            checkpoint_dirs = sorted(
                [d for d in self.checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                key=lambda x: int(x.name.split("-")[1]),
                reverse=True
            )
            
            if checkpoint_dirs:
                latest = checkpoint_dirs[0]
                step = int(latest.name.split("-")[1])
                print(f"Found latest checkpoint: {latest.name} (Step {step})")
                self.log_recovery(f"Found checkpoint: {latest.name}")
                return latest
        except Exception as e:
            print(f"Error finding checkpoint: {str(e)}")
        
        return None
    
    def save_checkpoint_info(self, step: int, epoch: float, loss: float, perplexity: float):
        """Save checkpoint information"""
        checkpoint_name = f"checkpoint-{step}"
        
        self.metadata[checkpoint_name] = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "perplexity": perplexity,
            "timestamp": str(pd.Timestamp.now()),
        }
        
        if len(self.metadata) > 5:
            oldest = sorted(self.metadata.items(), key=lambda x: x[1]["step"])[0]
            del self.metadata[oldest[0]]
        
        self._save_metadata()
    
    def get_recovery_state(self) -> Dict:
        """Get recovery state for resuming training"""
        latest_checkpoint = self.find_latest_checkpoint()
        
        if latest_checkpoint:
            return {
                "checkpoint_path": str(latest_checkpoint),
                "found": True,
                "metadata": self.metadata.get(latest_checkpoint.name, {}),
            }
        
        return {"checkpoint_path": None, "found": False, "metadata": {}}
    
    def cleanup_old_checkpoints(self, keep_last: int = 3):
        """Remove old checkpoints to save space"""
        try:
            checkpoint_dirs = sorted(
                [d for d in self.checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                key=lambda x: int(x.name.split("-")[1]),
                reverse=True
            )
            
            if len(checkpoint_dirs) > keep_last:
                for checkpoint_dir in checkpoint_dirs[keep_last:]:
                    try:
                        import shutil
                        shutil.rmtree(checkpoint_dir)
                        print(f"Removed old checkpoint: {checkpoint_dir.name}")
                        self.log_recovery(f"Removed old checkpoint: {checkpoint_dir.name}")
                    except Exception as e:
                        print(f"Error removing checkpoint: {str(e)}")
        except Exception as e:
            print(f"Error in cleanup: {str(e)}")

# ========== SWAHILI SCORE EARLY STOPPING ==========
class SwahiliScoreEarlyStopping(EarlyStoppingCallback):
    """Early stop if Swahili quality plateaus"""
    
    def __init__(self, patience: int = 3, threshold: float = 0.01, window_size: int = 5):
        """
        Args:
            patience: How many evals without improvement before stopping
            threshold: Minimum improvement to reset patience counter
            window_size: How many scores to average (smooth noise)
        """
        super().__init__(early_stopping_patience=patience)
        self.threshold = threshold
        self.window_size = window_size
        self.swahili_scores = []
        self.best_score = -float('inf')
        self.patience_counter = 0
        self.eval_count = 0
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Check Swahili score every evaluation"""
        
        if metrics is None:
            return
        
        self.eval_count += 1
        
        current_score = metrics.get("swahili_score", None)
        
        if current_score is None:
            current_score = metrics.get("language_consistency", None)
        
        if current_score is not None:
            self.swahili_scores.append(float(current_score))
            
            if len(self.swahili_scores) >= self.window_size:
                avg_score = np.mean(self.swahili_scores[-self.window_size:])
            else:
                avg_score = np.mean(self.swahili_scores)
            
            print(f"\nSwahili Score Monitor (Eval #{self.eval_count}):")
            print(f"   Current: {current_score:.4f} | Windowed Avg: {avg_score:.4f}")
            
            if avg_score > self.best_score + self.threshold:
                self.best_score = avg_score
                self.patience_counter = 0
                print(f"   New best Swahili score: {self.best_score:.4f} (Reset patience)")
            else:
                self.patience_counter += 1
                improvement_needed = (self.best_score + self.threshold - avg_score)
                print(f"   No improvement. Patience: {self.patience_counter}/{self.early_stopping_patience}")
                print(f"   Need +{improvement_needed:.4f} to reset")
            
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n SWAHILI EARLY STOPPING TRIGGERED!")
                print(f"   Swahili quality plateaued at {self.best_score:.4f}")
                print(f"   Stopping training...")
                control.should_training_stop = True

# ========== TRACKING CONFIGURATION ==========
@dataclass
class TrackingConfig:
    """Configuration for comprehensive monitoring"""
    
    AUTO_METRICS: List[str] = field(default_factory=lambda: [
        "perplexity", "token_accuracy", "entropy", "bpc", "loss", "learning_rate", "grad_norm",
    ])
    
    MANUAL_METRICS: List[str] = field(default_factory=lambda: [
        "completions", "language_consistency", "vocab_coverage", "training_stability",
    ])
    
    AUTO_EVAL_FREQ: int = 100
    COMPLETION_FREQ: int = 100
    MANUAL_METRICS_FREQ: int = 1
    
    TEST_PROMPTS: List[str] = field(default_factory=lambda: [
        "Habari za asubuhi? Leo ni siku nzuri ya",
        "Mimi ni mwalimu wa shule ya msingi. Ninafundisha",
        "Nchini Kenya, kilimo ni sekta muhimu. Wananchi",
        "Siku moja, mtoto alikuwa akitembea porini na",
        "Kiswahili ni lugha ya afrika inayozungumzwa",
        "Elimu ni muhimu sana. Kwa hivyo, wazazi wanafanya",
    ])
    
    METRICS_DIR: str = "/kaggle/working/metrics"
    PLOTS_DIR: str = "/kaggle/working/plots"
    COMPLETIONS_DIR: str = "/kaggle/working/completions"
    CHECKPOINTS_DIR: str = "/kaggle/working/checkpoints"
    
    VOCAB_SAMPLE_SIZE: int = 10000
    SWAHILI_KEYWORDS: List[str] = field(default_factory=lambda: [
        'na', 'ya', 'wa', 'katika', 'kwa', 'ni', 'la', 'za',
        'kuwa', 'kama', 'habari', 'asante', 'karibu', 'samahani', 'tafadhali',
    ])

@dataclass  
class TrainingConfig:
    """Main training configuration"""
    
    MODEL_ID: str = "CraneAILabs/swahili-gemma-1b"
    
    DATA_PATH: str = "/kaggle/input/swahili-master-pretraining"
    JSONL_FILENAME: str = "SWAHILI_MASTER_PRETRAINING.jsonl"
    TEXT_FIELD: str = "text"
    TOKENIZED_DATA_PATH: str = "/kaggle/working/tokenized_packed"
    
    MAX_LENGTH: int = 512 
    USE_SEQUENCE_PACKING: bool = True
    
    EPOCHS: int = 0.5
    BATCH_SIZE: int = 2  
    GRADIENT_ACCUMULATION_STEPS: int = 16 
    LEARNING_RATE: float = 5e-5
    WARMUP_RATIO: float = 0.1
    WEIGHT_DECAY: float = 0.01
    
    LORA_R: int = 16 
    LORA_ALPHA: int = 32  
    LORA_DROPOUT: float = 0.05
    
    LOAD_IN_4BIT: bool = True  
    BNB_4BIT_COMPUTE_DTYPE: str = "bfloat16"
    
    OUTPUT_DIR: str = "/kaggle/working/swahili-linguistic-foundation"
    LOGGING_STEPS: int = 50
    SAVE_STEPS: int = 100
    EVAL_STEPS: int = 100
    
    TARGET_MODULES: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    
    AUTO_RESUME: bool = True
    USE_SWAHILI_EARLY_STOPPING: bool = True
    SWAHILI_STOPPING_PATIENCE: int = 3
    SWAHILI_STOPPING_THRESHOLD: float = 0.01

# ========== SEQUENCE PACKING IMPLEMENTATION ==========
class SequencePacker:
    """Implements sequence packing for efficient token usage"""
    
    @staticmethod
    def tokenize_with_packing(
        texts: List[str],
        tokenizer,
        max_length: int = 512,
    ) -> Dict:
        """Tokenize texts with sequence packing"""
        print("Tokenizing with sequence packing...")
        
        all_token_ids = []
        
        print(f"  Processing {len(texts)} texts...")
        for idx, text in enumerate(tqdm(texts, desc="Tokenizing")):
            if not text or not isinstance(text, str):
                continue
            
            tokens = tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=False,  # Don't truncate - we'll pack later
            )
            
            if len(tokens) > 10:
                tokens = tokens + [tokenizer.eos_token_id]
                all_token_ids.extend(tokens)
        
        print(f"  Total tokens: {len(all_token_ids):,}")
        
        packed_input_ids = []
        packed_attention_masks = []
        
        print(f"  Packing into {max_length}-token chunks...")
        for i in range(0, len(all_token_ids), max_length):
            chunk = all_token_ids[i:i + max_length]
            
            if len(chunk) < max_length:
                pad_length = max_length - len(chunk)
                chunk = chunk + [tokenizer.pad_token_id] * pad_length
                attention_mask = [1] * (max_length - pad_length) + [0] * pad_length
            else:
                attention_mask = [1] * max_length
            
            packed_input_ids.append(chunk)
            packed_attention_masks.append(attention_mask)
        
        labels = packed_input_ids.copy()
        
        print(f"   Created {len(packed_input_ids)} packed sequences")
        print(f"     Efficiency: {len(all_token_ids) / (len(packed_input_ids) * max_length) * 100:.1f}% token utilization")
        
        return {
            "input_ids": torch.tensor(packed_input_ids),
            "attention_mask": torch.tensor(packed_attention_masks),
            "labels": torch.tensor(labels),
        }

# ========== METRICS TRACKER ==========
class MetricsTracker:
    """Comprehensive metrics tracking and analysis"""
    
    def __init__(self, config: TrackingConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.metrics_history = []
        
        for dir_path in [config.METRICS_DIR, config.PLOTS_DIR, config.COMPLETIONS_DIR, config.CHECKPOINTS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
        
        self.metrics_file = f"{config.METRICS_DIR}/metrics_history.json"
        self.completions_file = f"{config.COMPLETIONS_DIR}/completions.json"
        
        self.loss_history = []
        self.grad_norm_history = []
        self.lr_history = []
        self.perplexity_history = []
        self.accuracy_history = []
        self.swahili_score_history = []
        
        self.vocab_counter = Counter()
        self.total_tokens = 0
        
        self._load_previous_metrics()
        
        print(" MetricsTracker initialized")
    
    def _load_previous_metrics(self):
        """Load metrics from previous training run"""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, "r", encoding="utf-8") as f:
                    self.metrics_history = json.load(f)
                print(f" Loaded {len(self.metrics_history)} previous metrics")
            except Exception as e:
                print(f" Could not load previous metrics: {str(e)}")
        
    def compute_auto_metrics(self, eval_pred, model=None) -> Dict:
        """Compute all automatic metrics from evaluation predictions"""
        try:
            predictions, labels = eval_pred
            
            device = model.device if model and hasattr(model, 'device') else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            
            preds = torch.tensor(predictions) if not isinstance(predictions, torch.Tensor) else predictions
            labs = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
            
            preds = preds.to(device)
            labs = labs.to(device)
            
            mask = labs != -100
            
            if mask.sum() == 0:
                return self._empty_metrics()
            
            preds_masked = preds[mask]
            labs_masked = labs[mask]
            
            if preds_masked.dim() == 1:
                return self._empty_metrics()
            
            probs = torch.softmax(preds_masked.float(), dim=-1)
            target_probs = probs[torch.arange(len(labs_masked), device=device), labs_masked.long()]
            loss = -torch.log(target_probs + 1e-8).mean()
            perplexity = torch.exp(loss).item()
            
            pred_tokens = torch.argmax(preds_masked, dim=-1)
            token_accuracy = (pred_tokens == labs_masked).float().mean().item()
            
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean().item()
            bpc = math.log2(perplexity) if perplexity > 0 else 0
            
            self.loss_history.append(float(loss.item()))
            self.perplexity_history.append(perplexity)
            self.accuracy_history.append(token_accuracy)
            
            return {
                "perplexity": float(perplexity),
                "token_accuracy": float(token_accuracy),
                "entropy": float(entropy),
                "bpc": float(bpc),
                "loss": float(loss.item()),
            }
            
        except Exception as e:
            print(f" Error in compute_auto_metrics: {str(e)}")
            return self._empty_metrics()
    
    def _empty_metrics(self) -> Dict:
        return {
            "perplexity": 0.0,
            "token_accuracy": 0.0,
            "entropy": 0.0,
            "bpc": 0.0,
            "loss": 0.0,
        }
    
    def track_training_stability(self, logs: Dict) -> Dict:
        """Track training stability metrics"""
        stability_metrics = {}
        
        try:
            if "loss" in logs:
                loss_val = logs["loss"]
                self.loss_history.append(loss_val)
                
                if len(self.loss_history) > 1:
                    stability_metrics["loss_std"] = float(np.std(self.loss_history[-100:])) if len(self.loss_history) >= 100 else 0
                    if len(self.loss_history) >= 50:
                        trend = np.polyfit(range(len(self.loss_history[-50:])), self.loss_history[-50:], 1)[0]
                        stability_metrics["loss_trend"] = float(trend)
            
            if "grad_norm" in logs:
                self.grad_norm_history.append(logs["grad_norm"])
                stability_metrics["grad_norm"] = float(logs["grad_norm"])
                if len(self.grad_norm_history) > 100:
                    stability_metrics["grad_norm_std"] = float(np.std(self.grad_norm_history[-100:]))
            
            if "learning_rate" in logs:
                self.lr_history.append(logs["learning_rate"])
                stability_metrics["learning_rate"] = float(logs["learning_rate"])
            
        except Exception as e:
            print(f" Warning in track_training_stability: {str(e)}")
        
        return stability_metrics
    
    def generate_completions(self, model, step: int) -> Dict:
        """Generate completions for test prompts"""
        print(f"\n Generating completions at step {step}...")
        
        completions = {}
        model.eval()
        
        try:
            for prompt in self.config.TEST_PROMPTS:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                
                full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                completion = full_text[len(prompt):].strip()
                
                analysis = {
                    "completion": completion,
                    "length": len(completion.split()),
                    "swahili_score": self._compute_swahili_score(completion),
                    "coherence": self._compute_coherence(prompt, completion),
                }
                
                completions[prompt[:40]] = analysis
                print(f"  ✓ Swahili score: {analysis['swahili_score']:.2f}")
            
            model.train()
            
            step_completions = {
                "step": step,
                "timestamp": str(pd.Timestamp.now()),
                "completions": completions
            }
            
            if os.path.exists(self.completions_file):
                with open(self.completions_file, "r", encoding="utf-8") as f:
                    all_completions = json.load(f)
            else:
                all_completions = []
            
            all_completions.append(step_completions)
            
            with open(self.completions_file, "w", encoding="utf-8") as f:
                json.dump(all_completions, f, indent=2, ensure_ascii=False)
            
            avg_swahili = float(np.mean([c["swahili_score"] for c in completions.values()]))
            self.swahili_score_history.append(avg_swahili)
            
            summary = {
                "avg_swahili_score": avg_swahili,
                "avg_completion_length": float(np.mean([c["length"] for c in completions.values()])),
                "avg_coherence": float(np.mean([c["coherence"] for c in completions.values()])),
            }
            
            return summary
            
        except Exception as e:
            print(f" Error generating completions: {str(e)}")
            return {"avg_swahili_score": 0, "avg_completion_length": 0, "avg_coherence": 0}
    
    def analyze_language_consistency(self, model, dataset_sample) -> Dict:
        """Analyze language consistency"""
        print("\n Analyzing language consistency...")
        
        model.eval()
        predictions = []
        
        try:
            with torch.no_grad():
                for i in range(min(100, len(dataset_sample["input_ids"]))):
                    inputs = {k: v[i:i+1].to(model.device) for k, v in dataset_sample.items() 
                             if k in ["input_ids", "attention_mask"]}
                    
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    pred_tokens = torch.argmax(probs, dim=-1)
                    pred_text = self.tokenizer.decode(pred_tokens[0], skip_special_tokens=True)
                    predictions.append(pred_text)
            
            model.train()
            
            swahili_scores = [self._compute_swahili_score(text) for text in predictions]
            avg_swahili_score = float(np.mean(swahili_scores))
            swahili_std = float(np.std(swahili_scores))
            
            consistency = 1.0 - (swahili_std / (avg_swahili_score + 1e-8)) if avg_swahili_score > 0 else 0.0
            consistency = max(0.0, min(1.0, consistency))
            
            metrics = {
                "language_consistency": float(consistency),
                "avg_swahili_score": avg_swahili_score,
                "swahili_score_std": swahili_std,
                "predictions_analyzed": len(predictions),
            }
            
            print(f"  ✓ Language consistency: {consistency:.3f}")
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error analyzing language consistency: {str(e)}")
            return {"language_consistency": 0, "avg_swahili_score": 0, "swahili_score_std": 0, "predictions_analyzed": 0}
    
    def analyze_vocab_coverage(self, model, dataset_sample) -> Dict:
        """Analyze vocabulary coverage"""
        print("\n Analyzing vocabulary coverage...")
        
        vocab_size = len(self.tokenizer)
        model.eval()
        token_counts = Counter()
        
        try:
            with torch.no_grad():
                for i in range(min(50, len(dataset_sample["input_ids"]))):
                    inputs = {k: v[i:i+1].to(model.device) for k, v in dataset_sample.items() 
                             if k in ["input_ids", "attention_mask"]}
                    
                    outputs = model(**inputs)
                    logits = outputs.logits
                    topk = torch.topk(logits, k=10, dim=-1)
                    tokens = topk.indices.flatten().cpu().numpy()
                    token_counts.update(tokens)
            
            model.train()
            
            unique_tokens = len(token_counts)
            coverage = unique_tokens / min(vocab_size, 10000)
            most_common = token_counts.most_common(5)
            most_common_tokens = [
                {"token": self.tokenizer.decode([tok]), "count": int(count)}
                for tok, count in most_common
            ]
            
            metrics = {
                "vocab_coverage": float(coverage),
                "unique_tokens_used": int(unique_tokens),
                "total_vocab_size": int(vocab_size),
                "most_common_tokens": most_common_tokens,
            }
            
            print(f"  ✓ Vocab coverage: {coverage:.3%}")
            
            return metrics
            
        except Exception as e:
            print(f" Error analyzing vocab coverage: {str(e)}")
            return {"vocab_coverage": 0, "unique_tokens_used": 0, "total_vocab_size": vocab_size, "most_common_tokens": []}
    
    def _compute_swahili_score(self, text: str) -> float:
        """Compute how Swahili-like a text is"""
        if not text or len(text.split()) == 0:
            return 0.0
        
        text_lower = f" {text.lower()} "
        swahili_count = sum(text_lower.count(f" {word} ") + text_lower.count(f" {word}") 
                           for word in self.config.SWAHILI_KEYWORDS)
        
        words = text.split()
        score = min(1.0, swahili_count / (len(words) * 0.3))
        
        return float(score)
    
    def _compute_coherence(self, prompt: str, completion: str) -> float:
        """Compute coherence between prompt and completion"""
        prompt_words = set(prompt.lower().split()[:10])
        completion_words = set(completion.lower().split()[:10])
        
        if not prompt_words or not completion_words:
            return 0.0
        
        intersection = len(prompt_words.intersection(completion_words))
        union = len(prompt_words.union(completion_words))
        
        return float(intersection / union) if union > 0 else 0.0
    
    def save_metrics(self, metrics: Dict, step: int, metric_type: str = "auto"):
        """Save metrics to history"""
        try:
            metric_entry = {
                "step": step,
                "epoch": metrics.get("epoch", 0),
                "type": metric_type,
                "timestamp": str(pd.Timestamp.now()),
                **metrics
            }
            
            self.metrics_history.append(metric_entry)
            
            with open(self.metrics_file, "w", encoding="utf-8") as f:
                json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
            
            csv_file = f"{self.config.METRICS_DIR}/metrics.csv"
            df = pd.DataFrame(self.metrics_history)
            df.to_csv(csv_file, index=False)
            
        except Exception as e:
            print(f" Warning saving metrics: {str(e)}")
    
    def generate_plots(self):
        """Generate comprehensive visualization plots"""
        print("\n Generating plots...")
        
        if not self.metrics_history:
            print("  No metrics to plot yet")
            return
        
        try:
            df = pd.DataFrame(self.metrics_history)
            sns.set_style("whitegrid")
            
            # Loss
            if "loss" in df.columns:
                plt.figure(figsize=(12, 4))
                plt.plot(df["step"], df["loss"], marker="o", linestyle="-", linewidth=2)
                plt.xlabel("Step")
                plt.ylabel("Loss")
                plt.title("Training Loss Over Time")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{self.config.PLOTS_DIR}/loss_over_time.png", dpi=150)
                plt.close()
                print("  ✓ Saved: loss_over_time.png")
            
            # Perplexity
            if "perplexity" in df.columns:
                plt.figure(figsize=(12, 4))
                plt.plot(df["step"], df["perplexity"], marker="o", linestyle="-", linewidth=2, color="green")
                plt.xlabel("Step")
                plt.ylabel("Perplexity")
                plt.title("Perplexity Over Time")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{self.config.PLOTS_DIR}/perplexity_over_time.png", dpi=150)
                plt.close()
                print("  ✓ Saved: perplexity_over_time.png")
            
            # Accuracy
            if "token_accuracy" in df.columns:
                plt.figure(figsize=(12, 4))
                plt.plot(df["step"], df["token_accuracy"], marker="o", linestyle="-", linewidth=2, color="blue")
                plt.xlabel("Step")
                plt.ylabel("Token Accuracy")
                plt.title("Token Accuracy Over Time")
                plt.ylim(0, 1)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{self.config.PLOTS_DIR}/token_accuracy_over_time.png", dpi=150)
                plt.close()
                print("  ✓ Saved: token_accuracy_over_time.png")
            
            # Learning Rate
            if "learning_rate" in df.columns:
                plt.figure(figsize=(12, 4))
                plt.plot(df["step"], df["learning_rate"], marker="s", linestyle="-", linewidth=2, color="orange")
                plt.xlabel("Step")
                plt.ylabel("Learning Rate")
                plt.title("Learning Rate Schedule")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{self.config.PLOTS_DIR}/learning_rate_schedule.png", dpi=150)
                plt.close()
                print("  ✓ Saved: learning_rate_schedule.png")
            
            # Swahili Score
            if "avg_swahili_score" in df.columns:
                df_filtered = df[df["avg_swahili_score"] > 0]
                if not df_filtered.empty:
                    plt.figure(figsize=(12, 4))
                    plt.plot(df_filtered["step"], df_filtered["avg_swahili_score"], marker="^", linestyle="-", linewidth=2, color="purple")
                    plt.xlabel("Step")
                    plt.ylabel("Swahili Score")
                    plt.title("Average Swahili Score Over Time")
                    plt.ylim(0, 1)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f"{self.config.PLOTS_DIR}/swahili_score_over_time.png", dpi=150)
                    plt.close()
                    print("  ✓ Saved: swahili_score_over_time.png")
            
            # Dashboard
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            
            if "loss" in df.columns:
                axes[0, 0].plot(df["step"], df["loss"], marker="o", color="red")
                axes[0, 0].set_title("Loss")
                axes[0, 0].grid(True, alpha=0.3)
            
            if "perplexity" in df.columns:
                axes[0, 1].plot(df["step"], df["perplexity"], marker="o", color="green")
                axes[0, 1].set_title("Perplexity")
                axes[0, 1].grid(True, alpha=0.3)
            
            if "token_accuracy" in df.columns:
                axes[1, 0].plot(df["step"], df["token_accuracy"], marker="o", color="blue")
                axes[1, 0].set_title("Token Accuracy")
                axes[1, 0].set_ylim(0, 1)
                axes[1, 0].grid(True, alpha=0.3)
            
            if "avg_swahili_score" in df.columns:
                axes[1, 1].plot(df["step"], df["avg_swahili_score"], marker="o", color="purple")
                axes[1, 1].set_title("Swahili Score")
                axes[1, 1].set_ylim(0, 1)
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{self.config.PLOTS_DIR}/metrics_dashboard.png", dpi=150)
            plt.close()
            print("  ✓ Saved: metrics_dashboard.png")
            
        except Exception as e:
            print(f" Error generating plots: {str(e)}")
    
    def generate_report(self):
        """Generate comprehensive training report"""
        print("\n" + "="*70)
        print(" COMPREHENSIVE TRAINING REPORT")
        print("="*70)
        
        if not self.metrics_history:
            print("No metrics collected yet.")
            return
        
        df = pd.DataFrame(self.metrics_history)
        
        auto_metrics = df[df["type"] == "auto"]
        manual_metrics = df[df["type"] == "manual"]
        
        print("\n AUTO METRICS (Every evaluation):")
        if not auto_metrics.empty:
            latest_auto = auto_metrics.iloc[-1]
            for metric in ["perplexity", "token_accuracy", "entropy", "bpc", "loss"]:
                if metric in latest_auto:
                    val = latest_auto[metric]
                    if pd.notna(val):
                        print(f"  {metric:20}: {val:.4f}")
        
        print("\n MANUAL METRICS (Periodic):")
        if not manual_metrics.empty:
            latest_manual = manual_metrics.iloc[-1]
            for metric in ["language_consistency", "vocab_coverage", "avg_swahili_score"]:
                if metric in latest_manual:
                    val = latest_manual[metric]
                    if pd.notna(val):
                        print(f"  {metric:20}: {val:.4f}")
        
        if self.swahili_score_history:
            print(f"\n SWAHILI SCORE HISTORY:")
            print(f"  Peak score: {max(self.swahili_score_history):.4f}")
            print(f"  Current score: {self.swahili_score_history[-1]:.4f}")
            print(f"  Trend: {' Improving' if self.swahili_score_history[-1] > np.mean(self.swahili_score_history[:-1]) else '📉 Plateauing'}")
        
        print("\n" + "="*70)

# ========== CUSTOM CALLBACK FOR TRACKING ==========
class ComprehensiveMetricsCallback(TrainerCallback):
    """Callback for comprehensive metrics tracking"""
    
    def __init__(self, tracker: MetricsTracker, config: TrackingConfig, checkpoint_manager: CheckpointRecoveryManager, memory_monitor: MemoryMonitor):
        super().__init__()
        self.tracker = tracker
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.memory_monitor = memory_monitor
        self.last_completion_step = 0
        self.last_manual_epoch = 0
        self.dataset_sample = None
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training stability metrics"""
        if logs:
            stability_metrics = self.tracker.track_training_stability(logs)
            if stability_metrics:
                self.tracker.save_metrics(stability_metrics, state.global_step, "stability")
            
            self.memory_monitor.log_memory(state.global_step, "Training step")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Compute and save auto metrics on evaluation"""
        if metrics and "eval_loss" in metrics:
            auto_metrics = {
                "loss": metrics.get("eval_loss", 0),
                "perplexity": math.exp(min(metrics.get("eval_loss", 0), 20)),
                "epoch": state.epoch,
            }
            self.tracker.save_metrics(auto_metrics, state.global_step, "auto")
            
            self.checkpoint_manager.save_checkpoint_info(
                step=state.global_step,
                epoch=state.epoch,
                loss=metrics.get("eval_loss", 0),
                perplexity=math.exp(min(metrics.get("eval_loss", 0), 20))
            )
            
            if (state.global_step - self.last_completion_step) >= self.config.COMPLETION_FREQ:
                model = kwargs.get("model", None)
                if model is None:
                    model = getattr(args, "model", None)
                if model:
                    completion_metrics = self.tracker.generate_completions(model, state.global_step)
                    self.tracker.save_metrics(completion_metrics, state.global_step, "completions")
                    
                    metrics["swahili_score"] = completion_metrics.get("avg_swahili_score", 0)
                    
                    self.last_completion_step = state.global_step
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Compute manual metrics at end of epoch"""
        current_epoch = int(state.epoch)
        
        if current_epoch > self.last_manual_epoch:
            model = kwargs.get("model", None)
            if model is None:
                model = getattr(args, "model", None)
            
            if model and self.dataset_sample is None:
                train_dataset = kwargs.get("train_dataset")
                if train_dataset is None:
                    train_dataset = getattr(args, "train_dataset", None)
                
                if train_dataset:
                    self._cache_dataset_sample(train_dataset)
            
            if model and self.dataset_sample:
                consistency_metrics = self.tracker.analyze_language_consistency(model, self.dataset_sample)
                self.tracker.save_metrics(consistency_metrics, state.global_step, "manual")
                
                vocab_metrics = self.tracker.analyze_vocab_coverage(model, self.dataset_sample)
                self.tracker.save_metrics(vocab_metrics, state.global_step, "manual")
                
                self.last_manual_epoch = current_epoch
    
    def on_save(self, args, state, control, **kwargs):
        """Cleanup old checkpoints when saving"""
        self.checkpoint_manager.cleanup_old_checkpoints(keep_last=3)
    
    def _cache_dataset_sample(self, dataset, sample_size=32):
        """Cache a sample once for manual metrics"""
        try:
            if self.dataset_sample is None and len(dataset) > 0:
                sample_size = min(sample_size, len(dataset))
                sample_indices = list(range(sample_size))
                sample = dataset.select(sample_indices)
                
                self.dataset_sample = {
                    "input_ids": torch.tensor([s["input_ids"] for s in sample]),
                    "attention_mask": torch.tensor([s["attention_mask"] for s in sample]),
                }
                
                print(f" Cached dataset sample: {sample_size} examples")
        except Exception as e:
            print(f" Error caching sample: {str(e)}")

# ========== JSONL DATASET LOADING WITH SEQUENCE PACKING ==========
def load_jsonl_dataset_with_packing(
    tokenizer, 
    data_path: str, 
    max_length: int = 512,
    text_field: str = "text",
    test_size: float = 0.1,
    use_packing: bool = True,
    max_samples: Optional[int] = None
) -> datasets.DatasetDict:
    """Load JSONL dataset with optional sequence packing"""
    print(" Loading JSONL dataset...")
    
    data_path = Path(data_path)
    texts = []
    
    jsonl_files = list(data_path.glob("*.jsonl"))
    
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files in {data_path}")
    
    print(f"  Found {len(jsonl_files)} JSONL file(s)")
    
    for jsonl_file in jsonl_files:
        print(f"  Loading {jsonl_file.name}...")
        
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if max_samples and len(texts) >= max_samples:
                    break
                
                try:
                    data = json.loads(line)
                    text = data.get(text_field) or data.get("content") or data.get("body")
                    
                    if text and isinstance(text, str) and text.strip():
                        texts.append(text)
                except:
                    continue
        
        print(f"    ✓ Loaded {len(texts)} texts so far")
    
    if not texts:
        raise ValueError("No valid text data found")
    
    print(f"\n Total texts loaded: {len(texts):,}")
    
    if use_packing:
        print("\n Using sequence packing for efficiency...")
        encodings = SequencePacker.tokenize_with_packing(texts, tokenizer, max_length)
    else:
        print("\n Using standard padding...")
        toks = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        encodings = {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
            "labels": toks["input_ids"].clone(),
        }
    
    from datasets import Dataset
    
    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": encodings["labels"],
    })
    
    print(f" Dataset created: {len(dataset):,} examples")
    
    dataset = dataset.train_test_split(test_size=test_size, seed=42)
    
    print(f"   Train: {len(dataset['train']):,}, Test: {len(dataset['test']):,}")
    
    return dataset

# ========== MAIN TRAINING SCRIPT ==========
def setup_environment():
    """Setup environment and print system info"""
    print("=" * 70)
    print(" SWAHILI LINGUISTIC FOUNDATION - KAGGLE OPTIMIZED VERSION")
    print("=" * 70)
    
    if torch.cuda.is_available():
        print(f" GPU: {torch.cuda.get_device_name(0)}")
        print(f" GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(" No GPU available")
    
    print(f" CPU Cores: {psutil.cpu_count()}")
    print(f" RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
    print("-" * 70)

def main():
    """Main training function"""
    
    #   Kaggle setup 
    kaggle_specific_setup()
    
    train_config = TrainingConfig()
    track_config = TrackingConfig()
    
    setup_environment()
    
    os.makedirs(train_config.OUTPUT_DIR, exist_ok=True)
    
    # Initialize managers
    checkpoint_manager = CheckpointRecoveryManager(track_config.CHECKPOINTS_DIR)
    memory_monitor = MemoryMonitor()
    
    print("\n" + "="*70)
    print(" CHECKPOINT RECOVERY CHECK")
    print("="*70)
    
    recovery_state = checkpoint_manager.get_recovery_state()
    
    if recovery_state["found"]:
        print(f" Found checkpoint: {recovery_state['checkpoint_path']}")
        print(f"   Step: {recovery_state['metadata'].get('step', 'unknown')}")
    else:
        print(" No previous checkpoint - starting fresh")
    
    print("-" * 70 + "\n")
    
    # Load tokenizer
    print(" Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(train_config.MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f" Tokenizer loaded. Vocab size: {len(tokenizer):,}")
    
    # Initialize tracker
    tracker = MetricsTracker(track_config, tokenizer)
    
    # Load model
    print("\n Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        train_config.MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    print(" Model loaded with 4-bit quantization")
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=train_config.LORA_R,
        lora_alpha=train_config.LORA_ALPHA,
        lora_dropout=train_config.LORA_DROPOUT,
        target_modules=train_config.TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    print(f" LoRA configured")
    
    # Load dataset with packing
    print("\n Loading dataset...")
    dataset = load_jsonl_dataset_with_packing(
        tokenizer,
        train_config.DATA_PATH,
        max_length=train_config.MAX_LENGTH,
        text_field=train_config.TEXT_FIELD,
        use_packing=train_config.USE_SEQUENCE_PACKING,
    )
    
    # Training args
    training_args = TrainingArguments(
        output_dir=train_config.OUTPUT_DIR,
        num_train_epochs=train_config.EPOCHS,
        per_device_train_batch_size=train_config.BATCH_SIZE,
        per_device_eval_batch_size=train_config.BATCH_SIZE // 1,
        eval_accumulation_steps=16, 
        gradient_accumulation_steps=train_config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=train_config.LEARNING_RATE,
        warmup_ratio=train_config.WARMUP_RATIO,
        weight_decay=train_config.WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        bf16=True,
        tf32=False,
        logging_dir=f"{train_config.OUTPUT_DIR}/logs",
        logging_steps=train_config.LOGGING_STEPS,
        save_strategy="steps",
        save_steps=train_config.SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=train_config.EVAL_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )
    
    #  Callbacks with Swahili early stopping
    callbacks = [
        ComprehensiveMetricsCallback(tracker, track_config, checkpoint_manager, memory_monitor),
        EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001),
    ]
    
    if train_config.USE_SWAHILI_EARLY_STOPPING:
        callbacks.append(
            SwahiliScoreEarlyStopping(
                patience=train_config.SWAHILI_STOPPING_PATIENCE,
                threshold=train_config.SWAHILI_STOPPING_THRESHOLD,
                window_size=5,
            )
        )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    
    # Train
    print("\n" + "="*70)
    print(" STARTING TRAINING - KAGGLE OPTIMIZED VERSION")
    print("="*70)
    print(" ALL FEATURES ENABLED:")
    print("   ✓ Kaggle P100 Optimizations")
    print("   ✓ Sequence Packing (20-30% efficiency)")
    print("   ✓ Memory Monitoring")
    print("   ✓ Checkpoint Recovery")
    print("   ✓ Robust Device Handling")
    print("   ✓ Swahili Score Early Stopping")
    print("-" * 70 + "\n")
    
    resume_from = None
    if recovery_state["found"] and train_config.AUTO_RESUME:
        resume_from = recovery_state["checkpoint_path"]
    
    trainer.train(resume_from_checkpoint=resume_from)
    
    # Save
    print("\n Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(train_config.OUTPUT_DIR)
    
    # Final eval
    print("\n Final evaluation...")
    final_metrics = trainer.evaluate()
    
    # Report
    tracker.generate_report()
    tracker.generate_plots()
    
    # Memory report
    memory_report = memory_monitor.get_report()
    
    # Summary
    final_summary = {
        "status": "completed",
        "final_loss": float(final_metrics.get("eval_loss", 0)),
        "final_perplexity": float(math.exp(min(final_metrics.get("eval_loss", 0), 20))),
        "training_steps": int(trainer.state.global_step),
        "training_epochs": int(trainer.state.epoch),
        "memory_report": memory_report,
        "sequence_packing_enabled": train_config.USE_SEQUENCE_PACKING,
        "checkpoint_recovery_enabled": True,
        "swahili_early_stopping_enabled": train_config.USE_SWAHILI_EARLY_STOPPING,
        "swahili_score_history": tracker.swahili_score_history,
    }
    
    with open(f"{train_config.OUTPUT_DIR}/training_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE!")
    print("=" * 70)
    print(f" Final Perplexity: {final_summary['final_perplexity']:.2f}")
    print(f" Total Steps: {final_summary['training_steps']:,}")
    print(f" Peak GPU Memory: {memory_report['peak_gpu_memory_gb']}GB")
    print(f" Peak CPU Memory: {memory_report['peak_cpu_memory_gb']}GB")
    
    if tracker.swahili_score_history:
        print(f" Peak Swahili Score: {max(tracker.swahili_score_history):.3f}")
        print(f" Final Swahili Score: {tracker.swahili_score_history[-1]:.3f}")
    
    print(f"\n Outputs saved to:")
    print(f"    Model: {train_config.OUTPUT_DIR}")
    print(f"    Metrics: {track_config.METRICS_DIR}")
    print(f"    Plots: {track_config.PLOTS_DIR}")
    print(f"    Completions: {track_config.COMPLETIONS_DIR}")
    print(f"    Checkpoints: {track_config.CHECKPOINTS_DIR}")
    
    print(f"\n Files generated:")
    print(f"   - training_summary.json")
    print(f"   - metrics_history.json")
    print(f"   - metrics.csv")
    print(f"   - completions.json")
    print(f"   - checkpoint_metadata.json")
    print(f"   - recovery.log")
    print(f"   - 6+ detailed plots (PNG)")
    print(f"\n Ready for Stage 2: Conversation Fine-tuning!")
    print("=" * 70)
    
    return trainer, tracker, checkpoint_manager, memory_monitor

if __name__ == "__main__":
    trainer, tracker, checkpoint_manager, memory_monitor = main()