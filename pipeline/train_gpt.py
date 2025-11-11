#!/usr/bin/env python3
"""
General-purpose GPT Fine-tuning

This script either pre-trains or fine-tunes a GPT model on any text corpus using an existing
tokenizer and model. Data is assumed to be already properly chunked.

Model Architecture Parameters (for creating new models from scratch):
- n_embd: Embedding dimension (width of the model, default: 1024)
- n_layer: Number of transformer layers (depth of the model, default: 16)  
- n_head: Number of attention heads per layer (default: 16)
- n_ctx: Context length - maximum number of tokens the model can process at once (default: 1024)
- n_positions: Maximum position embeddings, usually same as n_ctx (default: same as n_ctx)

Note: These parameters are only used when creating a new model. If loading a pre-trained
model with --model_path, the architecture is determined by the existing model.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
import numpy as np
from typing import List, Dict
import logging
import os
from pathlib import Path
import argparse
import warnings
from contextlib import nullcontext

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS"] = "1"
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.recompile_limit = 16

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """Dataset for text data."""
    
    def __init__(self, texts: List[str]):
        self.texts = texts
        logger.info(f"Dataset created with {len(self.texts)} texts")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {"text": self.texts[idx]}

class DataCollator:
    """Tokenizes text on-the-fly."""
    
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, examples):
        texts = [example["text"] for example in examples]
        tokenized = self.tokenizer(
            texts, max_length=self.max_length, padding=True, 
            truncation=True, add_special_tokens=False, return_tensors="pt"
        )
        
        labels = tokenized["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels
        }

class GPTTrainer:
    """Main training class."""
    
    def __init__(self, data_file: str, tokenizer_path: str, model_path: str = None, output_dir: str = "training_output",
                 compile_model: bool = False, model_config: dict = None):
        self.data_file = data_file
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.compile_model = compile_model
        self.model_config = model_config or {}
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading tokenizer from: {self.tokenizer_path}")
        if self.model_path:
            logger.info(f"Loading pre-trained model from: {self.model_path}")
        else:
            logger.info("Will create new model from scratch")
        
        self.tokenizer = None
        self.model = None
        self.use_autocast = self.model_config.get('use_autocast', False)
    
    def load_data(self) -> List[str]:
        """Load texts from JSON file."""
        logger.info(f"Loading data from {self.data_file}")
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = [item["text"] for item in data]
        logger.info(f"Loaded {len(texts)} texts")
        return texts
    
    def prepare_data(self, texts: List[str], max_length: int, min_length: int = 64):
        """Load tokenizer and prepare dataset."""
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.tokenizer_path)
        logger.info(f"Loaded tokenizer with {len(self.tokenizer)} tokens")
        
        training_chunks = []
        for text in texts:
            if len(text) <= max_length:
                if len(text) >= min_length:
                    training_chunks.append(text)
            else:
                for i in range(0, len(text), max_length):
                    chunk = text[i:i + max_length]
                    if len(chunk) >= min_length:
                        training_chunks.append(chunk)
        
        logger.info(f"Original texts: {len(texts)}, after chunking and filtering (min_length={min_length}): {len(training_chunks)}")
        
        dataset = TextDataset(training_chunks)
        return dataset
    
    def initialize_model(self):
        """Load or create the GPT model."""
        attn_implementation = self.model_config.get('attn_implementation', 'eager')
        
        torch_dtype = None
        if 'flash_attention' in attn_implementation:
            if torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16
        
        if self.model_path:
            try:
                model_kwargs = {
                    'attn_implementation': attn_implementation
                }
                if torch_dtype is not None:
                    model_kwargs['torch_dtype'] = torch_dtype
                    
                self.model = GPT2LMHeadModel.from_pretrained(
                    self.model_path, 
                    **model_kwargs
                ).to(self.device)
                logger.info(f"Loaded pre-trained model with {sum(p.numel() for p in self.model.parameters())} parameters")
                logger.info(f"Using attention implementation: {attn_implementation}")
                if torch_dtype is not None:
                    logger.info(f"Specified torch_dtype: {torch_dtype}")
                logger.info(f"Model dtype: {self.model.dtype}, autocast: {self.use_autocast}")
            except Exception as e:
                logger.warning(f"Failed to load model with {attn_implementation}: {e}")
                logger.info("Falling back to default attention implementation")
                self.model = GPT2LMHeadModel.from_pretrained(self.model_path).to(self.device)
                logger.info(f"Loaded pre-trained model with {sum(p.numel() for p in self.model.parameters())} parameters")
        else:
            n_embd = self.model_config.get('n_embd', 1024)
            n_layer = self.model_config.get('n_layer', 16)
            n_head = self.model_config.get('n_head', 16)
            n_ctx = self.model_config.get('n_ctx', 1024)
            n_positions = self.model_config.get('n_positions', n_ctx)
            
            config = GPT2Config(
                vocab_size=len(self.tokenizer),
                n_positions=n_positions,
                n_ctx=n_ctx,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                attn_implementation=attn_implementation
            )
            try:
                if torch_dtype is not None:
                    config.torch_dtype = torch_dtype
                self.model = GPT2LMHeadModel(config).to(self.device)
                logger.info(f"Created new model with {sum(p.numel() for p in self.model.parameters())} parameters")
                logger.info(f"Model config: {n_layer} layers, {n_embd} embedding dim, {n_head} heads, {n_ctx} context length")
                logger.info(f"Using attention implementation: {attn_implementation}")
                if torch_dtype is not None:
                    logger.info(f"Specified torch_dtype: {torch_dtype}")
                logger.info(f"Model dtype: {self.model.dtype}, autocast: {self.use_autocast}")
            except Exception as e:
                logger.warning(f"Failed to create model with {attn_implementation}: {e}")
                logger.info("Falling back to default attention implementation")
                config.attn_implementation = 'eager'
                self.model = GPT2LMHeadModel(config).to(self.device)
                logger.info(f"Created new model with {sum(p.numel() for p in self.model.parameters())} parameters")
                logger.info(f"Model config: {n_layer} layers, {n_embd} embedding dim, {n_head} heads, {n_ctx} context length")
        
        if self.compile_model:
            try:
                logger.info("Compiling model with torch.compile...")
                self.model = torch.compile(self.model)
                logger.info("Model compilation completed")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
                logger.info("Continuing without compilation")
    
    def train(self, num_epochs: int = 5, batch_size: int = 16, start_lr: float = 2e-4,
              max_length: int = 1024, min_length: int = 64, shuffle: bool = True, warmup_batches: int = 0, grad_clip: float = 1.0, 
              end_lr: float = None, grad_accum_steps: int = 1, max_steps: int = None):
        """Main training loop."""
        logger.info("Starting training...")
        
        if end_lr is None:
            end_lr = start_lr
        
        enable_decay = (start_lr != end_lr)
        
        texts = self.load_data()
        dataset = self.prepare_data(texts, max_length, min_length)
        self.initialize_model()
        
        collator = DataCollator(self.tokenizer, max_length)
        train_loader = DataLoader(dataset, 
                                  batch_size=batch_size, 
                                  shuffle=shuffle, 
                                  collate_fn=collator, 
                                  num_workers=1,
                                  pin_memory=True,
                                  prefetch_factor=4)
        
        total_batches = len(train_loader) * num_epochs
        total_steps = total_batches // grad_accum_steps
        warmup_steps = warmup_batches // grad_accum_steps if warmup_batches > 0 else 0
        decay_steps = max(0, total_steps - warmup_steps) if enable_decay else 0
        effective_batch_size = batch_size * grad_accum_steps
        
        optimizer = AdamW(self.model.parameters(), lr=start_lr if warmup_steps == 0 else start_lr * 0.01)
        
        autocast_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        scaler_enabled = self.use_autocast and (autocast_dtype == torch.float16)
        scaler = GradScaler(enabled=scaler_enabled)
        
        autocast_ctx = nullcontext() if (not self.use_autocast or self.device.type == 'cpu') else autocast(device_type=self.device.type, dtype=autocast_dtype)
        
        logger.info(f"Training for {num_epochs} epochs, \nbatch_size={batch_size}, grad_accum_steps={grad_accum_steps}, effective_batch_size={effective_batch_size}, \nstart_lr={start_lr}, end_lr={end_lr}, \nmax_length={max_length}, min_length={min_length}, \nshuffle={shuffle}, \ngrad_clip={grad_clip}")
        logger.info(f"Total batches: {total_batches}, total_steps: {total_steps}, warmup_steps: {warmup_steps}, decay_enabled: {enable_decay}")
        if max_steps is not None:
            logger.info(f"Max steps: {max_steps} (training will stop early if reached)")
        else:
            logger.info("Max steps: None (will train for full epochs)")
        logger.info(f"Using gradient scaler: {scaler_enabled} (autocast: {self.use_autocast}, autocast_dtype: {autocast_dtype if self.use_autocast else 'N/A'}, model dtype: float32)")
        if warmup_steps > 0:
            logger.info(f"Warmup: {warmup_steps} steps ({start_lr * 0.01:.2e} to {start_lr:.2e})")
        if enable_decay and decay_steps > 0:
            logger.info(f"Decay: {decay_steps} steps ({start_lr:.2e} to {end_lr:.2e})")
        elif not enable_decay:
            logger.info(f"Constant LR after warmup: {start_lr:.2e}")
        
        global_batch = 0
        global_step = 0
        for epoch in range(num_epochs):
            logger.info(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
            
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0
            accum_loss = 0.0
            last_norm = 0.0
            
            total_steps_this_epoch = len(train_loader) // grad_accum_steps
            if max_steps is not None:
                remaining_steps = max_steps - global_step
                total_steps_this_epoch = min(total_steps_this_epoch, remaining_steps)
            progress_bar = tqdm(total=total_steps_this_epoch, desc=f"Training Epoch {epoch}", unit="step")
            
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx % grad_accum_steps == 0:
                    optimizer.zero_grad()
                
                current_lr = optimizer.param_groups[0]['lr']
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                with autocast_ctx:
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / grad_accum_steps
                
                scaler.scale(loss).backward()
                
                epoch_loss += loss.item() * grad_accum_steps
                accum_loss += loss.item() * grad_accum_steps
                batch_count += 1
                global_batch += 1
                
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    if grad_clip != 0.0:
                        scaler.unscale_(optimizer)
                        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
                        last_norm = norm
                    else:
                        norm = last_norm
                    
                    scaler.step(optimizer)
                    scaler.update()
                    global_step += 1
                    
                    if max_steps is not None and global_step >= max_steps:
                        logger.info(f"Reached max_steps ({max_steps}), stopping training early")
                        model_path = self.output_dir / f"model_epoch_{epoch}"
                        model_path.mkdir(exist_ok=True)
                        self.model.save_pretrained(str(model_path))
                        logger.info(f"Model checkpoint saved: {model_path}")
                        break
                    
                    new_lr = self._get_scheduled_lr(global_step, warmup_steps, decay_steps, start_lr, end_lr, enable_decay)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    
                    current_accum_batches = min(grad_accum_steps, (batch_idx % grad_accum_steps) + 1)
                    avg_accum_loss = accum_loss / current_accum_batches
                    accum_loss = 0.0
                else:
                    norm = last_norm
                    current_accum_batches = (batch_idx % grad_accum_steps) + 1
                    avg_accum_loss = accum_loss / current_accum_batches
                
                display_lr = optimizer.param_groups[0]['lr']
                
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    progress_bar.set_postfix({
                        "Loss": f"{loss.item() * grad_accum_steps:.4f}",
                        "Avg Loss": f"{epoch_loss/batch_count:.4f}",
                        "Step": f"{global_step}",
                        "LR": f"{display_lr:.2e}",
                        "Norm": f"{norm:.4f}"
                    })
                    progress_bar.update(1)
            
            progress_bar.close()
            
            avg_loss = epoch_loss / batch_count
            logger.info(f"Epoch {epoch} completed - Average Loss: {avg_loss:.4f}")
            
            if max_steps is not None and global_step >= max_steps:
                break
            
            model_path = self.output_dir / f"model_epoch_{epoch}"
            model_path.mkdir(exist_ok=True)
            self.model.save_pretrained(str(model_path))
        
        logger.info(f"Training completed! Results saved in {self.output_dir}")
        
        # Save training metadata
        metadata = {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "effective_batch_size": effective_batch_size,
            "start_lr": start_lr,
            "end_lr": end_lr,
            "warmup_batches": warmup_batches,
            "warmup_steps": warmup_steps,
            "total_batches": total_batches,
            "total_steps": total_steps,
            "max_steps": max_steps,
            "actual_steps": global_step,
            "max_length": max_length,
            "vocab_size": len(self.tokenizer),
            "device": str(self.device),
            "model_path": self.model_path,
            "training_epochs": list(range(num_epochs)),
            "model_config": self.model_config
        }
        
        metadata_file = self.output_dir / "training_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    def _get_scheduled_lr(self, global_step: int, warmup_steps: int, decay_steps: int, start_lr: float, end_lr: float, enable_decay: bool) -> float:
        """Calculate learning rate based on current step and schedule."""
        if global_step < warmup_steps:
            warmup_progress = global_step / warmup_steps if warmup_steps > 0 else 1.0
            return start_lr * (0.01 + 0.99 * warmup_progress)
        elif enable_decay and decay_steps > 0:
            decay_step = global_step - warmup_steps
            decay_progress = min(decay_step / decay_steps, 1.0)
            return start_lr + (end_lr - start_lr) * decay_progress
        else:
            return start_lr

def main():
    parser = argparse.ArgumentParser(description='Fine-tune GPT model using existing tokenizer')
    parser.add_argument('--data_file', type=str, required=True, help='Path to JSON file with texts')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to existing tokenizer directory')
    parser.add_argument('--model_path', type=str, default=None, help='Path to existing pre-trained model directory (optional, will create new model if not provided)')
    parser.add_argument('--output_dir', type=str, default='training_output', help='Output directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--start_lr', type=float, default=2e-4, help='Learning rate after warmup')
    parser.add_argument('--end_lr', type=float, default=None, help='Final learning rate after decay (default: same as --start_lr, no decay)')
    parser.add_argument('--max_length', type=int, default=1024, help='Max length for training chunks')
    parser.add_argument('--min_length', type=int, default=64, help='Min length for training chunks (chunks shorter than this will be filtered out)')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile to optimize the model (requires PyTorch 2.0+)')
    parser.add_argument('--noshuffle', action='store_true', help='Do not shuffle the data in DataLoader')
    parser.add_argument('--warmup_batches', type=int, default=0, help='Number of batches for warmup (default: 0)')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Maximum norm for gradient clipping (default: 1.0)')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Number of gradient accumulation steps (default: 1)')
    parser.add_argument('--autocast', action='store_true', help='Use mixed precision autocast during forward pass (uses bfloat16 if supported, else float16)')
    parser.add_argument('--max_steps', type=int, default=None, help='Maximum number of training steps (default: None, train for full epochs)')
    
    # Model architecture parameters (only used when creating new model from scratch)
    parser.add_argument('--n_embd', type=int, default=1024, help='Embedding dimension (default: 1024)')
    parser.add_argument('--n_layer', type=int, default=16, help='Number of transformer layers (default: 24)')
    parser.add_argument('--n_head', type=int, default=16, help='Number of attention heads (default: 16)')
    parser.add_argument('--n_ctx', type=int, default=1024, help='Context length - max tokens processed at once (default: 1024)')
    parser.add_argument('--n_positions', type=int, default=None, help='Max position embeddings (default: same as n_ctx)')
    parser.add_argument('--attn_implementation', type=str, default='eager', choices=['eager', 'flash_attention_2', 'flash_attention_3'], help='Attention implementation to use (default: eager)')
    
    args = parser.parse_args()
    
    # Prepare model config dictionary
    model_config = {
        'n_embd': args.n_embd,
        'n_layer': args.n_layer,
        'n_head': args.n_head,
        'n_ctx': args.n_ctx,
        'n_positions': args.n_positions if args.n_positions is not None else args.n_ctx,
        'attn_implementation': args.attn_implementation,
        'use_autocast': args.autocast
    }
    
    trainer = GPTTrainer(
        data_file=args.data_file,
        tokenizer_path=args.tokenizer_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        compile_model=args.compile,
        model_config=model_config
    )
    
    trainer.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        start_lr=args.start_lr,
        max_length=args.max_length,
        min_length=args.min_length,
        shuffle=not args.noshuffle,
        warmup_batches=args.warmup_batches,
        grad_clip=args.grad_clip,
        grad_accum_steps=args.grad_accum_steps,
        end_lr=args.end_lr,
        max_steps=args.max_steps
    )

if __name__ == "__main__":
    main() 