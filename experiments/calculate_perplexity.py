#!/usr/bin/env python3
"""
Experiment 1: Macroscopic Analysis of Maospeak - Perplexity Calculation

This script calculates per-token perplexities for test corpora using trained GPT models
across different epochs to analyze familiarization and defamiliarization effects.

Corresponds to the macroscopic analysis described in the manuscript where we track
how the model's perplexity on Mao corpus decreases while perplexity on novels increases.
"""

import json
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import numpy as np
from typing import List, Dict, Optional
import logging
import os
from pathlib import Path
import argparse
import warnings
import random
from contextlib import nullcontext

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerplexityCalculator:
    """Calculates per-token perplexities efficiently for cognitive stylometry analysis."""
    
    def __init__(self, model, tokenizer, device, context_size=256):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.context_size = context_size
        
        if device.type == 'cpu':
            self.autocast_ctx = nullcontext()
        else:
            autocast_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            self.autocast_ctx = autocast(device_type=device.type, dtype=autocast_dtype)
    
    def calculate_perplexities_batched(self, input_ids, batch_size=64):
        """Calculate per-token perplexity using batched inference with full context.
        Only calculates perplexity for tokens with sufficient context (>= context_size).
        Tokens without enough context get np.nan perplexity."""
        self.model.eval()
        perplexities = np.full(len(input_ids), np.nan)
        
        input_ids = input_ids.to(self.device)
        
        contexts = []
        targets = []
        positions = []
        
        for pos in range(len(input_ids)):
            if pos >= self.context_size:
                start_pos = pos - self.context_size
                context = input_ids[start_pos:pos]
                
                contexts.append(context)
                targets.append(input_ids[pos])
                positions.append(pos)
        
        for i in range(0, len(contexts), batch_size):
            batch_contexts = torch.stack(contexts[i:i+batch_size])
            batch_targets = torch.stack(targets[i:i+batch_size])
            
            with torch.no_grad():
                with self.autocast_ctx:
                    outputs = self.model(input_ids=batch_contexts)
                    logits = outputs.logits[:, -1, :]
                
                losses = F.cross_entropy(logits, batch_targets, reduction='none')
                batch_perplexities = torch.exp(losses)
                
                batch_positions = positions[i:i+batch_size]
                for j, pos in enumerate(batch_positions):
                    perplexity_val = batch_perplexities[j]
                    if torch.isfinite(perplexity_val) and perplexity_val < 1e6:
                        perplexities[pos] = perplexity_val.item()
                    else:
                        perplexities[pos] = 1e6
        
        return perplexities

    def _trim_past_key_values(self, past_key_values, max_len):
        """Trim past_key_values to keep only the last max_len tokens."""
        if past_key_values is None:
            return None
        
        if past_key_values[0][0].size(-2) <= max_len:
            return past_key_values
        
        return tuple(
            (key[..., -max_len:, :], value[..., -max_len:, :])
            for key, value in past_key_values
        )

    def calculate_perplexities_sequential(self, input_ids):
        """Calculate per-token perplexity using sequential processing with cache.
        Only calculates perplexity for tokens with sufficient context (>= context_size).
        Tokens without enough context get np.nan perplexity.
        """
        self.model.eval()
        perplexities = np.full(len(input_ids), np.nan)
        
        input_ids = input_ids.to(self.device)
        
        if len(input_ids) <= self.context_size:
            return perplexities
        
        with torch.no_grad():
            initial_context = input_ids[:self.context_size].unsqueeze(0)
            
            with self.autocast_ctx:
                outputs = self.model(
                    input_ids=initial_context,
                    past_key_values=None,
                    use_cache=True
                )
            
            past_key_values = outputs.past_key_values
            
            if self.context_size < len(input_ids):
                first_prediction_logits = outputs.logits[0, -1, :]
                actual_token = input_ids[self.context_size]
                
                loss = F.cross_entropy(first_prediction_logits.unsqueeze(0), actual_token.unsqueeze(0))
                perplexity_val = torch.exp(loss)
                
                if torch.isfinite(perplexity_val) and perplexity_val < 1e6:
                    perplexities[self.context_size] = perplexity_val.item()
                else:
                    perplexities[self.context_size] = 1e6
            
            for pos in range(self.context_size, len(input_ids) - 1):
                curr_token = input_ids[pos:pos+1].unsqueeze(0)
                
                with self.autocast_ctx:
                    outputs = self.model(
                        input_ids=curr_token,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                
                next_pos = pos + 1
                next_token_logits = outputs.logits[0, -1, :]
                actual_next_token = input_ids[next_pos]
                
                loss = F.cross_entropy(next_token_logits.unsqueeze(0), actual_next_token.unsqueeze(0))
                perplexity_val = torch.exp(loss)
                
                if torch.isfinite(perplexity_val) and perplexity_val < 1e6:
                    perplexities[next_pos] = perplexity_val.item()
                else:
                    perplexities[next_pos] = 1e6
                
                past_key_values = outputs.past_key_values
                past_key_values = self._trim_past_key_values(past_key_values, self.context_size)
        
        return perplexities

def process_texts_to_perplexities(texts_data: List[Dict], model_path: str, tokenizer_path: str, 
                                context_size: int = 256, batch_size: int = 64, use_compile: bool = False, 
                                use_sequential: bool = False) -> List[Dict]:
    """Process texts and return perplexity data for cognitive stylometry analysis.
    
    Args:
        use_sequential: If True, use sequential processing with cache (more efficient).
                       If False, use batched processing (default behavior).
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info(f"Loading model from {model_path}")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    
    if use_compile:
        if hasattr(torch, 'compile'):
            logger.info("Applying torch.compile optimization...")
            model = torch.compile(model)
        else:
            logger.warning("torch.compile requested but not available (requires PyTorch 2.0+)")
    
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    
    calculator = PerplexityCalculator(model, tokenizer, device, context_size)
    
    method_name = "sequential (with cache)" if use_sequential else "batched"
    logger.info(f"Using {method_name} processing method")
    
    results = []
    running_losses = []
    
    progress_bar = tqdm(texts_data, desc="Processing texts")
    for idx, text_data in enumerate(progress_bar):
        text_id = f"chunk_{idx}"
        text = text_data['text']
        
        tokens = tokenizer.encode(text, add_special_tokens=False)
        input_ids = torch.tensor(tokens)
        
        if len(tokens) == 0:
            logger.warning(f"Empty token sequence for text {text_id}, skipping")
            continue
        
        if use_sequential:
            perplexities = calculator.calculate_perplexities_sequential(input_ids)
        else:
            perplexities = calculator.calculate_perplexities_batched(input_ids, batch_size)
        
        token_data = []
        for pos, (token_id, perplexity) in enumerate(zip(tokens, perplexities)):
            decoded_token = tokenizer.decode([token_id])
            token_data.append((decoded_token, token_id, float(perplexity) if not np.isnan(perplexity) else None))

        valid_perplexities = perplexities[~np.isnan(perplexities)]
        num_valid_tokens = len(valid_perplexities)
        num_skipped_tokens = len(perplexities) - num_valid_tokens

        text_mean_loss = None
        if num_valid_tokens > 0:
            valid_losses = np.log(valid_perplexities)
            text_mean_loss = float(np.mean(valid_losses))
            running_losses.append(text_mean_loss)
            current_avg_loss = np.mean(running_losses)
            progress_bar.set_description(f"Processing texts (avg loss: {current_avg_loss:.3f})")

        result = {
            'text_id': text_id,
            'text': text,
            'token_data': token_data,
            'metadata': {
                'num_tokens': len(tokens),
                'num_characters': len(text),
                'num_processed_tokens': num_valid_tokens,
                'num_skipped_tokens': num_skipped_tokens,
                'mean_perplexity': float(np.mean(valid_perplexities)) if num_valid_tokens > 0 else None,
                'std_perplexity': float(np.std(valid_perplexities)) if num_valid_tokens > 0 else None,
                'mean_loss': text_mean_loss,
                'context_size': context_size
            }
        }
        
        results.append(result)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Calculate Perplexities for Macroscopic Analysis")
    parser.add_argument("--input", required=True, help="Input JSON file with texts")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer")
    parser.add_argument("--output", required=True, help="Output JSON file for perplexities")
    parser.add_argument("--context-size", type=int, default=256, help="Context size for perplexity calculation")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for processing")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile optimization (requires PyTorch 2.0+)")
    parser.add_argument("--sequential", action="store_true", help="Use sequential processing with cache (more efficient)")
    parser.add_argument("--debug", type=int, help="Debug mode: process only N documents including those with target phrase")
    
    args = parser.parse_args()
    
    # Load input texts
    logger.info(f"Loading texts from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        texts_data = json.load(f)
    
    # Validate that texts_data is a list of dicts with 'text' key (from chunk_documents.py)
    if not isinstance(texts_data, list):
        raise ValueError("Expected input JSON to be a list of text chunks from chunk_documents.py")
    if texts_data and (not isinstance(texts_data[0], dict) or 'text' not in texts_data[0]):
        raise ValueError("Expected each item to be a dict with 'text' key (from chunk_documents.py)")
    
    # Debug mode: sample N documents including those with target phrase
    if args.debug is not None:
        target_phrase = "坚决、彻底、干净、全部地粉碎帝国主义者及其走狗"
        logger.info(f"Debug mode: sampling {args.debug} documents including those with target phrase")
        
        # Find texts containing the target phrase
        target_texts = []
        remaining_texts = []
        
        for i, text_data in enumerate(texts_data):
            if target_phrase in text_data['text']:
                target_texts.append((i, text_data))
            else:
                remaining_texts.append((i, text_data))
        
        logger.info(f"Found {len(target_texts)} texts with target phrase")
        
        # Set random seed for deterministic sampling
        random.seed(42)
        
        # Sample remaining texts to reach target total (or use all if fewer available)
        remaining_needed = max(0, args.debug - len(target_texts))
        if remaining_needed > 0 and remaining_texts:
            sampled_remaining = random.sample(remaining_texts, min(remaining_needed, len(remaining_texts)))
        else:
            sampled_remaining = []
        
        # Combine target texts and sampled remaining texts
        selected_texts = target_texts + sampled_remaining
        selected_texts.sort(key=lambda x: x[0])  # Sort by original index
        
        # Print sampled IDs for verification
        target_ids = [idx for idx, _ in target_texts]
        sampled_ids = [idx for idx, _ in sampled_remaining]
        all_selected_ids = [idx for idx, _ in selected_texts]
        
        logger.info(f"Target phrase document IDs: {target_ids}")
        logger.info(f"Randomly sampled document IDs: {sampled_ids}")
        logger.info(f"All selected document IDs: {all_selected_ids}")
        
        # Extract just the text data
        texts_data = [text_data for _, text_data in selected_texts]
        
        logger.info(f"Debug mode: processing {len(texts_data)} documents total")
    
    # Process texts
    results = process_texts_to_perplexities(
        texts_data, 
        args.model, 
        args.tokenizer,
        args.context_size,
        args.batch_size,
        args.compile,
        args.sequential
    )
    
    # Save results
    logger.info(f"Saving results to {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Print summary statistics
    logger.info("\n=== Experiment 1 Summary ===")
    total_texts = len(results)
    total_tokens = sum(r['metadata']['num_tokens'] for r in results)
    total_processed = sum(r['metadata']['num_processed_tokens'] for r in results)
    total_skipped = sum(r['metadata']['num_skipped_tokens'] for r in results)
    
    # Only include texts that have valid perplexities, convert to losses
    valid_mean_perplexities = [r['metadata']['mean_perplexity'] for r in results 
                              if r['metadata']['mean_perplexity'] is not None]
    
    logger.info(f"Processed {total_texts} texts with {total_tokens} total tokens")
    logger.info(f"Calculated perplexity for {total_processed} tokens, skipped {total_skipped} tokens")
    
    if valid_mean_perplexities:
        # Use consistent mean losses instead of converting perplexities back to losses
        valid_mean_losses = [r['metadata']['mean_loss'] for r in results 
                            if r['metadata']['mean_loss'] is not None]
        logger.info(f"Overall mean loss: {np.mean(valid_mean_losses):.3f} ± {np.std(valid_mean_losses):.3f}")
        logger.info(f"Texts with valid perplexities: {len(valid_mean_perplexities)}/{total_texts}")
    else:
        logger.info("No texts had sufficient context for perplexity calculation")
    
    logger.info("Results saved for further analysis and visualization")

if __name__ == "__main__":
    main() 