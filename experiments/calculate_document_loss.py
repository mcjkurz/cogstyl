#!/usr/bin/env python3
"""
Calculate Document-Level Loss

This script loads a pre-trained model and tokenizer, samples documents from a JSON file,
and calculates the mean loss across all documents. Each document is processed individually
without batching.
"""

import json
import torch
import argparse
import logging
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from typing import List
import numpy as np
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_file: str) -> List[str]:
    """Load texts from JSON file."""
    logger.info(f"Loading data from {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [item["text"] for item in data]
    logger.info(f"Loaded {len(texts)} texts")
    return texts

def calculate_document_loss(model, tokenizer, text: str, max_length: int = 1024, device: torch.device = None) -> float:
    """Calculate loss for a single document."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenized = tokenizer(
        text, 
        max_length=max_length, 
        truncation=True, 
        add_special_tokens=False, 
        return_tensors="pt"
    )
    
    input_ids = tokenized["input_ids"].to(device)
    
    labels = input_ids.clone()
    
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss.item()
    
    return loss

def calculate_corpus_loss(model_path: str, tokenizer_path: str, data_file: str, 
                         max_length: int = 1024, sample_size: int = None, 
                         device: torch.device = None) -> float:
    """Calculate mean loss across all documents in a corpus.
    
    Args:
        model_path: Path to pre-trained model directory
        tokenizer_path: Path to tokenizer directory  
        data_file: Path to JSON file with documents
        max_length: Maximum sequence length (default: 1024)
        sample_size: Number of documents to sample (default: all)
        device: Device to use (default: auto-detect)
        
    Returns:
        Mean loss across all processed documents
    """
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}")
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    
    # Load data
    texts = load_data(data_file)
    
    # Sample documents if requested
    if sample_size is not None and sample_size < len(texts):
        import random
        texts = random.sample(texts, sample_size)
        logger.info(f"Sampled {len(texts)} documents")
    
    # Calculate losses for each document
    logger.info("Calculating losses...")
    losses = []
    progress_bar = tqdm(texts, desc="Processing documents", position=0, leave=True)
    for text in progress_bar:   
        loss = calculate_document_loss(model, tokenizer, text, max_length, device)
        losses.append(loss)
        progress_bar.set_postfix(loss=loss)
    
    # Calculate and print mean loss
    mean_loss = np.mean(losses)
    
    logger.info(f"Processed {len(texts)} documents")
    logger.info(f"Mean loss: {mean_loss:.4f}")
    
    return mean_loss


def main():
    parser = argparse.ArgumentParser(description='Calculate document-level loss using a pre-trained model')
    parser.add_argument('model_path', type=str, help='Path to pre-trained model directory')
    parser.add_argument('tokenizer_path', type=str, help='Path to tokenizer directory')
    parser.add_argument('data_file', type=str, help='Path to JSON file with documents')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length (default: 1024)')
    parser.add_argument('--sample_size', type=int, default=None, help='Number of documents to sample (default: all)')
    
    args = parser.parse_args()
    
    # Call the main calculation function
    mean_loss = calculate_corpus_loss(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        data_file=args.data_file,
        max_length=args.max_length,
        sample_size=args.sample_size
    )
    
    # Print result for command line usage
    print(f"{mean_loss:.4f}")
    
    return mean_loss

if __name__ == "__main__":
    main() 