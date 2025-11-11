#!/usr/bin/env python3
"""Document chunking script."""

import json
import argparse
import random
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def chunk_document(text: str, max_length: int = 1024, 
                   context_size: int = 256, 
                   min_chunk_length: int = 64) -> List[str]:
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    stride = max_length - context_size
    
    for i in range(0, len(text), stride):
        chunk = text[i:i + max_length]
        if len(chunk) > min_chunk_length:
            chunks.append(chunk)
    
    return chunks

def process_documents(input_file: str, output_file: str, num_samples: int, 
                     text_key: str = "text", max_length: int = 1024, 
                     context_size: int = 256, seed: int = 42) -> None:
    random.seed(seed)
    
    logger.info(f"Loading data from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        if isinstance(data[0], dict):
            texts = [item[text_key] for item in data if text_key in item]
        else:
            texts = [str(item) for item in data]
    elif isinstance(data, dict):
        texts = [data[text_key]] if text_key in data else [str(data)]
    else:
        raise ValueError("Unsupported JSON structure")
    
    logger.info(f"Loaded {len(texts)} documents")
    
    all_chunks = []
    total_chunks_before_sampling = 0
    
    for doc_id, text in enumerate(texts):
        chunks = chunk_document(text, max_length, context_size)
        doc_chunks_count = len(chunks)
        total_chunks_before_sampling += len(chunks)
        
        if len(chunks) > num_samples:
            sampled_chunks = random.sample(chunks, num_samples)
        else:
            sampled_chunks = chunks
        
        for chunk in sampled_chunks:
            all_chunks.append({"text": chunk})
        logger.info(f"Doc {doc_id} generated {len(sampled_chunks)} chunks out of {doc_chunks_count} total chunks")
    
    logger.info(f"Created {total_chunks_before_sampling} total chunks")
    logger.info(f"Sampled down to {len(all_chunks)} chunks ({num_samples} per document max)")
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(all_chunks)} chunks to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Chunk documents by sampling N chunks from each document')
    parser.add_argument('source_json', type=str, help='Source JSON file with documents')
    parser.add_argument('output_json', type=str, help='Output JSON file for chunks')
    parser.add_argument('num_samples', type=int, help='Number of chunks to sample from each document')
    parser.add_argument('--text_key', type=str, default='text', help='Key for text field in JSON')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum character length for chunks')
    parser.add_argument('--context_size', type=int, default=256, help='Context size for overlapping chunks')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    process_documents(
        input_file=args.source_json,
        output_file=args.output_json,
        num_samples=args.num_samples,
        text_key=args.text_key,
        max_length=args.max_length,
        context_size=args.context_size,
        seed=args.seed
    )

if __name__ == "__main__":
    main() 