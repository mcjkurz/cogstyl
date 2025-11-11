#!/usr/bin/env python3
"""
Generate clean base datasets for contamination experiments.
"""

import argparse
import json
import multiprocessing
import os
import pickle
import random
import re
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple
from tqdm import tqdm

# Add the project root to Python path to allow imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from misc.utils import in_test_data, convert_to_chinese_punctuation, is_chinese_text, has_many_numbers
from download_dataset import download_dataset


def _sample_clean_documents_chunk(args: Tuple[List[Dict[str, str]], Set[str], int, int, int, int]) -> List[Dict[str, str]]:
    """Worker function to sample clean documents from a dataset chunk.
    
    Args:
        args: Tuple containing:
            - dataset_chunk: List of documents from the dataset
            - test_data_hashes: Set of n-gram hashes to avoid contamination
            - ngram_size: Size of n-grams for contamination detection
            - min_doc_length: Minimum character length for documents
            - max_docs_per_chunk: Maximum documents to return from this chunk
            - max_doc_length: Maximum character length before splitting
    """
    dataset_chunk, test_data_hashes, ngram_size, min_doc_length, max_docs_per_chunk, max_doc_length = args
    
    clean_documents = []
    
    for document in dataset_chunk:
        if len(clean_documents) >= max_docs_per_chunk:
            break
            
        try:
            # Extract and preprocess text
            text = document.get('text', '')
            # remove spaces, but not new lines etc
            text = re.sub(r'\s{2,}', '。', text)
            text = re.sub(r"\s+", "", text)
            text = convert_to_chinese_punctuation(text)  # Convert to Chinese punctuation

            if len(text) > min_doc_length:
                # Split long documents into fragments with random sampling
                if len(text) > max_doc_length:
                    # Calculate all possible non-overlapping fragment positions
                    max_fragments_per_doc = 12
                    possible_starts = list(range(0, len(text), max_doc_length))
                    
                    # Randomly sample at most 12 non-overlapping fragments
                    num_fragments = min(max_fragments_per_doc, len(possible_starts))
                    selected_starts = random.sample(possible_starts, num_fragments)
                    
                    fragments = [text[start:start + max_doc_length] for start in selected_starts]
                else:
                    fragments = [text]

                # Filter fragments by quality criteria
                for fragment in fragments:
                    if (len(fragment) >= min_doc_length and len(fragment) <= max_doc_length and
                        not in_test_data(fragment, test_data_hashes, ngram_size) and  # No contamination
                        is_chinese_text(fragment) and  # Must be Chinese text
                        not has_many_numbers(fragment)):  # Not too many numbers
                        clean_documents.append({"text": fragment})
                
        except Exception as e:
            # Silently skip errors to avoid flooding output
            continue
    
    return clean_documents


def make_base_dataset(dataset_name: str, 
                     data_dir: str = None,
                     split: str = "train",
                     cache_dir: str = "./dataset_cache", 
                     hf_token: str = None,
                     target_size: int = 1000000,
                     min_doc_length: int = 256,
                     max_doc_length: int = 1024,
                     hash_file_path: str = None,
                     ngram_size: int = 16,
                     num_workers: int = 4,
                     output_path: str = "./datasets/base_dataset.json",
                     seed: int = 42) -> int:
    """Generate a clean base dataset for contamination experiments.
    
    Args:
        dataset_name: HuggingFace dataset name
        data_dir: Subdirectory within the dataset  
        split: Dataset split to use
        cache_dir: Directory where dataset is cached
        hf_token: HuggingFace authentication token
        target_size: Target number of clean documents
        min_doc_length: Minimum document length to keep
        max_doc_length: Maximum document length before splitting
        test_datasets_dir: Directory containing test data hashes
        ngram_size: N-gram size for contamination detection
        num_workers: Number of worker processes
        output_path: Output path for base dataset
        seed: Random seed for reproducibility
        
    Returns:
        Number of documents in the generated base dataset
    """
    print("Base Dataset Generator")
    print("=" * 50)
    print(f"Dataset: {dataset_name}")
    print(f"Target size: {target_size:,} documents")
    print(f"Workers: {num_workers}")
    print()
    
    # Set random seed
    random.seed(seed)
    
    # Step 1: Load test data hashes for contamination filtering
    print("STEP 1: Loading test data hashes...")
    
    if hash_file_path is None:
        print("No hash file path provided, skipping contamination filtering")
        test_data_hashes = set()
    else:
        try:
            with open(hash_file_path, 'rb') as f:
                test_data_hashes = pickle.load(f)
            print(f"Loaded {len(test_data_hashes)} test data hashes from {hash_file_path}")
        except FileNotFoundError:
            print(f"Hash file not found: {hash_file_path}")
            print("Run create_test_datasets.py first to generate the hash file")
            test_data_hashes = set()
        except Exception as e:
            print(f"Error loading hash file: {e}")
            test_data_hashes = set()
    
    # Step 2: Load base dataset
    print(f"\nSTEP 2: Loading base dataset...")
    dataset = download_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        split=split,
        cache_dir=cache_dir,
        hf_token=hf_token
    )
    
    # Step 3: Sample clean base corpus with dynamic sampling
    print(f"\nSTEP 3: Sampling clean base corpus...")
    
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)  # Shuffle for better distribution and to avoid bias
    
    clean_documents = []
    processed_count = 0
    
    batch_size = max(2000, num_workers * 10000)
    
    print(f"Target: {target_size:,} clean documents from {len(all_indices):,} total documents")
    print(f"Processing in batches of {batch_size:,} documents with {num_workers} workers")
    print(f"Max document length: {max_doc_length:,} tokens, min document length: {min_doc_length:,} tokens")
    print(f"Shuffling enabled for better sampling distribution")
    print()
    
    while len(clean_documents) < target_size and processed_count < len(all_indices):
        # Calculate batch size for this iteration
        remaining_indices = len(all_indices) - processed_count
        current_batch_size = min(batch_size, remaining_indices)
        
        # Get the next batch of indices
        batch_indices = all_indices[processed_count:processed_count + current_batch_size]
        
        print(f"Processing batch {processed_count//batch_size + 1}: {current_batch_size:,} documents")
        print(f"   Indices {processed_count:,} to {processed_count + current_batch_size - 1:,}")
        
        # Convert indices to documents for this batch
        batch_documents = []
        for idx in batch_indices:
            try:
                batch_documents.append(dataset[idx])
            except Exception:
                # Skip invalid indices silently
                continue
        
        # Split batch into chunks for workers
        if num_workers == 1:
            document_chunks = [batch_documents]
        else:
            chunk_size = max(100, len(batch_documents) // num_workers)
            document_chunks = [batch_documents[i:i + chunk_size] for i in range(0, len(batch_documents), chunk_size)]
        
        # Prepare worker arguments (no per-chunk limit since we want all clean docs from this batch)
        worker_args = [
            (chunk_data, test_data_hashes, ngram_size, min_doc_length, float('inf'), max_doc_length)
            for chunk_data in document_chunks
        ]
        
        # Process this batch
        batch_clean_documents = []
        
        if num_workers == 1:
            # Single-threaded processing
            for args in worker_args:
                chunk_documents = _sample_clean_documents_chunk(args)
                batch_clean_documents.extend(chunk_documents)
        else:
            # Multi-threaded processing
            with multiprocessing.Pool(num_workers) as pool:
                for chunk_docs in pool.map(_sample_clean_documents_chunk, worker_args):
                    batch_clean_documents.extend(chunk_docs)
        
        # Add clean documents from this batch
        clean_documents.extend(batch_clean_documents)
        processed_count += current_batch_size
        
        # Progress report
        print(f"   Batch complete: +{len(batch_clean_documents):,} clean documents found")
        print(f"Progress: {len(clean_documents):,} / {target_size:,} clean documents ({len(clean_documents)/target_size*100:.1f}%)")
        print(f"Processed: {processed_count:,} / {len(all_indices):,} total documents ({processed_count/len(all_indices)*100:.1f}%)")
        if len(batch_clean_documents) > 0:
            avg_length = sum(len(doc["text"]) for doc in batch_clean_documents) / len(batch_clean_documents)
            max_length_in_batch = max(len(doc["text"]) for doc in batch_clean_documents)
            print(f"Average length of the clean documents: {avg_length:,} tokens, max length: {max_length_in_batch:,} tokens")
        # Calculate yield metrics
        if processed_count > 0:
            yield_ratio = len(clean_documents) / processed_count
            print(f"Yield: {yield_ratio:.2f} clean documents per one element of the dataset")
        print()

        # check if all processed documents are shorter or equal to max_doc_length
        # if not, print a warning
        if any(len(doc["text"]) > max_doc_length for doc in batch_clean_documents):
            print(f"Warning: Some processed documents are longer than max_doc_length. This should not happen.")
        
        # Early stopping if we have enough
        if len(clean_documents) >= target_size:
            print(f"Target reached! Collected {len(clean_documents):,} clean documents")
            break
    
    # Final status
    if len(clean_documents) < target_size:
        print(f"Dataset exhausted. Collected {len(clean_documents):,} clean documents (target was {target_size:,})")
    
    # Take only the target number of documents (in case we collected slightly more)
    clean_documents = clean_documents[:target_size]
    
    # Step 4: Save base dataset
    print(f"\nSTEP 4: Saving base dataset...")
    # Punctuation conversion is now done during processing
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {len(clean_documents):,} documents to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clean_documents, f, ensure_ascii=False, indent=2)
    
    # Summary
    print("\n" + "=" * 50)
    print("Base Dataset Generation Completed!")
    print("=" * 50)
    print(f"Dataset saved to: {output_path}")
    print(f"Total documents: {len(clean_documents):,}")
    print(f"Minimum length: {min_doc_length} characters")
    print(f"Workers used: {num_workers}")
    if test_data_hashes:
        print(f"Test data filtering: {len(test_data_hashes):,} n-grams")
    
    return len(clean_documents)


def main():
    parser = argparse.ArgumentParser(description="Generate clean base dataset for contamination experiments")
    
    # Base dataset configuration
    parser.add_argument("--dataset", "-d", required=True,
                       help="HuggingFace dataset name (e.g., opencsg/Fineweb-Edu-Chinese-V2.1)")
    parser.add_argument("--data-dir", default=None,
                       help="Subdirectory within the dataset (e.g., 4_5)")
    parser.add_argument("--split", "-s", default="train",
                       help="Dataset split to use (default: train)")
    parser.add_argument("--cache-dir", "-c", default="./datasets/train_cache",
                       help="Directory where dataset is cached (default: ./datasets/train_cache)")
    parser.add_argument("--hf-token", default=None,
                       help="HuggingFace authentication token")
    
    # Base corpus configuration
    parser.add_argument("--target-size", type=int, default=100000,
                       help="Target number of clean documents (default: 100000)")
    parser.add_argument("--min-doc-length", type=int, default=256,
                       help="Minimum document length to keep (default: 256)")
    parser.add_argument("--max-doc-length", type=int, default=1024,
                       help="Maximum document length before splitting (default: 1024)")
    
    # Contamination detection configuration
    parser.add_argument("--hash-file-path", default=None,
                       help="Path to test data hash file for contamination filtering")
    parser.add_argument("--ngram-size", type=int, default=16,
                       help="N-gram size for contamination detection (default: 16)")
    
    # Processing configuration
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of worker processes (default: 4)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    # Output configuration
    parser.add_argument("--output-path", default="./datasets/base_dataset.json",
                       help="Output path for base dataset (default: ./datasets/base_dataset.json)")
    
    args = parser.parse_args()
    
    # Get auth token from argument or environment
    auth_token = args.hf_token or os.getenv("HF_TOKEN")
    
    try:
        make_base_dataset(
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            split=args.split,
            cache_dir=args.cache_dir,
            hf_token=auth_token,
            target_size=args.target_size,
            min_doc_length=args.min_doc_length,
            max_doc_length=args.max_doc_length,
            hash_file_path=args.hash_file_path,
            ngram_size=args.ngram_size,
            num_workers=args.num_workers,
            output_path=args.output_path,
            seed=args.seed
        )
        return 0
    except Exception as e:
        print(f"Failed to generate base dataset: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 