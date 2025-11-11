#!/usr/bin/env python3
"""
MD5 Hash Generator for Contamination Detection

This script generates MD5 hashes from n-grams in test datasets to enable
contamination detection in training corpora. The hashes are used to identify
and remove any training documents that contain sequences from test data.
"""

import argparse
import json
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Set
from tqdm import tqdm
import sys

# Add the project root to Python path to allow imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from misc.utils import generate_ngrams

def generate_hashes_from_dataset(filepath: str, ngram_size: int = 16) -> Set[bytes]:
    """Generate MD5 hashes from n-grams in a single dataset file.
    
    Args:
        filepath: Path to JSON dataset file
        ngram_size: Size of character n-grams to extract
        
    Returns:
        Set of MD5 hash digests
    """
    hashes = set()
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        for doc in tqdm(documents, desc=f"Extracting {ngram_size}-grams"):
            text = doc.get('text', '')
            if text:
                for ngram in generate_ngrams(text, ngram_size):
                    hash_val = hashlib.md5(ngram.encode('utf-8')).digest()
                    hashes.add(hash_val)
                    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return set()
    
    return hashes

def generate_hashes_from_multiple_datasets(dataset_paths: List[str], ngram_size: int = 16) -> Set[bytes]:
    """Generate MD5 hashes from n-grams across multiple dataset files.
    
    Args:
        dataset_paths: List of paths to JSON dataset files
        ngram_size: Size of character n-grams to extract
        
    Returns:
        Set of MD5 hash digests from all datasets
    """
    print(f"Generating {ngram_size}-gram hashes for contamination detection...")
    
    all_hashes = set()
    
    for filepath in dataset_paths:
        dataset_name = Path(filepath).stem
        print(f"\nProcessing {dataset_name}...")
        
        dataset_hashes = generate_hashes_from_dataset(filepath, ngram_size)
        print(f"Generated {len(dataset_hashes)} unique hashes from {dataset_name}")
        
        all_hashes.update(dataset_hashes)
    
    print(f"\nTotal unique {ngram_size}-gram hashes: {len(all_hashes)}")
    return all_hashes

def save_hashes(hashes: Set[bytes], output_path: str, ngram_size: int = 16) -> None:
    """Save hash set to pickle file.
    
    Args:
        hashes: Set of MD5 hash digests
        output_path: Path to save the pickle file
        ngram_size: Size of n-grams (for filename)
    """
    output_file = Path(output_path)
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Add ngram size to filename if not present
    if f"{ngram_size}gram" not in output_file.name:
        stem = output_file.stem
        suffix = output_file.suffix
        output_file = output_file.parent / f"{stem}_{ngram_size}gram{suffix}"
    
    print(f"Saving {len(hashes)} hashes to {output_file}")
    
    with open(output_file, 'wb') as f:
        pickle.dump(hashes, f)
    
    print(f"Hash file saved: {output_file}")

def load_hashes(hash_file_path: str) -> Set[bytes]:
    """Load hash set from pickle file.
    
    Args:
        hash_file_path: Path to pickle file containing hashes
        
    Returns:
        Set of MD5 hash digests
    """
    try:
        with open(hash_file_path, 'rb') as f:
            hashes = pickle.load(f)
        print(f"Loaded {len(hashes)} hashes from {hash_file_path}")
        return hashes
    except Exception as e:
        print(f"Error loading hashes from {hash_file_path}: {e}")
        return set()

def create_hash_file_from_directory(test_data_dir: str, output_path: str, ngram_size: int = 16) -> str:
    """Create hash file from all JSON datasets in a directory.
    
    Args:
        test_data_dir: Directory containing test dataset JSON files
        output_path: Path to save hash file
        ngram_size: Size of character n-grams
        
    Returns:
        Path to created hash file
    """
    test_dir = Path(test_data_dir)
    
    if not test_dir.exists():
        raise FileNotFoundError(f"Test data directory not found: {test_data_dir}")
    
    # Find all JSON files in the directory
    json_files = list(test_dir.glob("*.json"))
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {test_data_dir}")
    
    print(f"Found {len(json_files)} test dataset files:")
    for json_file in json_files:
        print(f"  {json_file.name}")
    
    # Generate hashes from all datasets
    all_hashes = generate_hashes_from_multiple_datasets([str(f) for f in json_files], ngram_size)
    
    # Save hash file
    save_hashes(all_hashes, output_path, ngram_size)
    
    return str(output_path)

def main():
    """Main function for MD5 hash generation."""
    parser = argparse.ArgumentParser(description="Generate MD5 hashes from test datasets for contamination detection")
    
    parser.add_argument("--input-files", nargs='+', 
                       help="Input JSON dataset files to process")
    parser.add_argument("--input-dir",
                       help="Directory containing JSON dataset files")
    parser.add_argument("--output", required=True,
                       help="Output path for hash file (pickle format)")
    parser.add_argument("--ngram-size", type=int, default=16,
                       help="Size of character n-grams to extract (default: 16)")
    
    args = parser.parse_args()
    
    try:
        if args.input_dir:
            # Process all JSON files in directory
            hash_file = create_hash_file_from_directory(args.input_dir, args.output, args.ngram_size)
            
        elif args.input_files:
            # Process specific files
            all_hashes = generate_hashes_from_multiple_datasets(args.input_files, args.ngram_size)
            save_hashes(all_hashes, args.output, args.ngram_size)
            hash_file = args.output
            
        else:
            print("Error: Must specify either --input-files or --input-dir")
            return 1
        
        # Summary
        print("\n" + "=" * 50)
        print("MD5 Hash Generation Completed!")
        print("=" * 50)
        print(f"Hash file: {hash_file}")
        print(f"N-gram size: {args.ngram_size}")
        print("\nUse this hash file with create_base_dataset.py to prevent contamination")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 