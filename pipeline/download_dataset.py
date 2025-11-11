#!/usr/bin/env python3
"""
Simple script to download and cache HuggingFace datasets.
"""

import argparse
import os
import shutil
from pathlib import Path
from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download
from typing import Optional
import random


def download_dataset(dataset_name: str, 
                    data_dir: Optional[str] = None,
                    split: str = "train", 
                    cache_dir: str = "./datasets/train_cache",
                    output_dir: Optional[str] = None,
                    hf_token: Optional[str] = None,
                    num_proc: int = 1) -> Dataset:
    """Download and cache a HuggingFace dataset.
    
    Args:
        dataset_name: HuggingFace dataset name
        data_dir: Subdirectory within the dataset
        split: Dataset split to download
        cache_dir: Directory to cache the dataset
        output_dir: Directory to download files directly (if specified, files are downloaded instead of cached)
        hf_token: HuggingFace authentication token
        num_proc: Number of processes for downloading
        
    Returns:
        Downloaded dataset object or path to downloaded files
        
    Raises:
        Exception: If dataset download fails
    """
    print("HuggingFace Dataset Downloader")
    print("=" * 50)
    
    # Get auth token from argument or environment
    auth_token = hf_token or os.getenv("HF_TOKEN")
    
    print(f"Downloading dataset: {dataset_name}")
    print(f"Split: {split}")
    if data_dir:
        print(f"Data directory: {data_dir}")
    if auth_token:
        print("Using HuggingFace authentication token")
    
    # If output_dir is specified, download files directly
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_path}")
        
        try:
            # Build the allow_patterns based on data_dir
            allow_patterns = None
            if data_dir:
                # Download only files from the specified data_dir
                allow_patterns = [f"{data_dir}/**"]
                print(f"Downloading only files from: {data_dir}/")
            
            downloaded_path = snapshot_download(
                repo_id=dataset_name,
                local_dir=str(output_path),
                repo_type="dataset",
                token=auth_token,
                allow_patterns=allow_patterns,
                local_dir_use_symlinks=False  # Download actual files, not symlinks
            )
            
            # If data_dir was specified, move files up from the subdirectory
            if data_dir:
                data_subdir = output_path / data_dir
                if data_subdir.exists():
                    print(f"Moving files from {data_dir}/ to output directory...")
                    # Move all files from data_subdir to output_path
                    for item in data_subdir.iterdir():
                        target = output_path / item.name
                        if target.exists():
                            if target.is_dir():
                                shutil.rmtree(target)
                            else:
                                target.unlink()
                        item.rename(target)
                    # Remove the now-empty data_dir
                    data_subdir.rmdir()
            
            # Clean up any HuggingFace cache directories that may have been created
            cache_dir_path = output_path / ".cache"
            if cache_dir_path.exists():
                print("🧹 Cleaning up HuggingFace cache files...")
                shutil.rmtree(cache_dir_path)
            
            print(f"Dataset files downloaded successfully to: {output_path}")
            
            # List downloaded files (exclude cache and metadata files)
            files = list(output_path.rglob("*"))
            data_files = [f for f in files if f.is_file() and 
                         not any(part.startswith('.') for part in f.parts) and  # No hidden dirs
                         not f.suffix in ['.lock', '.metadata'] and  # No lock/metadata files
                         not '.cache' in str(f)]  # No cache files
            print(f"Downloaded {len(data_files)} files:")
            for f in sorted(data_files)[:10]:  # Show first 10 files
                print(f"  - {f.relative_to(output_path)}")
            if len(data_files) > 10:
                print(f"  ... and {len(data_files) - 10} more files")
            
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Failed to download dataset files: {e}")
            raise
    
    # Original caching behavior when output_dir is not specified
    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory: {cache_path}")
    
    try:
        dataset = load_dataset(
            dataset_name,
            data_dir=data_dir,
            split=split,
            cache_dir=str(cache_path),
            token=auth_token,
            num_proc=num_proc
        )
        
        print(f"Dataset loaded successfully")
        print(f"Number of examples: {len(dataset)}")
        print(f"Features: {dataset.features}")
        
        # Show 5 random examples from the dataset
        if len(dataset) > 0:
            print("\nSample data:")
            for i in range(min(5, len(dataset))):
                sample = dataset[random.randint(0, len(dataset) - 1)]
                if 'text' in sample and len(str(sample['text'])) > 200:
                    sample_copy = sample.copy()
                    sample_copy['text'] = str(sample['text'])[:200] + "... [truncated]"
                    print(f"Sample {i+1}: {sample_copy}")
                else:
                    print(f"Sample {i+1}: {sample}")
        
        print(f"\nDataset successfully downloaded and cached")
        return dataset
        
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Try using a HuggingFace token")
        print("3. Try reducing num_proc if getting rate limited")
        print("4. Check if the dataset exists and you have access")
        raise


def main():
    parser = argparse.ArgumentParser(description="Download and cache HuggingFace datasets")
    parser.add_argument("--dataset", "-d", required=True,
                       help="HuggingFace dataset name (e.g., opencsg/Fineweb-Edu-Chinese-V2.1)")
    parser.add_argument("--data-dir", default=None,
                       help="Subdirectory within the dataset (e.g., 4_5)")
    parser.add_argument("--split", "-s", default="train",
                       help="Dataset split to download (default: train)")
    parser.add_argument("--cache-dir", "-c", default="./datasets/train_cache",
                       help="Directory to cache the dataset (default: ./datasets/train_cache)")
    parser.add_argument("--output-dir", "-o", default=None,
                       help="Directory to download files directly (alternative to caching)")
    parser.add_argument("--hf-token", default=None,
                       help="HuggingFace authentication token")
    parser.add_argument("--num-proc", type=int, default=1,
                       help="Number of processes for downloading (default: 1)")
    
    args = parser.parse_args()
    
    try:
        result = download_dataset(
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            split=args.split,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            hf_token=args.hf_token,
            num_proc=args.num_proc
        )
        return 0
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 