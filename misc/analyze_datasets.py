#!/usr/bin/env python3
"""
Analyzes JSON datasets to count the number of documents and characters.
"""

import json
from pathlib import Path
from datasets import load_dataset
import os
from tqdm.auto import tqdm
import gc

def analyze_dataset(dataset_path: str, dataset_name: str):
    """
    Analyzes a single dataset file or cached HuggingFace dataset.

    Args:
        dataset_path (str): The path to the dataset file or cached dataset directory.
        dataset_name (str): The name of the dataset for display purposes.
    """
    print(f"Analyzing {dataset_name} from {dataset_path}...")

    try:
        # Check if it's a JSON file or a cached HuggingFace dataset
        if dataset_path.endswith('.json'):
            # Load JSON dataset
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                print(f"❌ Error: Dataset in {dataset_path} is not a list.")
                return
            documents = []
            for item in tqdm(data, desc="Loading dataset"):
                documents.append(item.get("text", ""))
        else:
            # Load cached HuggingFace dataset
            dataset = dataset = load_dataset(
                dataset_name,
                data_dir="4_5",
                split="train",
                cache_dir=dataset_path
            )
            documents = []
            for item in tqdm(dataset, desc="Loading dataset"):
                documents.append(item.get("text", ""))
            # clean memory
            del dataset
            gc.collect()

        num_documents = len(documents)
        num_characters = sum(len(item) for item in documents)
        
        # Calculate length statistics
        text_lengths = [len(item) for item in documents]
        longest_length = max(text_lengths) if text_lengths else 0
        shortest_length = min(text_lengths) if text_lengths else 0
        mean_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        median_length = sorted(text_lengths)[len(text_lengths) // 2] if text_lengths else 0

        print(f"{dataset_name}:")
        print(f"  Number of documents: {num_documents:,}")
        print(f"  Number of characters: {num_characters:,}")
        print(f"  Longest document: {longest_length:,} characters")
        print(f"  Shortest document: {shortest_length:,} characters")
        print(f"  Mean document length: {mean_length:,.2f} characters")
        print(f"  Median document length: {median_length:,} characters")
        print("-" * 20)

    except FileNotFoundError:
        print(f"❌ Error: File not found at {dataset_path}")
    except json.JSONDecodeError:
        print(f"❌ Error: Could not decode JSON from {dataset_path}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


def main():
    """
    Main function to analyze all specified datasets.
    """
    datasets = {
        "Base Corpus (Fineweb-Edu)": "datasets/base_dataset.json",
        "Test Corpus (Mao)": "datasets/test/C_mao.json",
        "Test Corpus (Mao - Chunked)": "datasets/chunked/C_chunked_mao.json",
        "Test Corpus (Novels)": "datasets/chinese_novels/C_chinese_novels.json",
        "Test Corpus (Novels - Chunked)": "datasets/chunked/C_chunked_chinese_novels.json",
        "opencsg/Fineweb-Edu-Chinese-V2.1": "datasets/fineweb_edu"
    }

    print("="*20)
    print("Dataset Analysis")
    print("="*20)

    for name, path in datasets.items():
        analyze_dataset(path, name)

if __name__ == "__main__":
    main() 