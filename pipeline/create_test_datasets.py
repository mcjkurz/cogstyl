#!/usr/bin/env python3
"""
Test Dataset Creator

This script creates and cleans test datasets from various sources (Mao texts, Chinese novels, 
case studies) with type-specific cleaning. The datasets are saved as JSON files for use
in cognitive stylometry experiments.

For MD5 hash generation (contamination detection), use create_md5_hashes.py separately.
"""

import argparse
import json
import re
import random
import opencc
import sys
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Add the project root to Python path to allow imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from misc.utils import convert_to_chinese_punctuation, is_chinese_text, has_many_numbers


def _clean_mao_line(line: str, min_length: int = 16) -> str:
    """Clean a line from Mao texts - remove footnotes and short lines."""
    line = line.strip()
    
    # Line-level patterns (lines to completely remove)
    line_level_patterns = [
        r"^\*",        # Lines starting with *
        r"^〔\d+〕",    # Lines starting with footnote numbers
        r"^（.{5,}）",        # Lines starting with something like （一九三七年十月二十五日）
    ]
    
    # Check if line should be completely removed
    for pattern in line_level_patterns:
        if re.match(pattern, line):
            return ""
    
    # Remove inline patterns
    inline_patterns = [
        r"〔\d+〕",     # Inline footnote numbers
        r"（[○一二三四五六七八九十百千万\d]+）",
        r"\([○一二三四五六七八九十百千万\d]+\)", # Inline Chinese numbers in brackets (e.g. （三五） )
    ]
    
    for pattern in inline_patterns:
        line = re.sub(pattern, "", line)
    
    # Remove all spaces within the line (between characters)
    line = re.sub(r"\s+", "", line)
    
    if len(line.strip()) < min_length \
        or not is_chinese_text(line) \
        or has_many_numbers(line):
        return ""
    
    return line.strip()


def _clean_novel_text(text: str, percentage_to_remove: float = 0.05) -> str:
    """Clean novel text by removing front/back matter and chapter headings. Also, take only the middle part to avoid intros and endings."""
    
    # remove the first and last 5% of the text to avoid intros and endings
    text = text[int(len(text) * percentage_to_remove):int(len(text) * (1 - percentage_to_remove))]
    
    lines = text.split("\n")
    cleaned_lines = []
    
    wrong_patterns = [
        r"^第\s*[一二三四五六七八九十百千万\d]+\s*章",
        r"^第\s*[一二三四五六七八九十百千万\d]+\s*回",
        r"^第\s*[一二三四五六七八九十百千万\d]+\s*部",
        r"^第\s*[一二三四五六七八九十百千万\d]+\s*卷",
        r"《我城》",
        r"─{4,}",
        r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]+",
        r"※",
        r"□",
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        is_wrong = False
        for pattern in wrong_patterns:
            if re.match(pattern, line):
                is_wrong = True
                break
        if is_wrong:
            continue

        line = re.sub(r"\s+", "", line)
        if len(line) <= 1:
            continue

        cleaned_lines.append(line)
    
    return "".join(cleaned_lines)


def _process_mao(directory: str, min_doc_length: int = 100) -> List[Dict[str, str]]:
    """Process Mao documents with footnote removal and traditional->simplified conversion."""
    documents = []
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"Warning: Directory {directory} does not exist")
        return documents
    
    # Initialize OpenCC converter
    converter = opencc.OpenCC('t2s')  # traditional to simplified
    
    print(f"Processing Mao documents from: {directory}")
    
    text_files = list(dir_path.glob("*.txt"))
    for file_path in tqdm(text_files, desc="Processing Mao texts"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            lines = [line.strip() for line in lines if line.strip()]

            current_doc = []
            for line in lines:
                cleaned_line = _clean_mao_line(line, min_length=14)
                if cleaned_line:
                    cleaned_line = converter.convert(cleaned_line)
                    current_doc.append(cleaned_line)
                elif current_doc:
                    doc_text = "".join(current_doc)
                    if len(doc_text) > min_doc_length:
                        documents.append({"text": doc_text})
                    current_doc = []
            
            if current_doc:
                doc_text = "".join(current_doc)
                if len(doc_text) > min_doc_length:
                    documents.append({"text": doc_text})
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    print(f"Processed {len(documents)} Mao documents")
    random.shuffle(documents)
    return documents


def _process_novels(directory: str, min_doc_length: int = 256, percentage_to_remove: float = 0.05) -> List[Dict[str, str]]:
    """Process novel documents with front/back matter removal and chapter heading cleaning."""
    documents = []
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"Warning: Directory {directory} does not exist")
        return documents
    
    print(f"Processing novel documents from: {directory}")
    
    text_files = list(dir_path.glob("*.txt"))

    print(f"Processing {len(text_files)} novels")
    for file_path in tqdm(text_files, desc="Processing novels"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            # Apply novel-specific cleaning
            cleaned_text = _clean_novel_text(text, percentage_to_remove)
            
            if len(cleaned_text) > min_doc_length:
                documents.append({"text": cleaned_text})
            else:
                print(f"Skipping {file_path} because it is too short ({len(cleaned_text)} characters after cleaning.)")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    print(f"Processed {len(documents)} novel documents")
    return documents


def _process_case_studies(directory: str, min_doc_length: int = 100) -> List[Dict[str, str]]:
    """Process case study documents with minimal preprocessing (just strip)."""
    documents = []
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"Warning: Directory {directory} does not exist")
        return documents
    
    print(f"Processing case study documents from: {directory}")
    
    text_files = list(dir_path.glob("*.txt"))
    for file_path in tqdm(text_files, desc="Processing case studies"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if len(text) > min_doc_length:
                documents.append({"text": text})
            else:
                print(f"Skipping {file_path} because it is too short ({len(text)} characters after cleaning.)")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    print(f"Processed {len(documents)} case study documents")
    return documents


def _save_dataset(documents: List[Dict[str, str]], output_path: str):
    """Save dataset to JSON file."""
    print(f"Saving {len(documents)} documents to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)


def create_test_datasets(output_dir: str = "./datasets") -> Dict[str, str]:
    """Create test datasets from built-in sources (Mao texts and Chinese novels).
    
    Args:
        output_dir: Directory to save processed datasets
    
    Returns:
        Dict mapping dataset names to their output file paths
    """
    print("Test Dataset Creator")
    print("=" * 50)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    test_corpora = {}
    
    # Built-in datasets to process
    datasets = {
        "mao": ("./datasets/mao", _process_mao),
        "chinese_novels": ("./datasets/chinese_novels", _process_novels),
        "case_studies": ("./datasets/case_studies", _process_case_studies),
    }
    
    print("\nProcessing datasets:")
    for name in datasets.keys():
        print(f"  {name}")
    
    # Process each dataset
    for name, (directory, processor_func) in datasets.items():
        print(f"\nProcessing {name} dataset...")
        
        # Process the dataset using each processor's default min_doc_length
        output_file = output_path / f"C_{name}.json"
        documents = processor_func(directory)
        documents = [{"text": convert_to_chinese_punctuation(doc["text"])} for doc in documents]
        
        if documents:
            _save_dataset(documents, str(output_file))
            test_corpora[f"C_{name}"] = str(output_file)
            print(f"Saved {len(documents)} {name} documents to {output_file}")
            # print 3 random examples truncated to first 100 characters 
            print(f"3 random examples:")
            for doc in random.sample(documents, 3):
                print(f"  {doc['text'][:100]}...")
        else:
            print(f"No documents found for {name} in {directory}")
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Dataset Creation Completed!")
    print("=" * 50)
    
    if test_corpora:
        print("\nTest datasets created:")
        for name, filepath in test_corpora.items():
            print(f"  {name}: {filepath}")
        print(f"\nNext step: Generate MD5 hashes using create_md5_hashes.py")
        print(f"   python pipeline/create_md5_hashes.py --input-dir {output_dir} --output {output_dir}/test_hashes.pkl")
    else:
        print("\nNo test datasets were created")
    
    return test_corpora

def main():
    """Main function for creating test datasets."""
    parser = argparse.ArgumentParser(description="Create test datasets for cognitive stylometry evaluation")

    parser.add_argument("--output-dir", required=True,
                       help="Output directory for test datasets")
    
    args = parser.parse_args()
    
    # Create datasets
    create_test_datasets(
        output_dir=args.output_dir
    )
    
    return 0

if __name__ == "__main__":
    exit(main()) 