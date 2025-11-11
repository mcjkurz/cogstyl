#!/usr/bin/env python3
"""
Utility functions for cognitive stylometry research.
"""

import hashlib
import json
from typing import Dict, Set, Iterator
from pathlib import Path


def generate_ngrams(text: str, n: int) -> Iterator[str]:
    """Generate character-level n-grams from text.
    
    Args:
        text: Input text
        n: Size of n-grams
        
    Yields:
        Character-level n-grams
    """
    for i in range(len(text) - n + 1):
        yield text[i:i + n]


def in_test_data(text: str, test_data_hashes: Set[bytes], ngram_size: int) -> bool:
    """Check if text contains any n-grams that are present in test data (using digest()).
    
    Args:
        text: Text to check
        test_data_hashes: Set of MD5 digest hashes from test data n-grams
        ngram_size: Size of n-grams to check
        
    Returns:
        True if text contains test data n-grams, False otherwise
    """
    for i in range(len(text) - ngram_size + 1):
        ngram = text[i:i + ngram_size]
        hash_val = hashlib.md5(ngram.encode('utf-8')).digest()
        if hash_val in test_data_hashes:
            if ngram in text: # additional check to avoid false positives
                return True
    return False


def load_sequences(filepath: str) -> Dict[str, str]:
    """Load text sequences from a JSON file or directory containing text files.
    
    Args:
        filepath: Path to JSON file containing text sequences OR directory containing .txt files
        
    Returns:
        Dictionary mapping sequence names to text content
    """
    filepath_obj = Path(filepath)
    
    # Handle directory containing text files
    if filepath_obj.is_dir():
        print(f"Loading text files from directory: {filepath}")
        sequences = {}
        
        # Look for .txt files in the directory
        txt_files = list(filepath_obj.glob("*.txt"))
        
        if not txt_files:
            print(f"No .txt files found in directory: {filepath}")
            return {}
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                # Use filename without extension as sequence name
                sequence_name = txt_file.stem
                sequences[sequence_name] = content
                print(f"  Loaded: {sequence_name} ({len(content)} characters)")
            except Exception as e:
                print(f"  Error reading {txt_file}: {e}")
                continue
        
        return sequences
    
    # Handle JSON file (original functionality)
    elif filepath_obj.is_file():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON formats
            if isinstance(data, list):
                # List of documents with text field
                return {f"seq_{i}": doc.get('text', str(doc)) for i, doc in enumerate(data)}
            elif isinstance(data, dict):
                # Dictionary format
                return data
            else:
                # Single string or other format
                return {"sequence": str(data)}
                
        except Exception as e:
            print(f"Error loading sequences from {filepath}: {e}")
            return {}
    
    else:
        print(f"Error: Path does not exist: {filepath}")
        return {}


def extract_all_chunks(text_dict: Dict[str, str], chunk_size: int) -> Set[str]:
    """Extract character-level chunks from text dictionary for contamination detection.
    
    Args:
        text_dict: Dictionary mapping names to text content
        chunk_size: Size of character chunks to extract
        
    Returns:
        Set of unique text chunks
    """
    chunks = set()
    
    for text in text_dict.values():
        if not text:
            continue
            
        # Extract character-level chunks
        for i in range(len(text) - chunk_size + 1):
            chunk = text[i:i + chunk_size]
            chunks.add(chunk)
    
    return chunks


def is_chinese_char(char):
    """Returns True if char is a CJK Unified Ideograph."""
    codepoint = ord(char)
    # Basic CJK Unified Ideographs
    if 0x4E00 <= codepoint <= 0x9FFF:
        return True
    # Extensions A-F (rare, but for completeness)
    if 0x3400 <= codepoint <= 0x4DBF:
        return True
    if 0x20000 <= codepoint <= 0x2A6DF:
        return True
    if 0x2A700 <= codepoint <= 0x2B73F:
        return True
    if 0x2B740 <= codepoint <= 0x2B81F:
        return True
    if 0x2B820 <= codepoint <= 0x2CEAF:
        return True
    return False


def is_latin_char(char):
    """Returns True if char is a basic Latin letter or digit."""
    return ('A' <= char <= 'Z') or ('a' <= char <= 'z') or ('0' <= char <= '9')


def is_chinese_text(text, threshold=0.5):
    """Returns True if text is predominantly Chinese based on character analysis."""
    chinese_count = 0
    latin_count = 0
    for char in text:
        if is_chinese_char(char):
            chinese_count += 1
        elif is_latin_char(char):
            latin_count += 1
        # else: ignore punctuation, spaces, etc.

    total = chinese_count + latin_count
    if total == 0:
        return False

    proportion = chinese_count / total
    return proportion >= threshold


def has_many_numbers(text, min_count=20):
    """Returns True if text has more than min_count numbers."""
    number_count = 0
    for char in text:
        if char.isdigit():
            number_count += 1
    return number_count >= min_count


def convert_to_chinese_punctuation(text: str) -> str:
    """Convert Western punctuation marks to their Chinese equivalents.
    
    This function converts common Western (half-width) punctuation marks
    to their Chinese (full-width) equivalents, which are standard in
    Chinese text processing.
    
    Args:
        text: Input text containing Western punctuation
        
    Returns:
        Text with Chinese punctuation marks
        
    Example:
        >>> convert_to_chinese_punctuation("Hello, world! How are you?")
        "Hello，world！How are you？"
    """
    # Mapping from Western punctuation to Chinese punctuation
    punctuation_map = {
        ',': '，',    # comma
        '.': '。',    # period/full stop  
        '?': '？',    # question mark
        '!': '！',    # exclamation mark
        ';': '；',    # semicolon
        ':': '：',    # colon
        '(': '（',    # left parenthesis
        ')': '）',    # right parenthesis
        '[': '【',    # left square bracket
        ']': '】',    # right square bracket
        '{': '｛',    # left curly brace
        '}': '｝',    # right curly brace
        '…': '……',   # ellipsis
        '—': '——',   # em dash
        '–': '——',   # en dash to em dash
        '-': '－',    # hyphen
        '「': '“',
        '」': '”',
    }
    
    # Apply punctuation conversion
    result = text
    for western, chinese in punctuation_map.items():
        result = result.replace(western, chinese)
    
    return result 