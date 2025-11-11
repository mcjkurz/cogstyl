#!/usr/bin/env python3
"""
Experiment 2: Find Learning Patterns

This script analyzes perplexity data to find text patterns that showed the biggest
changes during training. It ranks patterns by improvement/deterioration and selects
the top patterns based on the specified mode, then saves them as JSON files for further analysis.

Additionally, it supports including user-specified phrases from a text file
alongside the ranked patterns found.

Corresponds to the microscopic analysis described in the manuscript where we identify
specific sequences that demonstrate familiarization through perplexity changes.
"""

import json
import numpy as np
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict

from tqdm import tqdm
import glob
import re

sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LearningPattern:
    """Represents a text pattern with perplexity change during training."""
    text_id: str
    start_idx: int
    end_idx: int
    early_perplexity: float
    late_perplexity: float
    improvement: float
    relative_improvement: float
    pattern_text: str
    full_document_text: Optional[str] = None
    full_document_tokens: Optional[List[str]] = None
    full_document_epoch_data: Optional[Dict[str, List[Tuple]]] = None
    metadata: Optional[Dict] = None
    direction: str = "improvement"

def load_epoch_data_from_directory(directory: str, specific_epochs: Optional[List[int]] = None) -> Dict[str, Dict]:
    """Load perplexity data from all or specific epoch files in directory.
    
    Args:
        directory: Directory containing epoch files
        specific_epochs: If provided, only load these specific epoch numbers (memory optimization)
    """
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist")
    
    pattern = "epoch_*_perplexities.json"
    epoch_files = list(directory_path.glob(pattern))
    
    if not epoch_files:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found in {directory}")
    
    logger.info(f"Found {len(epoch_files)} epoch files in {directory}")
    
    epoch_file_pairs = []
    for file_path in epoch_files:
        filename = file_path.stem
        match = re.search(r'epoch_(-?\d+)_perplexities', filename)
        if match:
            epoch_num = int(match.group(1))
            if specific_epochs is None or epoch_num in specific_epochs:
                epoch_file_pairs.append((epoch_num, str(file_path)))
        else:
            logger.warning(f"Could not extract epoch number from {filename}, skipping")
    
    epoch_file_pairs.sort(key=lambda x: x[0])
    file_paths = [pair[1] for pair in epoch_file_pairs]
    
    if specific_epochs:
        logger.info(f"Loading specific epochs: {[pair[0] for pair in epoch_file_pairs]}")
    else:
        logger.info(f"Loading all epochs: {[pair[0] for pair in epoch_file_pairs]}")
    
    return load_epoch_data(file_paths)

def load_epoch_data(file_paths: List[str]) -> Dict[str, Dict]:
    """Load perplexity data from multiple epoch files with new token_data format."""
    epoch_data = {}
    
    for file_path in file_paths:
        logger.info(f"Loading data from {file_path}")
        
        filename = Path(file_path).stem
        if 'epoch' in filename:
            match = re.search(r'epoch_(-?\d+)_perplexities', filename)
            if match:
                epoch_num = int(match.group(1))
            else:
                epoch_str = filename.split('epoch')[1].split('_')[0]
                try:
                    epoch_num = int(epoch_str)
                except ValueError:
                    epoch_num = filename
        else:
            epoch_num = filename
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        epoch_dict = {}
        for doc in data:
            text_id = doc.get('text_id', 'unknown')
            
            perplexities = []
            tokens = []
            for decoded_token, token_id, perplexity in doc['token_data']:
                if perplexity is None:
                    perplexities.append(np.nan)
                else:
                    perplexities.append(perplexity)
                tokens.append(decoded_token)
            
            epoch_dict[text_id] = {
                'text': doc['text'],
                'perplexities': perplexities,
                'tokens': tokens,
                'token_data': doc['token_data'],
                'metadata': doc.get('metadata', {})
            }
        
        epoch_data[epoch_num] = epoch_dict
        
    return epoch_data

def find_learning_patterns(early_data: Dict, late_data: Dict, early_epoch: str, late_epoch: str, 
                          all_epoch_data: Optional[Dict] = None, pattern_length: int = 16, 
                          min_context: int = 256, mode: str = "top-bottom", 
                          store_full_data: bool = False) -> Tuple[List[LearningPattern], Dict]:
    """Find patterns with perplexity changes during training, ranked by improvement.
    
    Args:
        store_full_data: If False, only stores minimal data (memory optimization).
                        Set to True only for final patterns after filtering.
    """
    
    logger.info(f"Finding patterns of length {pattern_length}")
    logger.info(f"Mode: {mode}")
    if not store_full_data:
        logger.info("Memory optimization: storing minimal data only")
    
    common_text_ids = set(early_data.keys()) & set(late_data.keys())
    logger.info(f"Analyzing {len(common_text_ids)} common texts")
    
    if not common_text_ids:
        logger.error("No common texts found between epochs")
        return [], {}
    
    all_patterns = []
    
    for text_id in tqdm(common_text_ids, desc="Processing texts"):
        early_text_data = early_data[text_id]
        late_text_data = late_data[text_id]
        
        early_perplexities = np.array(early_text_data['perplexities'])
        late_perplexities = np.array(late_text_data['perplexities'])
        
        if len(early_perplexities) != len(late_perplexities):
            logger.warning(f"Mismatched perplexity lengths for {text_id}, skipping")
            continue
        
        if len(early_perplexities) < pattern_length + min_context:
            continue
        
        num_positions = len(early_perplexities) - pattern_length + 1
        for start_idx in range(min_context, num_positions):
            end_idx = start_idx + pattern_length
            
            early_slice = early_perplexities[start_idx:end_idx]
            late_slice = late_perplexities[start_idx:end_idx]
            
            if not isinstance(early_slice, np.ndarray):
                early_slice = np.array([early_slice])
            if not isinstance(late_slice, np.ndarray):
                late_slice = np.array([late_slice])
            
            if np.all(np.isfinite(early_slice)) and np.all(np.isfinite(late_slice)) and np.all(early_slice > 0) and np.all(late_slice > 0):
                log_early = np.log(early_slice).mean()
                log_late = np.log(late_slice).mean()
                log_improvement = log_early - log_late
                
                early_avg = np.exp(log_early)
                late_avg = np.exp(log_late)
                abs_improvement = early_avg - late_avg
                relative_improvement = abs_improvement / early_avg if early_avg > 0 else 0
                
                direction = "improvement" if log_improvement > 0 else "deterioration"
                
                if store_full_data:
                    full_document_epoch_dict = {}
                    
                    if all_epoch_data is not None:
                        for epoch_name, epoch_dict in all_epoch_data.items():
                            if text_id in epoch_dict:
                                epoch_text_data = epoch_dict[text_id]
                                full_document_epoch_dict[str(epoch_name)] = epoch_text_data['token_data']
                    else:
                        full_document_epoch_dict = {
                            str(early_epoch): early_text_data['token_data'],
                            str(late_epoch): late_text_data['token_data']
                        }
                    
                    pattern = LearningPattern(
                        text_id=text_id,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        early_perplexity=early_avg,
                        late_perplexity=late_avg,
                        improvement=log_improvement,
                        relative_improvement=relative_improvement,
                        pattern_text=''.join(early_text_data['tokens'][start_idx:end_idx]),
                        full_document_text=early_text_data['text'],
                        full_document_tokens=early_text_data['tokens'],
                        full_document_epoch_data=full_document_epoch_dict,
                        metadata=early_text_data['metadata'],
                        direction=direction
                    )
                else:
                    pattern = LearningPattern(
                        text_id=text_id,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        early_perplexity=early_avg,
                        late_perplexity=late_avg,
                        improvement=log_improvement,
                        relative_improvement=relative_improvement,
                        pattern_text=''.join(early_text_data['tokens'][start_idx:end_idx]),
                        direction=direction
                    )
                
                all_patterns.append(pattern)
    
    if not all_patterns:
        logger.error("No valid patterns found")
        return [], {}
    
    logger.info(f"Collected {len(all_patterns)} patterns")
    
    log_improvements = np.array([p.improvement for p in all_patterns])
    
    statistics = {
        'total_patterns': len(all_patterns),
        'mean_log_improvement': float(np.mean(log_improvements)),
        'std_log_improvement': float(np.std(log_improvements)),
        'median_log_improvement': float(np.median(log_improvements)),
        'min_log_improvement': float(np.min(log_improvements)),
        'max_log_improvement': float(np.max(log_improvements)),
        'pattern_length': pattern_length,
        'mode': mode,
        'method': 'simple_ranking'
    }
    
    if mode == "top":
        all_patterns.sort(key=lambda p: p.improvement, reverse=True)
    elif mode == "bottom":
        all_patterns.sort(key=lambda p: p.improvement)
    else:
        all_patterns.sort(key=lambda p: abs(p.improvement), reverse=True)
    
    improvements = sum(1 for p in all_patterns if p.improvement > 0)
    deteriorations = sum(1 for p in all_patterns if p.improvement < 0)
    statistics['total_improvements'] = improvements
    statistics['total_deteriorations'] = deteriorations
    
    if all_patterns and store_full_data and all_patterns[0].full_document_epoch_data:
        epochs_available = len(all_patterns[0].full_document_epoch_data)
        statistics['epochs_per_pattern'] = epochs_available
        logger.info(f"Each pattern includes data from {epochs_available} epochs")
    
    logger.info(f"Found {len(all_patterns)} patterns total")
    logger.info(f"  - Improvements: {improvements}")
    logger.info(f"  - Deteriorations: {deteriorations}")
    
    return all_patterns, statistics

def hydrate_patterns_with_full_data(patterns: List[LearningPattern], all_epoch_data: Dict, 
                                   early_epoch: str, late_epoch: str) -> List[LearningPattern]:
    """Add full document data to patterns after filtering (memory optimization)."""
    logger.info(f"Hydrating {len(patterns)} patterns with full document data")
    
    unique_text_ids = set(p.text_id for p in patterns)
    
    for pattern in tqdm(patterns, desc="Hydrating patterns"):
        if pattern.full_document_text is not None:
            continue
        
        text_id = pattern.text_id
        
        if early_epoch in all_epoch_data and text_id in all_epoch_data[early_epoch]:
            early_text_data = all_epoch_data[early_epoch][text_id]
            
            pattern.full_document_text = early_text_data['text']
            pattern.full_document_tokens = early_text_data['tokens']
            pattern.metadata = early_text_data.get('metadata', {})
            
            full_document_epoch_dict = {}
            for epoch_name, epoch_dict in all_epoch_data.items():
                if text_id in epoch_dict:
                    epoch_text_data = epoch_dict[text_id]
                    full_document_epoch_dict[str(epoch_name)] = epoch_text_data['token_data']
            
            pattern.full_document_epoch_data = full_document_epoch_dict
        else:
            logger.warning(f"Could not hydrate pattern for text_id {text_id}")
    
    logger.info("Pattern hydration complete")
    return patterns

def filter_patterns_by_ngram_frequency(patterns: List[LearningPattern], ngram_length: int, epoch_data: Dict) -> List[LearningPattern]:
    """Filter patterns by checking if their n-grams appear more than once in the dataset.
    
    Args:
        patterns: List of patterns to filter
        ngram_length: Length of n-grams to extract and check
        epoch_data: All epoch data (only one epoch will be used for text data)
    
    Returns:
        Filtered list containing only patterns with n-grams that appear more than once
    """
    if not patterns or ngram_length <= 0:
        return patterns
    
    first_epoch = list(epoch_data.keys())[0]
    epoch_dict = epoch_data[first_epoch]
    
    all_texts = []
    for text_id, text_data in epoch_dict.items():
        all_texts.append(text_data["text"])
    
    combined_text = ''.join(all_texts)
    
    filtered_patterns = []
    min_count = 3
    
    for pattern in tqdm(patterns, desc="Filtering by n-gram frequency"):
        pattern_text = pattern.pattern_text
        
        if len(pattern_text) < ngram_length:
            filtered_patterns.append(pattern)
            continue
        
        pattern_ngrams = []
        for i in range(len(pattern_text) - ngram_length + 1):
            ngram = pattern_text[i:i + ngram_length]
            pattern_ngrams.append(ngram)
        
        keep_pattern = False
        for ngram in pattern_ngrams:
            count = combined_text.count(ngram)
            if count >= min_count:
                keep_pattern = True
                break
        
        if keep_pattern:
            filtered_patterns.append(pattern)
    
    logger.info(f"Filtered from {len(patterns)} to {len(filtered_patterns)} patterns")
    return filtered_patterns

def filter_overlapping_patterns(patterns: List[LearningPattern], substring_length: int = 0) -> List[LearningPattern]:
    """Filter out overlapping patterns using exact substring matching.
    
    Args:
        patterns: List of patterns to filter
        substring_length: Length of substrings to check for overlap. 
                         If 0, no filtering is performed.
    
    Returns:
        Filtered list with no patterns sharing substrings of the specified length
    """
    if not patterns or substring_length == 0:
        return patterns
    
    logger.info(f"Filtering patterns with substring length {substring_length}")
    
    patterns.sort(key=lambda p: p.improvement, reverse=True)
    
    filtered = []
    seen_substrings = set()
    
    for pattern in tqdm(patterns, desc="Filtering overlapping patterns"):
        pattern_text = pattern.pattern_text
        
        if len(pattern_text) < substring_length:
            continue
        
        pattern_substrings = set()
        for i in range(len(pattern_text) - substring_length + 1):
            substring = pattern_text[i:i+substring_length]
            pattern_substrings.add(substring)
        
        if pattern_substrings & seen_substrings:
            continue
        
        filtered.append(pattern)
        seen_substrings.update(pattern_substrings)
    
    logger.info(f"Filtered from {len(patterns)} to {len(filtered)} patterns")
    return filtered

def load_user_phrases(phrases_file: str) -> List[str]:
    """Load user-specified phrases from a text file (one phrase per line)."""
    if not phrases_file:
        return []
    
    phrases_path = Path(phrases_file)
    if not phrases_path.exists():
        logger.warning(f"Phrases file {phrases_file} does not exist")
        return []
    
    phrases = []
    with open(phrases_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            phrase = line.strip()
            if phrase:
                phrases.append(phrase)
    
    logger.info(f"Loaded {len(phrases)} user-specified phrases from {phrases_file}")
    return phrases

def find_user_specified_patterns(phrases: List[str], epoch_data: Dict, early_epoch: str, late_epoch: str) -> List[LearningPattern]:
    """Find occurrences of user-specified phrases in the epoch data."""
    if not phrases:
        return []
    
    logger.info(f"Finding occurrences of {len(phrases)} user-specified phrases")
    
    early_data = epoch_data[early_epoch]
    late_data = epoch_data[late_epoch]
    common_text_ids = set(early_data.keys()) & set(late_data.keys())
    
    user_patterns = []
    
    for phrase in tqdm(phrases, desc="Processing user phrases"):
        for text_id in common_text_ids:
            early_text_data = early_data[text_id]
            late_text_data = late_data[text_id]
            
            tokens = early_text_data['tokens']
            full_text = ''.join(tokens)
            
            start_pos = 0
            while True:
                pos = full_text.find(phrase, start_pos)
                if pos == -1:
                    break
                
                char_count = 0
                start_idx = 0
                end_idx = len(tokens)
                
                for i, token in enumerate(tokens):
                    if char_count <= pos < char_count + len(token):
                        start_idx = i
                    char_count += len(token)
                    if char_count >= pos + len(phrase):
                        end_idx = i + 1
                        break
                
                if start_idx < end_idx and end_idx <= len(tokens):
                    early_perplexities = np.array(early_text_data['perplexities'][start_idx:end_idx])
                    late_perplexities = np.array(late_text_data['perplexities'][start_idx:end_idx])
                    
                    if (len(early_perplexities) > 0 and len(late_perplexities) > 0 and 
                        np.all(np.isfinite(early_perplexities)) and np.all(np.isfinite(late_perplexities)) and 
                        np.all(early_perplexities > 0) and np.all(late_perplexities > 0)):
                        
                        early_avg = float(np.exp(np.log(early_perplexities).mean()))
                        late_avg = float(np.exp(np.log(late_perplexities).mean()))
                        abs_improvement = early_avg - late_avg
                        relative_improvement = abs_improvement / early_avg if early_avg > 0 else 0
                        log_improvement = np.log(early_avg) - np.log(late_avg)
                        
                        full_document_epoch_dict = {}
                        for epoch_name, epoch_dict in epoch_data.items():
                            if text_id in epoch_dict:
                                epoch_text_data = epoch_dict[text_id]
                                full_document_epoch_dict[str(epoch_name)] = epoch_text_data['token_data']
                        
                        direction = "improvement" if log_improvement > 0 else "deterioration"
                        
                        pattern = LearningPattern(
                            text_id=text_id,
                            start_idx=start_idx,
                            end_idx=end_idx,
                            early_perplexity=early_avg,
                            late_perplexity=late_avg,
                            improvement=log_improvement,
                            relative_improvement=relative_improvement,
                            pattern_text=phrase,
                            full_document_text=early_text_data['text'],
                            full_document_tokens=early_text_data['tokens'],
                            full_document_epoch_data=full_document_epoch_dict,
                            metadata={**early_text_data['metadata'], 'user_specified': True},
                            direction=direction
                        )
                        
                        user_patterns.append(pattern)
                
                start_pos = pos + 1
    
    logger.info(f"Found {len(user_patterns)} occurrences of user-specified phrases")
    return user_patterns

def save_pattern_analysis(patterns: List[LearningPattern], statistics: Dict, 
                         output_file: str, n_patterns: int = 50, mode: str = "top-bottom") -> None:
    """Save pattern analysis results to JSON file with appropriate patterns based on mode."""
    
    regular_patterns = [p for p in patterns if not p.metadata.get('user_specified', False)]
    user_specified_patterns = [p for p in patterns if p.metadata.get('user_specified', False)]
    
    if mode == "top-bottom":
        n_top = n_patterns // 2
        n_bottom = n_patterns - n_top
        
        improvements = [p for p in regular_patterns if p.improvement > 0]
        deteriorations = [p for p in regular_patterns if p.improvement < 0]
        
        improvements.sort(key=lambda p: p.improvement, reverse=True)
        deteriorations.sort(key=lambda p: p.improvement)
        
        selected_improvements = improvements[:n_top] if len(improvements) >= n_top else improvements
        selected_deteriorations = deteriorations[:n_bottom] if len(deteriorations) >= n_bottom else deteriorations
        
        selected_regular = selected_improvements + selected_deteriorations
        
        # Combine with all user-specified patterns
        all_patterns = selected_regular + user_specified_patterns
        logger.info(f"Converting {len(all_patterns)} patterns to serializable format...")
        patterns_data = [asdict(pattern) for pattern in tqdm(all_patterns, desc="Serializing patterns")]
        
        # Create analysis report
        analysis_data = {
            'experiment': 'exp2_find_learning_patterns',
            'description': 'Learning patterns found through microscopic analysis (simple ranking)',
            'statistics': statistics,
            'patterns': patterns_data,
            'metadata': {
                'num_patterns': len(patterns_data),
                'num_regular_patterns': len(selected_regular),
                'num_user_specified_patterns': len(user_specified_patterns),
                'num_improvements': len(selected_improvements),
                'num_deteriorations': len(selected_deteriorations),
                'n_patterns_requested': n_patterns,
                'n_top_requested': n_top,
                'n_bottom_requested': n_bottom,
                'analysis_method': 'simple_ranking',
                'mode': mode
            }
        }
    else:
        # Single direction mode (top or bottom)
        selected_regular = regular_patterns[:n_patterns] if len(regular_patterns) >= n_patterns else regular_patterns
        all_patterns = selected_regular + user_specified_patterns
        logger.info(f"Converting {len(all_patterns)} patterns to serializable format...")
        patterns_data = [asdict(pattern) for pattern in tqdm(all_patterns, desc="Serializing patterns")]
        
        direction_desc = "improvements" if mode == "top" else "deteriorations"
        
        # Create analysis report
        analysis_data = {
            'experiment': 'exp2_find_learning_patterns',
            'description': f'Learning patterns ({direction_desc}) found through microscopic analysis (simple ranking)',
            'statistics': statistics,
            'patterns': patterns_data,
            'metadata': {
                'num_patterns': len(patterns_data),
                'num_regular_patterns': len(selected_regular),
                'num_user_specified_patterns': len(user_specified_patterns),
                'n_patterns_requested': n_patterns,
                'analysis_method': 'simple_ranking',
                'mode': mode,
                'direction': direction_desc
            }
        }
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to file
    logger.info(f"Writing {len(patterns_data)} patterns to {output_file}...")
    logger.info("This may take a moment for large datasets...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(patterns_data)} patterns to {output_file} ({len(selected_regular)} regular + {len(user_specified_patterns)} user-specified)")

def main():
    """Main function for finding learning patterns."""
    parser = argparse.ArgumentParser(description="Experiment 2: Find Learning Patterns")
    parser.add_argument("--input-directory", required=True, help="Directory containing epoch perplexity files")
    parser.add_argument("--output", required=True, help="Output JSON file for learning patterns")
    parser.add_argument("--pattern-length", type=int, default=16,
                       help="Length of patterns to analyze (default: 16)")
    parser.add_argument("--min-context", type=int, default=256,
                       help="Minimum context for token prediction (default: 256)")
    parser.add_argument("--top-n", type=int, default=50,
                       help="Save only top N patterns (default: 50)")
    parser.add_argument("--max-overlap", type=int, default=0,
                       help="Maximum substring length for overlap filtering (default: 0, no filtering)")
    parser.add_argument("--noise-filter-length", type=int, default=0,
                       help="N-gram length for noise filtering (default: 0, no filtering)")
    parser.add_argument("--mode", type=str, default="top-bottom", choices=["top", "bottom", "top-bottom"],
                       help="Mode for finding patterns (top: improvements, bottom: deteriorations, top-bottom: both)")
    parser.add_argument("--phrases-file", type=str, default=None,
                       help="Optional path to text file containing user-specified phrases (one per line)")
    
    args = parser.parse_args()
    
    # PHASE 1: Load only early and late epochs for initial pattern finding (memory optimization)
    logger.info("=== PHASE 1: Initial pattern finding with minimal memory ===")
    logger.info(f"Loading perplexity data from {args.input_directory}")
    
    try:
        # First, identify available epochs
        from pathlib import Path
        directory_path = Path(args.input_directory)
        epoch_files = list(directory_path.glob("epoch_*_perplexities.json"))
        
        epoch_numbers = []
        for file_path in epoch_files:
            match = re.search(r'epoch_(-?\d+)_perplexities', file_path.stem)
            if match:
                epoch_numbers.append(int(match.group(1)))
        
        if len(epoch_numbers) < 2:
            logger.error("Need at least 2 epochs for comparison")
            return 1
        
        epoch_numbers.sort()
        early_epoch = epoch_numbers[0]
        late_epoch = epoch_numbers[-1]
        
        logger.info(f"Found {len(epoch_numbers)} epochs: {epoch_numbers}")
        logger.info(f"Will compare epoch {early_epoch} vs epoch {late_epoch}")
        
        # Load only early and late epochs initially
        logger.info("Loading only early and late epochs for pattern finding...")
        epoch_data_minimal = load_epoch_data_from_directory(
            args.input_directory, 
            specific_epochs=[early_epoch, late_epoch]
        )
        
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        return 1
    
    early_data = epoch_data_minimal[early_epoch]
    late_data = epoch_data_minimal[late_epoch]
    
    # Find learning patterns with minimal data storage
    logger.info("Finding learning patterns (storing minimal data)...")
    patterns, statistics = find_learning_patterns(
        early_data=early_data,
        late_data=late_data,
        early_epoch=early_epoch,
        late_epoch=late_epoch,
        all_epoch_data=None,  # Don't pass all epochs yet
        pattern_length=args.pattern_length,
        min_context=args.min_context,
        mode=args.mode,
        store_full_data=False  # Memory optimization
    )
    
    if not patterns:
        logger.error("No patterns found")
        return 1
    
    # Select top patterns to reduce dataset size before expensive filtering
    pre_filter_count = 10000
    logger.info("Selecting top patterns before filtering to improve performance...")
    
    if args.mode == "top-bottom":
        # Get top improvements and bottom deteriorations
        improvements = [p for p in patterns if p.improvement > 0]
        deteriorations = [p for p in patterns if p.improvement < 0]
        
        improvements.sort(key=lambda p: p.improvement, reverse=True)
        deteriorations.sort(key=lambda p: p.improvement)
        
        top_improvements = improvements[:pre_filter_count] if len(improvements) >= pre_filter_count else improvements
        top_deteriorations = deteriorations[:pre_filter_count] if len(deteriorations) >= pre_filter_count else deteriorations
        
        patterns_for_filtering = top_improvements + top_deteriorations
        
    elif args.mode == "top":
        patterns.sort(key=lambda p: p.improvement, reverse=True)
        patterns_for_filtering = patterns[:pre_filter_count] if len(patterns) >= pre_filter_count else patterns
        
    elif args.mode == "bottom":
        patterns.sort(key=lambda p: p.improvement)
        patterns_for_filtering = patterns[:pre_filter_count] if len(patterns) >= pre_filter_count else patterns
    
    logger.info(f"Selected {len(patterns_for_filtering)} patterns for filtering (from {len(patterns)} total)")
    
    # Apply filtering on the reduced set
    filtered_patterns = patterns_for_filtering
    
    # Filter patterns by n-gram frequency if noise_filter_length is specified
    if args.noise_filter_length > 0:
        logger.info(f"Filtering patterns by n-gram frequency (n-gram length: {args.noise_filter_length})")
        original_count = len(filtered_patterns)
        filtered_patterns = filter_patterns_by_ngram_frequency(filtered_patterns, args.noise_filter_length, epoch_data_minimal)
        logger.info(f"Filtered from {original_count} to {len(filtered_patterns)} patterns")
    
    # Filter overlapping patterns if max_overlap is specified
    if args.max_overlap > 0:
        logger.info(f"Filtering overlapping patterns (max substring length: {args.max_overlap})")
        original_count = len(filtered_patterns)
        filtered_patterns = filter_overlapping_patterns(filtered_patterns, args.max_overlap)
        logger.info(f"Filtered from {original_count} to {len(filtered_patterns)} patterns")
    
    patterns = filtered_patterns
    
    # PHASE 2: Load all epochs and hydrate only the filtered patterns (memory optimization)
    logger.info("\n=== PHASE 2: Loading all epochs for filtered patterns ===")
    logger.info(f"Loading all {len(epoch_numbers)} epochs for {len(patterns)} filtered patterns...")
    
    # Get unique text_ids from filtered patterns to optimize loading
    unique_text_ids_needed = set(p.text_id for p in patterns)
    logger.info(f"Need data for {len(unique_text_ids_needed)} unique texts")
    
    # Load all epoch data
    epoch_data_full = load_epoch_data_from_directory(args.input_directory)
    
    # Hydrate filtered patterns with full document data
    patterns = hydrate_patterns_with_full_data(patterns, epoch_data_full, early_epoch, late_epoch)
    
    # Load and find user-specified phrases if provided
    user_patterns = []
    if args.phrases_file:
        logger.info("Processing user-specified phrases...")
        user_phrases = load_user_phrases(args.phrases_file)
        if user_phrases:
            user_patterns = find_user_specified_patterns(user_phrases, epoch_data_full, early_epoch, late_epoch)
    
    # Combine user-specified and regular patterns
    all_patterns_to_save = patterns + user_patterns
    
    # Save results
    logger.info("Saving pattern analysis...")
    save_pattern_analysis(all_patterns_to_save, statistics, args.output, args.top_n, args.mode)
    
    # Print summary
    logger.info("\n=== Experiment 2 Summary ===")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Pattern length: {args.pattern_length}")
    logger.info(f"Epochs compared: {early_epoch} vs {late_epoch}")
    if 'epochs_per_pattern' in statistics:
        logger.info(f"Epochs of data saved per pattern: {statistics['epochs_per_pattern']}")
    logger.info(f"Total patterns analyzed: {statistics['total_patterns']:,}")
    logger.info(f"  - Improvements: {statistics['total_improvements']:,}")
    logger.info(f"  - Deteriorations: {statistics['total_deteriorations']:,}")
    logger.info(f"Mean improvement: {statistics['mean_log_improvement']:.4f} ± {statistics['std_log_improvement']:.4f}")
    logger.info(f"Pre-filtering selection: top {pre_filter_count} patterns per direction")
    if args.noise_filter_length > 0:
        logger.info(f"Noise filtering applied (n-gram length: {args.noise_filter_length})")
    if args.max_overlap > 0:
        logger.info(f"Overlap filtering applied (max substring length: {args.max_overlap})")
    
    if user_patterns:
        logger.info(f"User-specified patterns found: {len(user_patterns)}")
    
    if patterns:
        logger.info(f"\nTop 5 patterns by improvement:")
        for i, pattern in enumerate(patterns[:5]):
            direction_str = "↑" if pattern.improvement > 0 else "↓"
            logger.info(f"{i+1}. {pattern.text_id}[{pattern.start_idx}:{pattern.end_idx}] {direction_str} "
                       f"(improvement: {pattern.improvement:.4f})")
    
    logger.info("Experiment 2 completed successfully")
    logger.info(f"Results saved to: {args.output}")
    logger.info(f"Total patterns in output: {len(all_patterns_to_save)} (regular: {len(patterns)}, user-specified: {len(user_patterns)})")
    
    return 0

if __name__ == "__main__":
    exit(main()) 