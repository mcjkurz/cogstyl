#!/usr/bin/env python3
"""
Experiment 3: N-gram Entropy Analysis

This script compares Shannon entropy of n-grams between different corpora
to quantify linguistic repetitiveness and formulaic patterns.

Corresponds to the n-gram entropy experiment described in the manuscript
where we compare Maospeak against literary texts to show that political
discourse is more formulaic and repetitive.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict, Tuple
import argparse
from pathlib import Path
from tqdm.auto import tqdm
import logging
import sys

# Add the project root to Python path to allow imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from misc.utils import generate_ngrams

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_corpus(filepath: str) -> List[str]:
    """Load text corpus from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        # List of documents with 'text' field
        texts = [item['text'] for item in data if 'text' in item and item['text']]
    elif isinstance(data, dict):
        # Dictionary format
        texts = [text for text in data.values() if text]
    else:
        # Single text
        texts = [str(data)]
    
    logger.info(f"Loaded {len(texts)} texts from {filepath}")
    return texts

def extract_all_ngrams(texts: List[str], n_values: List[int]) -> Dict[int, List[str]]:
    """Extract all n-grams for all n-values from texts. Done once per corpus."""
    logger.info(f"Extracting n-grams for n-values: {n_values}")
    all_ngrams = {}
    
    for n in tqdm(n_values, desc="Extracting n-grams"):
        ngrams = []
        for text in texts:
            if len(text) >= n:
                ngrams.extend([text[i:i+n] for i in range(len(text) - n + 1)])
        all_ngrams[n] = ngrams
    
    return all_ngrams

def calculate_entropy_from_sample(ngrams: List[str], sample_size: int, seed: int) -> float:
    """Calculate Shannon entropy from a random sample of n-grams."""
    if not ngrams:
        return 0.0
    
    # Sample n-grams
    if len(ngrams) <= sample_size:
        sample = ngrams
    else:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(ngrams), size=sample_size, replace=False)
        sample = [ngrams[i] for i in indices]
    
    # Count frequencies and calculate entropy
    counter = Counter(sample)
    total = len(sample)
    
    entropy = 0.0
    for count in counter.values():
        prob = count / total
        entropy -= prob * np.log2(prob)
    
    return entropy

def run_entropy_experiment(
    corpus1_texts: List[str], 
    corpus2_texts: List[str],
    corpus1_name: str = "Corpus 1",
    corpus2_name: str = "Corpus 2", 
    n_values: List[int] = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16],
    sample_size: int = 100000,
    num_trials: int = 1000,
    random_seed: int = 42
) -> Dict:
    """Run n-gram entropy comparison experiment between two corpora."""
    
    logger.info(f"Running entropy experiment: {corpus1_name} vs {corpus2_name}")
    logger.info(f"Parameters: n_values={n_values}, sample_size={sample_size}, trials={num_trials}")
    
    # Pre-compute all n-grams once
    logger.info("Pre-computing all n-grams for corpus 1...")
    corpus1_ngrams = extract_all_ngrams(corpus1_texts, n_values)
    
    logger.info("Pre-computing all n-grams for corpus 2...")
    corpus2_ngrams = extract_all_ngrams(corpus2_texts, n_values)
    
    results = {
        'corpus1_name': corpus1_name,
        'corpus2_name': corpus2_name,
        'parameters': {
            'n_values': n_values,
            'sample_size': sample_size,
            'num_trials': num_trials,
            'random_seed': random_seed
        },
        'n_values': n_values,
        'corpus1_entropies': {},
        'corpus2_entropies': {},
        'entropy_differences': {},
        'statistics': {}
    }
    
    # Run trials for each n-gram size
    for n in tqdm(n_values, desc="Processing n-gram sizes"):
        if n not in corpus1_ngrams or n not in corpus2_ngrams:
            logger.warning(f"No {n}-grams found, skipping")
            continue
            
        if not corpus1_ngrams[n] or not corpus2_ngrams[n]:
            logger.warning(f"Empty {n}-gram sets, skipping")
            continue
            
        logger.info(f"Processing {n}-grams: {len(corpus1_ngrams[n]):,} vs {len(corpus2_ngrams[n]):,}, sample size: {sample_size}")
        
        corpus1_entropies = []
        corpus2_entropies = []
        differences = []
        
        for trial in tqdm(range(num_trials), desc=f"{n}-gram trials", leave=False):
            trial_seed = random_seed + trial
            
            # Calculate entropy for both corpora
            entropy1 = calculate_entropy_from_sample(corpus1_ngrams[n], sample_size, trial_seed)
            entropy2 = calculate_entropy_from_sample(corpus2_ngrams[n], sample_size, trial_seed + 1000)
            
            corpus1_entropies.append(entropy1)
            corpus2_entropies.append(entropy2)
            differences.append(entropy2 - entropy1)
        
        # Store results
        results['corpus1_entropies'][n] = {
            'mean': float(np.mean(corpus1_entropies)),
            'std': float(np.std(corpus1_entropies)),
            'trials': corpus1_entropies
        }
        
        results['corpus2_entropies'][n] = {
            'mean': float(np.mean(corpus2_entropies)),
            'std': float(np.std(corpus2_entropies)),
            'trials': corpus2_entropies
        }
        
        results['entropy_differences'][n] = {
            'mean': float(np.mean(differences)),
            'std': float(np.std(differences)),
            'trials': differences
        }
        

    
    return results

def create_entropy_plots(results: Dict, output_dir: str = None) -> None:
    """Create academic-style visualization plots for entropy analysis."""
    
    # Only create output directory if output_dir is provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    n_values = results['n_values']
    corpus1_name = results['corpus1_name']
    corpus2_name = results['corpus2_name']
    

    
    # Filter n_values to match available data (handle both string and int keys)
    available_n = []
    for n in n_values:
        if n in results['corpus1_entropies'] or str(n) in results['corpus1_entropies']:
            available_n.append(n)
    
    # Re-extract data using the correct keys (string or int)
    corpus1_means = []
    corpus1_stds = []
    corpus2_means = []
    corpus2_stds = []
    diff_means = []
    diff_stds = []
    
    for n in available_n:
        # Try both integer and string keys
        key = n if n in results['corpus1_entropies'] else str(n)
        
        corpus1_means.append(results['corpus1_entropies'][key]['mean'])
        corpus1_stds.append(results['corpus1_entropies'][key]['std'])
        corpus2_means.append(results['corpus2_entropies'][key]['mean'])
        corpus2_stds.append(results['corpus2_entropies'][key]['std'])
        diff_means.append(results['entropy_differences'][key]['mean'])
        diff_stds.append(results['entropy_differences'][key]['std'])
    
    # Convert to numpy arrays for vector operations
    corpus1_means, corpus1_stds = np.array(corpus1_means), np.array(corpus1_stds)
    corpus2_means, corpus2_stds = np.array(corpus2_means), np.array(corpus2_stds)
    diff_means, diff_stds = np.array(diff_means), np.array(diff_stds)
    
    # Set academic plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Define professional, colorblind-safe colors
    color_corpus1 = '#d62728'  # Red
    color_corpus2 = '#1f77b4'  # Blue
    color_diff = '#2ca02c'     # Green
    
    # --- Plot 1: N-gram Entropy Comparison ---
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    
    # Plot data with error bars
    ax1.errorbar(available_n, corpus1_means, yerr=corpus1_stds,
                 color=color_corpus1, marker='o', markersize=5, linewidth=1.5,
                 capsize=3, label=corpus1_name)
    
    ax1.errorbar(available_n, corpus2_means, yerr=corpus2_stds,
                 color=color_corpus2, marker='^', markersize=5, linewidth=1.5,
                 capsize=3, label=corpus2_name)
    
    # Shade the area between the lines to emphasize the "entropy gap"
    ax1.fill_between(available_n, corpus1_means, corpus2_means,
                     color='gray', alpha=0.15, label='Entropy Gap')
    
    # Customize aesthetics with larger fonts
    ax1.set_xlabel('N-gram Size (n)', fontsize=16)
    ax1.set_ylabel('Shannon Entropy (bits)', fontsize=16)
    ax1.set_xticks(available_n)
    ax1.set_xticklabels(available_n, fontsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Use a legend without a frame for a cleaner look
    handles, labels = ax1.get_legend_handles_labels()
    order = [1, 0, 2] if len(handles) == 3 else [1, 0]  # Reorder legend items
    ax1.legend([handles[idx] for idx in order], [labels[idx] for idx in order], 
               loc='best', frameon=False, fontsize=12)
    
    fig1.tight_layout(pad=1.0)
    
    # Save as JPG with 300 DPI if output directory is provided
    if output_dir:
        output_path1 = output_path / 'ngram_entropy_comparison.jpg'
        plt.savefig(output_path1, format='jpg', dpi=300, bbox_inches='tight')
        logger.info(f"Entropy comparison plot saved to {output_path1}")
    
    # Show the plot
    plt.show()
    plt.close()
    
    # --- Plot 2: Entropy Difference ---
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    
    ax2.errorbar(available_n, diff_means, yerr=diff_stds,
                 color=color_diff, marker='D', markersize=5, linewidth=1.5,
                 capsize=3, linestyle='-')
    
    # Add a zero line for reference
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    
    # Customize aesthetics with larger fonts
    ax2.set_xlabel('N-gram Size (n)', fontsize=16)
    ax2.set_ylabel('Entropy Difference (bits)', fontsize=16)
    ax2.set_xticks(available_n)
    ax2.set_xticklabels(available_n, fontsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Add significance annotations
    for i, n in enumerate(available_n):
        if abs(diff_means[i]) > 2 * diff_stds[i]:  # Rough significance test
            ax2.annotate('**', (n, diff_means[i]), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=14, color='red', fontweight='bold')
    
    fig2.tight_layout(pad=1.0)
    
    # Save as JPG with 300 DPI if output directory is provided
    if output_dir:
        output_path2 = output_path / 'ngram_entropy_difference.jpg'
        plt.savefig(output_path2, format='jpg', dpi=300, bbox_inches='tight')
        logger.info(f"Entropy difference plot saved to {output_path2}")
    
    # Show the plot
    plt.show()
    plt.close()

def load_results_from_json(json_file: str) -> Dict:
    """Load experiment results from JSON file."""
    logger.info(f"Loading results from {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    logger.info("Results loaded successfully")
    return results

def run_ngram_entropy_analysis(
    corpus1_file: str = None,
    corpus2_file: str = None,
    corpus1_name: str = "Corpus 1",
    corpus2_name: str = "Corpus 2",
    output_dir: str = "results/ngram_entropy",
    n_values: List[int] = None,
    sample_size: int = 100000,
    num_trials: int = 1000,
    random_seed: int = 42,
    load_results_file: str = None,
    create_plots: bool = True,
    save_results: bool = True
) -> Dict:
    """Run n-gram entropy analysis between two corpora or load existing results.
    
    Args:
        corpus1_file: Path to first corpus JSON file
        corpus2_file: Path to second corpus JSON file  
        corpus1_name: Name for first corpus
        corpus2_name: Name for second corpus
        output_dir: Output directory for results and plots
        n_values: List of n-gram sizes to process
        sample_size: Sample size per trial
        num_trials: Number of trials
        random_seed: Random seed for reproducibility
        load_results_file: Path to existing results JSON file to load
        create_plots: Whether to create visualization plots
        save_results: Whether to save results to JSON file
        
    Returns:
        Dictionary containing experiment results
    """
    if n_values is None:
        n_values = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]
    
    # Check if we're loading existing results or running new experiment
    if load_results_file:
        # Load results from existing JSON file
        results = load_results_from_json(load_results_file)
    else:
        # Validate required arguments for new experiment
        if not corpus1_file or not corpus2_file:
            raise ValueError("corpus1_file and corpus2_file are required when not loading existing results")
        
        # Load corpora
        logger.info("Loading corpora...")
        corpus1_texts = load_corpus(corpus1_file)
        corpus2_texts = load_corpus(corpus2_file)
        
        if not corpus1_texts or not corpus2_texts:
            raise ValueError("Failed to load corpus data")
        
        # Run entropy experiment
        results = run_entropy_experiment(
            corpus1_texts,
            corpus2_texts,
            corpus1_name=corpus1_name,
            corpus2_name=corpus2_name,
            n_values=n_values,
            sample_size=sample_size,
            num_trials=num_trials,
            random_seed=random_seed
        )
        
        # Save results if requested
        if save_results:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            results_file = output_path / 'ngram_entropy_results.json'
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Results saved to {results_file}")
    
    # Create visualizations if requested
    if create_plots:
        logger.info("Creating visualizations...")
        # Pass output_dir - plots will be saved only if output_dir is provided
        create_entropy_plots(results, output_dir)
    
    # Print summary
    logger.info("\n=== Summary ===")
    # Sort keys, handling both string and integer keys
    keys = list(results['entropy_differences'].keys())
    sorted_keys = sorted(keys, key=lambda x: int(x) if isinstance(x, str) else x)
    
    corpus1_name = results['corpus1_name']
    corpus2_name = results['corpus2_name']
    
    for n in sorted_keys:
        corpus1_data = results['corpus1_entropies'][n]
        corpus2_data = results['corpus2_entropies'][n]
        diff_data = results['entropy_differences'][n]
        significance = "**" if abs(diff_data['mean']) > 2 * diff_data['std'] else ""
        
        logger.info(f"N={n}:")
        logger.info(f"  {corpus1_name}: {corpus1_data['mean']:.3f} ± {corpus1_data['std']:.3f}")
        logger.info(f"  {corpus2_name}: {corpus2_data['mean']:.3f} ± {corpus2_data['std']:.3f}")
        logger.info(f"  Difference: {diff_data['mean']:.3f} ± {diff_data['std']:.3f} {significance}")
    
    logger.info("Experiment 3 completed successfully")
    return results


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: N-gram Entropy Analysis")
    parser.add_argument("--corpus1", help="First corpus JSON file")
    parser.add_argument("--corpus2", help="Second corpus JSON file")
    parser.add_argument("--corpus1-name", default="Corpus 1", help="Name for first corpus")
    parser.add_argument("--corpus2-name", default="Corpus 2", help="Name for second corpus")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--n-values", type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16], help="N-gram sizes to process")
    parser.add_argument("--sample-size", type=int, default=100000, help="Sample size per trial")
    parser.add_argument("--num-trials", type=int, default=1000, help="Number of trials")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument("--load-results", help="Load results from existing JSON file and recreate visualizations")
    
    args = parser.parse_args()
    
    # Validate required arguments when not loading results
    if not args.load_results and (not args.corpus1 or not args.corpus2):
        parser.error("--corpus1 and --corpus2 are required when not using --load-results")
    
    # Call the main analysis function
    try:
        run_ngram_entropy_analysis(
            corpus1_file=args.corpus1,
            corpus2_file=args.corpus2,
            corpus1_name=args.corpus1_name,
            corpus2_name=args.corpus2_name,
            output_dir=args.output_dir,
            n_values=args.n_values,
            sample_size=args.sample_size,
            num_trials=args.num_trials,
            random_seed=args.random_seed,
            load_results_file=args.load_results,
            create_plots=True,
            save_results=True
        )
    except ValueError as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main() 