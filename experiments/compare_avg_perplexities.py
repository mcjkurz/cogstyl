#!/usr/bin/env python3
"""
Compare Average Perplexities Across Datasets and Epochs

This script analyzes perplexity results from multiple datasets across different epochs,
calculating statistics for character sequences. Only sequences with valid perplexity
scores (no null/nan values) are included in the analysis.
"""

import json
import os
import re
import argparse
import gc
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from collections import defaultdict
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_epoch_files(directory: str) -> Dict[int, str]:
    """Find all epoch perplexity files in a directory.
    
    Returns:
        Dict mapping epoch numbers to file paths
    """
    epoch_files = {}
    directory_path = Path(directory)
    
    if not directory_path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return epoch_files
    
    # Pattern to match epoch files: epoch_NUMBER_perplexities.json
    pattern = re.compile(r'epoch_(-?\d+)_perplexities\.json')
    
    for file_path in directory_path.iterdir():
        if file_path.is_file():
            match = pattern.match(file_path.name)
            if match:
                epoch_num = int(match.group(1))
                epoch_files[epoch_num] = str(file_path)
    
    return epoch_files


def load_all_epoch_data(epoch_files: Dict[int, str], verbose: bool = True) -> Dict[int, List]:
    """Load all epoch files into memory and convert token_data to numpy arrays."""
    epoch_data = {}
    
    print(f"Loading {len(epoch_files)} epoch files...")
    iterator = tqdm(epoch_files.items(), desc="Loading epochs", disable=not verbose)
    for epoch_num, file_path in iterator:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert token_data to numpy arrays for fast slicing
            for chunk in data:
                if 'token_data' in chunk:
                    # Extract perplexity values (3rd column) into numpy array
                    perplexity_values = []
                    for row in chunk['token_data']:
                        if len(row) > 2 and row[2] is not None:
                            perplexity_values.append(float(row[2]))
                        else:
                            perplexity_values.append(np.nan)
                    chunk['perplexity_array'] = np.array(perplexity_values)
                    # Remove the original token_data to save memory
                    del chunk['token_data']
            
            epoch_data[epoch_num] = data
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            continue
    
    print(f"✓ Loaded {len(epoch_data)} epoch files")
    return epoch_data


def extract_sequence_structure_fast(data: List, sequence_length: int = 16, verbose: bool = True) -> Tuple[List[Tuple[int, int, int]], Dict]:
    """Extract sequence structure using pre-loaded data with numpy arrays."""
    sequence_positions = []
    total_chunks = len(data)
    valid_chunks = 0
    
    print(f"Extracting {sequence_length}-character sequences from {total_chunks} chunks...")
    chunk_pbar = tqdm(data, desc="Extracting sequences", position=2, leave=False, disable=not verbose)
    
    for chunk_idx, chunk in enumerate(chunk_pbar):
        if 'perplexity_array' not in chunk:
            continue
        
        perplexity_array = chunk['perplexity_array']
        chunk_sequences = 0
        
        if len(perplexity_array) >= sequence_length:
            # Check each possible sequence position using vectorized operations
            for start_idx in range(len(perplexity_array) - sequence_length + 1):
                end_idx = start_idx + sequence_length
                seq = perplexity_array[start_idx:end_idx]
                
                # Fast check: if no NaN values, sequence is valid
                if not np.isnan(seq).any():
                    sequence_positions.append((chunk_idx, start_idx, end_idx))
                    chunk_sequences += 1
        
        if chunk_sequences > 0:
            valid_chunks += 1
        
        if verbose:
            chunk_pbar.set_description(f"Found: {len(sequence_positions)} sequences")
    
    chunk_pbar.close()
    print(f"✓ Found {len(sequence_positions)} valid sequences in {valid_chunks}/{total_chunks} chunks")
    
    metadata = {
        'num_chunks': total_chunks,
        'valid_chunks': valid_chunks,
        'num_sequences': len(sequence_positions)
    }
    
    return sequence_positions, metadata


def analyze_epoch_with_structure_fast(data: List, sequence_positions: List[Tuple[int, int, int]], 
                                     sequence_length: int = 16) -> Dict:
    """Analyze perplexities using pre-loaded data with numpy arrays."""
    
    if not sequence_positions:
        return {
            'num_sequences': 0,
            'mean_perplexity': None,
            'std_perplexity': None,
            'median_perplexity': None
        }
    
    # Extract perplexities for each sequence using fast numpy operations
    sequence_averages = []
    valid_sequences = 0
    
    for chunk_idx, start_pos, end_pos in sequence_positions:
        if chunk_idx >= len(data) or 'perplexity_array' not in data[chunk_idx]:
            continue
        
        perplexity_array = data[chunk_idx]['perplexity_array']
        if end_pos > len(perplexity_array):
            continue
        
        # Fast numpy slicing
        seq = perplexity_array[start_pos:end_pos]
        
        if not np.isnan(seq).any():
            # Fast vectorized operations
            log_losses = np.log(seq)
            avg_log_loss = np.mean(log_losses)
            sequence_perplexity = np.exp(avg_log_loss)
            sequence_averages.append(sequence_perplexity)
            valid_sequences += 1
    
    if not sequence_averages:
        return {
            'num_sequences': 0,
            'mean_perplexity': None,
            'std_perplexity': None,
            'median_perplexity': None
        }
    
    # Convert to numpy for fast statistics
    sequence_averages = np.array(sequence_averages)
    
    results = {
        'num_sequences': valid_sequences,
        'mean_perplexity': float(np.mean(sequence_averages)),
        'std_perplexity': float(np.std(sequence_averages)),
        'median_perplexity': float(np.median(sequence_averages)),
        'sequence_averages': sequence_averages  # Keep for potential sampling
    }
    
    return results


def run_multiple_trials_fast(data: List, all_sequence_positions: List[Tuple[int, int, int]], 
                            sample_size: int, num_trials: int, sequence_length: int, verbose: bool = True) -> Dict:
    """Run multiple trials by sampling positions first, then computing only needed perplexities."""
    
    if not all_sequence_positions:
        return {
            'mean_perplexity': None,
            'std_perplexity': None,
            'num_sequences': 0,
            'trial_means': [],
            'trial_stds': []
        }
    
    # Sample positions first, then compute perplexities only for what we need
    actual_sample_size = min(sample_size, len(all_sequence_positions)) if sample_size else len(all_sequence_positions)
    
    if sample_size:
        print(f"Running {num_trials} trials with {sample_size} sequences each...")
    else:
        print(f"Running {num_trials} trials with all {len(all_sequence_positions):,} sequences...")
    trial_means = []
    
    iterator = tqdm(range(num_trials), desc="Trials", position=2, leave=False, disable=not verbose)
    for trial in iterator:
        # Sample positions for this trial
        rng = np.random.RandomState(42 + trial)
        if len(all_sequence_positions) > actual_sample_size:
            sampled_indices = rng.choice(len(all_sequence_positions), size=actual_sample_size, replace=False)
            sampled_positions = [all_sequence_positions[i] for i in sampled_indices]
        else:
            sampled_positions = all_sequence_positions
        
        # Now compute perplexities ONLY for the sampled positions
        trial_perplexities = []
        
        for chunk_idx, start_pos, end_pos in sampled_positions:
            if chunk_idx >= len(data) or 'perplexity_array' not in data[chunk_idx]:
                continue
            
            perplexity_array = data[chunk_idx]['perplexity_array']
            if end_pos > len(perplexity_array):
                continue
            
            # Fast numpy slicing and computation
            seq = perplexity_array[start_pos:end_pos]
            
            if not np.isnan(seq).any():
                log_losses = np.log(seq)
                avg_log_loss = np.mean(log_losses)
                sequence_perplexity = np.exp(avg_log_loss)
                trial_perplexities.append(sequence_perplexity)
        
        if trial_perplexities:
            # Calculate trial mean from sampled perplexities
            trial_perplexities = np.array(trial_perplexities)
            log_perplexities = np.log(trial_perplexities)
            avg_log_perplexity = np.mean(log_perplexities)
            trial_mean = np.exp(avg_log_perplexity)
            trial_means.append(trial_mean)
    
    if not trial_means:
        return {
            'mean_perplexity': None,
            'std_perplexity': None,
            'num_sequences': 0,
            'trial_means': [],
            'trial_stds': []
        }
    
    # Compute statistics across trials
    aggregated_results = {
        'mean_perplexity': float(np.mean(trial_means)),
        'std_perplexity': float(np.std(trial_means)),
        'num_sequences': actual_sample_size,
        'trial_means': trial_means,
        'trial_stds': [],
        'num_trials': len(trial_means)
    }
    
    return aggregated_results


def create_perplexity_visualization(epoch_results: Dict, all_epochs: set, sequence_length: int, labels: Optional[Dict[str, str]] = None, savefig_path: Optional[str] = None):
    """Create a bar chart visualization comparing perplexities across datasets and epochs."""
    
    # Get all datasets and sort them
    datasets = list(set(dataset for epoch_data in epoch_results.values() for dataset in epoch_data.keys()))
    datasets.sort()
    
    if len(datasets) == 0:
        logger.warning("No datasets found for visualization")
        return
    
    # Prepare data for plotting
    epochs_sorted = sorted(all_epochs)
    epoch_labels = [str(e) if e >= 0 else 'Pre-trained' for e in epochs_sorted]
    
    # Extract perplexities for each dataset
    dataset_perplexities = {}
    for dataset in datasets:
        perplexities = []
        for epoch in epochs_sorted:
            if (dataset in epoch_results[epoch] and 
                epoch_results[epoch][dataset]['mean_perplexity'] is not None):
                perplexities.append(epoch_results[epoch][dataset]['mean_perplexity'])
            else:
                perplexities.append(np.nan)  # Missing data
        dataset_perplexities[dataset] = perplexities
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Bar parameters
    bar_width = 0.35
    bar_gap = 0.06
    x = np.arange(len(epochs_sorted))
    
    # Colors (colorblind-friendly, expandable for more datasets)
    colors = ['#AF48CF', '#176ba0', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    
    # Create bars for each dataset
    bars = []
    for i, dataset in enumerate(datasets):
        offset = (i - len(datasets)/2 + 0.5) * (bar_width + bar_gap)
        
        # Filter out NaN values for plotting
        y_values = dataset_perplexities[dataset]
        x_positions = x + offset
        
        # Only plot non-NaN values
        valid_mask = ~np.isnan(y_values)
        if np.any(valid_mask):
            # Check if we have error bar data (from multiple trials)
            has_error_bars = False
            error_values = []
            
            for epoch in epochs_sorted:
                if (dataset in epoch_results[epoch] and 
                    epoch_results[epoch][dataset]['mean_perplexity'] is not None):
                    if 'num_trials' in epoch_results[epoch][dataset]:
                        has_error_bars = True
                        error_values.append(epoch_results[epoch][dataset]['std_perplexity'])
                    else:
                        error_values.append(0)  # No error for single trial
                else:
                    error_values.append(0)  # No error for missing data
            
            # Only use error values where we have valid data
            if has_error_bars:
                error_values = np.array(error_values)[valid_mask]
            else:
                error_values = None
            
            rects = ax.bar(
                x_positions[valid_mask], 
                np.array(y_values)[valid_mask], 
                bar_width,
                label=dataset if not labels else labels.get(dataset, dataset),
                color=colors[i % len(colors)],
                edgecolor='black',
                linewidth=0.7,
                zorder=3,
                yerr=error_values if has_error_bars else None,
                capsize=3 if has_error_bars else 0
            )
            bars.append(rects)
            
            # Add data labels
            ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=12)
    
    # Formatting
    ax.set_ylabel(f'Average Perplexity on {sequence_length}-grams', fontsize=16)
    ax.set_xlabel('Fine-Tuning Epoch', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(epoch_labels, fontsize=14)
    ax.legend(fontsize=15, frameon=False, loc='upper left')
    
    # Subtle grid
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5, zorder=0)
    ax.xaxis.grid(False)
    
    # Remove top/right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Format y axis
    ax.tick_params(axis='y', labelsize=11)
    
    # Layout
    fig.tight_layout()
    
    # Save the plot if path is provided
    if savefig_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(savefig_path), exist_ok=True)
        
        plt.savefig(savefig_path, format="png", bbox_inches="tight", dpi=300)
        print(f"\n📊 Visualization saved to: {savefig_path}")
    
    # Show the plot
    plt.show()


def process_datasets(directories: List[str], sequence_length: int, sample_size: Optional[int], 
                     num_trials: int, seed: int, verbose: bool = True) -> Tuple[Dict, set]:
    """Process all datasets and return epoch results and all epochs found."""
    # Set random seed if sampling is used
    if sample_size:
        np.random.seed(seed)
        if num_trials > 1:
            print(f"Configuration: {num_trials} trials with {sample_size} sequences per trial")
        else:
            print(f"Configuration: Sampling {sample_size} sequences per dataset")
    
    epoch_results = defaultdict(dict)  # epoch -> {dataset_name: results}
    all_epochs = set()
    
    print(f"\nProcessing {len(directories)} datasets...")
    dataset_pbar = tqdm(directories, desc="Processing datasets", position=0, disable=not verbose)
    
    for directory in dataset_pbar:
        dataset_name = Path(directory).name
        print(f"\n=== Processing dataset: {dataset_name} ===")
        if verbose:
            dataset_pbar.set_description(f"Processing: {dataset_name}")
        
        epoch_files = find_epoch_files(directory)
        if not epoch_files:
            logger.warning(f"No epoch files found in {directory}")
            continue
        
        print(f"Found {len(epoch_files)} epochs: {sorted(epoch_files.keys())}")
        all_epochs.update(epoch_files.keys())
        
        # Load all epoch data into memory once
        all_epoch_data = load_all_epoch_data(epoch_files, verbose)
        
        # Extract sequence structure once from the first epoch data
        first_epoch = min(epoch_files.keys())
        first_data = all_epoch_data[first_epoch]
        sequence_positions, metadata = extract_sequence_structure_fast(first_data, sequence_length, verbose)
        
        if not sequence_positions:
            logger.warning(f"No valid sequences found in dataset {dataset_name}")
            continue
        
        # Apply sampling to sequence positions only for single trial
        if sample_size and num_trials == 1 and len(sequence_positions) > sample_size:
            indices = np.random.choice(len(sequence_positions), size=sample_size, replace=False)
            sequence_positions = [sequence_positions[i] for i in indices]
            print(f"Sampled {len(sequence_positions)} sequences for analysis")
        elif num_trials > 1:
            print(f"Using all {len(sequence_positions)} sequences for {num_trials} trials")
        
        # Progress bar for epoch files within current dataset  
        epoch_pbar = tqdm(epoch_files.items(), desc=f"Processing epochs", 
                         position=1, leave=False, disable=not verbose)
        
        for epoch_num, file_path in epoch_pbar:
            if verbose:
                epoch_pbar.set_description(f"Epoch {epoch_num}")
            
            # Use pre-loaded data for this epoch
            epoch_data = all_epoch_data[epoch_num]
            
            # Run multiple trials for this epoch
            if num_trials > 1:
                results = run_multiple_trials_fast(epoch_data, sequence_positions, sample_size, num_trials, sequence_length, verbose)
                # Add metadata from structure extraction
                results.update({
                    'num_chunks': metadata['num_chunks'],
                    'valid_chunks': metadata['valid_chunks']
                })
                # Print results for this epoch
                if results['mean_perplexity'] is not None:
                    print(f"  Epoch {epoch_num}: {results['mean_perplexity']:.3f} ± {results['std_perplexity']:.3f} "
                          f"({results['num_trials']} trials)")
                else:
                    print(f"  Epoch {epoch_num}: No valid data")
            else:
                results = analyze_epoch_with_structure_fast(epoch_data, sequence_positions, sequence_length)
                # Add metadata from structure extraction
                results.update({
                    'num_chunks': metadata['num_chunks'],
                    'valid_chunks': metadata['valid_chunks']
                })
                # Print results for this epoch
                if results['mean_perplexity'] is not None:
                    print(f"  Epoch {epoch_num}: {results['mean_perplexity']:.3f} ± {results['std_perplexity']:.3f}")
                else:
                    print(f"  Epoch {epoch_num}: No valid data")
            
            # Remove sequence_averages from results to save memory if present
            if 'sequence_averages' in results:
                del results['sequence_averages']
            # Remove trial data to save memory if present
            if 'trial_means' in results:
                del results['trial_means']
            if 'trial_stds' in results:
                del results['trial_stds']
            
            epoch_results[epoch_num][dataset_name] = results
        
        epoch_pbar.close()
        
        # Clean up memory after processing this dataset
        del all_epoch_data
        del sequence_positions
        gc.collect()
        print(f"✓ Completed {dataset_name} (memory cleaned)")
    
    print(f"\n✓ Completed processing all datasets!")
    return epoch_results, all_epochs


def print_and_save_results(epoch_results: Dict, all_epochs: set, sequence_length: int, 
                          sample_size: Optional[int] = None, output_file: str = "temp_stats.txt"):
    """Display results to console and save to file."""
    output_lines = []
    
    def print_and_save(text="", end="\n"):
        """Print to console and save to output lines"""
        print(text, end=end)
        if end == "\n":
            output_lines.append(text)
        else:
            if output_lines:
                output_lines[-1] += text
            else:
                output_lines.append(text)
    
    print_and_save("\n" + "="*80)
    print_and_save(f"PERPLEXITY COMPARISON - {sequence_length}-CHARACTER SEQUENCES")
    print_and_save("="*80)
    
    if sample_size:
        print_and_save(f"Sample size: {sample_size} sequences per dataset")
    
    datasets = list(set(dataset for epoch_data in epoch_results.values() for dataset in epoch_data.keys()))
    datasets.sort()
    
    # Header
    header_line = f"{'Epoch':<8}"
    for dataset in datasets:
        header_line += f"{dataset:<25}"
    print_and_save(f"\n{header_line}")
    print_and_save("-" * (8 + 25 * len(datasets)))
    
    # Results by epoch
    for epoch in sorted(all_epochs):
        line = f"{epoch:<8}"
        
        for dataset in datasets:
            if dataset in epoch_results[epoch]:
                results = epoch_results[epoch][dataset]
                if results['mean_perplexity'] is not None:
                    mean_pp = results['mean_perplexity']
                    std_pp = results['std_perplexity']
                    n_seq = results['num_sequences']
                    line += f"{mean_pp:6.2f}±{std_pp:5.2f} (n={n_seq:<6})  "
                else:
                    line += f"{'No data':<24}"
            else:
                line += f"{'---':<24}"
        print_and_save(line)
    
    # Summary statistics
    print_and_save("\n" + "="*50)
    print_and_save("SUMMARY STATISTICS")
    print_and_save("="*50)
    
    for dataset in datasets:
        print_and_save(f"\nDataset: {dataset}")
        print_and_save("-" * 30)
        
        dataset_results = []
        for epoch in sorted(all_epochs):
            if dataset in epoch_results[epoch] and epoch_results[epoch][dataset]['mean_perplexity'] is not None:
                results = epoch_results[epoch][dataset]
                dataset_results.append({
                    'epoch': epoch,
                    'mean_perplexity': results['mean_perplexity'],
                    'num_sequences': results['num_sequences'],
                    'valid_chunks': results['valid_chunks'],
                    'total_chunks': results['num_chunks']
                })
        
        if dataset_results:
            perplexities = [r['mean_perplexity'] for r in dataset_results]
            print_and_save(f"Epochs analyzed: {len(dataset_results)}")
            print_and_save(f"Average perplexity across epochs: {np.mean(perplexities):.3f} ± {np.std(perplexities):.3f}")
            print_and_save(f"Min perplexity: {np.min(perplexities):.3f} (epoch {dataset_results[np.argmin(perplexities)]['epoch']})")
            print_and_save(f"Max perplexity: {np.max(perplexities):.3f} (epoch {dataset_results[np.argmax(perplexities)]['epoch']})")
            
            # Show chunk utilization
            total_sequences = sum(r['num_sequences'] for r in dataset_results)
            total_valid_chunks = sum(r['valid_chunks'] for r in dataset_results)
            total_chunks = sum(r['total_chunks'] for r in dataset_results)
            print_and_save(f"Total sequences analyzed: {total_sequences:,}")
            print_and_save(f"Chunk utilization: {total_valid_chunks}/{total_chunks} ({100*total_valid_chunks/total_chunks:.1f}%)")
        else:
            print_and_save("No valid data found")
    
    # Save to file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_lines))
        print(f"\n📁 Results saved to: {output_file}")
    except Exception as e:
        logger.error(f"Error saving results to {output_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Compare average perplexities across datasets and epochs")
    parser.add_argument("directories", nargs='+', help="List of directories containing perplexity files")
    parser.add_argument("--sequence-length", type=int, default=16, help="Length of character sequences to analyze (default: 16)")
    parser.add_argument("--sample-size", type=int, help="Sample this many sequences from each dataset (optional)")
    parser.add_argument("--num-trials", type=int, default=1, help="Number of trials with different samples (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument("--labels", nargs='+', help="Custom labels for datasets in legend (must match number of directories)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.labels and len(args.labels) != len(args.directories):
        parser.error(f"Number of labels ({len(args.labels)}) must match number of directories ({len(args.directories)})")
    
    if args.num_trials > 1 and not args.sample_size:
        parser.error("--sample-size must be specified when using multiple trials (--num-trials > 1)")
    
    # Create mapping from directory to label if labels are provided
    dir_to_label = {}
    if args.labels:
        for directory, label in zip(args.directories, args.labels):
            dataset_name = Path(directory).name
            dir_to_label[dataset_name] = label
    
    # Process all datasets
    epoch_results, all_epochs = process_datasets(
        directories=args.directories,
        sequence_length=args.sequence_length,
        sample_size=args.sample_size,
        num_trials=args.num_trials,
        seed=args.seed,
        verbose=True  # Always verbose when run from command line
    )
    
    # Display and save results
    print_and_save_results(
        epoch_results=epoch_results,
        all_epochs=all_epochs,
        sequence_length=args.sequence_length,
        sample_size=args.sample_size
    )
    
    # Create visualization
    savefig_path = f"results/figures/perplexity_comparison_{args.sequence_length}.png"
    create_perplexity_visualization(epoch_results, all_epochs, args.sequence_length, dir_to_label, savefig_path)


if __name__ == "__main__":
    main() 