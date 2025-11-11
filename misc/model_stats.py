#!/usr/bin/env python3
"""
Model Statistics Analyzer

This script loads a pre-trained model and displays comprehensive statistics about its
architecture, parameters, and configuration.
"""

import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_number(num):
    """Format large numbers with appropriate suffixes."""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)

def get_model_size_mb(model):
    """Calculate model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def analyze_model(model_path: str):
    """Analyze model and print comprehensive statistics."""
    print("=" * 60)
    print(f"MODEL STATISTICS: {model_path}")
    print("=" * 60)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    config = model.config
    
    print(f"\n📊 ARCHITECTURE OVERVIEW")
    print(f"{'Model Type:':<25} GPT2LMHeadModel")
    print(f"{'Vocabulary Size:':<25} {config.vocab_size:,}")
    print(f"{'Number of Layers:':<25} {config.n_layer}")
    print(f"{'Embedding Dimension:':<25} {config.n_embd}")
    print(f"{'Attention Heads:':<25} {config.n_head}")
    print(f"{'Context Length:':<25} {config.n_ctx}")
    print(f"{'Max Position Embeddings:':<25} {config.n_positions}")
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print(f"\n🔢 PARAMETER STATISTICS")
    print(f"{'Total Parameters:':<25} {total_params:,} ({format_number(total_params)})")
    print(f"{'Trainable Parameters:':<25} {trainable_params:,} ({format_number(trainable_params)})")
    print(f"{'Non-trainable Parameters:':<25} {non_trainable_params:,} ({format_number(non_trainable_params)})")
    
    # Memory usage
    model_size_mb = get_model_size_mb(model)
    print(f"\n💾 MEMORY USAGE")
    print(f"{'Model Size:':<25} {model_size_mb:.2f} MB")
    print(f"{'Model Size:':<25} {model_size_mb/1024:.2f} GB")
    
    # Layer breakdown
    print(f"\n🏗️  LAYER BREAKDOWN")
    layer_counts = {}
    param_counts = {}
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type not in layer_counts:
            layer_counts[module_type] = 0
            param_counts[module_type] = 0
        layer_counts[module_type] += 1
        param_counts[module_type] += sum(p.numel() for p in module.parameters())
    
    # Sort by parameter count
    sorted_layers = sorted(param_counts.items(), key=lambda x: x[1], reverse=True)
    
    for layer_type, param_count in sorted_layers:
        if param_count > 0:  # Only show layers with parameters
            count = layer_counts[layer_type]
            print(f"{'  ' + layer_type + ':':<25} {count:>3} layers, {param_count:>12,} params ({format_number(param_count)})")
    
    # Configuration details
    print(f"\n⚙️  CONFIGURATION DETAILS")
    config_dict = config.to_dict()
    important_configs = [
        'activation_function', 'attn_pdrop', 'embd_pdrop', 'resid_pdrop',
        'layer_norm_epsilon', 'initializer_range', 'use_cache'
    ]
    
    for key in important_configs:
        if key in config_dict:
            print(f"{'  ' + key.replace('_', ' ').title() + ':':<25} {config_dict[key]}")
    
    # Calculate theoretical FLOPS (rough estimate)
    print(f"\n🧮 COMPUTATIONAL COMPLEXITY (per forward pass)")
    # Rough FLOPS calculation for transformer: 2 * n_params * sequence_length
    flops_per_token = 2 * total_params
    print(f"{'FLOPs per token:':<25} {format_number(flops_per_token)}")
    print(f"{'FLOPs at max context:':<25} {format_number(flops_per_token * config.n_ctx)}")
    
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Analyze model architecture and statistics')
    parser.add_argument('model_path', type=str, help='Path to pre-trained model directory')
    
    args = parser.parse_args()
    
    try:
        analyze_model(args.model_path)
    except Exception as e:
        logger.error(f"Error analyzing model: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 