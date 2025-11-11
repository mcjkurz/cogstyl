#!/usr/bin/env python3
"""Download pre-trained models and tokenizer from HuggingFace."""

import argparse
import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download


def save_tokenizer_from_model(model_path: str, output_dir: str = "./tokenizers/fineweb_char_tokenizer"):
    """Copy tokenizer files from model directory to tokenizer directory.
    
    Args:
        model_path: Path to model directory containing tokenizer files
        output_dir: Directory to save tokenizer
    """
    tokenizer_dir = Path(output_dir)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = Path(model_path)
    tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']
    
    print(f"Copying tokenizer files to {tokenizer_dir}...")
    
    for file in tokenizer_files:
        src = model_path / file
        if src.exists():
            shutil.copy2(src, tokenizer_dir / file)
    
    print(f"✓ Tokenizer saved to {tokenizer_dir}")
    return str(tokenizer_dir)


def download_model(epoch: int, output_dir: str = "./models", hf_token: str = None):
    """Download a specific model epoch from HuggingFace.
    
    Args:
        epoch: Model epoch (-1 for pretrained base model, 0-4 for fine-tuned)
        output_dir: Base directory to save models
        hf_token: HuggingFace authentication token
    """
    repo_id = f"qhchina/fineweb_edu_mao_{epoch}"
    model_dir = Path(output_dir) / f"model_epoch_{epoch}"
    
    print(f"Downloading {repo_id}...")
    
    auth_token = hf_token or os.getenv("HF_TOKEN")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_dir),
            repo_type="model",
            token=auth_token,
            local_dir_use_symlinks=False
        )
        print(f"✓ Downloaded to {model_dir}")
        return str(model_dir)
        
    except Exception as e:
        print(f"✗ Failed to download {repo_id}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Download pre-trained models and tokenizer from HuggingFace")
    parser.add_argument("--epochs", "-e", type=int, nargs="+", default=[-1, 0, 1, 2, 3, 4],
                       help="Model epochs to download (default: -1 0 1 2 3 4)")
    parser.add_argument("--output-dir", "-o", default="./models",
                       help="Directory to save models (default: ./models)")
    parser.add_argument("--tokenizer-dir", "-t", default="./tokenizers/fineweb_char_tokenizer",
                       help="Directory to save tokenizer (default: ./tokenizers/fineweb_char_tokenizer)")
    parser.add_argument("--skip-tokenizer", action="store_true",
                       help="Skip tokenizer extraction if already exists")
    parser.add_argument("--hf-token", default=None,
                       help="HuggingFace authentication token (or set HF_TOKEN env var)")
    
    args = parser.parse_args()
    
    print("Downloading models and tokenizer from HuggingFace")
    print("=" * 50)
    
    # Download models
    base_model_path = None
    for epoch in args.epochs:
        model_path = download_model(epoch, args.output_dir, args.hf_token)
        # Track base model for tokenizer extraction
        if epoch == -1:
            base_model_path = model_path
    
    # Extract tokenizer from base model if needed
    tokenizer_exists = Path(args.tokenizer_dir).exists()
    if tokenizer_exists and args.skip_tokenizer:
        print(f"\nTokenizer already exists at {args.tokenizer_dir}, skipping extraction")
    else:
        if base_model_path:
            print()
            save_tokenizer_from_model(base_model_path, args.tokenizer_dir)
        elif -1 not in args.epochs:
            print(f"\n⚠ Warning: Base model (epoch -1) not downloaded. Tokenizer not extracted.")
            print(f"  To extract tokenizer, download epoch -1 or run:")
            print(f"  python pipeline/download_models.py -e -1")
    
    print("\n✓ All downloads completed successfully")


if __name__ == "__main__":
    main()

