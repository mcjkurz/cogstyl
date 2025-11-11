#!/usr/bin/env python3
"""
Simple script to upload a model and tokenizer to Hugging Face Hub.

Usage:
    python upload_to_hf.py --model_path /path/to/model --tokenizer_path /path/to/tokenizer --repo_name your-username/model-name

Requirements:
    pip install transformers huggingface_hub
"""

import argparse
import os
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import HfApi


def upload_model_and_tokenizer(model_path, tokenizer_path, repo_name, private=False):
    """
    Upload model and tokenizer to Hugging Face Hub.
    
    Args:
        model_path (str): Path to the model directory
        tokenizer_path (str): Path to the tokenizer directory
        repo_name (str): Repository name on HF Hub (format: username/model-name)
        private (bool): Whether to make the repository private
    """
    print(f"Loading model from: {model_path}")
    print(f"Loading tokenizer from: {tokenizer_path}")
    print(f"Uploading to repository: {repo_name}")
    
    try:
        # Load model and tokenizer
        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        print("✓ Model and tokenizer loaded successfully")
        
        # Upload to Hugging Face Hub
        print("Uploading model...")
        model.push_to_hub(repo_name, private=private)
        
        print("Uploading tokenizer...")
        tokenizer.push_to_hub(repo_name, private=private)
        
        print(f"✓ Successfully uploaded to https://huggingface.co/{repo_name}")
        
    except Exception as e:
        print(f"❌ Error during upload: {str(e)}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Upload model and tokenizer to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python upload_to_hf.py --model_path ./my_model --tokenizer_path ./my_tokenizer --repo_name username/my-model
    python upload_to_hf.py --model_path ./my_model --tokenizer_path ./my_tokenizer --repo_name username/my-model --private
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory"
    )
    
    parser.add_argument(
        "--tokenizer_path", 
        type=str,
        required=True,
        help="Path to the tokenizer directory"
    )
    
    parser.add_argument(
        "--repo_name",
        type=str,
        required=True,
        help="Repository name on HF Hub (format: username/model-name)"
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private (default: public)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.model_path):
        print(f"❌ Model path does not exist: {args.model_path}")
        return
    
    if not os.path.exists(args.tokenizer_path):
        print(f"❌ Tokenizer path does not exist: {args.tokenizer_path}")
        return
    
    # Check if user is logged in to HF
    try:
        api = HfApi()
        user = api.whoami()
        print(f"✓ Logged in as: {user['name']}")
    except Exception as e:
        print("❌ Not logged in to Hugging Face. Please run: huggingface-cli login")
        return
    
    # Upload
    success = upload_model_and_tokenizer(
        args.model_path,
        args.tokenizer_path, 
        args.repo_name,
        args.private
    )
    
    if success:
        print("\n🎉 Upload completed successfully!")
    else:
        print("\n💥 Upload failed!")


if __name__ == "__main__":
    main() 