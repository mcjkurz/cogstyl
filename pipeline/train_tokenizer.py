#!/usr/bin/env python3
"""Train tokenizers (BPE or character-based) from HuggingFace datasets."""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator
from tqdm.auto import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers, processors
from transformers import PreTrainedTokenizerFast, BertTokenizerFast

script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from misc.utils import convert_to_chinese_punctuation


def showcase_tokenizer(tokenizer, tokenizer_name: str = "Tokenizer"):
    print(f"\n=== {tokenizer_name} Showcase ===")
    
    test_sentences = [
        "你好",
        "你好世界",
        "我爱中国",
        "今天天气很好",
        "学习中文很有趣",
        "人工智能的发展很快",
    ]
    
    print(f"\nVocabulary size: {tokenizer.get_vocab_size()}")
    
    special_tokens = ["<PAD>", "<UNK>"]
    print("\nSpecial tokens:")
    for token in special_tokens:
        token_id = tokenizer.token_to_id(token)
        print(f"  {token}: {token_id}")
    
    print("\n" + "="*60)
    print("Tokenization Examples:")
    print("="*60)
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{i}. Input: '{sentence}'")
        
        encoded = tokenizer.encode(sentence)
        print(f"   Token IDs: {encoded.ids}")
        
        tokens = encoded.tokens
        print(f"   Tokens: {tokens}")
        
        decoded = tokenizer.decode(encoded.ids)
        print(f"   Decoded: '{decoded}'")
        
        print(f"   Length: {len(encoded.ids)} total")
    
    print("\n" + "="*60)
    print("✓ Tokenizer showcase complete!")
    print("="*60)

def set_seed(seed):
    random.seed(seed)


def round_to_nearest_multiple_of_32(size):
    return ((size + 16) // 32) * 32


def train_character_tokenizer(dataset, topn_chars: int = 20000, max_examples=None, append_bert_tokens=False) -> Tokenizer:
    print("Counting character frequencies...")
    char_counter = Counter()
    
    count = 0
    total = max_examples
    if total is None:
        try:
            total = len(dataset)
        except:
            total = None
    
    for example in tqdm(dataset, total=total):
        if max_examples is not None and count >= max_examples:
            break
        text = example["text"]
        text = convert_to_chinese_punctuation(text)
        char_counter.update(text)
        count += 1
    
    if max_examples is not None:
        print(f"Processed {count} examples (limited by max_examples={max_examples})")
    
    most_common_chars = char_counter.most_common(topn_chars)
    
    sorted_chars = [char for char, _ in most_common_chars]
    
    if append_bert_tokens:
        print("Adding BERT Chinese characters to vocabulary...")
        bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
        
        bert_standalone_chars = set(
            token for token in bert_tokenizer.vocab.keys() 
            if not token.startswith("##") and len(token) == 1
        )
        
        original_count = len(sorted_chars)
        for char in bert_standalone_chars:
            if char not in sorted_chars:
                sorted_chars.append(char)
        
        print(f"Added {len(sorted_chars) - original_count} BERT characters")
        print(f"Total characters before rounding: {len(sorted_chars)} (original {topn_chars} + BERT additions)")
    
    total_vocab_size = 2 + len(sorted_chars)
    
    rounded_vocab_size = round_to_nearest_multiple_of_32(total_vocab_size)
    target_char_count = rounded_vocab_size - 2
    
    if target_char_count != len(sorted_chars):
        if target_char_count < len(sorted_chars):
            sorted_chars = sorted_chars[:target_char_count]
            print(f"Trimmed vocabulary to {len(sorted_chars)} characters")
        else:
            original_count = len(sorted_chars)
            while len(sorted_chars) < target_char_count:
                sorted_chars.append(f"<UNUSED_{len(sorted_chars) - original_count}>")
            print(f"Padded vocabulary with {target_char_count - original_count} unused tokens")
    
    print(f"Final vocabulary size: {rounded_vocab_size} (divisible by 32)")
    print(f"Character tokens: {len(sorted_chars)}, Special tokens: 2")
    
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }
    
    for i, char in enumerate(sorted_chars):
        vocab[char] = i + 2
    
    print(f"Building character tokenizer with {len(vocab)} tokens...")
    
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<UNK>"))
    
    tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="removed")
    
    print(f"Character tokenizer trained with {tokenizer.get_vocab_size()} tokens")
    return tokenizer


def get_training_corpus(dataset, max_examples=None) -> Iterator[str]:
    count = 0
    for example in dataset:
        if max_examples is not None and count >= max_examples:
            break
        yield example["text"]
        count += 1


def save_tokenizer(tokenizer: Tokenizer, output_path: Path, tokenizer_name: str):
    tokenizer_path = output_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    hf_tokenizer.pad_token = "<PAD>"
    hf_tokenizer.unk_token = "<UNK>"
    hf_tokenizer.pad_token_id = tokenizer.token_to_id("<PAD>")
    hf_tokenizer.unk_token_id = tokenizer.token_to_id("<UNK>")
    
    hf_tokenizer.save_pretrained(output_path)
    
    print(f"{tokenizer_name} tokenizer saved to {output_path}")
    print(f"Pad token ID: {hf_tokenizer.pad_token_id}")
    print(f"UNK token ID: {hf_tokenizer.unk_token_id}")


def train_bpe_tokenizer(dataset, vocab_size: int = 20000, max_examples=None) -> Tokenizer:
    print("Training BPE tokenizer...")
    
    rounded_vocab_size = round_to_nearest_multiple_of_32(vocab_size)
    if rounded_vocab_size != vocab_size:
        print(f"Rounded vocabulary size from {vocab_size} to {rounded_vocab_size} (divisible by 32)")
    else:
        print(f"Vocabulary size {vocab_size} is already divisible by 32")
    
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    trainer = trainers.BpeTrainer(
        vocab_size=rounded_vocab_size,
        special_tokens=["<PAD>", "<UNK>"],
        show_progress=True
    )
    
    if max_examples is not None:
        print(f"Training on max {max_examples} examples")
    tokenizer.train_from_iterator(get_training_corpus(dataset, max_examples), trainer)
    
    print(f"BPE tokenizer trained with {tokenizer.get_vocab_size()} tokens")
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train tokenizers from local datasets")
    parser.add_argument("--dataset-path", "-d", required=True,
                       help="Path to locally cached dataset directory")
    parser.add_argument("--dataset-name", default=None,
                       help="Original dataset name (for loading from cache)")
    parser.add_argument("--data-dir", default=None,
                       help="Subdirectory within the dataset (if applicable)")
    parser.add_argument("--split", "-s", default="train",
                       help="Dataset split to use (default: train)")
    parser.add_argument("--tokenizer-type", "-t", choices=["character", "bpe"], 
                       default="character",
                       help="Tokenizer type (default: character)")
    parser.add_argument("--vocab-size", "-v", type=int, default=20000,
                       help="For BPE: vocabulary size. For character: top N characters (default: 20000)")
    parser.add_argument("--append-bert-tokens", action="store_true",
                       help="For character tokenizer: append BERT Chinese characters")
    parser.add_argument("--output", "-o", required=True,
                       help="Output directory to save tokenizer")
    parser.add_argument("--no-streaming", action="store_true",
                       help="Disable streaming mode (loads entire dataset into memory)")
    parser.add_argument("--max-examples", type=int, default=None,
                       help="Maximum number of examples to use for training (default: use all)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    print(f"Setting random seed to {args.seed}")
    set_seed(args.seed)
    
    args.streaming = not args.no_streaming
    
    print(f"Loading dataset from: {args.dataset_path}")
    print(f"Split: {args.split}")
    print(f"Streaming: {args.streaming}")
    
    if args.dataset_name:
        dataset = load_dataset(
            args.dataset_name,
            data_dir=args.data_dir,
            split=args.split,
            cache_dir=args.dataset_path,
            streaming=args.streaming
        )
    else:
        try:
            dataset = load_dataset(
                args.dataset_path,
                split=args.split,
                streaming=args.streaming
            )
        except Exception as e:
            print(f"Failed to load from path directly: {e}")
            print("Please provide --dataset-name if loading from cache directory")
            return
    
    print("Dataset loaded successfully")
    if not args.streaming:
        print(f"Dataset size: {len(dataset)} examples")
    else:
        print("Dataset in streaming mode - size not available")
    
    try:
        print(f"Dataset features: {dataset.features}")
    except:
        print("Could not retrieve dataset features")
    
    if args.max_examples is not None:
        print(f"Training will use maximum {args.max_examples} examples")
    else:
        print("Training will use all available examples")
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.tokenizer_type == "character":
        tokenizer = train_character_tokenizer(dataset, args.vocab_size, args.max_examples, args.append_bert_tokens)
        save_tokenizer(tokenizer, output_path, "Character")
        showcase_tokenizer(tokenizer, "Character Tokenizer")
        
    elif args.tokenizer_type == "bpe":
        tokenizer = train_bpe_tokenizer(dataset, args.vocab_size, args.max_examples)
        save_tokenizer(tokenizer, output_path, "BPE")
        showcase_tokenizer(tokenizer, "BPE Tokenizer")

if __name__ == "__main__":
    main() 