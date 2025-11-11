#!/bin/bash

set -e

echo "=== GPT-2 Mao Corpus Fine-tuning ==="
echo "Stage: Fine-tuning on Mao dataset"
echo

if [[ ! -f "./datasets/test/C_mao.json" ]]; then
    echo "Error: ./datasets/test/C_mao.json not found"
    exit 1
fi

if [[ ! -d "./tokenizers/fineweb_char_tokenizer" ]]; then
    echo "Error: ./tokenizers/fineweb_char_tokenizer not found"
    exit 1
fi

if [[ ! -d "./models/model_epoch_-1" ]]; then
    echo "Error: ./models/model_epoch_-1 not found"
    echo "Please run train_base_corpus.sh first to create the base model"
    exit 1
fi

mkdir -p ./models

echo "=== Fine-tuning ==="
echo

python3 pipeline/train_gpt.py \
    --data_file "./datasets/test/C_mao.json" \
    --tokenizer_path "./tokenizers/fineweb_char_tokenizer" \
    --model_path "./models/model_epoch_-1" \
    --output_dir "./models" \
    --epochs 5 \
    --batch_size 8 \
    --start_lr 5e-5 \
    --end_lr 1e-6 \
    --grad_accum_steps 1 \
    --max_length 1024 \

echo "✓ Fine-tuning completed: Models saved as:"
echo "  - ./models/model_epoch_0"
echo "  - ./models/model_epoch_1" 
echo "  - ./models/model_epoch_2"
echo "  - ./models/model_epoch_3"
echo "  - ./models/model_epoch_4"
echo

echo "=== Fine-tuning Complete ==="
echo "Fine-tuned models available:"
echo "  - ./models/model_epoch_0 to model_epoch_4 (fine-tuned on Mao corpus)" 