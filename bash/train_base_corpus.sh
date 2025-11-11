#!/bin/bash

set -e

echo "=== GPT-2 Base Corpus Training ==="
echo "Stage: Training from scratch on base dataset"
echo

if [[ ! -f "./datasets/base_dataset.json" ]]; then
    echo "Error: ./datasets/base_dataset.json not found"
    exit 1
fi

if [[ ! -d "./tokenizers/fineweb_char_tokenizer" ]]; then
    echo "Error: ./tokenizers/fineweb_char_tokenizer not found"
    exit 1
fi

mkdir -p ./models

echo "=== Training from scratch ==="

echo
python3 pipeline/train_gpt.py \
    --data_file "./datasets/base_dataset.json" \
    --tokenizer_path "./tokenizers/fineweb_char_tokenizer" \
    --output_dir "./models/temp_base_training" \
    --epochs 1 \
    --batch_size 32 \
    --start_lr 2e-4 \
    --end_lr 5e-5 \
    --max_length 1024 \
    --grad_accum_steps 8 \
    --warmup_batches 4096 \
    --n_layer 16 \
    --n_head 16 \
    --n_embd 1024 \
    --n_ctx 1024 \
    --n_positions 1024 \
    --attn_implementation "flash_attention_2" \
    --compile

echo "Moving base model to ./models/model_epoch_-1"
mv "./models/temp_base_training/model_epoch_0" "./models/model_epoch_-1"
rm -rf "./models/temp_base_training"

echo "✓ Base corpus training completed: Model saved as ./models/model_epoch_-1"
echo
echo "=== Training Complete ==="
echo "Base model available at: ./models/model_epoch_-1" 