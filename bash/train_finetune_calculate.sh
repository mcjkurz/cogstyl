#!/bin/bash

set -e

echo "=========================================="
echo "=== COMBINED TRAINING PIPELINE ==="
echo "=========================================="
echo "This script will:"
echo "1. Train GPT-2 base model from scratch"
echo "2. Fine-tune on Mao corpus"
echo "3. Calculate perplexities for all datasets"
echo ""

echo "=========================================="
echo "=== STAGE 1: GPT-2 Base Corpus Training ==="
echo "=========================================="
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
    --min_length 64 \
    --grad_accum_steps 8 \
    --warmup_batches 4096 \
    --n_layer 24 \
    --n_head 16 \
    --n_embd 1024 \
    --n_ctx 1024 \
    --n_positions 1024 \
    --attn_implementation "flash_attention_2" \
    --compile \
    --autocast

if [ -d "./models/model_epoch_-1" ]; then
    mv "./models/model_epoch_-1" "./models/model_epoch_-1_old"
    echo "Renamed model_epoch_-1 to model_epoch_-1_old"
fi
echo "Moving base model to ./models/model_epoch_-1"
mv "./models/temp_base_training/model_epoch_0" "./models/model_epoch_-1"
rm -rf "./models/temp_base_training"

echo "✓ Base corpus training completed: Model saved as ./models/model_epoch_-1"
echo
echo "=== Stage 1 Complete ==="
echo "Base model available at: ./models/model_epoch_-1"
echo

echo "=========================================="
echo "=== STAGE 2: GPT-2 Mao Corpus Fine-tuning ==="
echo "=========================================="
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
    echo "Base model training must have failed"
    exit 1
fi

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
    --min_length 64 \
    --autocast

echo "✓ Fine-tuning completed: Models saved as:"
echo "  - ./models/model_epoch_0"
echo "  - ./models/model_epoch_1" 
echo "  - ./models/model_epoch_2"
echo "  - ./models/model_epoch_3"
echo "  - ./models/model_epoch_4"
echo

echo "=== Stage 2 Complete ==="
echo "Fine-tuned models available:"
echo "  - ./models/model_epoch_0 to model_epoch_4 (fine-tuned on Mao corpus)"
echo

echo "=========================================="
echo "=== STAGE 3: Perplexity Calculations ==="
echo "=========================================="

TOKENIZER_PATH="./tokenizers/fineweb_char_tokenizer"
CONTEXT_SIZE=256
BATCH_SIZE=512

DATA_FILES=(
    "./datasets/chunked/C_chunked_mao.json"
    "./datasets/chunked/C_chunked_case_studies.json"
    "./datasets/chunked/C_chunked_chinese_novels.json"
)

OUTPUT_FOLDERS=(
    "mao"
    "case_studies"
    "chinese_novels"
)

EPOCHS=(-1 0 1 2 3 4)

echo "Starting perplexity calculations for ${#DATA_FILES[@]} datasets and ${#EPOCHS[@]} model epochs..."
echo "Tokenizer: $TOKENIZER_PATH"
echo "Context size: $CONTEXT_SIZE"
echo "Batch size: $BATCH_SIZE"
echo ""

if [ ! -d "$TOKENIZER_PATH" ]; then
    echo "ERROR: Tokenizer path $TOKENIZER_PATH does not exist."
    exit 1
fi

for i in "${!DATA_FILES[@]}"; do
    DATA_FILE="${DATA_FILES[$i]}"
    OUTPUT_FOLDER="${OUTPUT_FOLDERS[$i]}"
    
    echo "=========================================="
    echo "Processing dataset: $DATA_FILE"
    echo "Output folder: results/$OUTPUT_FOLDER"
    echo "=========================================="
    
    if [ ! -f "$DATA_FILE" ]; then
        echo "WARNING: Data file $DATA_FILE does not exist. Skipping."
        continue
    fi
    
    mkdir -p "results"
    mkdir -p "results/$OUTPUT_FOLDER"
    
    for epoch in "${EPOCHS[@]}"; do
        echo "----------------------------------------"
        echo "Processing epoch $epoch for $(basename $DATA_FILE)"
        echo "----------------------------------------"
        
        MODEL_PATH="./models/model_epoch_${epoch}"
        
        if [ ! -d "$MODEL_PATH" ]; then
            echo "WARNING: Model path $MODEL_PATH does not exist. Skipping epoch $epoch."
            continue
        fi
        
        OUTPUT_FILE="results/$OUTPUT_FOLDER/epoch_${epoch}_perplexities.json"
        
        echo "Running perplexity calculation..."
        echo "  Model: $MODEL_PATH"
        echo "  Data: $DATA_FILE"
        echo "  Output: $OUTPUT_FILE"
        
        python experiments/calculate_perplexity.py \
            --input "$DATA_FILE" \
            --model "$MODEL_PATH" \
            --tokenizer "$TOKENIZER_PATH" \
            --output "$OUTPUT_FILE" \
            --context-size "$CONTEXT_SIZE" \
            --batch-size "$BATCH_SIZE" \
            --compile
        
        echo "✓ Completed epoch $epoch for $(basename $DATA_FILE)"
        echo ""
    done
    
    echo "✓ Completed all epochs for $DATA_FILE"
    echo ""
done

echo "=========================================="
echo "=== STAGE 3 Complete ==="
echo "All perplexity calculations completed!"
echo ""
echo "Results saved in:"
for i in "${!DATA_FILES[@]}"; do
    OUTPUT_FOLDER="${OUTPUT_FOLDERS[$i]}"
    echo "  - results/$OUTPUT_FOLDER/ (epochs: ${EPOCHS[*]})"
done
echo "=========================================="

echo ""
echo "=========================================="
echo "=== ENTIRE PIPELINE COMPLETE ==="
echo "=========================================="
echo "Successfully completed:"
echo "1. ✓ Base model training (./models/model_epoch_-1)"
echo "2. ✓ Fine-tuning on Mao corpus (./models/model_epoch_0 to 4)"
echo "3. ✓ Perplexity calculations for all datasets and epochs"
echo ""
echo "All models and results are ready for analysis!"
echo "=========================================="