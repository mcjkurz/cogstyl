#!/bin/bash

set -e

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
echo "All perplexity calculations completed!"
echo ""
echo "Results saved in:"
for i in "${!DATA_FILES[@]}"; do
    OUTPUT_FOLDER="${OUTPUT_FOLDERS[$i]}"
    echo "  - results/$OUTPUT_FOLDER/ (epochs: ${EPOCHS[*]})"
done
echo "==========================================" 