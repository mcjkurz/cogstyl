#!/bin/bash

echo "Running perplexity comparison experiments for different sequence lengths..."

DIR1="./results/mao"
DIR2="./results/chinese_novels"

LABEL1="Selected Works of Mao Zedong"
LABEL2="Top 100 Chinese Novels"

SAMPLE_SIZE=1000
NUM_TRIALS=10

if [ ! -d "$DIR1" ]; then
    echo "Warning: Directory $DIR1 does not exist"
fi

if [ ! -d "$DIR2" ]; then
    echo "Warning: Directory $DIR2 does not exist"
fi

for length in 16 64 128 256; do
    echo ""
    echo "=================================="
    echo "Running experiment for $length-character sequences..."
    echo "Sample size: $SAMPLE_SIZE sequences per trial"
    echo "Number of trials: $NUM_TRIALS"
    echo "=================================="
    
    python3 experiments/compare_avg_perplexities.py "$DIR1" "$DIR2" \
        --sequence-length $length \
        --labels "$LABEL1" "$LABEL2" \
        --sample-size $SAMPLE_SIZE \
        --num-trials $NUM_TRIALS
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed $length-character analysis"
    else
        echo "✗ Failed $length-character analysis"
    fi
done

echo ""
echo "All experiments completed!"
echo "Generated files:"
ls -la perplexity_comparison_*.png 2>/dev/null || echo "No visualization files found" 