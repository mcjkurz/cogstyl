#!/bin/bash

source cogstyl/bin/activate

echo "Epoch,Corpus,Mean_Loss"

epochs=(-1 0 1 2 3 4)
corpora=("mao" "chinese_novels")

for epoch in "${epochs[@]}"; do
    for corpus in "${corpora[@]}"; do
        echo -n "${epoch},${corpus},"
        
        python experiments/calculate_document_loss.py \
            "./models/model_epoch_${epoch}" \
            "./tokenizers/fineweb_char_tokenizer" \
            "./datasets/chunked/C_chunked_${corpus}.json" \
            --sample_size 1000 \
            --max_length 1024 2>/dev/null | tail -1
    done
done 