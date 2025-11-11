# Cognitive Stylometry

This repository accompanies the article:

Kurzynski, Maciej, "Cognitive Stylometry: A Computational Study of Defamiliarization in Modern Chinese," *Computational Humanities Research* (forthcoming).

## Abstract

Autoregressive language models generate text by predicting the next word from the preceding context. The regularities internalized from specific training data make this mechanism a useful proxy for historically situated readerly expectations, reflecting what earlier linguistic communities would find probable or meaningful. In this article, I pre-train a GPT model (223M parameters) on a broad corpus of Chinese texts (*FineWeb Edu V2.1*) and fine-tune it on the collected writings of Mao Zedong (1893–1976) to simulate the evolving linguistic landscape of post-1949 China. Identifying token sequences with the sharpest drops in perplexity—a measure of the model's surprise—allows me to identify the core phraseology of "Maospeak," the militant language style that developed from Mao's writings and pronouncements. A comparative analysis of modern Chinese fiction reveals how literature becomes unfamiliar to the fine-tuned model, generating perplexity spikes of increasing magnitude. The findings suggest a mechanism of attentional control: whereas propaganda backgrounds meaning through repetition (cognitive overfitting), literature foregrounds it through deviation (non-anomalous surprise). By visualizing token sequences as perplexity landscapes with peaks and valleys, the article reconceives style as a probabilistic phenomenon and showcases the potential of "cognitive stylometry" for literary theory and close reading.

## Setup

Install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Optional - install Flash Attention for faster training (the installation might take some time):

```bash
pip install psutil
MAX_JOBS=4 pip install flash-attn --no-build-isolation
pip install -r requirements.txt
```

## Pipeline

**Note:** All HuggingFace downloads require authentication. Set `export HF_TOKEN=your_token` or use `--hf-token YOUR_TOKEN`.

### Option A: Quick Start (Download Pre-made Dataset & Models)

```bash
# Download pre-made base dataset
python3 pipeline/download_dataset.py \
  --dataset qhchina/cogstyl_base \
  --output-dir ./datasets/cogstyl_base \
  --hf-token YOUR_TOKEN
mv ./datasets/cogstyl_base/data/base_dataset.json ./datasets/base_dataset.json
rm -rf ./datasets/cogstyl_base

# Download pre-trained models (all epochs)
python3 pipeline/download_models.py \
  --epochs -1 0 1 2 3 4 \
  --output-dir ./models \
  --hf-token YOUR_TOKEN

# Skip to step 3 below
```

### Option B: Full Pipeline (Build from Scratch)

#### 1. Download Source Datasets

Downloads the FineWeb Edu Chinese corpus that will serve as the broad pre-training dataset (may require multiple requests due to HuggingFace rate limits).

```bash
# Download FineWeb for base dataset creation
python3 pipeline/download_dataset.py \
  --dataset "opencsg/Fineweb-Edu-Chinese-V2.1" \
  --data-dir "4_5" \
  --cache-dir "./datasets/fineweb_edu" \
  --hf-token YOUR_TOKEN \
  --num-proc 3
```

#### 2. Train Tokenizer and Create Base Dataset

Trains a character-level tokenizer on the FineWeb corpus and creates a deduplicated base dataset.

```bash
# Train character-level tokenizer
python3 pipeline/train_tokenizer.py \
  --dataset-path "./datasets/fineweb_edu/" \
  --dataset-name "opencsg/Fineweb-Edu-Chinese-V2.1" \
  --data-dir "4_5" \
  --tokenizer-type character \
  --vocab-size 20000 \
  --append-bert-tokens \
  --max-examples 2000000 \
  --output "./tokenizers/fineweb_char_tokenizer" \
  --no-streaming

# Create base training dataset
python3 pipeline/create_base_dataset.py \
  --dataset "opencsg/Fineweb-Edu-Chinese-V2.1" \
  --data-dir "4_5" \
  --target-size 1000000 \
  --min-doc-length 256 \
  --max-doc-length 1024 \
  --ngram-size 13 \
  --num-workers 10 \
  --output-path "./datasets/base_dataset.json" \
  --cache-dir "./datasets/fineweb_edu" \
  --hash-file-path "./datasets/test/test_hashes_13gram.pkl"
```

#### 3. Train Models

Pre-trains a GPT model on the broad FineWeb corpus, then fine-tunes it on the Mao corpus to simulate post-1949 linguistic expectations.

```bash
# Train base model (pretrained)
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

mv "./models/temp_base_training/model_epoch_0" "./models/model_epoch_-1"
rm -rf "./models/temp_base_training"

# Fine-tune on Mao corpus
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
  --max_length 1024
```

### 3. Download Test Datasets and Create Test Sets

Downloads the Mao corpus and Chinese novels corpus, then creates test datasets with n-gram hashes for deduplication validation.

```bash
# Download additional datasets
python3 pipeline/download_dataset.py \
  --dataset qhchina/maoxuan \
  --data-dir data \
  --output-dir datasets/mao \
  --hf-token YOUR_TOKEN

python3 pipeline/download_dataset.py \
  --dataset qhchina/100_top_chinese_novels \
  --data-dir data \
  --output-dir datasets/chinese_novels \
  --hf-token YOUR_TOKEN

# Create test datasets and hashes
python pipeline/create_test_datasets.py --output-dir "./datasets/test/"

python pipeline/create_md5_hashes.py \
  --input-dir ./datasets/test/ \
  --output ./datasets/test/test_hashes.pkl \
  --ngram-size 13
```

### 4. Chunk Test Documents

Divides test documents into overlapping chunks to ensure each token has sufficient context for perplexity calculation.

```bash
python pipeline/chunk_documents.py \
  ./datasets/test/C_chinese_novels.json \
  ./datasets/chunked/C_chunked_chinese_novels.json 100

python pipeline/chunk_documents.py \
  ./datasets/test/C_mao.json \
  ./datasets/chunked/C_chunked_mao.json 1000

python pipeline/chunk_documents.py \
  ./datasets/test/C_case_studies.json \
  ./datasets/chunked/C_chunked_case_studies.json 1000
```

### 5. Calculate Perplexities

Calculates token-level perplexities across all test corpora for each fine-tuning epoch to track how model expectations evolve.

```bash
bash bash/calculate_perplexities.sh
```

### 6. Analyze Results

Analyzes perplexity patterns to identify key phraseology, measure corpus-level surprise, and compare linguistic characteristics.

**Calculate document losses:**
Computes aggregate perplexity scores for entire corpora to measure overall model surprise.

```bash
python experiments/calculate_document_loss.py \
  ./models/model_epoch_2 \
  ./tokenizers/fineweb_char_tokenizer \
  ./datasets/chunked/C_chunked_mao.json \
  --sample_size 1000 \
  --max_length 1024

python experiments/calculate_document_loss.py \
  ./models/model_epoch_2 \
  ./tokenizers/fineweb_char_tokenizer \
  ./datasets/chunked/C_chunked_chinese_novels.json \
  --sample_size 1000 \
  --max_length 1024
```

**Find significant patterns:**
Identifies token sequences with the sharpest perplexity drops to extract characteristic phraseology.

```bash
python3 experiments/find_significant_patterns.py \
  --input-directory ./results/mao \
  --output ./results/patterns/mao_patterns.json \
  --top-n 200 \
  --mode "top" \
  --max-overlap 5
```

**Calculate n-gram entropy:**
Measures lexical diversity and repetition patterns across corpora using n-gram frequency distributions.

```bash
python experiments/ngram_entropy.py \
  --corpus1 "./datasets/chunked/C_chunked_mao.json" \
  --corpus2 "./datasets/chunked/C_chunked_chinese_novels.json" \
  --corpus1-name "Mao" \
  --corpus2-name "Novels" \
  --output-dir "./results/ngram_entropy"
```

**Compare average perplexities:**
Contrasts mean perplexity trajectories across fine-tuning epochs to visualize diverging model expectations.

```bash
python experiments/compare_avg_perplexities.py \
  --dir1 "./results/mao" \
  --dir2 "./results/chinese_novels" \
  --label1 "Selected Works of Mao Zedong" \
  --label2 "Top 100 Chinese Novels"
```

## Visualization

Launches an interactive Streamlit interface for exploring perplexity patterns and token-level surprises across texts (requires a .json file produced by find_significant_patterns.py).

```bash
cd app
streamlit run pattern_viewer.py
```

## License

MIT License
