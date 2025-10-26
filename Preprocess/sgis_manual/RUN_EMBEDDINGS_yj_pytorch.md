# Running Local Embeddings with yj_pytorch Environment

## Environment Setup Complete

**Environment**: `yj_pytorch` (Python 3.12)
**GPU**: NVIDIA GeForce RTX 4070 (12GB VRAM)
**Status**: Installing PyTorch with CUDA 12.1

---

## Step 1: Verify Installation (After conda/pip finishes)

In Anaconda Prompt with `yj_pytorch` activated:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

**Expected output:**
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4070
```

---

## Step 2: Set Hugging Face Token

You need a Hugging Face token to download Llama-3.2-3B-Instruct:

### Get Token:
1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access is enough)
3. Request access to meta-llama/Llama-3.2-3B-Instruct at:
   https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

### Set Token (in Anaconda Prompt):

**Windows (Anaconda Prompt):**
```bash
set HF_TOKEN=your_token_here
```

**Or edit the script directly:**
Open `generate_local_embeddings.py` and on line 44, add your token:
```python
hf_token = os.environ.get('HF_TOKEN', 'your_token_here')
```

---

## Step 3: Generate Embeddings

In Anaconda Prompt:

```bash
# Activate environment
conda activate yj_pytorch

# Navigate to directory
cd C:\Users\jour\Documents\GitHub\airbnb\Preprocess\sgis_manual

# Run embedding generation
python generate_local_embeddings.py
```

---

## What to Expect

### First Run (Model Download):
```
Loading model configuration from meta-llama/Llama-3.2-3B-Instruct...
Loading tokenizer...
Loading model (this may take a few minutes)...
Downloading: 100%|████████████| 6.0GB/6.0GB [XX:XX<00:00, XXMiB/s]
✓ Model loaded successfully
  Embedding dimension: 3072
```

This downloads ~6GB model (one-time only).

### Embedding Generation Progress:
```
Generating embeddings for 28,274 prompts...
Processing local prompts:  35%|████     | 9895/28274 [12:34<23:45, 12.89it/s]
```

**Time estimate with RTX 4070:**
- Processing speed: ~10-15 prompts/second
- Total time: **30-45 minutes**
- GPU memory usage: ~6-8 GB

### Completion:
```
✓ Embedding generation complete
  Total prompts: 28274
  Failed: 0
  Success rate: 100.0%

Saving embeddings to sgis_local_llm_embeddings.csv...

================================================================================
COMPLETE!
================================================================================
✓ Output saved to: sgis_local_llm_embeddings.csv
  Shape: (28274, 3074)
  Embedding dimensions: 3072
```

---

## Step 4: Verify Output

```bash
python -c "import pandas as pd; df = pd.read_csv('sgis_local_llm_embeddings.csv'); print(f'Shape: {df.shape}'); print(f'Expected: (28274, 3074)'); print(f'Columns: {df.columns[:5].tolist()}...')"
```

Should show:
```
Shape: (28274, 3074)
Expected: (28274, 3074)
Columns: ['Reporting Month', 'Dong_name', 'dim_0', 'dim_1', 'dim_2']...
```

---

## Step 5: Test with Model

After embeddings are generated, test with the model:

```bash
# Switch back to yj_env for model training
conda activate yj_env

cd C:\Users\jour\Documents\GitHub\airbnb\Model

# Test: Raw + Local Embeddings
python main.py \
  --embed1 raw \
  --embed2 sgis_local_llm \
  --model transformer \
  --epochs 5 \
  --batch_size 8 \
  --window_size 9 \
  --mode 3m \
  --label all
```

**Compare results to baseline (RMSE = 0.810)**

---

## Troubleshooting

### "CUDA out of memory"
The script automatically handles this by reducing chunk size. If it persists:
- Close other GPU-using applications (browsers, etc.)
- The script will fall back to smaller chunk sizes automatically

### "Access denied" for Llama model
- Check HF_TOKEN is set correctly
- Verify you requested access at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- Wait for approval (usually instant, but can take hours)

### Model download is slow
- First download takes time (~6GB)
- Model is cached in `~/.cache/huggingface/`
- Subsequent runs use cached model (fast)

### Script crashes
Check available GPU memory:
```bash
nvidia-smi
```
Should show ~3-4GB free before starting.

---

## File Locations

**Input**: `Preprocess/sgis_manual/sgis_local_prompts.csv` (28,274 prompts)
**Output**: `Preprocess/sgis_manual/sgis_local_llm_embeddings.csv` (28,274 × 3,074)
**Model cache**: `C:\Users\jour\.cache\huggingface\hub\`

---

## Quick Commands Summary

```bash
# 1. Verify CUDA
conda activate yj_pytorch
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 2. Set token
set HF_TOKEN=your_token_here

# 3. Generate embeddings
cd C:\Users\jour\Documents\GitHub\airbnb\Preprocess\sgis_manual
python generate_local_embeddings.py

# 4. Test with model (in yj_env)
conda activate yj_env
cd C:\Users\jour\Documents\GitHub\airbnb\Model
python main.py --embed1 raw --embed2 sgis_local_llm --model transformer --epochs 5 --batch_size 8 --window_size 9 --mode 3m --label all
```

---

Good luck! The hard part (environment setup) is done. Now it's just waiting for the embeddings to generate!
