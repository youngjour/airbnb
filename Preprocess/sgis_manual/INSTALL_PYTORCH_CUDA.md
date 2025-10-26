# Install PyTorch with CUDA in yj_env

## Current Status
- Environment: `yj_env` (already active)
- GPU: NVIDIA GeForce RTX 4070 (12GB VRAM)
- CUDA Driver: 12.8
- Issue: PyTorch 2.9.0+cpu (CPU-only, no CUDA support)

## Solution: Install PyTorch with CUDA

### Option 1: Using Conda (RECOMMENDED)

Open **Anaconda Prompt** (not regular command prompt) and run:

```bash
# Activate your environment (if not already active)
conda activate yj_env

# Install PyTorch with CUDA 12.1 support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

This will install CUDA-enabled PyTorch from official conda channels.

### Option 2: Using pip

If conda doesn't work, try pip in Anaconda Prompt:

```bash
conda activate yj_env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Option 3: PyTorch 2.0 with CUDA 11.8 (More Compatible)

If CUDA 12.1 has issues, use the more widely compatible CUDA 11.8:

```bash
conda activate yj_env
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Verify Installation

After installation, verify CUDA is working:

```python
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4070
```

## Then Resume Embedding Generation

Once PyTorch with CUDA is installed:

```bash
cd C:\Users\jour\Documents\GitHub\airbnb\Preprocess\sgis_manual
python generate_local_embeddings.py
```

This will use your RTX 4070 GPU for fast embedding generation (~30-45 minutes instead of 3-4 hours on CPU).
