# HJ Baseline LLM Embeddings Generation Guide

## Overview

This guide explains how to generate all LLM embeddings for the HJ baseline model using meta-llama/Llama-3.2-3B-Instruct.

## What is the HJ Baseline?

The HJ baseline (Hongju's baseline) is a transformer model that uses **LLM embeddings** from three types of features:
1. **road_llm**: Road network and transportation infrastructure
2. **hf_llm**: Human flow (floating population demographics)
3. **llm_w** or **llm_wo**: Airbnb features (with or without listing titles)

The original paper didn't upload the generated embedding files, so we need to recreate them.

## Your Goal

Add **local district embeddings (sgis_local_llm)** to the HJ baseline to test if local business composition and market saturation improve Airbnb demand forecasting.

---

## Complete Pipeline

### Phase 1: Prompt Generation (**COMPLETED**)

Generated natural language prompts from structured data.

**Files Created:**
```
Preprocess/prompt_construct/prompt_construct/
├── dong_prompts_new/
│   ├── AirBnB_SSP_wo_prompts.csv    (28,408 prompts - Airbnb without listings)
│   └── AirBnB_SSP_w_prompts.csv     (28,408 prompts - Airbnb with listings)
└── dong_prompts/
    ├── road_prompts.csv             (28,408 prompts - Road network)
    └── human_flow_prompts.csv       (28,408 prompts - Human flow)
```

**Generation Scripts:**
- `generate_airbnb_prompts.py` - Airbnb features (Category, Binary, Numerical attributes)
- `generate_road_prompts.py` - Road types, tunnels, bridges, bus/subway ridership
- `generate_hf_prompts.py` - Domestic and foreign floating population by age/gender

**Status:**
- [X] Road network prompts - COMPLETE (28,408 prompts)
- [X] Human flow prompts - COMPLETE (28,408 prompts)
- [ ] Airbnb prompts - IN PROGRESS (~3%, estimated 30-35 min remaining)

### Phase 2: LLM Embedding Generation (**PENDING**)

Convert prompts to 3,072-dimensional embeddings using Llama-3.2-3B-Instruct.

**Tool:** `generate_hj_embeddings.py`

**Usage:**
```bash
# Activate PyTorch environment
conda activate yj_pytorch

# Set Hugging Face token
set HF_TOKEN=your_token_here

# Navigate to directory
cd C:\Users\jour\Documents\GitHub\airbnb\Preprocess\prompt_construct\prompt_construct

# Generate all embeddings (sequential)
python generate_hj_embeddings.py

# OR generate specific embedding type
python generate_hj_embeddings.py llm_wo      # Airbnb without listings
python generate_hj_embeddings.py llm_w       # Airbnb with listings
python generate_hj_embeddings.py road_llm    # Road network
python generate_hj_embeddings.py hf_llm      # Human flow
```

**Time Estimates** (with RTX 4070 GPU):
- llm_wo: ~30-45 minutes
- llm_w: ~30-45 minutes (longer prompts due to listing titles)
- road_llm: ~30-45 minutes
- hf_llm: ~30-45 minutes

**Total time: ~2-3 hours for all four**

**Output Files:**
```
Data/Preprocessed_data/Dong/llm_embeddings_new/
├── Airbnb_SSP_wo.csv      (28,408 × 3,074)
├── Airbnb_SSP_w.csv       (28,408 × 3,074)
├── road_llm.csv           (28,408 × 3,074)
└── human_flow_llm.csv     (28,408 × 3,074)
```

---

## Testing Strategy

### Option 1: HJ Baseline Only (Reproduce Original)

Test the original HJ baseline configuration:

```bash
cd C:\Users\jour\Documents\GitHub\airbnb\Model

python main.py \
  --embed1 road_llm \
  --embed2 hf_llm \
  --embed3 llm_w \
  --model transformer \
  --epochs 5 \
  --batch_size 8 \
  --window_size 9 \
  --mode 3m \
  --label all
```

**Purpose:** Establish HJ baseline performance

### Option 2: HJ Baseline + Local Embeddings (Your Hypothesis)

Add SGIS local district embeddings to HJ baseline:

```bash
python main.py \
  --embed1 road_llm \
  --embed2 hf_llm \
  --embed3 sgis_local_llm \
  --model transformer \
  --epochs 5 \
  --batch_size 8 \
  --window_size 9 \
  --mode 3m \
  --label all
```

**Purpose:** Test if local features improve HJ baseline

**Hypothesis:** Local district characteristics (business composition, market saturation) provide additional signal for Airbnb demand prediction.

### Option 3: Full Comparison (4 Embeddings)

Test with all four embedding types:

```bash
python main.py \
  --embed1 road_llm \
  --embed2 hf_llm \
  --embed3 llm_w \
  --embed4 sgis_local_llm \
  --model transformer \
  --epochs 5 \
  --batch_size 8 \
  --window_size 9 \
  --mode 3m \
  --label all
```

**Note:** Check if main.py supports 4 embeddings (may need code modification)

---

## Expected Results

### Success Criteria

1. **HJ baseline reproduces paper results**
   - Achieves similar RMSE to paper (need to check paper for exact value)
   - Validates our embedding generation is correct

2. **Local embeddings improve HJ baseline**
   - Lower RMSE than HJ baseline alone
   - Even 2-5% improvement is significant
   - Shows local district characteristics are predictive

3. **Ablation studies**
   - Test each embedding type individually
   - Understand contribution of each feature source

### Comparison Baselines

- **Raw-only**: RMSE = 0.810 (simplest baseline)
- **Raw + SGIS raw features**: RMSE = 1.097 (degraded performance)
- **Raw + SGIS local LLM**: ? (to be tested)
- **HJ baseline**: ? (to be established)
- **HJ baseline + local LLM**: ? (main hypothesis)

---

## Technical Details

### Model Configuration

**LLM Model:** meta-llama/Llama-3.2-3B-Instruct
- **Embedding dimension:** 3,072
- **Context length:** 512 tokens (truncated if longer)
- **Pooling method:** Mean pooling over sequence length
- **Precision:** bfloat16 during inference, float32 for storage

### Memory Requirements

**GPU Memory (RTX 4070 - 12GB):**
- Model: ~6-8GB
- Batch processing: ~2-4GB
- Total: ~8-10GB (fits comfortably)

**Disk Space:**
- Model cache: ~6GB (one-time download)
- Each embedding file: ~900MB
- Total for all 4 embeddings: ~3.6GB

### Prompt Structure Examples

**Airbnb (llm_wo):**
```
[2017-01-01 | 혜화동] AirBnB Feature Summary: Total number of AirBnB: 42

Category Column Attributes
Category: Property Type Information: Total number with data: 42
   Entire home/apt: 30
   Private room: 12

Binary Column Attributes
Airbnb Superhost Information: Total number with data: 42
number of Airbnb Superhost: 15

Numerical Column Attributes
Bedrooms Information: Total number with data: 42
   Mean: 1.52
   Median: 1.00
   ...

Assume you are a data analyst that is familiar with AirBnB market.
Give me the embedding of this 혜화동 at 2017-01-01
```

**Road Network (road_llm):**
```
[2017-01-01 | 혜화동] Road and Transportation Overview:
- Number of road nodes near AirBnBs: 177.0
- Total number of roads in the dong: 102, Total length: 15032.778
- Number of tunnels in the dong: 0 general, 0 building passage
- Number of bridges in the dong: 0 general, 0 viaducts
- Road types:
  • Residential: 44.0
  • Primary: 2.0
  ...
- Bus ridership (on/off): 183338 / 202583
- Subway ridership (on/off): 0.0 / 0.0
```

**Human Flow (hf_llm):**
```
[2017-01-01 | 혜화동] Average Floating Population Summary:
- Domestic Floating Population Summary
- Total Domestic Floating Population: 15948.60
- Domestic Floating Population by Age and Gender
- Teens Male: 348.77, Female: 119.62
- 20s Male: 687.54, Female: 841.58
...
- Long-term Foreign Resident Floating Population Summary
- Total Long-term Foreign Resident Floating Population: 400.11
...
```

---

## Next Steps (Checklist)

### Immediate Tasks

- [ ] Wait for Airbnb prompt generation to complete (~30 min)
- [ ] Verify all 4 prompt files exist and have correct format
- [ ] Generate all 4 LLM embeddings using RTX 4070 (~2-3 hours)
- [ ] Verify embedding file shapes (28,408 × 3,074)

### Testing Tasks

- [ ] Update main.py with LLM embedding paths (already added in session)
- [ ] Test HJ baseline reproduction (road_llm + hf_llm + llm_w)
- [ ] Test HJ baseline + local embeddings (road_llm + hf_llm + sgis_local_llm)
- [ ] Compare RMSE scores to establish improvement

### Analysis Tasks

- [ ] Document HJ baseline RMSE
- [ ] Document improvement with local embeddings
- [ ] Run ablation studies (each embedding type alone)
- [ ] Analyze which features contribute most

---

## Files Overview

### Scripts Created This Session

```
Preprocess/prompt_construct/prompt_construct/
├── generate_airbnb_prompts.py    # Generate Airbnb prompts
├── generate_road_prompts.py      # Generate road network prompts
├── generate_hf_prompts.py        # Generate human flow prompts
└── generate_hj_embeddings.py     # Generate LLM embeddings for all types
```

### Data Flow

```
Raw Data
    ↓
[1] Prompt Generation Scripts
    ↓
Prompt CSV Files (28,408 prompts each)
    ↓
[2] LLM Embedding Generation (generate_hj_embeddings.py)
    ↓
Embedding CSV Files (28,408 × 3,074)
    ↓
[3] Model Training (main.py)
    ↓
Results & Evaluation
```

---

## Troubleshooting

### CUDA Out of Memory

The script automatically handles this with smaller batches. If it persists:
- Close other GPU applications
- Use CPU fallback (slower but works)

### Token Authentication Failed

```bash
# Verify token is set
echo %HF_TOKEN%

# Re-set if needed
set HF_TOKEN=your_token_here

# Verify access to Llama model
# Visit: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
```

### Prompt File Not Found

Check paths in `generate_hj_embeddings.py`:
- llm_wo: `../dong_prompts_new/AirBnB_SSP_wo_prompts.csv`
- llm_w: `../dong_prompts_new/AirBnB_SSP_w_prompts.csv`
- road_llm: `../dong_prompts/road_prompts.csv`
- hf_llm: `../dong_prompts/human_flow_prompts.csv`

---

## Summary

This pipeline generates all HJ baseline LLM embeddings needed to:

1. **Reproduce HJ baseline** - Validate methodology
2. **Test local embeddings** - Test your hypothesis
3. **Compare approaches** - Establish best configuration

**Key Insight:** The real baseline is not raw-only (0.810), but HJ's LLM-based approach. Your contribution is adding **local district context** to improve predictions.

Good luck with the embedding generation!
