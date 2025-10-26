# SGIS Local Embeddings - Complete Guide

## Overview

This guide explains how to create and use "local embeddings" for SGIS census features to enhance the HJ baseline model. The approach follows Hongju's methodology of converting structured data into natural language prompts, then generating LLM embeddings using meta-llama/Llama-3.2-3B-Instruct.

## What Are Local Embeddings?

**Local embeddings** are LLM-generated vector representations of local district characteristics derived from SGIS census data. They capture semantic information about:

- **Business composition**: Retail, accommodation, and restaurant ratios
- **Market characteristics**: Housing units, Airbnb penetration
- **Tourism potential**: Combined attractiveness and competition indicators
- **Investment context**: Market saturation and growth opportunity

## Why Local Embeddings?

1. **Consistent with HJ baseline**: Uses same LLM model and methodology
2. **Semantic richness**: Captures relationships between features that raw numbers can't
3. **Domain knowledge**: Prompts include interpretations and business insights
4. **Dimensionality**: Produces 3,072-dimensional embeddings matching other LLM features

---

## Pipeline Architecture

```
SGIS Raw Data
    ↓
[1] Feature Engineering (COMPLETE)
    → sgis_improved_final.csv (6 features × 28,274 dong-months)
    ↓
[2] Prompt Generation (COMPLETE)
    → sgis_local_prompts.csv (28,274 natural language prompts)
    ↓
[3] LLM Embedding Generation (PENDING - USER ACTION REQUIRED)
    → sgis_local_llm_embeddings.csv (3,072 dimensions)
    ↓
[4] Model Training
    → Test HJ baseline + local embeddings
```

---

## Files Created

### In `Preprocess/sgis_manual/`:

1. **sgis_improved_final.csv** (28,274 rows × 8 columns)
   - Reporting Month, Dong_name
   - retail_ratio, accommodation_ratio, restaurant_ratio
   - housing_units, airbnb_listing_count, airbnb_per_1k_housing

2. **generate_local_prompts.py** (COMPLETED)
   - Converts SGIS features to natural language descriptions
   - Creates structured prompts with business interpretations

3. **sgis_local_prompts.csv** (28,274 rows × 3 columns)
   - Reporting Month, Dong_name, prompt
   - Natural language descriptions of local district characteristics
   - Average ~1,500 characters per prompt

4. **generate_local_embeddings.py** (READY TO RUN)
   - Loads meta-llama/Llama-3.2-3B-Instruct model
   - Processes prompts through LLM
   - Generates 3,072-dimensional embeddings
   - Saves to sgis_local_llm_embeddings.csv

5. **sgis_local_llm_embeddings.csv** (TO BE GENERATED)
   - Format: [Reporting Month, Dong_name, dim_0, ..., dim_3071]
   - Will contain 28,274 rows × 3,074 columns

### In `Model/`:

- **main.py** (UPDATED)
  - Added 'sgis_local_llm' to embedding_paths_dict
  - Path: '../Preprocess/sgis_manual/sgis_local_llm_embeddings.csv'

---

## Step-by-Step Instructions

### Step 1: Verify Prerequisites (BEFORE RUNNING)

#### Hardware Requirements:
- **GPU**: Highly recommended (NVIDIA with CUDA support)
  - Minimum: 8GB VRAM
  - Recommended: 16GB+ VRAM
- **RAM**: 16GB+ system memory
- **Disk space**: ~10GB for model + embeddings

#### Software Requirements:
```bash
# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Install required packages if needed
pip install transformers torch pandas numpy tqdm
```

#### Hugging Face Setup:
1. Create account at https://huggingface.co/
2. Request access to meta-llama/Llama-3.2-3B-Instruct: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
3. Get your access token: https://huggingface.co/settings/tokens
4. Set environment variable:
   ```bash
   # Windows
   set HF_TOKEN=your_token_here

   # Linux/Mac
   export HF_TOKEN=your_token_here
   ```

### Step 2: Generate Local Embeddings (USER ACTION REQUIRED)

**Estimated time**: 30-45 minutes on GPU, 3-4 hours on CPU

```bash
cd C:\Users\jour\Documents\GitHub\airbnb\Preprocess\sgis_manual

# Run the embedding generation script
python generate_local_embeddings.py
```

**What to expect**:
1. Model loading (2-3 minutes)
2. Processing bar showing progress through 28,274 prompts
3. GPU memory management with periodic cache clearing
4. Final output: sgis_local_llm_embeddings.csv

**Monitoring progress**:
```python
# Check partial progress if interrupted
import pandas as pd
df = pd.read_csv('sgis_local_llm_embeddings.csv')
print(f"Generated {len(df)}/28274 embeddings ({100*len(df)/28274:.1f}%)")
```

**Troubleshooting**:
- **Out of memory**: Script automatically reduces chunk size
- **No CUDA**: Will use CPU (much slower but works)
- **Access denied**: Check HF_TOKEN and model access approval
- **Connection error**: Check internet connection for model download

### Step 3: Verify Generated Embeddings

```bash
cd Preprocess/sgis_manual

python -c "import pandas as pd; df = pd.read_csv('sgis_local_llm_embeddings.csv'); print(f'Shape: {df.shape}'); print(f'Expected: (28274, 3074)'); print(f'Match: {df.shape == (28274, 3074)}')"
```

Expected output:
```
Shape: (28274, 3074)
Expected: (28274, 3074)
Match: True
```

### Step 4: Test with HJ Baseline Configuration

#### Option A: HJ Baseline Only (Reproduce Original)
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

**Note**: This requires the original LLM embedding files (road_llm, hf_llm, llm_w) which Hongju didn't upload. You may need to generate these first using the same process.

#### Option B: HJ Baseline + Local Embeddings (Your Hypothesis)
```bash
cd C:\Users\jour\Documents\GitHub\airbnb\Model

# Add local embeddings to HJ baseline
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

**Note**: Replaces `llm_w` (Airbnb SSP with listings) with `sgis_local_llm` (SGIS local features)

#### Option C: Test Local Embeddings Standalone (Simpler First Test)
```bash
cd C:\Users\jour\Documents\GitHub\airbnb\Model

# Test local embeddings alone
python main.py \
  --embed1 sgis_local_llm \
  --model transformer \
  --epochs 5 \
  --batch_size 8 \
  --window_size 9 \
  --mode 3m \
  --label all
```

This tests if local embeddings have standalone predictive value.

#### Option D: Raw + Local Embeddings (Practical Alternative)
```bash
cd C:\Users\jour\Documents\GitHub\airbnb\Model

# Combine raw Airbnb features with local embeddings
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

This is a practical test without needing other LLM embeddings.

---

## Expected Results Analysis

### Success Criteria:

1. **Local embeddings improve HJ baseline**:
   - HJ baseline RMSE: [Unknown - needs to be established]
   - HJ + local embeddings RMSE: [Should be lower]
   - Improvement margin: Even 2-5% is significant

2. **Local embeddings standalone performance**:
   - If RMSE < 2.0: Shows SGIS has predictive value
   - Compare to raw-only baseline (0.810)

3. **Raw + local embeddings**:
   - Target: Beat raw-only (0.810)
   - Previous raw + SGIS improved (6 features): 1.097
   - Hypothesis: LLM embeddings will capture relationships better than raw features

### Interpretation:

**If local embeddings improve performance**:
- ✓ Confirms your hypothesis: Local district characteristics matter
- ✓ LLM embeddings capture semantic relationships
- ✓ Business composition, competition, and saturation are predictive signals

**If local embeddings don't improve performance**:
- May indicate SGIS features aren't relevant to this prediction task
- Could try alternative features (POI data, accessibility)
- Consider ensemble or different architecture

---

## Technical Details

### Prompt Structure

Each prompt contains 4 sections:

1. **BUSINESS COMPOSITION**
   - Retail, accommodation, restaurant ratios
   - Interpretation (shopping destination, dining hub, etc.)
   - Competition assessment

2. **MARKET CHARACTERISTICS**
   - Housing units (market size)
   - Airbnb listing count and penetration rate
   - Saturation level interpretation

3. **TOURISM & INVESTMENT POTENTIAL**
   - Combined attractiveness score
   - Market opportunity assessment
   - Competitive dynamics

4. **ANALYST PERSPECTIVE**
   - Instruction to consider all factors
   - Focus on short-term rental context

### Example Prompt:
```
[2017-01-01 | 혜화동] Local District Feature Summary:

BUSINESS COMPOSITION:
This district has 6.31% retail businesses, 0.70% accommodation facilities,
and 17.56% restaurants and food services.
The low retail ratio indicates limited commercial shopping options. The
healthy restaurant presence suggests a vibrant food scene. There is
moderate competition from other accommodation providers.

MARKET CHARACTERISTICS:
The district contains 3,609 housing units, representing the potential
market size. This is a medium-sized residential district. There are
currently 42 Airbnb listings, resulting in 11.64 listings per 1,000
housing units.
This represents very high Airbnb market penetration and saturation.

TOURISM & INVESTMENT POTENTIAL:
The moderate retail and dining options (total 23.9%) provide basic
tourism amenities. Both high tourism amenities and high Airbnb
saturation indicate a mature, competitive market.

Based on these local district characteristics, consider the competitive
dynamics, tourism attractiveness, market saturation, and investment
potential for short-term rental properties.
```

### LLM Embedding Process:

1. **Tokenization**: Prompt → token IDs (max 512 tokens)
2. **Forward pass**: Through Llama-3.2-3B-Instruct model
3. **Pooling**: Mean over sequence length from last hidden state
4. **Dimension**: 3,072 (model.config.hidden_size)
5. **Dtype**: Float32 for compatibility
6. **Output**: Single vector per dong-month

---

## Alternative Approaches (If Current Method Doesn't Work)

### Plan B: Feature Selection with Raw Features
Already prepared in `Preprocess/sgis_manual/`:
- sgis_improved_subset_competition.csv (2 features)
- sgis_improved_subset_attractiveness.csv (2 features)
- sgis_improved_subset_ratios.csv (3 features)
- sgis_improved_subset_penetration.csv (1 feature)
- sgis_improved_subset_no_redundancy.csv (4 features)

Test these with:
```bash
python main.py --embed1 raw --embed2 sgis_penetration --model transformer ...
```

### Plan C: Tourism POI Data
If SGIS census data doesn't help, consider:
- Tourist attractions (museums, landmarks)
- Public transportation accessibility
- Neighborhood amenities

### Plan D: Hybrid Approach
Combine raw SGIS features with local embeddings:
```bash
python main.py --embed1 raw --embed2 sgis_penetration --embed3 sgis_local_llm ...
```

---

## Frequently Asked Questions

**Q: Do I need to generate road_llm, hf_llm, and llm_w embeddings too?**
A: For a full HJ baseline comparison, yes. But you can start with Option D (raw + sgis_local_llm) as a simpler first test.

**Q: How long will embedding generation take?**
A: ~30-45 minutes on modern GPU, 3-4 hours on CPU. Based on Hongju's notebook: ~12 prompts/second on GPU.

**Q: What if I don't have GPU access?**
A: The script will use CPU (slower but works). Consider using Google Colab or cloud GPU (AWS, Azure).

**Q: Can I pause and resume embedding generation?**
A: Not currently implemented. The script processes all prompts in one run. Plan accordingly.

**Q: What's the file size of the output?**
A: ~900MB for 28,274 rows × 3,074 columns (float32).

**Q: How do I know if it's working?**
A: You'll see a progress bar showing "Processing local prompts: X%". GPU memory usage should be steady.

---

## Next Steps

1. **Generate embeddings** (requires GPU + Hugging Face token)
2. **Test Option D first**: Raw + local embeddings (simplest valid test)
3. **Compare to baseline**: RMSE should improve over raw-only (0.810)
4. **If successful**: Consider generating other LLM embeddings (road_llm, hf_llm)
5. **If unsuccessful**: Try alternative features or feature selection

---

## Files Summary

```
Preprocess/sgis_manual/
├── sgis_improved_final.csv                 ✓ COMPLETE
├── generate_local_prompts.py               ✓ COMPLETE
├── sgis_local_prompts.csv                  ✓ COMPLETE
├── generate_local_embeddings.py            ✓ READY TO RUN
└── sgis_local_llm_embeddings.csv           ⏳ PENDING USER ACTION

Model/
└── main.py                                 ✓ UPDATED (sgis_local_llm path added)
```

---

## Contact & Support

For issues with:
- **Model loading**: Check HF_TOKEN and model access
- **CUDA errors**: Update torch and CUDA drivers
- **Memory errors**: Reduce batch size or use CPU
- **File paths**: Verify relative paths from Model/ directory

Good luck with your local embeddings generation!
