# SGIS Local Embeddings - Quick Start

## Current Status

✓ **COMPLETE**: Feature engineering, prompt generation, scripts ready
⏳ **PENDING**: LLM embedding generation (requires user action)

## Files Ready to Use

```
Preprocess/sgis_manual/
├── sgis_improved_final.csv          # 6 SGIS features (28,274 dong-months)
├── sgis_local_prompts.csv           # 28,274 natural language prompts
├── generate_local_embeddings.py     # LLM embedding generator (READY TO RUN)
└── LOCAL_EMBEDDINGS_GUIDE.md        # Complete documentation
```

## Quick Action: Generate Embeddings

### Prerequisites
1. **GPU with CUDA** (or use CPU - will be slower)
2. **Hugging Face token** with access to meta-llama/Llama-3.2-3B-Instruct
3. **Packages**: `pip install transformers torch pandas numpy tqdm`

### Run This Command

```bash
# Set your Hugging Face token
set HF_TOKEN=your_token_here

# Generate embeddings (30-45 min on GPU)
cd C:\Users\jour\Documents\GitHub\airbnb\Preprocess\sgis_manual
python generate_local_embeddings.py
```

### Output
- Creates: `sgis_local_llm_embeddings.csv`
- Size: 28,274 rows × 3,074 columns (Reporting Month, Dong_name, dim_0 ... dim_3071)

## Quick Test: After Embeddings Generated

### Option 1: Raw + Local Embeddings (Simplest Valid Test)
```bash
cd C:\Users\jour\Documents\GitHub\airbnb\Model

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

**Compare**: Results vs raw-only baseline (RMSE = 0.810)

### Option 2: HJ Baseline + Local Embeddings (Your Original Goal)
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

**Note**: Requires road_llm and hf_llm embeddings (not yet generated)

## What We Created

### 1. SGIS Local Features (Tourism-Focused)
- **retail_ratio**: Shopping attractiveness
- **accommodation_ratio**: Competition indicator
- **restaurant_ratio**: Dining attractiveness
- **housing_units**: Market size
- **airbnb_listing_count**: Current presence
- **airbnb_per_1k_housing**: Market saturation

### 2. Natural Language Prompts
Converted features to structured business narratives:
```
[2017-01-01 | 혜화동] Local District Feature Summary:

BUSINESS COMPOSITION:
This district has 6.31% retail businesses, 0.70% accommodation
facilities, and 17.56% restaurants...

MARKET CHARACTERISTICS:
The district contains 3,609 housing units... There are currently
42 Airbnb listings, resulting in 11.64 listings per 1,000
housing units... very high market penetration...

TOURISM & INVESTMENT POTENTIAL:
The moderate retail and dining options suggest... mature,
competitive market...
```

### 3. LLM Embedding Generator
- Uses meta-llama/Llama-3.2-3B-Instruct (same as HJ baseline)
- Generates 3,072-dimensional embeddings
- Compatible with existing model architecture

## Why This Approach?

1. **Consistent methodology**: Same LLM model as HJ baseline
2. **Semantic richness**: Captures relationships raw features can't
3. **Domain knowledge**: Prompts include business interpretations
4. **Testable hypothesis**: Does local context improve predictions?

## Success Criteria

If `raw + sgis_local_llm` beats `raw-only` (0.810):
- ✓ Local features are predictive
- ✓ LLM embeddings capture useful signals
- ✓ Approach validated

If it doesn't:
- Try feature selection subsets (already prepared)
- Consider alternative features (POI, accessibility)
- Test different architectures

## Next Steps

1. **Generate embeddings** (see LOCAL_EMBEDDINGS_GUIDE.md for details)
2. **Test Option 1** (raw + local embeddings)
3. **Compare results** to baseline
4. **If successful**: Generate other LLM embeddings (road_llm, hf_llm) for full HJ baseline test

## Key Insight from Session

The **real baseline** is not raw-only (0.810), but rather:
- HJ Baseline = LLM embeddings of road network + human flow + Airbnb features
- Your hypothesis: Adding LLM embeddings of SGIS local features will improve this

We've now built the infrastructure to test this hypothesis!

## Need Help?

See `LOCAL_EMBEDDINGS_GUIDE.md` for:
- Detailed troubleshooting
- Hardware requirements
- Alternative approaches
- FAQ section
