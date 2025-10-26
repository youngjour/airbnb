# Airbnb Demand Forecasting with Local District Embeddings

## Project Overview

This project enhances Hongju's baseline Airbnb demand forecasting model by adding **LLM embeddings of local district characteristics** (SGIS census data). The goal is to test whether local business composition and market saturation improve prediction accuracy.

**Research Question:** Do local district characteristics (retail/accommodation/restaurant ratios, housing units, Airbnb market penetration) provide additional predictive signal beyond road networks and human flow data?

---

## Current Status (2025-10-25)

### ✅ Completed

1. **SGIS Local Feature Engineering**
   - Collected SGIS census data for 422 Seoul dongs × 67 months (2017-01 to 2022-07)
   - Created 6 engineered features: business composition ratios + market indicators
   - Output: `sgis_improved_final.csv` (28,408 rows × 8 columns)

2. **SGIS Local Prompt Generation**
   - Converted SGIS features to natural language descriptions
   - Created business interpretations and market analysis narratives
   - Output: `sgis_local_prompts.csv` (28,408 prompts, ~1,500 chars each)

3. **SGIS Local LLM Embedding Generation**
   - Used meta-llama/Llama-3.2-3B-Instruct to generate embeddings
   - GPU-accelerated with RTX 4070 (CUDA 12.1, PyTorch with bfloat16)
   - Output: `sgis_local_llm_embeddings.csv` (28,408 rows × 3,074 columns)
   - **Status: ✅ VERIFIED COMPLETE**

4. **HJ Baseline Prompt Generation**
   - Generated prompts for Airbnb features (with/without listings)
   - Generated prompts for road network and transportation
   - Generated prompts for human flow demographics
   - All outputs: 28,408 prompts each
   - **Status: ✅ ALL PROMPTS COMPLETE**

5. **Infrastructure Setup**
   - Created `yj_pytorch` environment with CUDA-enabled PyTorch
   - Configured Hugging Face authentication for Llama model access
   - Updated `Model/main.py` with `sgis_local_llm` embedding path

### ⏳ In Progress

6. **HJ Baseline LLM Embedding Generation**
   - Script ready: `generate_hj_embeddings.py`
   - Prompts ready for all 4 embedding types
   - **Next step: Run embedding generation (~2-3 hours total)**

### 📋 Pending

7. **Model Testing & Evaluation**
   - Test HJ baseline reproduction (road_llm + hf_llm + llm_w)
   - Test HJ baseline + local embeddings (road_llm + hf_llm + sgis_local_llm)
   - Compare RMSE scores to establish improvement
   - Run ablation studies

---

## Repository Structure

```
airbnb/
├── PROJECT_README.md                     # This file - project overview and progress
│
├── Model/                                # Training and evaluation
│   ├── main.py                          # Main training script (updated with sgis_local_llm)
│   └── ...
│
├── Preprocess/
│   ├── sgis_manual/                     # SGIS local embeddings workflow
│   │   ├── generate_local_prompts.py   # SGIS → Natural language prompts
│   │   ├── generate_local_embeddings.py # Prompts → LLM embeddings
│   │   ├── sgis_improved_final.csv      # SGIS engineered features (source)
│   │   ├── sgis_local_prompts.csv       # Generated prompts
│   │   ├── sgis_local_llm_embeddings.csv # Generated embeddings (COMPLETE)
│   │   ├── LOCAL_EMBEDDINGS_GUIDE.md    # Detailed guide
│   │   ├── QUICK_START.md               # Quick reference
│   │   ├── RUN_EMBEDDINGS.bat           # Windows helper script
│   │   ├── _archive_experimental/       # Old/experimental code (cleaned up)
│   │   └── ...
│   │
│   └── prompt_construct/prompt_construct/  # HJ baseline embeddings workflow
│       ├── generate_airbnb_prompts.py   # Airbnb → Prompts
│       ├── generate_road_prompts.py     # Road network → Prompts
│       ├── generate_hf_prompts.py       # Human flow → Prompts
│       ├── generate_hj_embeddings.py    # ALL prompts → LLM embeddings
│       ├── HJ_BASELINE_GUIDE.md         # Complete HJ baseline guide
│       ├── dong_prompts_new/            # Airbnb prompts (COMPLETE)
│       │   ├── AirBnB_SSP_wo_prompts.csv
│       │   └── AirBnB_SSP_w_prompts.csv
│       └── dong_prompts/                # Road & HF prompts (COMPLETE)
│           ├── road_prompts.csv
│           └── human_flow_prompts.csv
│
└── Data/
    └── Preprocessed_data/Dong/
        └── llm_embeddings_new/          # HJ baseline embeddings (TO BE GENERATED)
            ├── Airbnb_SSP_wo.csv        # ⏳ PENDING
            ├── Airbnb_SSP_w.csv         # ⏳ PENDING
            ├── road_llm.csv             # ⏳ PENDING
            └── human_flow_llm.csv       # ⏳ PENDING
```

---

## Methodology: LLM Embedding Pipeline

All embeddings (SGIS local + HJ baseline) use **identical methodology** for fair comparison:

### 1. Feature Engineering
Convert raw census/property data into meaningful features:
- SGIS: Business ratios, housing units, Airbnb penetration
- Airbnb: Property characteristics, amenities, pricing
- Road: Network topology, transportation ridership
- Human Flow: Population demographics by age/gender

### 2. Prompt Generation
Transform structured features → natural language descriptions:

**Example SGIS Local Prompt:**
```
[2017-01-01 | 혜화동] Local District Feature Summary:

BUSINESS COMPOSITION:
This district has 6.31% retail businesses, 0.70% accommodation
facilities, and 17.56% restaurants. The low retail ratio indicates
limited commercial shopping options. The healthy restaurant presence
suggests a vibrant food scene. There is moderate competition from
other accommodation providers.

MARKET CHARACTERISTICS:
The district contains 3,609 housing units, representing the potential
market size. There are currently 42 Airbnb listings, resulting in
11.64 listings per 1,000 housing units. This represents very high
Airbnb market penetration and saturation.

TOURISM & INVESTMENT POTENTIAL:
The moderate retail and dining options provide basic tourism amenities.
Both high tourism amenities and high Airbnb saturation indicate a
mature, competitive market.
```

### 3. LLM Embedding Generation
- **Model:** meta-llama/Llama-3.2-3B-Instruct
- **Method:** Mean pooling over last hidden states
- **Dimensions:** 3,072 (model.config.hidden_size)
- **Precision:** bfloat16 during inference, float32 for storage
- **Hardware:** NVIDIA RTX 4070 with CUDA 12.1
- **Time:** ~30-45 minutes per embedding type (28,408 prompts each)

### 4. Model Training
- **Architecture:** Transformer with multi-embedding fusion
- **Task:** 3-month ahead Airbnb demand forecasting
- **Evaluation:** RMSE on test set

---

## SGIS Local Features

### 6 Engineered Features (Business-Oriented)

| Feature | Description | Purpose |
|---------|-------------|---------|
| `retail_ratio` | % of businesses in retail | Shopping attractiveness |
| `accommodation_ratio` | % of businesses in accommodation | Competition indicator |
| `restaurant_ratio` | % of businesses in food services | Dining attractiveness |
| `housing_units` | Number of housing units in dong | Market size |
| `airbnb_listing_count` | Current Airbnb listings | Market presence |
| `airbnb_per_1k_housing` | Listings per 1,000 housing units | Market saturation |

### Why LLM Embeddings?

**Hypothesis:** LLM embeddings capture semantic relationships that raw features cannot:
- Business composition patterns (e.g., "tourist destination" vs "residential area")
- Market maturity levels (e.g., "emerging market" vs "saturated market")
- Investment potential context (e.g., "high competition, low growth" vs "underserved market")

---

## Next Steps

### 1. Generate HJ Baseline LLM Embeddings (~2-3 hours)

```bash
# Activate PyTorch environment
conda activate yj_pytorch

# Set Hugging Face token
set HF_TOKEN=your_token_here

# Navigate to directory
cd C:\Users\jour\Documents\GitHub\airbnb\Preprocess\prompt_construct\prompt_construct

# Generate all 4 HJ baseline embeddings
python generate_hj_embeddings.py
```

**Expected outputs:**
- `Data/Preprocessed_data/Dong/llm_embeddings_new/Airbnb_SSP_wo.csv` (28,408 × 3,074)
- `Data/Preprocessed_data/Dong/llm_embeddings_new/Airbnb_SSP_w.csv` (28,408 × 3,074)
- `Data/Preprocessed_data/Dong/llm_embeddings_new/road_llm.csv` (28,408 × 3,074)
- `Data/Preprocessed_data/Dong/llm_embeddings_new/human_flow_llm.csv` (28,408 × 3,074)

### 2. Test HJ Baseline (Reproduce Original Paper)

```bash
conda activate yj_env
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

**Purpose:** Establish HJ baseline RMSE (expected: competitive with paper results)

### 3. Test HJ Baseline + Local Embeddings (Main Hypothesis)

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

**Purpose:** Test if local district features improve predictions

### 4. Ablation Studies

Test each embedding type individually to understand contribution:
- Raw-only baseline (existing: RMSE = 0.810)
- Road LLM only
- Human flow LLM only
- Local LLM only
- Various combinations

---

## Key Results to Compare

| Configuration | RMSE | Status |
|--------------|------|--------|
| Raw-only | 0.810 | ✅ Established baseline |
| Raw + SGIS raw features (6 features) | 1.097 | ✅ Degraded (raw features don't help) |
| HJ baseline (road_llm + hf_llm + llm_w) | ? | ⏳ To be established |
| **HJ baseline + local LLM** | ? | ⏳ **Main hypothesis** |

**Success criteria:** HJ baseline + local LLM < HJ baseline alone
- Even 2-5% improvement is significant
- Validates that local district context adds predictive value

---

## Technical Specifications

### Environment: yj_pytorch
- Python 3.12
- PyTorch 2.5.0 with CUDA 12.1
- Transformers (Hugging Face)
- GPU: NVIDIA RTX 4070 (12GB VRAM)

### Environment: yj_env
- Python 3.13
- TensorFlow/PyTorch for model training
- Standard ML libraries (pandas, numpy, sklearn)

### LLM Model
- **Model ID:** meta-llama/Llama-3.2-3B-Instruct
- **Parameters:** 3 billion
- **License:** Llama license (gated model - requires HF access request)
- **Cache:** ~/.cache/huggingface/hub/ (~6GB first download)

---

## Data Sources

1. **Airbnb Listings:** `DATA/AirBnB_data.csv` (953,953 listings)
2. **Road Network:** `Data/Preprocessed_data/Dong/Road_Embeddings_with_flow.csv`
3. **Human Flow:** `Data/Preprocessed_data/Dong/Human_flow.csv`
4. **SGIS Census:** Collected via SGIS OpenAPI3
   - Korean Statistical Geographic Information Service
   - Business distribution, industry codes, housing data
   - 422 Seoul administrative dongs, monthly from 2017-01 to 2022-07

---

## Documentation

### Quick References
- `Preprocess/sgis_manual/QUICK_START.md` - SGIS local embeddings quick start
- `Preprocess/sgis_manual/RUN_EMBEDDINGS.bat` - Windows helper script
- `Preprocess/sgis_manual/RUN_EMBEDDINGS_yj_pytorch.md` - Environment-specific guide

### Detailed Guides
- `Preprocess/sgis_manual/LOCAL_EMBEDDINGS_GUIDE.md` - Complete SGIS workflow (300+ lines)
- `Preprocess/prompt_construct/prompt_construct/HJ_BASELINE_GUIDE.md` - Complete HJ baseline workflow

### Installation Guides
- `Preprocess/sgis_manual/INSTALL_PYTORCH_CUDA.md` - PyTorch CUDA setup

---

## Methodology Alignment

**Critical:** SGIS local embeddings use **identical methodology** to HJ baseline:

| Aspect | SGIS Local | HJ Baseline | Match |
|--------|-----------|-------------|-------|
| LLM Model | Llama-3.2-3B-Instruct | Llama-3.2-3B-Instruct | ✅ |
| Embedding Dim | 3,072 | 3,072 | ✅ |
| Pooling | Mean pooling | Mean pooling | ✅ |
| Approach | Features → Prompts → Embeddings | Features → Prompts → Embeddings | ✅ |
| Output Format | CSV [Date, Dong, dim_0...dim_3071] | CSV [Date, Dong, dim_0...dim_3071] | ✅ |

**Only difference:** Source features (local district vs road/flow/Airbnb properties)

---

## Repository Cleanup

### Archived Files

**SGIS Manual Directory** (`Preprocess/sgis_manual/_archive_experimental/`):
- Data collection experiments (collect_sgis_*.py)
- Testing scripts (test_*.py, check_*.py, diagnose_*.py)
- Preprocessing experiments (align_*.py, analyze_*.py, fill_*.py)
- Intermediate CSV files (sgis_complete_data.csv, sgis_ratios_*.csv)
- Old documentation (API_LIMITATIONS, WORK_LOG, etc.)

**Prompt Construct Directory** (`Preprocess/prompt_construct/prompt_construct/_archive_notebooks/`):
- Experimental notebooks (additional_embedding_gen.ipynb, airbnb_embedding_gen*.ipynb, etc.)
- Old scripts (refined_prompt_gen.py)
- Development/testing notebooks (airbnb_prompt_preprocess.ipynb, make_additional_llm_prompt.ipynb)

### Active Files (Essential LLM Workflow)
Only production-ready code remains in main directories:
- Prompt generation scripts (`generate_*.py`)
- LLM embedding generation scripts
- Final feature/prompt/embedding CSV files
- Documentation for current workflow (*.md guides)

---

## Project Timeline

**2025-10-21:** SGIS data collection and feature engineering
**2025-10-22:** Feature refinement and alignment with labels
**2025-10-23:** Final SGIS features created (sgis_improved_final.csv)
**2025-10-25:**
- SGIS local prompt generation ✅
- SGIS local LLM embedding generation ✅
- HJ baseline prompt generation ✅
- Repository cleanup (sgis_manual + prompt_construct) ✅
- Documentation updates ✅
- **Current:** Ready to generate HJ baseline LLM embeddings

---

## Contact & References

**Original Paper:** Hongju's Airbnb demand forecasting (repository cloned and enhanced)
**Methodology:** LLM embedding approach following Hongju's baseline
**Enhancement:** Adding local district characteristics (SGIS census data)

---

## Quick Command Reference

### Generate HJ Baseline Embeddings
```bash
conda activate yj_pytorch
set HF_TOKEN=your_token_here
cd C:\Users\jour\Documents\GitHub\airbnb\Preprocess\prompt_construct\prompt_construct
python generate_hj_embeddings.py
```

### Test Model (HJ Baseline + Local)
```bash
conda activate yj_env
cd C:\Users\jour\Documents\GitHub\airbnb\Model
python main.py --embed1 road_llm --embed2 hf_llm --embed3 sgis_local_llm --model transformer --epochs 5 --batch_size 8 --window_size 9 --mode 3m --label all
```

---

**Last Updated:** 2025-10-25
**Status:** Ready for HJ baseline embedding generation → Model testing
