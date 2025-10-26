# HJ Baseline Prompt Generation Directory - Reorganization Summary

**Date:** 2025-10-25
**Purpose:** Clean up experimental notebooks, keep only essential HJ baseline embedding workflow files

---

## What Was Organized

### Files Moved to `_archive_notebooks/`

**Experimental Notebooks (Development/Testing):**
- additional_embedding_gen.ipynb
- airbnb_embedding_gen1.ipynb
- airbnb_embedding_gen2.ipynb
- airbnb_prompt_preprocess.ipynb
- make_additional_llm_prompt.ipynb

**Old Scripts:**
- refined_prompt_gen.py (superseded by generate_*.py scripts)

---

## Files Kept (Essential HJ Baseline Workflow)

### Production Scripts
- `generate_airbnb_prompts.py` - Airbnb features → Natural language prompts
- `generate_road_prompts.py` - Road network → Natural language prompts
- `generate_hf_prompts.py` - Human flow → Natural language prompts
- `generate_hj_embeddings.py` - All prompts → LLM embeddings (unified script)

### Documentation
- `HJ_BASELINE_GUIDE.md` - Complete HJ baseline embedding generation guide

### Data Files (In parent directories)
- `../dong_prompts/road_prompts.csv` - Road network prompts (28,408 rows)
- `../dong_prompts/human_flow_prompts.csv` - Human flow prompts (28,408 rows)
- `../dong_prompts_new/AirBnB_SSP_wo_prompts.csv` - Airbnb prompts without listings (28,408 rows)
- `../dong_prompts_new/AirBnB_SSP_w_prompts.csv` - Airbnb prompts with listings (28,408 rows)

---

## Current Directory Structure

```
Preprocess/prompt_construct/
├── dong_prompts/                           # Road & Human flow prompts
│   ├── road_prompts.csv                   # 28,408 road network prompts
│   └── human_flow_prompts.csv             # 28,408 human flow prompts
│
├── dong_prompts_new/                       # Airbnb prompts
│   ├── AirBnB_SSP_wo_prompts.csv          # 28,408 Airbnb prompts (without listings)
│   └── AirBnB_SSP_w_prompts.csv           # 28,408 Airbnb prompts (with listings)
│
└── prompt_construct/                       # Production scripts
    ├── generate_airbnb_prompts.py         # PRODUCTION: Airbnb prompt generation
    ├── generate_road_prompts.py           # PRODUCTION: Road prompt generation
    ├── generate_hf_prompts.py             # PRODUCTION: Human flow prompt generation
    ├── generate_hj_embeddings.py          # PRODUCTION: LLM embedding generation
    ├── HJ_BASELINE_GUIDE.md               # DOCS: Complete workflow guide
    └── _archive_notebooks/                # ARCHIVE: Old notebooks & scripts
        ├── additional_embedding_gen.ipynb
        ├── airbnb_embedding_gen1.ipynb
        ├── airbnb_embedding_gen2.ipynb
        ├── airbnb_prompt_preprocess.ipynb
        ├── make_additional_llm_prompt.ipynb
        └── refined_prompt_gen.py
```

---

## Benefits of Reorganization

1. **Clarity:** Clear separation between production code and experimental notebooks
2. **Maintainability:** Easy to find essential HJ baseline workflow files
3. **Documentation:** Self-documenting structure (production vs archive)
4. **Onboarding:** New users can quickly understand the workflow
5. **Git History:** Experimental iterations preserved but organized

---

## Essential HJ Baseline Workflow (Post-Cleanup)

### Step 1: Feature Data (Already Available)
**Inputs:**
- `Data/Preprocessed_data/Dong/Road_Embeddings_with_flow.csv` (Road network)
- `Data/Preprocessed_data/Dong/Human_flow.csv` (Human flow demographics)
- `DATA/AirBnB_data.csv` (Airbnb property features)

### Step 2: Prompt Generation
**Scripts:** `generate_*.py` files
**Outputs:**
- `dong_prompts/road_prompts.csv` (28,408 prompts)
- `dong_prompts/human_flow_prompts.csv` (28,408 prompts)
- `dong_prompts_new/AirBnB_SSP_wo_prompts.csv` (28,408 prompts)
- `dong_prompts_new/AirBnB_SSP_w_prompts.csv` (28,408 prompts)
**Status:** ✅ COMPLETE

### Step 3: LLM Embedding Generation
**Script:** `generate_hj_embeddings.py`
**Inputs:** All prompt CSV files
**Outputs:**
- `Data/Preprocessed_data/Dong/llm_embeddings_new/road_llm.csv` (28,408 × 3,074)
- `Data/Preprocessed_data/Dong/llm_embeddings_new/human_flow_llm.csv` (28,408 × 3,074)
- `Data/Preprocessed_data/Dong/llm_embeddings_new/Airbnb_SSP_wo.csv` (28,408 × 3,074)
- `Data/Preprocessed_data/Dong/llm_embeddings_new/Airbnb_SSP_w.csv` (28,408 × 3,074)
**Status:** ⏳ PENDING (~2-3 hours to generate)

### Step 4: Model Training
**Location:** `Model/main.py`
**Usage:** `--embed1 road_llm --embed2 hf_llm --embed3 llm_w` (HJ baseline)
**Status:** ⏳ Pending embedding generation

---

## Accessing Archived Files

If you need to access experimental notebooks or old scripts:

```bash
cd Preprocess/prompt_construct/prompt_construct/_archive_notebooks/
ls -la
```

All experimental iterations are preserved but organized separately from production workflow.

---

## Next Steps

With clean repository structure:
1. ✅ All HJ baseline prompts generated (4 types, 28,408 each)
2. ⏳ Generate HJ baseline embeddings (~2-3 hours)
3. ⏳ Test HJ baseline model
4. ⏳ Test HJ baseline + SGIS local embeddings

---

**Note:** This reorganization preserves all code and data - nothing was deleted. Experimental notebooks are archived for reference but separated from the main workflow.
