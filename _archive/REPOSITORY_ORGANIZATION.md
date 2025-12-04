# Repository Organization Guide

**Date:** 2025-10-25
**Purpose:** Clean, organized repository structure focused on LLM embedding workflow

---

## Overview

This repository has been organized to focus exclusively on the **LLM embedding workflow** for enhancing Airbnb demand forecasting. All experimental code, testing scripts, and intermediate data files have been archived while maintaining a clean production-ready codebase.

---

## Current Repository Structure

```
airbnb/
├── PROJECT_README.md                     # Main project documentation & progress tracker
├── REPOSITORY_ORGANIZATION.md            # This file - organization overview
├── README.md                             # Original repository README
│
├── Model/                                # Training and evaluation
│   ├── main.py                          # Main training script (updated with sgis_local_llm)
│   └── ...
│
├── Preprocess/
│   │
│   ├── sgis_manual/                     # SGIS Local Embeddings Workflow
│   │   ├── generate_local_prompts.py   # PRODUCTION: SGIS → Prompts
│   │   ├── generate_local_embeddings.py # PRODUCTION: Prompts → LLM embeddings
│   │   │
│   │   ├── sgis_improved_final.csv      # SOURCE: Engineered features (28,408 rows)
│   │   ├── sgis_local_prompts.csv       # OUTPUT: Natural language prompts
│   │   ├── sgis_local_llm_embeddings.csv # OUTPUT: LLM embeddings (COMPLETE)
│   │   ├── 한국행정구역분류_행정동코드(7자리)_20210701기준_extracted.csv # REFERENCE
│   │   │
│   │   ├── LOCAL_EMBEDDINGS_GUIDE.md    # Complete SGIS workflow guide
│   │   ├── QUICK_START.md               # Quick reference
│   │   ├── RUN_EMBEDDINGS_yj_pytorch.md # Environment-specific guide
│   │   ├── INSTALL_PYTORCH_CUDA.md      # PyTorch setup guide
│   │   ├── RUN_EMBEDDINGS.bat           # Windows helper script
│   │   ├── README.md                    # SGIS directory README
│   │   ├── REORGANIZATION_SUMMARY.md    # SGIS cleanup summary
│   │   │
│   │   ├── ref_code/                    # Reference implementations
│   │   └── _archive_experimental/       # Archived experimental code
│   │       ├── collect_sgis_*.py        # Data collection experiments
│   │       ├── test_*.py                # Testing scripts
│   │       ├── check_*.py               # Diagnostic scripts
│   │       ├── align_*.py, analyze_*.py, etc. # Preprocessing experiments
│   │       ├── sgis_*.csv (intermediate) # Old/intermediate data
│   │       └── *.md (old docs)          # Old documentation
│   │
│   └── prompt_construct/                 # HJ Baseline Embeddings Workflow
│       ├── dong_prompts/                 # Road & Human flow prompts (OUTPUT)
│       │   ├── road_prompts.csv         # 28,408 road network prompts
│       │   └── human_flow_prompts.csv   # 28,408 human flow prompts
│       │
│       ├── dong_prompts_new/             # Airbnb prompts (OUTPUT)
│       │   ├── AirBnB_SSP_wo_prompts.csv # 28,408 Airbnb prompts (w/o listings)
│       │   └── AirBnB_SSP_w_prompts.csv  # 28,408 Airbnb prompts (w/ listings)
│       │
│       └── prompt_construct/             # Production scripts
│           ├── generate_airbnb_prompts.py  # PRODUCTION: Airbnb prompts
│           ├── generate_road_prompts.py    # PRODUCTION: Road prompts
│           ├── generate_hf_prompts.py      # PRODUCTION: Human flow prompts
│           ├── generate_hj_embeddings.py   # PRODUCTION: LLM embeddings
│           │
│           ├── HJ_BASELINE_GUIDE.md        # Complete HJ baseline guide
│           ├── REORGANIZATION_SUMMARY.md   # HJ baseline cleanup summary
│           │
│           └── _archive_notebooks/         # Archived experimental notebooks
│               ├── additional_embedding_gen.ipynb
│               ├── airbnb_embedding_gen*.ipynb
│               ├── airbnb_prompt_preprocess.ipynb
│               ├── make_additional_llm_prompt.ipynb
│               └── refined_prompt_gen.py   # Old script
│
├── Data/
│   └── Preprocessed_data/Dong/
│       ├── Road_Embeddings_with_flow.csv   # Road network features (INPUT)
│       ├── Human_flow.csv                  # Human flow features (INPUT)
│       └── llm_embeddings_new/             # HJ baseline embeddings (TO BE GENERATED)
│           ├── Airbnb_SSP_wo.csv           # ⏳ PENDING
│           ├── Airbnb_SSP_w.csv            # ⏳ PENDING
│           ├── road_llm.csv                # ⏳ PENDING
│           └── human_flow_llm.csv          # ⏳ PENDING
│
└── DATA/
    └── AirBnB_data.csv                     # Raw Airbnb listings (INPUT)
```

---

## Key Documentation Files

### Root Level
- **PROJECT_README.md** - Complete project overview, methodology, current status, next steps
- **REPOSITORY_ORGANIZATION.md** - This file (organization guide)
- **README.md** - Original repository README

### SGIS Local Embeddings
- **Preprocess/sgis_manual/LOCAL_EMBEDDINGS_GUIDE.md** - Complete SGIS workflow (300+ lines)
- **Preprocess/sgis_manual/QUICK_START.md** - Quick reference
- **Preprocess/sgis_manual/RUN_EMBEDDINGS_yj_pytorch.md** - Environment guide
- **Preprocess/sgis_manual/REORGANIZATION_SUMMARY.md** - SGIS cleanup documentation

### HJ Baseline Embeddings
- **Preprocess/prompt_construct/prompt_construct/HJ_BASELINE_GUIDE.md** - Complete HJ baseline workflow
- **Preprocess/prompt_construct/prompt_construct/REORGANIZATION_SUMMARY.md** - HJ baseline cleanup documentation

---

## Production Workflow Summary

### 1. SGIS Local Embeddings (COMPLETE ✅)

**Purpose:** Generate LLM embeddings of local district characteristics

**Pipeline:**
```
sgis_improved_final.csv
    ↓ [generate_local_prompts.py]
sgis_local_prompts.csv
    ↓ [generate_local_embeddings.py]
sgis_local_llm_embeddings.csv (28,408 × 3,074)
```

**Status:** ✅ COMPLETE

### 2. HJ Baseline Embeddings (PENDING ⏳)

**Purpose:** Generate LLM embeddings for HJ baseline features

**Pipeline:**
```
Raw Features (Road, Human Flow, Airbnb)
    ↓ [generate_*.py scripts]
Prompt CSV files (4 types, 28,408 each)
    ↓ [generate_hj_embeddings.py]
Embedding CSV files (4 types, 28,408 × 3,074 each)
```

**Status:** ⏳ Prompts ready, embeddings pending (~2-3 hours to generate)

### 3. Model Training (PENDING ⏳)

**Purpose:** Test HJ baseline + local embeddings

**Command:**
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

**Status:** ⏳ Pending HJ baseline embedding generation

---

## What Was Archived

### SGIS Manual Directory
- **30+ experimental scripts** moved to `_archive_experimental/`
  - Data collection experiments (collect_sgis_*.py)
  - Testing scripts (test_*.py, check_*.py, diagnose_*.py)
  - Preprocessing experiments (align_*.py, analyze_*.py, fill_*.py, etc.)
- **Intermediate CSV files** moved to archive
  - sgis_complete_data.csv, sgis_ratios_*.csv, test_*.csv, etc.
- **Old documentation** moved to archive
  - API_LIMITATIONS, WORK_LOG, PREPROCESSING_COMPLETE, etc.
- **Test output directories** moved to archive

### Prompt Construct Directory
- **5 experimental notebooks** moved to `_archive_notebooks/`
  - additional_embedding_gen.ipynb
  - airbnb_embedding_gen1.ipynb, airbnb_embedding_gen2.ipynb
  - airbnb_prompt_preprocess.ipynb
  - make_additional_llm_prompt.ipynb
- **1 old script** moved to archive
  - refined_prompt_gen.py (superseded by generate_*.py)

**Total archived:** 35+ files
**Nothing deleted:** All experimental code preserved for reference

---

## Benefits of Organization

1. **Clarity** - Clear separation between production code and experiments
2. **Maintainability** - Easy to find essential workflow files
3. **Documentation** - Self-documenting structure
4. **Onboarding** - New users can quickly understand the workflow
5. **Focus** - Emphasis on LLM embedding methodology
6. **Git History** - Experimental iterations preserved but organized

---

## Accessing Archived Files

All archived files are preserved and accessible:

```bash
# SGIS experimental code
cd Preprocess/sgis_manual/_archive_experimental/
ls -la

# HJ baseline experimental notebooks
cd Preprocess/prompt_construct/prompt_construct/_archive_notebooks/
ls -la
```

---

## Quick Start Commands

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

## Next Steps

1. ⏳ **Generate HJ baseline embeddings** (~2-3 hours)
   - Run `generate_hj_embeddings.py` to create 4 embedding files

2. ⏳ **Test HJ baseline reproduction**
   - Establish baseline RMSE with road_llm + hf_llm + llm_w

3. ⏳ **Test HJ baseline + local embeddings**
   - Compare RMSE with sgis_local_llm addition

4. ⏳ **Run ablation studies**
   - Test individual embedding contributions

---

**Last Updated:** 2025-10-25
**Organization Status:** ✅ COMPLETE
**Next Priority:** Generate HJ baseline LLM embeddings
