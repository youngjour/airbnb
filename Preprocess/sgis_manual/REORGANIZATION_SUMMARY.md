# Repository Reorganization Summary

**Date:** 2025-10-25
**Purpose:** Clean up experimental code, keep only essential LLM embedding workflow files

---

## What Was Organized

### Files Moved to `_archive_experimental/`

**Data Collection Experiments:**
- collect_sgis_complete.py
- collect_sgis_data.py
- collect_sgis_data_v2.py
- collect_sgis_improved.py
- collect_sgis_ratios_simple.py
- sgis_api_client.py

**Testing & Diagnostic Scripts:**
- test_*.py (all test scripts)
- check_*.py (all check scripts)
- diagnose_*.py
- explore_*.py
- find_correct_code_format.py

**Preprocessing Experiments:**
- align_*.py
- analyze_*.py
- deduplicate_*.py
- fill_*.py
- preprocess_sgis_*.py
- create_feature_subsets.py
- create_custom_subsets.py

**Intermediate/Old Data Files:**
- sgis_complete_data.csv
- sgis_monthly_embedding.csv
- sgis_monthly_embedding_aligned.csv
- sgis_monthly_embedding_aligned_dates.csv
- sgis_monthly_embedding_complete.csv
- sgis_ratios_latest.csv
- sgis_ratios_snapshot.csv
- sgis_improved_subset_*.csv (all subset files)
- test_*.csv

**Old Documentation:**
- API_LIMITATIONS_AND_NEXT_STEPS.md
- FINAL_DATA_COLLECTION_SUMMARY.md
- PREPROCESSING_COMPLETE.md
- WORK_LOG_2025_10_21.md

**Log Files:**
- sgis_collection_20251021_020838.log

**Test Output Directories:**
- test_output/
- test_output_v2/

---

## Files Kept (Essential LLM Workflow)

### Production Scripts
- `generate_local_prompts.py` - SGIS features → Natural language prompts
- `generate_local_embeddings.py` - Prompts → LLM embeddings (3,072 dims)

### Data Files (Essential)
- `sgis_improved_final.csv` - Final SGIS engineered features (source)
- `sgis_local_prompts.csv` - Generated natural language prompts
- `sgis_local_llm_embeddings.csv` - Generated LLM embeddings ✅ COMPLETE
- `한국행정구역분류_행정동코드(7자리)_20210701기준_extracted.csv` - Administrative codes

### Documentation (Current Workflow)
- `LOCAL_EMBEDDINGS_GUIDE.md` - Complete SGIS embedding guide
- `QUICK_START.md` - Quick reference
- `RUN_EMBEDDINGS_yj_pytorch.md` - Environment-specific instructions
- `INSTALL_PYTORCH_CUDA.md` - PyTorch setup guide
- `RUN_EMBEDDINGS.bat` - Windows helper script
- `README.md` - Original SGIS directory README

### Reference Code
- `ref_code/` - Reference implementations (kept for documentation)

---

## Current Directory Structure

```
Preprocess/sgis_manual/
├── generate_local_prompts.py           # PRODUCTION: Prompt generation
├── generate_local_embeddings.py        # PRODUCTION: LLM embedding generation
│
├── sgis_improved_final.csv             # SOURCE: Engineered features (28,408 rows)
├── sgis_local_prompts.csv              # OUTPUT: Natural language prompts
├── sgis_local_llm_embeddings.csv       # OUTPUT: LLM embeddings (COMPLETE)
├── 한국행정구역분류_행정동코드(7자리)_20210701기준_extracted.csv  # REFERENCE: Admin codes
│
├── LOCAL_EMBEDDINGS_GUIDE.md           # DOCS: Complete guide
├── QUICK_START.md                      # DOCS: Quick reference
├── RUN_EMBEDDINGS_yj_pytorch.md        # DOCS: Environment guide
├── INSTALL_PYTORCH_CUDA.md             # DOCS: PyTorch setup
├── RUN_EMBEDDINGS.bat                  # HELPER: Windows script
├── README.md                           # DOCS: Directory README
├── REORGANIZATION_SUMMARY.md           # DOCS: This file
│
├── ref_code/                           # REFERENCE: Example code
└── _archive_experimental/              # ARCHIVE: Old/experimental code
    ├── collect_sgis_*.py               # Old data collection scripts
    ├── test_*.py                       # Old test scripts
    ├── check_*.py                      # Old check scripts
    ├── align_*.py, analyze_*.py, etc.  # Old preprocessing experiments
    ├── sgis_*.csv (intermediate files) # Old/intermediate data
    └── *.md (old documentation)        # Old documentation
```

---

## Benefits of Reorganization

1. **Clarity:** Clear separation between production code and experiments
2. **Maintainability:** Easy to find essential LLM workflow files
3. **Documentation:** Self-documenting structure (production vs archive)
4. **Onboarding:** New users can quickly understand the workflow
5. **Git History:** Experimental iterations preserved but organized

---

## Essential LLM Embedding Workflow (Post-Cleanup)

### Step 1: Feature Engineering
**Script:** (Archived - already completed)
**Output:** `sgis_improved_final.csv` (28,408 rows × 8 columns)

### Step 2: Prompt Generation
**Script:** `generate_local_prompts.py`
**Input:** sgis_improved_final.csv
**Output:** `sgis_local_prompts.csv` (28,408 prompts)

### Step 3: LLM Embedding Generation
**Script:** `generate_local_embeddings.py`
**Input:** sgis_local_prompts.csv
**Output:** `sgis_local_llm_embeddings.csv` (28,408 × 3,074)
**Status:** ✅ COMPLETE

### Step 4: Model Training
**Location:** `Model/main.py`
**Usage:** `--embed1 road_llm --embed2 hf_llm --embed3 sgis_local_llm`
**Status:** ⏳ Pending HJ baseline embedding generation

---

## Accessing Archived Files

If you need to access experimental code or old data:

```bash
cd Preprocess/sgis_manual/_archive_experimental/
ls -la
```

All experimental iterations are preserved but organized separately from production workflow.

---

## Next Steps

With clean repository structure:
1. ✅ SGIS local embeddings complete
2. ⏳ Generate HJ baseline embeddings (4 types, ~2-3 hours)
3. ⏳ Test HJ baseline + local embeddings
4. ⏳ Evaluate improvement vs baseline

---

**Note:** This reorganization preserves all code and data - nothing was deleted. Experimental files are archived for reference but separated from the main workflow.
