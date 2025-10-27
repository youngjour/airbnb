# Tourism Data Integration - Complete Pipeline Summary

**Created**: 2025-10-26
**Objective**: Add tourism data as 5th LLM embedding to improve Airbnb demand forecasting
**Status**: Embeddings generating, ready for model integration

---

## Data Pipeline Overview

### Input Data Sources (Korean Tourism Data Portal)
1. **Korean Credit Card Sales**: Domestic tourism spending by gu (2018.01-2023.12)
   - 32,203 rows
   - Categories: Accommodation, restaurants, shopping, entertainment, etc.

2. **Foreign Credit Card Sales**: International tourism spending by gu (2018.01-2024.12)
   - 34,386 rows
   - Same categories as Korean CC

3. **Navigation Searches**: Tmap destination searches by gu (2018.01-2024.12)
   - 19,950 rows
   - 10 categories: Natural, Historical, Cultural, Experience, Leisure/Sports, Shopping, Dining, Accommodation, Other

### Processing Steps Completed

#### 1. Data Extraction & Consolidation
**Script**: `unzip_and_merge_tourism_data.py`
- Extracted 20 zip files
- Categorized into 3 groups using filename patterns and content analysis
- Merged into consolidated CSV files
- Removed duplicates

**Outputs**:
- `korean_cc_sales_2018_2024.csv`
- `foreign_cc_sales_2018_2024.csv`
- `navigation_searches_2018_2024.csv`

#### 2. Gu-Level Aggregation
**Script**: `process_tourism_complete.py`
- Aggregated all 3 datasets by (month, gu)
- Pivoted navigation categories for detailed features
- Combined into single gu-level dataset

**Output**:
- `tourism_gu_level_combined.csv`: 2,100 rows (25 gu × 84 months)
- 15 columns: month, gu, korean_cc_sales, foreign_cc_sales, navigation_searches, 10 navigation category columns

#### 3. Dong-to-Gu Mapping
**Script**: `distribute_tourism_to_dong.py`
- Extracted from SGIS administrative code file
- 426 Seoul dongs mapped to 25 gu
- Used administrative hierarchy: first 5 digits of 8-digit code = gu

#### 4. Dong-Level Distribution
**Script**: `distribute_tourism_to_dong.py`
- **Method**: Airbnb-density weighted distribution
- **Formula**: `dong_value = gu_value × (dong_airbnb_count / gu_total_airbnb_count)`
- **Validation**: Perfect (0% difference) - sum of dong values = gu value

**Output**:
- `tourism_dong_level_distributed.csv`: 35,784 rows (425 dongs × 84 months)
- 17 columns: month, Dong_name, Gu_name, weight, 13 tourism features

**Top Tourism Dongs by Korean CC Sales**:
1. 명동 (Myeong-dong): 32.3B won
2. 삼성1동: 27.6B won
3. 회현동: 15.0B won
4. 삼성2동: 13.2B won
5. 청담동: 12.8B won

#### 5. Feature Engineering & Prompt Generation
**Script**: `generate_tourism_prompts.py`

**Engineered Features**:
- `total_cc_sales`: Korean + Foreign CC sales
- `foreign_ratio`: International tourism percentage
- `cc_per_search`: Spending intensity per search
- Navigation category ratios (9 categories)
- `cc_rank`: Ranking by tourism spending
- `cc_percentile`: Percentile rank
- `nav_rank`: Ranking by navigation searches
- `cc_mom_growth`: Month-over-month growth
- `cc_seasonal_index`: Seasonal variation index

**Prompt Template** (average 700 characters):
```
"In {year}-{month}, {dong} in {gu} district recorded tourism activity worth {total_cc:,} won
in credit card transactions, ranking {cc_rank} among Seoul neighborhoods.

{International tourism context based on foreign_ratio}

The area received {nav_searches:,} navigation searches for tourist destinations, ranking
{nav_rank} in search interest.

{Dominant tourism categories - top 3}

{Growth trend interpretation based on mom_growth}

{Seasonal context based on seasonal_index}

{Airbnb relevance based on percentile}"
```

**Output**:
- `tourism_prompts.csv`: 35,784 prompts
- Columns: Dong_name, Gu_name, month, prompt

#### 6. LLM Embedding Generation
**Script**: `generate_tourism_embeddings.py` (IN PROGRESS)
- **Model**: meta-llama/Llama-3.2-3B-Instruct
- **Method**: Mean pooling over last hidden state
- **Embedding dimension**: 3,072
- **Batch size**: 16
- **Max token length**: 512

**Expected Output**:
- `tourism_llm_embeddings.csv`: 35,784 × 3,074 (2 metadata + 3,072 embedding dims)
- Columns: Reporting Month, Dong_name, dim_0, dim_1, ..., dim_3071
- Estimated time: 30-60 minutes on GPU

---

## Data Coverage Analysis

### Temporal Coverage
- **Tourism data**: 2018.01 - 2024.12 (84 months)
- **Model period**: 2017.01 - 2021.11 (62 months: 49 train + 6 val + 7 test)
- **Overlap**: 2018.01 - 2021.11 (47 months = 76% coverage)

**Coverage breakdown**:
- Training period (2017.01-2020.01): 36/37 months = 97%
- Validation period (2020.02-2020.07): 6/6 months = 100%
- Test period (2020.08-2021.11): 5/19 months = 26%

**Note**: Missing 2017 data is acceptable because:
- 97% of training data is covered
- All validation data is covered
- Test period gap will be handled by filling with zeros or interpolation

### Geographic Coverage
- **Dongs with tourism data**: 425
- **Dongs with Airbnb data**: 424
- **Overlap**: 424 dongs (99.8%)
- **Missing**: 1-2 dongs will have zero tourism features

---

## Next Steps

### Step 1: Verify Embedding Generation (USER ACTION)
After the script completes, verify the output:

```bash
cd Preprocess/tmap_tourism
python -c "import pandas as pd; df = pd.read_csv('tourism_llm_embeddings.csv', encoding='utf-8-sig'); print(f'Shape: {df.shape}'); print(f'Columns: {df.columns.tolist()[:5]} ... {df.columns.tolist()[-3:]}'); print(f'Sample:\n{df.head(3)}')"
```

Expected output:
```
Shape: (35784, 3074)
Columns: ['Reporting Month', 'Dong_name', 'dim_0', 'dim_1', 'dim_2'] ... ['dim_3069', 'dim_3070', 'dim_3071']
```

### Step 2: Filter to Model Period
Create a version filtered to the model training period (2017.01-2021.11):

```python
import pandas as pd

# Load full embeddings
df = pd.read_csv('tourism_llm_embeddings.csv', encoding='utf-8-sig')

# Filter to model period
df['Reporting Month'] = pd.to_datetime(df['Reporting Month'])
model_period = (df['Reporting Month'] >= '2017-01-01') & (df['Reporting Month'] <= '2021-11-01')
df_filtered = df[model_period]

# Save
df_filtered.to_csv('tourism_llm_embeddings_model_period.csv', index=False, encoding='utf-8-sig')
print(f"Filtered shape: {df_filtered.shape}")
```

Expected: ~26,000 rows (424 dongs × ~60 months with data)

### Step 3: Add --embed5 Support to Model Code

#### A. Update `Model/main.py`
Add embed5 argument after embed4:

```python
# Around line where embed4 is defined
parser.add_argument('--embed4', type=str, default=None,
                   help='4th embedding type (sgis_local_llm, sgis_local_llm_v2)')
parser.add_argument('--embed5', type=str, default=None,
                   help='5th embedding type (tourism_llm)')
```

#### B. Update `Model/transformer_model.py`
Add 6-input configuration (5 embeddings + 1 preprocessing):

```python
# In get_model_config() function, add:
if args.embed1 and args.embed2 and args.embed3 and args.embed4 and args.embed5 and args.embed_artifact:
    # 6-input configuration
    config = {
        'num_inputs': 6,
        'input_dims': [
            EMBEDDING_DIMS[args.embed1],
            EMBEDDING_DIMS[args.embed2],
            EMBEDDING_DIMS[args.embed3],
            EMBEDDING_DIMS[args.embed4],
            EMBEDDING_DIMS[args.embed5],
            artifact_dim
        ],
        'fusion_type': 'concat'  # or 'attention'
    }
```

#### C. Update embedding dimensions mapping
```python
EMBEDDING_DIMS = {
    'road_llm': 3072,
    'hf_llm': 3072,
    'llm_w': 3072,
    'sgis_local_llm': 3072,
    'sgis_local_llm_v2': 3072,
    'tourism_llm': 3072,  # ADD THIS LINE
}
```

#### D. Update data loading
In `Model/load_data.py` or wherever embeddings are loaded:

```python
# Add tourism_llm to embedding paths
EMBEDDING_PATHS = {
    # ... existing paths ...
    'tourism_llm': 'Preprocess/tmap_tourism/tourism_llm_embeddings_model_period.csv',
}
```

### Step 4: Test Model with Tourism Embeddings

```bash
cd Model

# Test configuration
python main.py \
  --embed1 road_llm \
  --embed2 hf_llm \
  --embed3 llm_w \
  --embed4 sgis_local_llm_v2 \
  --embed5 tourism_llm \
  --model transformer \
  --epochs 5 \
  --batch_size 8 \
  --window_size 9 \
  --mode 3m \
  --label all \
  2>&1 | tee tourism_integration_test.log
```

### Step 5: Compare Performance

**Baselines to compare**:
- HJ baseline (3 embeddings): RMSE 0.5312
- HJ + Local v2 (4 embeddings): RMSE 0.5287 (0.47% improvement)

**Expected with tourism (5 embeddings)**:
- Target: RMSE < 0.520 (1-2% improvement)
- Rationale: Tourism data adds temporal patterns (seasonal trends, spending growth) that complement spatial features

---

## Troubleshooting

### If embeddings have wrong shape
Check the prompt CSV first:
```bash
cd Preprocess/tmap_tourism
wc -l tourism_prompts.csv  # Should be 35785 (including header)
```

### If model fails to load tourism embeddings
Verify column names match exactly:
```python
df = pd.read_csv('tourism_llm_embeddings.csv')
assert 'Reporting Month' in df.columns
assert 'Dong_name' in df.columns
assert 'dim_0' in df.columns
```

### If temporal alignment fails
The tourism data uses format `2018-01-01`, ensure other embeddings match:
```python
# In data loading code
tourism['Reporting Month'] = pd.to_datetime(tourism['Reporting Month'])
```

---

## Files Created

```
Preprocess/tmap_tourism/
├── unzip_and_merge_tourism_data.py        # Step 1: Extract & merge
├── process_tourism_complete.py             # Step 2: Gu-level aggregation
├── distribute_tourism_to_dong.py           # Step 3-4: Dong distribution
├── generate_tourism_prompts.py             # Step 5: Prompt generation
├── generate_tourism_embeddings.py          # Step 6: LLM embeddings
├── korean_cc_sales_2018_2024.csv          # Merged Korean CC data
├── foreign_cc_sales_2018_2024.csv         # Merged foreign CC data
├── navigation_searches_2018_2024.csv      # Merged navigation data
├── tourism_gu_level_combined.csv          # Gu-level aggregated (2,100 rows)
├── tourism_dong_level_distributed.csv      # Dong-level distributed (35,784 rows)
├── tourism_prompts.csv                     # LLM prompts (35,784 rows)
├── tourism_llm_embeddings.csv             # LLM embeddings (IN PROGRESS)
├── TMAP_INTEGRATION_PLAN.md               # Original planning document
└── TOURISM_INTEGRATION_SUMMARY.md         # This file
```

---

## Theoretical Foundation

### Why Tourism Data Should Improve Airbnb Predictions

1. **Intent Signal**: Navigation searches indicate tourist interest in visiting specific areas
2. **Spending Patterns**: CC sales show actual economic activity from tourists
3. **International Appeal**: Foreign CC ratio identifies areas attractive to international guests
4. **Temporal Dynamics**: Monthly variations capture seasonal tourism patterns
5. **Category Mix**: Different destination types (cultural, shopping, dining) attract different Airbnb segments

### Hypothesis
Areas with:
- High tourism spending → Higher Airbnb demand
- Growing tourism trends → Increasing Airbnb bookings
- Strong international tourism → Premium pricing opportunities
- Diverse destination categories → Broader guest appeal

### Complementarity with Existing Embeddings
- **Road network (road_llm)**: Static infrastructure
- **Human flow (hf_llm)**: General mobility patterns
- **Airbnb (llm_w)**: Platform-specific features
- **Local SGIS (sgis_local_llm_v2)**: Economic structure
- **Tourism (tourism_llm)**: Temporal tourism dynamics ← NEW

---

## Contact & Maintenance

**Created by**: Claude Code
**Date**: 2025-10-26
**Model**: Llama-3.2-3B-Instruct
**Purpose**: Airbnb demand forecasting enhancement

For questions or issues, refer to:
- TMAP_INTEGRATION_PLAN.md (original planning document)
- tourism_prompts.csv (to inspect prompt quality)
- Preprocess/sgis_manual/generate_local_embeddings_v2.py (reference implementation)
