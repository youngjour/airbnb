# SGIS API Limitations & Recommended Next Steps

## Summary of Investigation

After thorough testing and reviewing the SGIS manual, I've discovered important limitations of the SGIS OpenAPI:

### ✅ What the API CAN Provide:
1. **Housing Units** - Household counts at enumeration district level (subsets of dongs)
   - Fields: `household_cnt`, `avg_family_member_cnt`, `family_member_cnt`

2. **Company Counts (Aggregate)** - Total business establishments and workers
   - Fields: `corp_cnt`, `tot_worker`
   - **NO industry breakdown** (accommodations, retail, restaurants, etc.)

### ❌ What the API CANNOT Provide:
- Industry-specific business counts (e.g., `cp2_bnu_55` for accommodations)
- Separate counts for accommodations vs. retail vs. restaurants
- Any granular business classification data

### 🔍 Why This Limitation Exists:

The SGIS platform has two separate systems:
1. **OpenAPI** - Returns only aggregate statistics
2. **Web Portal + SDC (Statistics Data Center)** - Provides detailed industry breakdowns through custom data requests

The industry codes like `cp2_bnu_55` (accommodations) and `cp2_bnu_56` (restaurants/bars) that you found in the reference manual are **column names in downloaded CSV files from the web portal**, not API parameters.

---

## Three Options Going Forward

### Option 1: Use Aggregate Company Counts (RECOMMENDED)

**Collect what's available via API:**
- Housing units (residential property supply)
- Total company counts (overall commercial activity)
- Total worker counts (employment/economic activity)

**Pros:**
- ✅ Fully automated via API
- ✅ Can collect immediately
- ✅ Still provides valuable economic activity indicators
- ✅ Total commercial activity is correlated with Airbnb demand

**Cons:**
- ❌ No distinction between accommodations and other businesses
- ❌ Less specific than originally planned

**Implementation:**
```bash
# Run the updated collection script
python collect_sgis_data_v2.py --years 2017 2018 2019 2020 2021 2022 2023
```

**Expected Variables for Model:**
1. `housing_units` - Total residential properties in dong
2. `total_companies` - Aggregate business establishment count
3. `total_workers` - Total employment in dong

---

### Option 2: Manual Web Portal Request

**Request industry-specific data through SGIS web portal:**

**Process:**
1. Go to https://sgis.kostat.go.kr/
2. Navigate to "자료제공" (Data Service)
3. Submit custom data request for:
   - `cp2_bnu_55` (accommodations)
   - `cp2_bnu_56` (restaurants/bars)
   - `cp2_bnu_47` (retail stores, if needed)
4. Receive data file via email/download (typically within 1-2 business days)

**Pros:**
- ✅ Get exact industry-specific data you wanted
- ✅ Official SGIS data with full granularity

**Cons:**
- ❌ Manual process, not automated
- ❌ May take 1-2 days to receive data
- ❌ One-time download (not easy to update)

---

### Option 3: Housing Units Only

**Simplest approach - skip company data entirely:**

**Rationale:**
- Housing units alone is still a valuable indicator
- Represents supply-side potential for Airbnb listings
- Avoids the complexity of ambiguous company data

**Pros:**
- ✅ Clean, simple, automated
- ✅ Strong theoretical justification
- ✅ Easy to implement and maintain

**Cons:**
- ❌ Missing commercial activity indicators
- ❌ Less comprehensive than original plan

---

## My Recommendation

I recommend **Option 1** (aggregate company counts) because:

1. **Pragmatic**: You can collect data immediately and start training
2. **Still Valuable**: Total commercial activity is a reasonable proxy for:
   - Tourist area attractiveness
   - Neighborhood vitality
   - Economic development level
3. **Automated**: Fully scriptable and repeatable
4. **Theory-Based**: Areas with high Airbnb demand likely have:
   - More overall businesses (including accommodations)
   - More restaurants, shops, services
   - Higher foot traffic

### Revised Research Rationale

Instead of:
> "We collect counts of **accommodations**, **restaurants/bars**, and **retail stores**"

You would say:
> "We collect **total business establishment counts** and **total employment** as proxies for commercial vitality and tourist area attractiveness, alongside **housing units** representing Airbnb supply potential"

This is methodologically sound and many urban economics papers use aggregate business density as an indicator of neighborhood characteristics.

---

## Next Steps to Proceed with Option 1

### Step 1: Run Full Data Collection (~30-45 minutes)

```bash
cd "C:\Users\jour\Documents\GitHub\airbnb\Preprocess\sgis_manual"

# Collect for all 426 Seoul dongs, years 2017-2023
python collect_sgis_data_v2.py --years 2017 2018 2019 2020 2021 2022 2023 --output-dir ./collected_data
```

This will create:
- `collected_data/housing_data_raw.csv` - Household/housing statistics
- `collected_data/company_data_raw.csv` - Company counts and worker totals
- `collected_data/collection_summary.txt` - Summary statistics

### Step 2: Preprocess to Monthly Time Series

```bash
# Convert yearly data to monthly (67 months: 2017-07 to 2023-01)
python preprocess_sgis_data.py \
  --input-dir ./collected_data \
  --output-dir ../../Data/Preprocessed_data/Dong \
  --reference-data ../../Data/Preprocessed_data/AirBnB_labels_dong.csv \
  --start-month 2017-07 \
  --end-month 2023-01
```

This will create:
- `SGIS_housing_units_monthly.csv`
- `SGIS_company_counts_monthly.csv`
- `SGIS_combined_embedding.csv`
- `SGIS_embedding_aligned.csv` (ready for model)

### Step 3: Update Model Configuration

In `Model/main.py`, add the SGIS embedding:

```python
embedding_paths_dict = {
    'road': '../Data/Preprocessed_data/Dong/Road_Embeddings_with_flow.csv',
    'hf': '../Data/Preprocessed_data/Dong/Human_flow.csv',
    'raw': '../Data/Preprocessed_data/Dong/AirBnB_raw_embedding.csv',
    'llm_w': '../Data/Preprocessed_data/Dong/llm_embeddings_new/Airbnb_SSP_w.csv',
    'llm_wo': '../Data/Preprocessed_data/Dong/llm_embeddings_new/Airbnb_SSP_wo.csv',
    'road_llm': '../Data/Preprocessed_data/Dong/llm_embeddings_new/road_llm.csv',
    'hf_llm': '../Data/Preprocessed_data/Dong/llm_embeddings_new/human_flow_llm.csv',
    'sgis': '../Data/Preprocessed_data/Dong/SGIS_embedding_aligned.csv'  # NEW!
}
```

### Step 4: Train and Evaluate

```bash
# Test with SGIS embedding
python main.py --embed1 road_llm --embed2 hf_llm --embed3 sgis --model lstm --window_size 6

# Compare with baseline (no SGIS)
python main.py --embed1 road_llm --embed2 hf_llm --embed3 llm_w --model lstm --window_size 6
```

---

## If You Want Industry-Specific Data (Option 2)

If you decide you really need the industry breakdowns and are willing to do the manual request:

1. Visit: https://sgis.kostat.go.kr/
2. Click "자료제공" → "맞춤 통계 생성"
3. Request data for Seoul (서울시)
4. Specify years: 2017-2023
5. Request these specific columns:
   - `cp2_bnu_55` - 숙박업 (Accommodation businesses)
   - `cp2_bnu_56` - 음식점 및 주점업 (Restaurants and bars)
   - Additional: `cp2_bnu_47` if you want retail

6. Wait 1-2 business days for data file
7. Once received, I can help you integrate it into the pipeline

---

## Questions?

Let me know which option you'd like to proceed with! I've prepared the code for Option 1 (aggregate data) and can help with either Option 2 (manual request) or Option 3 (housing only) if you prefer those approaches.

The collection script is ready to run whenever you give the go-ahead!
