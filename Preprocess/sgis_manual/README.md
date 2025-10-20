# SGIS Data Integration for Airbnb Prediction Model

This directory contains the complete pipeline for collecting and integrating SGIS (Statistical Geographic Information Service) census data into the Airbnb prediction model.

## 📊 Overview

### Rationale for Data Enhancement

The original Airbnb prediction model uses:
- **Urban Accessibility Data** (road networks, OSM data)
- **Human Mobility Data** (population flow)
- **Airbnb Listing Features** (from AirDNA)

This enhancement adds three new data sources from Korean Census (SGIS):
1. **Housing Units** - Total residential properties in each dong (potential Airbnb properties)
2. **Accommodations** - Hotels, motels, guesthouses (competitors to Airbnb)
3. **Retail Stores** - Shops and commercial establishments (POI indicators for tourism attractiveness)

### Why These Variables Matter

- **Housing Units**: Indicates the supply-side potential for Airbnb listings
- **Accommodations**: Traditional lodging competitors that affect Airbnb demand
- **Retail Stores**: Reflects commercial vitality and tourist attraction factors

---

## 🗂️ File Structure

```
sgis_manual/
├── README.md                                    # This file
├── sgis_api_client.py                           # SGIS API client classes
├── collect_sgis_data.py                         # Data collection script
├── preprocess_sgis_data.py                      # Data preprocessing pipeline
├── 한국행정구역분류_행정동코드(7자리)_20210701기준_extracted.csv  # Dong codes
└── collected_data/                              # Output directory (created after collection)
    ├── housing_units_raw.csv
    ├── company_data_raw.csv
    ├── accommodations_raw.csv
    └── retail_stores_raw.csv
```

---

## 🚀 Quick Start Guide

### Step 1: Collect SGIS Data

Run the data collection script to fetch data from SGIS API:

```bash
cd Preprocess/sgis_manual

# Collect data for 2017-2023 (matching your 67 months of Airbnb data)
python collect_sgis_data.py --years 2017 2018 2019 2020 2021 2022 2023
```

**Expected Output:**
- `collected_data/housing_units_raw.csv` - Raw housing unit counts
- `collected_data/company_data_raw.csv` - Raw company/business data
- `collected_data/accommodations_raw.csv` - Filtered accommodation businesses
- `collected_data/retail_stores_raw.csv` - Filtered retail stores

**Note:** This process will take time (approximately 30-60 minutes) as it makes API calls for all 424 Seoul dongs across 7 years.

### Step 2: Preprocess Data to Monthly Embeddings

Convert the yearly census data to monthly time series:

```bash
python preprocess_sgis_data.py \
  --input-dir ./collected_data \
  --output-dir ../../Data/Preprocessed_data/Dong \
  --reference-data ../../Data/Preprocessed_data/AirBnB_labels_dong.csv \
  --start-month 2017-07 \
  --end-month 2023-01
```

**Expected Output:**
- `SGIS_housing_units_monthly.csv` - Monthly housing unit counts per dong
- `SGIS_accommodations_monthly.csv` - Monthly accommodation counts per dong
- `SGIS_retail_monthly.csv` - Monthly retail store counts per dong
- `SGIS_combined_embedding.csv` - All three features combined
- `SGIS_embedding_aligned.csv` - Aligned with your existing Airbnb data structure

### Step 3: Update Model Configuration

Modify `Model/main.py` to include the new SGIS embedding:

```python
# In main.py, update the embedding_paths_dict:

embedding_paths_dict = {
    'road': '../Data/Preprocessed_data/Dong/Road_Embeddings_with_flow.csv',
    'hf': '../Data/Preprocessed_data/Dong/Human_flow.csv',
    'raw': '../Data/Preprocessed_data/Dong/AirBnB_raw_embedding.csv',
    'llm_w': '../Data/Preprocessed_data/Dong/llm_embeddings_new/Airbnb_SSP_w.csv',
    'llm_wo': '../Data/Preprocessed_data/Dong/llm_embeddings_new/Airbnb_SSP_wo.csv',
    'road_llm': '../Data/Preprocessed_data/Dong/llm_embeddings_new/road_llm.csv',
    'hf_llm': '../Data/Preprocessed_data/Dong/llm_embeddings_new/human_flow_llm.csv',
    # NEW: Add SGIS embedding
    'sgis': '../Data/Preprocessed_data/Dong/SGIS_embedding_aligned.csv'
}
```

### Step 4: Train Model with SGIS Features

Run training with the new SGIS embedding:

```bash
cd ../../Model

# Example: LSTM with road_llm + hf_llm + sgis embeddings
python main.py \
  --embed1 road_llm \
  --embed2 hf_llm \
  --embed3 sgis \
  --model lstm \
  --dim_opt 3 \
  --window_size 6 \
  --mode 3m \
  --label all \
  --output_dir outputs_with_sgis
```

---

## 📚 Detailed Component Documentation

### 1. SGIS API Client (`sgis_api_client.py`)

**Classes:**

- **`SGISAPIClient`**: Base client for SGIS API authentication and requests
  - Handles OAuth-like token authentication
  - Auto-refreshes expired tokens
  - Rate-limiting friendly

- **`HouseholdDataCollector`**: Collects household/housing statistics
  - Endpoint: `/OpenAPI3/stats/household.json`
  - Returns housing unit counts by administrative division

- **`CompanyDataCollector`**: Collects company/business statistics
  - Endpoint: `/OpenAPI3/stats/company.json`
  - Returns business establishments by industry classification

- **`IndustryCodeHelper`**: Helper for industry code lookups
  - Endpoint: `/OpenAPI3/stats/industrycode.json`
  - Useful for finding correct industry codes

**Example Usage:**

```python
from sgis_api_client import SGISAPIClient, HouseholdDataCollector

# Initialize client
client = SGISAPIClient(
    consumer_key="fbf9612b73e54fac8545",
    consumer_secret="0543b74f9984418da672"
)

# Collect data
collector = HouseholdDataCollector(client)
housing_data = collector.get_household_data(
    adm_cd="1101053",  # 사직동
    year="2020"
)
```

### 2. Data Collection Script (`collect_sgis_data.py`)

**Features:**
- Batch collection for all Seoul dongs
- Rate limiting to respect API quotas
- Progress tracking and logging
- Automatic filtering of accommodations and retail stores

**Command-line Arguments:**

```bash
python collect_sgis_data.py \
  --years 2017 2018 2019 2020 2021 2022 2023 \
  --dong-codes-csv 한국행정구역분류_행정동코드(7자리)_20210701기준_extracted.csv \
  --output-dir ./collected_data \
  --rate-limit 0.5
```

- `--years`: Years to collect data for (space-separated)
- `--dong-codes-csv`: Path to dong codes CSV file
- `--output-dir`: Directory to save collected data
- `--rate-limit`: Delay between API calls in seconds

### 3. Data Preprocessing Pipeline (`preprocess_sgis_data.py`)

**Key Functions:**

- **`interpolate_yearly_to_monthly()`**: Converts yearly census data to monthly time series
  - Uses linear interpolation between years
  - Forward/backward fills for edge cases

- **`process_housing_units()`**: Processes housing unit data
- **`process_company_counts()`**: Processes accommodation/retail counts
- **`create_combined_embedding()`**: Merges all features into one DataFrame
- **`align_with_existing_data()`**: Ensures compatibility with existing Airbnb data structure

**Example Usage:**

```python
from preprocess_sgis_data import SGISDataPreprocessor

preprocessor = SGISDataPreprocessor(
    start_month='2017-07',
    end_month='2023-01'
)

# Process housing units
housing_df = preprocessor.process_housing_units(
    housing_raw_path='./collected_data/housing_units_raw.csv',
    output_path='./SGIS_housing_monthly.csv'
)

# Create combined embedding
combined = preprocessor.create_combined_embedding(
    housing_df=housing_df,
    accommodations_df=accommodations_df,
    retail_df=retail_df
)
```

---

## 🔍 Data Schema

### Housing Units Monthly Data

| Column | Type | Description |
|--------|------|-------------|
| dong_code | str | 7-digit administrative dong code |
| dong_name | str | Dong name in Korean |
| Reporting Month | str | Month in YYYY-MM format |
| housing_units | float | Number of housing units |
| housing_units_normalized | float | Normalized housing units |

### Accommodations Monthly Data

| Column | Type | Description |
|--------|------|-------------|
| dong_code | str | 7-digit administrative dong code |
| dong_name | str | Dong name in Korean |
| Reporting Month | str | Month in YYYY-MM format |
| accommodations_count | float | Number of accommodation businesses |
| accommodations_count_normalized | float | Normalized count |

### Retail Stores Monthly Data

| Column | Type | Description |
|--------|------|-------------|
| dong_code | str | 7-digit administrative dong code |
| dong_name | str | Dong name in Korean |
| Reporting Month | str | Month in YYYY-MM format |
| retail_count | float | Number of retail stores |
| retail_count_normalized | float | Normalized count |

### Combined SGIS Embedding

Combines all three feature sets with both raw and normalized values.

---

## 🛠️ Troubleshooting

### Issue: "Authentication failed"

**Solution:** Check that your API credentials are correct:
```python
CONSUMER_KEY = "fbf9612b73e54fac8545"
CONSUMER_SECRET = "0543b74f9984418da672"
```

### Issue: "No data collected for some dongs"

**Possible Reasons:**
- The dong code doesn't exist in SGIS database
- No data available for the requested year
- API rate limiting (increase `--rate-limit` value)

**Solution:** Check the log files for specific errors and retry with higher rate limit.

### Issue: "Column not found during preprocessing"

**Solution:** Inspect the raw CSV files to see actual column names:
```python
import pandas as pd
df = pd.read_csv('collected_data/housing_units_raw.csv')
print(df.columns)
```

Adjust the `value_col` detection logic in `preprocess_sgis_data.py` accordingly.

### Issue: "Dong names don't match between SGIS and Airbnb data"

**Solution:** Use the `align_with_existing_data()` function which handles name matching automatically. If issues persist, create a manual mapping file.

---

## 📊 Expected Model Performance Improvement

Based on the rationale:

1. **Housing Units** should help predict:
   - Airbnb listing supply growth
   - Market saturation indicators

2. **Accommodations** should help predict:
   - Competition effects on pricing
   - Market dynamics in tourist areas

3. **Retail Stores** should help predict:
   - Tourist attraction factors
   - Commercial vitality of neighborhoods

**Suggested Experiments:**

```bash
# Baseline (without SGIS)
python main.py --embed1 road_llm --embed2 hf_llm --embed3 llm_w --model lstm

# With SGIS embedding
python main.py --embed1 road_llm --embed2 hf_llm --embed3 sgis --model lstm

# All features combined (4 embeddings)
# Note: You'll need to modify main.py to support 4+ embeddings
python main.py --embed1 road_llm --embed2 hf_llm --embed3 llm_w --embed4 sgis --model lstm
```

---

## 🔗 API Documentation References

- **SGIS Main Page**: https://sgis.kostat.go.kr/
- **Household API Docs**: https://sgis.kostat.go.kr/developer/html/newOpenApi/api/dataApi/census.html#household
- **Company API Docs**: https://sgis.kostat.go.kr/developer/html/newOpenApi/api/dataApi/census.html#company
- **Industry Codes**: https://sgis.kostat.go.kr/developer/html/openApi/api/dataCode/ThemeCode.html

---

## 📝 Notes for Code Table Integration

**TODO: After obtaining the actual code tables from SGIS API**

Once you retrieve the industry classification codes, update the filtering logic in `collect_sgis_data.py`:

```python
# In filter_accommodations():
accommodation_codes = [
    'I55',    # Accommodation (update with actual codes)
    'I551',   # Short-term accommodation
    'I552',   # Camping grounds, recreational vehicle parks
    # Add more specific codes here
]

# In filter_retail_stores():
retail_codes = [
    'G47',    # Retail trade (update with actual codes)
    'G471',   # Retail sale in non-specialized stores
    'G472',   # Food/beverage/tobacco retail
    # Add more specific codes here
]
```

To fetch industry codes:
```bash
curl "https://sgisapi.kostat.go.kr/OpenAPI3/stats/industrycode.json?consumer_key=fbf9612b73e54fac8545&consumer_secret=0543b74f9984418da672"
```

---

## 🤝 Contributing

If you improve this pipeline or find issues:
1. Document any modifications to API calls
2. Update filtering logic based on actual industry codes
3. Share performance improvements with the team

---

## 📧 Contact

For questions about this integration:
- **Author**: Airbnb Prediction Model Team
- **Date**: October 2025
- **Reference**: SGIS API v3.0

---

## ✅ Checklist

Before running the model with SGIS data:

- [ ] Collected raw data from SGIS API
- [ ] Preprocessed data to monthly time series
- [ ] Verified data alignment with existing Airbnb data
- [ ] Updated `main.py` with new embedding path
- [ ] Tested model training with SGIS features
- [ ] Compared performance with baseline model
- [ ] Documented any issues or improvements

Good luck with your model enhancement! 🚀
