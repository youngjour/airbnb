"""
Distribute gu-level tourism data to dong level using Airbnb-density weighting

Steps:
1. Load dong-to-gu mapping from administrative code file
2. Load Airbnb density by dong
3. Calculate distribution weights for each gu
4. Distribute tourism metrics to dong level
5. Save dong-level tourism data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print(f"{'='*100}")
print(f"TOURISM DATA DONG-LEVEL DISTRIBUTION")
print(f"{'='*100}\n")

# Paths
TOURISM_DIR = Path(r'C:\Users\jour\Documents\GitHub\airbnb\Preprocess\tmap_tourism')
DATA_DIR = Path(r'C:\Users\jour\Documents\GitHub\airbnb\Data\Preprocessed_data')
SGIS_DIR = Path(r'C:\Users\jour\Documents\GitHub\airbnb\Preprocess\sgis_manual')

# Input files
TOURISM_GU_FILE = TOURISM_DIR / 'tourism_gu_level_combined.csv'
AIRBNB_LABELS = DATA_DIR / 'AirBnB_labels_dong.csv'
SGIS_DATA = SGIS_DIR / 'sgis_improved_final.csv'
ADM_CODE_FILE = SGIS_DIR / '한국행정구역분류_행정동코드(7자리)_20210701기준_extracted.csv'

# Step 1: Load administrative code file for dong-to-gu mapping
print(f"STEP 1: LOADING DONG-TO-GU MAPPING")
print(f"{'-'*100}")

adm_codes = pd.read_csv(ADM_CODE_FILE, encoding='utf-8-sig')

# Filter Seoul dongs only (시도 == 11)
seoul_dongs = adm_codes[adm_codes['대분류'] == 11].copy()

# Keep only rows with dong codes (소분류 is not NaN)
seoul_dongs = seoul_dongs[seoul_dongs['소분류'].notna()].copy()

# Create mapping dataframe
dong_to_gu = seoul_dongs[['읍면동', '시군구']].copy()
dong_to_gu.columns = ['Dong_name', 'Gu_name']
dong_to_gu = dong_to_gu.drop_duplicates()

print(f"[OK] Loaded {len(dong_to_gu)} Seoul dongs")
print(f"Unique gu: {dong_to_gu['Gu_name'].nunique()}")
print(f"\nSample mapping:")
print(dong_to_gu.head(10).to_string(index=False))

# Step 2: Load Airbnb density data
print(f"\nSTEP 2: LOADING AIRBNB DENSITY DATA")
print(f"{'-'*100}")

sgis = pd.read_csv(SGIS_DATA, encoding='utf-8-sig')

# Calculate average Airbnb listings per dong across all months
airbnb_density = sgis.groupby('Dong_name')['airbnb_listing_count'].mean().reset_index()
airbnb_density.columns = ['Dong_name', 'avg_airbnb_count']

print(f"[OK] Loaded Airbnb density for {len(airbnb_density)} dongs")
print(f"\nTop 10 dongs by Airbnb density:")
print(airbnb_density.nlargest(10, 'avg_airbnb_count').to_string(index=False))

# Step 3: Merge dong-to-gu mapping with Airbnb density
print(f"\nSTEP 3: MERGING MAPPING WITH AIRBNB DENSITY")
print(f"{'-'*100}")

dong_info = dong_to_gu.merge(airbnb_density, on='Dong_name', how='left')

# Fill missing Airbnb counts with 0
dong_info['avg_airbnb_count'] = dong_info['avg_airbnb_count'].fillna(0)

print(f"[OK] Merged mapping: {len(dong_info)} dongs")
print(f"Dongs with Airbnb listings: {(dong_info['avg_airbnb_count'] > 0).sum()}")

# Calculate weights for each gu
dong_info['weight'] = dong_info.groupby('Gu_name')['avg_airbnb_count'].transform(
    lambda x: x / x.sum() if x.sum() > 0 else 1 / len(x)
)

print(f"\nSample weights (Gangnam-gu):")
gangnam_sample = dong_info[dong_info['Gu_name'].str.contains('강남', na=False)]
if len(gangnam_sample) > 0:
    print(gangnam_sample[['Dong_name', 'avg_airbnb_count', 'weight']].head(10).to_string(index=False))

# Step 4: Load gu-level tourism data
print(f"\nSTEP 4: LOADING GU-LEVEL TOURISM DATA")
print(f"{'-'*100}")

tourism_gu = pd.read_csv(TOURISM_GU_FILE, encoding='utf-8-sig')

print(f"[OK] Loaded tourism data: {tourism_gu.shape}")
print(f"Columns: {tourism_gu.columns.tolist()}")
print(f"Time range: {tourism_gu['month'].min()} to {tourism_gu['month'].max()}")

# Clean gu names in tourism data (remove "서울특별시 " prefix if present)
tourism_gu['gu_clean'] = tourism_gu['gu'].str.replace('서울특별시 ', '', regex=False).str.strip()

# Step 5: Distribute tourism data to dong level
print(f"\nSTEP 5: DISTRIBUTING TOURISM DATA TO DONG LEVEL")
print(f"{'-'*100}")

# Create empty list to store results
dong_tourism_list = []

# Get tourism feature columns (exclude month and gu columns)
tourism_cols = [col for col in tourism_gu.columns if col not in ['month', 'gu', 'gu_clean']]

print(f"Tourism features to distribute: {len(tourism_cols)}")
print(f"  {tourism_cols[:5]} ... {tourism_cols[-2:]}")

# For each month and gu combination
total_iterations = len(tourism_gu)
for idx, row in tourism_gu.iterrows():
    if idx % 500 == 0:
        print(f"  Processing {idx}/{total_iterations}...")

    month = row['month']
    gu_name = row['gu_clean']

    # Get dongs in this gu
    gu_dongs = dong_info[dong_info['Gu_name'] == gu_name].copy()

    if len(gu_dongs) == 0:
        # Try fuzzy matching if exact match fails
        gu_dongs = dong_info[dong_info['Gu_name'].str.contains(gu_name.replace('구', ''), na=False)].copy()

    if len(gu_dongs) == 0:
        print(f"  [WARNING] No dongs found for gu: {gu_name}")
        continue

    # Distribute each tourism metric to dongs
    for _, dong_row in gu_dongs.iterrows():
        dong_data = {
            'month': month,
            'Dong_name': dong_row['Dong_name'],
            'Gu_name': dong_row['Gu_name'],
            'weight': dong_row['weight']
        }

        # Distribute each tourism feature
        for col in tourism_cols:
            gu_value = row[col]
            dong_value = gu_value * dong_row['weight']
            dong_data[col] = dong_value

        dong_tourism_list.append(dong_data)

# Create DataFrame
tourism_dong = pd.DataFrame(dong_tourism_list)

print(f"\n[OK] Created dong-level tourism data: {tourism_dong.shape}")
print(f"Unique dongs: {tourism_dong['Dong_name'].nunique()}")
print(f"Unique months: {tourism_dong['month'].nunique()}")

# Step 6: Validate distribution
print(f"\nSTEP 6: VALIDATING DISTRIBUTION")
print(f"{'-'*100}")

# Check that sum of dong values equals gu value for each month-gu
sample_month = 201801
sample_gu = '강남구'

gu_total = tourism_gu[
    (tourism_gu['month'] == sample_month) &
    (tourism_gu['gu_clean'] == sample_gu)
]['korean_cc_sales'].values

if len(gu_total) > 0:
    gu_total = gu_total[0]

    dong_total = tourism_dong[
        (tourism_dong['month'] == sample_month) &
        (tourism_dong['Gu_name'] == sample_gu)
    ]['korean_cc_sales'].sum()

    print(f"Sample validation (201801, 강남구, Korean CC sales):")
    print(f"  Gu total: {gu_total:,.0f}")
    print(f"  Dong sum: {dong_total:,.0f}")
    print(f"  Difference: {abs(gu_total - dong_total):,.0f} ({abs(gu_total - dong_total) / gu_total * 100:.2f}%)")

# Step 7: Save dong-level tourism data
print(f"\nSTEP 7: SAVING DONG-LEVEL TOURISM DATA")
print(f"{'-'*100}")

output_file = TOURISM_DIR / 'tourism_dong_level_distributed.csv'
tourism_dong.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"[OK] Saved: {output_file}")
print(f"Shape: {tourism_dong.shape}")
print(f"Columns: {tourism_dong.columns.tolist()}")

# Show summary statistics
print(f"\n{'='*100}")
print(f"DISTRIBUTION SUMMARY")
print(f"{'='*100}\n")

print(f"Input:")
print(f"  - Gu-level data: {tourism_gu.shape[0]} rows × {len(tourism_cols)} features")
print(f"  - Unique gu: {tourism_gu['gu_clean'].nunique()}")
print(f"  - Unique months: {tourism_gu['month'].nunique()}")

print(f"\nOutput:")
print(f"  - Dong-level data: {tourism_dong.shape[0]} rows × {len(tourism_cols)} features")
print(f"  - Unique dongs: {tourism_dong['Dong_name'].nunique()}")
print(f"  - Unique months: {tourism_dong['month'].nunique()}")

print(f"\nTop 10 dongs by total tourism activity (Korean CC sales):")
top_dongs = tourism_dong.groupby('Dong_name')['korean_cc_sales'].sum().sort_values(ascending=False).head(10)
for dong, value in top_dongs.items():
    print(f"  {dong}: {value:,.0f}")

print(f"\n{'='*100}")
print(f"NEXT STEP: Feature engineering and prompt generation")
print(f"{'='*100}")
