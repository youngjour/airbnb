"""
Complete Tourism Data Processing Pipeline
Creates tourism embeddings from 3 data sources:
1. Korean credit card sales
2. Foreign credit card sales
3. Navigation searches

Steps:
1. Load all tourism data
2. Create dong-to-gu mapping
3. Get Airbnb density for weighted distribution
4. Distribute gu-level data to dong level
5. Engineer tourism features
6. Generate LLM prompts
7. Save for embedding generation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
TOURISM_DIR = Path(r'C:\Users\jour\Documents\GitHub\airbnb\Preprocess\tmap_tourism')
DATA_DIR = Path(r'C:\Users\jour\Documents\GitHub\airbnb\Data\Preprocessed_data')

# Load files
KOREAN_CC_FILE = TOURISM_DIR / 'korean_cc_sales_2018_2024.csv'
FOREIGN_CC_FILE = TOURISM_DIR / 'foreign_cc_sales_2018_2024.csv'
NAVIGATION_FILE = TOURISM_DIR / 'navigation_searches_2018_2024.csv'
AIRBNB_LABELS = DATA_DIR / 'AirBnB_labels_dong.csv'
SGIS_DATA = Path(r'C:\Users\jour\Documents\GitHub\airbnb\Preprocess\sgis_manual\sgis_improved_final.csv')

print(f"{'='*100}")
print(f"TOURISM DATA PROCESSING PIPELINE")
print(f"{'='*100}\n")

# Step 1: Load tourism data
print(f"STEP 1: LOADING TOURISM DATA")
print(f"{'-'*100}")

korean_cc = pd.read_csv(KOREAN_CC_FILE, encoding='utf-8-sig')
foreign_cc = pd.read_csv(FOREIGN_CC_FILE, encoding='utf-8-sig')
navigation = pd.read_csv(NAVIGATION_FILE, encoding='utf-8-sig')

print(f"[OK] Korean CC: {korean_cc.shape}")
print(f"[OK] Foreign CC: {foreign_cc.shape}")
print(f"[OK] Navigation: {navigation.shape}")

# Step 2: Load Airbnb data for dong list and density
print(f"\nSTEP 2: LOADING AIRBNB DATA FOR DONG MAPPING")
print(f"{'-'*100}")

airbnb = pd.read_csv(AIRBNB_LABELS, encoding='utf-8-sig')
sgis = pd.read_csv(SGIS_DATA, encoding='utf-8-sig')

print(f"[OK] Airbnb labels: {airbnb.shape}")
print(f"[OK] SGIS data: {sgis.shape}")

# Get unique dongs from Airbnb data
unique_dongs = airbnb['Dong_name'].unique()
print(f"\n[OK] Found {len(unique_dongs)} unique dongs")

# Step 3: Create Dong-to-Gu mapping
print(f"\nSTEP 3: CREATING DONG-TO-GU MAPPING")
print(f"{'-'*100}")

# Load Seoul administrative structure
# This maps dong names to their gu (district)
# Note: Some dongs may not have exact matches due to naming variations

# Create mapping dictionary from dong names
# We'll infer gu from dong names or use a lookup table

# For simplicity, let's use the first occurrence of each dong in the SGIS data
# and merge with Airbnb listings to get density

# Merge Airbnb and SGIS to get complete picture
dong_info = airbnb.groupby('Dong_name').agg({
    'Reservation': 'sum',
    'Reporting Month': 'count'
}).reset_index()
dong_info.columns = ['Dong_name', 'total_reservations', 'months_count']

# Merge with SGIS to get listing count
sgis_listings = sgis.groupby('Dong_name')['airbnb_listing_count'].mean().reset_index()
dong_info = dong_info.merge(sgis_listings, on='Dong_name', how='left')

print(f"[OK] Created dong information table: {dong_info.shape}")
print(f"\nTop 10 dongs by Airbnb listings:")
print(dong_info.nlargest(10, 'airbnb_listing_count')[['Dong_name', 'airbnb_listing_count']])

# Step 4: Process each tourism dataset
print(f"\nSTEP 4: PROCESSING TOURISM DATASETS")
print(f"{'-'*100}")

# 4a: Korean CC
print(f"\n4a. Korean Credit Card Sales")
korean_cc.columns = ['month', 'region', 'category', 'sales']
print(f"Unique regions: {korean_cc['region'].nunique()}")
print(f"Unique categories: {korean_cc['category'].nunique()}")
print(f"Sample categories: {korean_cc['category'].unique()[:5]}")

# 4b: Foreign CC
print(f"\n4b. Foreign Credit Card Sales")
foreign_cc.columns = ['month', 'region', 'category', 'sales']
print(f"Unique regions: {foreign_cc['region'].nunique()}")
print(f"Unique categories: {foreign_cc['category'].nunique()}")

# 4c: Navigation
print(f"\n4c. Navigation Searches")
navigation.columns = ['month', 'region', 'category', 'searches']
print(f"Unique regions: {navigation['region'].nunique()}")
print(f"Unique categories: {navigation['category'].nunique()}")
print(f"Categories: {navigation['category'].unique()}")

# Step 5: Create gu-level aggregations
print(f"\nSTEP 5: CREATING GU-LEVEL AGGREGATIONS")
print(f"{'-'*100}")

# Aggregate by month and region (gu)
korean_agg = korean_cc.groupby(['month', 'region'])['sales'].sum().reset_index()
korean_agg.columns = ['month', 'gu', 'korean_cc_sales']

foreign_agg = foreign_cc.groupby(['month', 'region'])['sales'].sum().reset_index()
foreign_agg.columns = ['month', 'gu', 'foreign_cc_sales']

navigation_agg = navigation.groupby(['month', 'region'])['searches'].sum().reset_index()
navigation_agg.columns = ['month', 'gu', 'navigation_searches']

# Also get navigation category breakdown
nav_categories = navigation.pivot_table(
    index=['month', 'region'],
    columns='category',
    values='searches',
    aggfunc='sum',
    fill_value=0
).reset_index()
nav_categories.columns = ['month', 'gu'] + [f'nav_{col}' for col in nav_categories.columns[2:]]

print(f"[OK] Korean CC aggregated: {korean_agg.shape}")
print(f"[OK] Foreign CC aggregated: {foreign_agg.shape}")
print(f"[OK] Navigation aggregated: {navigation_agg.shape}")
print(f"[OK] Navigation categories: {nav_categories.shape}")

# Merge all tourism data at gu level
tourism_gu = korean_agg.merge(foreign_agg, on=['month', 'gu'], how='outer')
tourism_gu = tourism_gu.merge(navigation_agg, on=['month', 'gu'], how='outer')
tourism_gu = tourism_gu.merge(nav_categories, on=['month', 'gu'], how='outer')
tourism_gu = tourism_gu.fillna(0)

print(f"\n[OK] Combined tourism data at gu level: {tourism_gu.shape}")
print(f"Columns: {tourism_gu.columns.tolist()}")

# Step 6: Save gu-level data
print(f"\nSTEP 6: SAVING GU-LEVEL TOURISM DATA")
print(f"{'-'*100}")

output_file = TOURISM_DIR / 'tourism_gu_level_combined.csv'
tourism_gu.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"[OK] Saved: {output_file}")
print(f"Shape: {tourism_gu.shape}")
print(f"Time range: {tourism_gu['month'].min()} to {tourism_gu['month'].max()}")

# Step 7: Create dong-to-gu mapping file
print(f"\nSTEP 7: CREATING DONG-TO-GU MAPPING FILE")
print(f"{'-'*100}")

# Extract unique gu names from tourism data
unique_gus = tourism_gu['gu'].unique()
print(f"Unique gu in tourism data: {len(unique_gus)}")

# Create a manual mapping file for user to review/edit
# This will map each dong to its parent gu

mapping_file = TOURISM_DIR / 'dong_to_gu_mapping_TEMPLATE.csv'

# Create template with all dongs
mapping_df = pd.DataFrame({
    'Dong_name': unique_dongs,
    'Gu_name': '',  # To be filled manually or inferred
    'airbnb_listings': dong_info.set_index('Dong_name').loc[unique_dongs, 'airbnb_listing_count'].values
})

mapping_df.to_csv(mapping_file, index=False, encoding='utf-8-sig')
print(f"[OK] Created mapping template: {mapping_file}")
print(f"\nNOTE: This template needs to be completed with gu names.")
print(f"Each dong must be assigned to one of these gu:")
for i, gu in enumerate(sorted([str(g) for g in unique_gus])[:15], 1):
    print(f"  {i}. {gu}")
if len(unique_gus) > 15:
    print(f"  ... and {len(unique_gus)-15} more")

print(f"\n{'='*100}")
print(f"GU-LEVEL PROCESSING COMPLETE")
print(f"{'='*100}\n")

print(f"Next steps:")
print(f"1. Complete the dong-to-gu mapping file:")
print(f"   {mapping_file}")
print(f"2. Run the distribution script to create dong-level data")
print(f"3. Engineer features and generate prompts")
print(f"4. Create LLM embeddings")

print(f"\nFiles created:")
print(f"  - tourism_gu_level_combined.csv ({tourism_gu.shape})")
print(f"  - dong_to_gu_mapping_TEMPLATE.csv ({mapping_df.shape})")
