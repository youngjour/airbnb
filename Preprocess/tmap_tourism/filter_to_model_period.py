"""
Filter tourism embeddings to match the model period (2017-01 to 2021-11)
and handle missing 2017 data by filling with zeros or forward-filling
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*100)
print("FILTERING TOURISM EMBEDDINGS TO MODEL PERIOD")
print("="*100 + "\n")

TOURISM_DIR = Path(r'C:\Users\jour\Documents\GitHub\airbnb\Preprocess\tmap_tourism')
LABEL_FILE = Path(r'C:\Users\jour\Documents\GitHub\airbnb\Data\Preprocessed_data\AirBnB_labels_dong.csv')

# Load tourism embeddings
print("Step 1: Loading tourism embeddings...")
tourism = pd.read_csv(TOURISM_DIR / 'tourism_llm_embeddings.csv', encoding='utf-8-sig')
print(f"[OK] Loaded: {tourism.shape}")
print(f"Columns: {tourism.columns.tolist()[:5]} ... {tourism.columns.tolist()[-3:]}")

# Convert Reporting Month to datetime
tourism['Reporting Month'] = pd.to_datetime(tourism['Reporting Month'])

print(f"\nDate range: {tourism['Reporting Month'].min()} to {tourism['Reporting Month'].max()}")
print(f"Unique months: {tourism['Reporting Month'].nunique()}")
print(f"Unique dongs: {tourism['Dong_name'].nunique()}")

# Load labels to get expected period
print("\nStep 2: Loading labels to determine model period...")
labels = pd.read_csv(LABEL_FILE, encoding='utf-8-sig')
labels['Reporting Month'] = pd.to_datetime(labels['Reporting Month'])

print(f"[OK] Labels shape: {labels.shape}")
print(f"Label period: {labels['Reporting Month'].min()} to {labels['Reporting Month'].max()}")
print(f"Expected months: {labels['Reporting Month'].nunique()}")

# Get the expected date range
expected_months = labels['Reporting Month'].unique()
expected_dongs = labels['Dong_name'].unique()

print(f"\nExpected dongs: {len(expected_dongs)}")

# Step 3: Create complete month-dong grid
print("\nStep 3: Creating complete month-dong grid...")
complete_grid = pd.MultiIndex.from_product(
    [expected_months, expected_dongs],
    names=['Reporting Month', 'Dong_name']
).to_frame(index=False)

print(f"[OK] Complete grid: {len(complete_grid)} rows ({len(expected_months)} months × {len(expected_dongs)} dongs)")

# Step 4: Merge with tourism embeddings
print("\nStep 4: Merging with tourism embeddings...")
merged = complete_grid.merge(
    tourism,
    on=['Reporting Month', 'Dong_name'],
    how='left'
)

print(f"[OK] Merged shape: {merged.shape}")

# Check missing data
embedding_cols = [col for col in tourism.columns if col.startswith('dim_')]
missing_count = merged[embedding_cols[0]].isna().sum()
print(f"Missing data: {missing_count} rows ({100*missing_count/len(merged):.1f}%)")

# Show missing months
missing_months = merged[merged[embedding_cols[0]].isna()]['Reporting Month'].unique()
print(f"\nMissing months: {sorted(missing_months)}")

# Step 5: Fill missing data with zeros
print("\nStep 5: Filling missing data with zeros...")
for col in embedding_cols:
    merged[col] = merged[col].fillna(0.0)

print(f"[OK] All embedding columns filled")

# Verify no missing data
still_missing = merged[embedding_cols].isna().sum().sum()
print(f"Remaining missing values: {still_missing}")

# Step 6: Sort by dong and month
print("\nStep 6: Sorting by dong and month...")
merged = merged.sort_values(['Dong_name', 'Reporting Month'])

# Step 7: Save filtered embeddings
print("\nStep 7: Saving filtered embeddings...")
output_file = TOURISM_DIR / 'tourism_llm_embeddings_model_period.csv'
merged.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"[OK] Saved: {output_file}")
print(f"Final shape: {merged.shape}")

# Verification
print("\n" + "="*100)
print("VERIFICATION")
print("="*100 + "\n")

print(f"Expected shape: {len(expected_months)} months × {len(expected_dongs)} dongs = {len(expected_months) * len(expected_dongs)} rows")
print(f"Actual shape: {merged.shape[0]} rows × {merged.shape[1]} columns")

print(f"\nDate range: {merged['Reporting Month'].min()} to {merged['Reporting Month'].max()}")
print(f"Unique months: {merged['Reporting Month'].nunique()}")
print(f"Unique dongs: {merged['Dong_name'].nunique()}")

# Show sample
print(f"\nSample data (first 5 rows):")
print(merged[['Reporting Month', 'Dong_name', 'dim_0', 'dim_1', 'dim_2']].head().to_string(index=False))

# Show statistics
print(f"\nEmbedding statistics:")
embedding_values = merged[embedding_cols].values
print(f"  Mean: {embedding_values.mean():.6f}")
print(f"  Std: {embedding_values.std():.6f}")
print(f"  Min: {embedding_values.min():.6f}")
print(f"  Max: {embedding_values.max():.6f}")
print(f"  Non-zero values: {(embedding_values != 0).sum()} / {embedding_values.size} ({100*(embedding_values != 0).sum()/embedding_values.size:.1f}%)")

print("\n" + "="*100)
print("FILTERING COMPLETE - Ready for model integration")
print("="*100)
