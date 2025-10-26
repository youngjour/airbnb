"""Check for NaN values in SGIS improved features"""

import pandas as pd
import numpy as np

df = pd.read_csv('sgis_improved_final.csv', encoding='utf-8-sig')

print(f"Data shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# Check for NaN values
print(f"\nNaN counts per column:")
nan_counts = df.isna().sum()
print(nan_counts)

# Check for inf values
print(f"\nInf counts per column:")
for col in df.select_dtypes(include=[np.number]).columns:
    inf_count = np.isinf(df[col]).sum()
    if inf_count > 0:
        print(f"  {col}: {inf_count}")

# Check specific feature columns
feature_cols = ['retail_ratio', 'accommodation_ratio', 'restaurant_ratio',
                'housing_units', 'airbnb_listing_count', 'airbnb_per_1k_housing']

print(f"\nFeature statistics:")
for col in feature_cols:
    if col in df.columns:
        nan_count = df[col].isna().sum()
        inf_count = np.isinf(df[col]).sum()
        print(f"{col}:")
        print(f"  NaN: {nan_count}, Inf: {inf_count}")
        print(f"  Min: {df[col].min():.2f}, Max: {df[col].max():.2f}")
