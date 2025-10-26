"""
Deduplicate SGIS Improved Features

Removes duplicate (Dong_name, Reporting Month) combinations by keeping the first occurrence.
"""

import pandas as pd

# Load data
df = pd.read_csv('sgis_improved_final.csv', encoding='utf-8-sig')

print(f"Original data shape: {df.shape}")
print(f"Unique (Dong, Month) combinations: {df.groupby(['Dong_name', 'Reporting Month']).size().shape[0]}")

# Check for duplicates
before_count = len(df)
dups = df.duplicated(subset=['Dong_name', 'Reporting Month'], keep='first')
dup_count = dups.sum()

print(f"\nDuplicate rows: {dup_count}")

# Remove duplicates - keep first occurrence
df_clean = df.drop_duplicates(subset=['Dong_name', 'Reporting Month'], keep='first')

print(f"\nCleaned data shape: {df_clean.shape}")
print(f"Rows removed: {before_count - len(df_clean)}")

# Verify no duplicates remain
dups_after = df_clean.duplicated(subset=['Dong_name', 'Reporting Month']).sum()
print(f"Duplicates remaining: {dups_after}")

# Save cleaned data
df_clean.to_csv('sgis_improved_final.csv', index=False, encoding='utf-8-sig')
print(f"\n✓ Saved deduplicated data to sgis_improved_final.csv")

# Show summary
print(f"\nFinal summary:")
print(f"  Unique dongs: {df_clean['Dong_name'].nunique()}")
print(f"  Unique months: {df_clean['Reporting Month'].nunique()}")
print(f"  Total records: {len(df_clean)}")
