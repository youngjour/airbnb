"""
Find dong mismatch between raw embedding and labels
"""
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Load files
raw = pd.read_csv('../Data/Preprocessed_data/Dong/AirBnB_raw_embedding.csv', encoding='utf-8-sig')
labels = pd.read_csv('../Data/Preprocessed_data/AirBnB_labels_dong.csv', encoding='utf-8-sig')

# Get unique dongs
raw_dongs = set(raw['Dong_name'].unique())
label_dongs = set(labels['Dong_name'].unique())

print(f"Raw embedding dongs: {len(raw_dongs)}")
print(f"Labels dongs: {len(label_dongs)}")

# Find differences
extra_in_raw = raw_dongs - label_dongs
extra_in_labels = label_dongs - raw_dongs

if extra_in_raw:
    print(f"\nDongs in raw but not in labels ({len(extra_in_raw)}):")
    for dong in sorted(extra_in_raw):
        count = len(raw[raw['Dong_name'] == dong])
        print(f"  - {dong} ({count} records)")

if extra_in_labels:
    print(f"\nDongs in labels but not in raw ({len(extra_in_labels)}):")
    for dong in sorted(extra_in_labels):
        count = len(labels[labels['Dong_name'] == dong])
        print(f"  - {dong} ({count} records)")
