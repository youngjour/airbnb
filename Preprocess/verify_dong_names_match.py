"""
Verify that dong names match exactly between SGIS complete and raw embedding
"""
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Load files
print("Loading files...")
sgis = pd.read_csv('sgis_manual/sgis_monthly_embedding_complete.csv', encoding='utf-8-sig')
raw = pd.read_csv('../Data/Preprocessed_data/Dong/AirBnB_raw_embedding.csv', encoding='utf-8-sig')

# Get unique dongs
sgis_dongs = set(sgis['Dong_name'].unique())
raw_dongs = set(raw['Dong_name'].unique())

print(f"\nSGIS dongs: {len(sgis_dongs)}")
print(f"Raw dongs: {len(raw_dongs)}")

# Check if they match
if sgis_dongs == raw_dongs:
    print("\n✓ Dong names match perfectly!")
else:
    print("\n✗ Dong names do NOT match")

    extra_in_sgis = sgis_dongs - raw_dongs
    extra_in_raw = raw_dongs - sgis_dongs

    if extra_in_sgis:
        print(f"\nDongs in SGIS but not in raw ({len(extra_in_sgis)}):")
        for dong in sorted(extra_in_sgis):
            print(f"  - '{dong}'")

    if extra_in_raw:
        print(f"\nDongs in raw but not in SGIS ({len(extra_in_raw)}):")
        for dong in sorted(extra_in_raw):
            print(f"  - '{dong}'")

# Also check a sample alignment to see if there are ordering issues
print("\n\nChecking sample alignment:")
print("SGIS first 5 dongs:")
for dong in sorted(sgis_dongs)[:5]:
    print(f"  '{dong}'")

print("\nRaw first 5 dongs:")
for dong in sorted(raw_dongs)[:5]:
    print(f"  '{dong}'")
