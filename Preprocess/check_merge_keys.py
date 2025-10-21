"""
Check for exact format differences in merge keys between labels and SGIS
"""
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Load files
print("Loading files...")
labels = pd.read_csv('../Data/Preprocessed_data/AirBnB_labels_dong.csv', encoding='utf-8-sig')
sgis = pd.read_csv('sgis_manual/sgis_monthly_embedding_complete.csv', encoding='utf-8-sig')

print(f"\nLabels shape: {labels.shape}")
print(f"SGIS shape: {sgis.shape}")

# Check Reporting Month formats
print("\n" + "="*70)
print("REPORTING MONTH COMPARISON")
print("="*70)

labels_months = set(labels['Reporting Month'].unique())
sgis_months = set(sgis['Reporting Month'].unique())

print(f"Labels months: {len(labels_months)}")
print(f"SGIS months: {len(sgis_months)}")

if labels_months == sgis_months:
    print("✓ Reporting Month values match")
else:
    print("✗ Reporting Month values DO NOT match")
    print(f"\nIn labels but not SGIS: {labels_months - sgis_months}")
    print(f"In SGIS but not labels: {sgis_months - labels_months}")

# Sample comparison
print("\nFirst 5 months in labels:")
for m in sorted(labels_months)[:5]:
    print(f"  '{m}' (type: {type(m)})")

print("\nFirst 5 months in SGIS:")
for m in sorted(sgis_months)[:5]:
    print(f"  '{m}' (type: {type(m)})")

# Check for whitespace issues
print("\n" + "="*70)
print("CHECKING FOR WHITESPACE IN DONG NAMES")
print("="*70)

labels_with_space = [d for d in labels['Dong_name'].unique() if str(d) != str(d).strip()]
sgis_with_space = [d for d in sgis['Dong_name'].unique() if str(d) != str(d).strip()]

print(f"Labels dongs with leading/trailing spaces: {len(labels_with_space)}")
if labels_with_space:
    for dong in labels_with_space[:5]:
        print(f"  '{dong}'")

print(f"SGIS dongs with leading/trailing spaces: {len(sgis_with_space)}")
if sgis_with_space:
    for dong in sgis_with_space[:5]:
        print(f"  '{dong}'")

# Try the actual merge to see what happens
print("\n" + "="*70)
print("TESTING ACTUAL MERGE")
print("="*70)

test_merge = pd.merge(
    labels[['Dong_name', 'Reporting Month']],
    sgis,
    on=['Dong_name', 'Reporting Month'],
    how='left'
)

nan_count = test_merge.isna().sum().sum()
print(f"Merged shape: {test_merge.shape}")
print(f"NaN count after merge: {nan_count}")

if nan_count > 0:
    print("\nNaN columns:")
    print(test_merge.isna().sum())

    # Find which combinations have NaNs
    nan_rows = test_merge[test_merge.isna().any(axis=1)]
    print(f"\nNumber of rows with NaN: {len(nan_rows)}")
    print("\nFirst 10 problematic combinations:")
    print(nan_rows[['Dong_name', 'Reporting Month']].head(10))
