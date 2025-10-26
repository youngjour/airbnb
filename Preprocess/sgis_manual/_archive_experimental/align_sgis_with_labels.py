"""
Align SGIS data with AirBnB labels
Filters SGIS data to only include dongs that exist in the labels file
"""

import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

def align_sgis_with_labels(
    sgis_file='sgis_monthly_embedding.csv',
    labels_file='../../DATA/AirBnB_labels_dong.csv',
    output_file='sgis_monthly_embedding_aligned.csv'
):
    """Filter SGIS data to match labels exactly."""

    print("=" * 70)
    print("ALIGNING SGIS DATA WITH AIRBNB LABELS")
    print("=" * 70)

    # Load files
    print(f"\n[1] Loading files...")
    sgis = pd.read_csv(sgis_file, encoding='utf-8-sig')
    labels = pd.read_csv(labels_file, encoding='utf-8-sig')

    print(f"    SGIS: {len(sgis)} records, {sgis['Dong_name'].nunique()} dongs")
    print(f"    Labels: {len(labels)} records, {labels['Dong_name'].nunique()} dongs")

    # Get valid dong names from labels
    valid_dongs = set(labels['Dong_name'].unique())
    print(f"\n[2] Valid dongs from labels: {len(valid_dongs)}")

    # Filter SGIS to only include valid dongs
    print(f"\n[3] Filtering SGIS data...")
    aligned = sgis[sgis['Dong_name'].isin(valid_dongs)].copy()

    print(f"    Filtered to {len(aligned)} records")
    print(f"    Unique dongs: {aligned['Dong_name'].nunique()}")
    print(f"    Unique months: {aligned['Reporting Month'].nunique()}")

    # Check coverage
    sgis_dongs = set(aligned['Dong_name'].unique())
    missing_in_sgis = valid_dongs - sgis_dongs
    extra_in_sgis = sgis_dongs - valid_dongs

    if missing_in_sgis:
        print(f"\n    [WARN] {len(missing_in_sgis)} dongs in labels but not in SGIS:")
        for dong in list(missing_in_sgis)[:10]:
            print(f"           {dong}")
        if len(missing_in_sgis) > 10:
            print(f"           ... and {len(missing_in_sgis) - 10} more")

    if extra_in_sgis:
        print(f"\n    [WARN] {len(extra_in_sgis)} dongs in SGIS but not in labels (filtered out):")
        for dong in list(extra_in_sgis)[:10]:
            print(f"           {dong}")

    # Validate
    print(f"\n[4] Validating alignment...")
    expected_records = len(valid_dongs) * 67  # 67 months
    actual_records = len(aligned)

    if actual_records == expected_records:
        print(f"    [OK] Perfect alignment: {actual_records} records")
    else:
        print(f"    [WARN] Record mismatch:")
        print(f"           Expected: {expected_records}")
        print(f"           Got: {actual_records}")

    # Save
    print(f"\n[5] Saving aligned data...")
    aligned.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"    [OK] Saved to {output_file}")

    print("\n" + "=" * 70)
    print("ALIGNMENT COMPLETE!")
    print("=" * 70)
    print(f"\nOutput: {output_file}")
    print(f"Records: {len(aligned)}")
    print(f"Dongs: {aligned['Dong_name'].nunique()}")
    print(f"Months: {aligned['Reporting Month'].nunique()}")
    print("\nNext: Use this file with --embed1 sgis in Model/main.py")
    print("=" * 70)

    return aligned

if __name__ == "__main__":
    aligned_df = align_sgis_with_labels()
