"""
Fill missing dong-month combinations in labels with zeros
The model expects a complete dense matrix: all dongs × all months
"""

import pandas as pd
import numpy as np
from itertools import product

def fill_missing_labels(
    input_file='../DATA/AirBnB_labels_dong.csv',
    output_file='../DATA/AirBnB_labels_dong_complete.csv',
    reference_file='../Data/Preprocessed_data/Dong/AirBnB_raw_embedding.csv'
):
    """Fill missing dong-month combinations with zeros.

    Uses reference_file (raw embedding) to get the complete list of dongs.
    """

    print("=" * 70)
    print("FILLING MISSING LABEL DATA")
    print("=" * 70)

    # Load original labels
    print(f"\n[1] Loading labels from {input_file}...")
    df = pd.read_csv(input_file, encoding='utf-8-sig')

    print(f"    Original records: {len(df)}")
    print(f"    Unique dongs: {df['Dong_name'].nunique()}")
    print(f"    Unique months: {df['Reporting Month'].nunique()}")

    # Load reference file to get complete dong list
    print(f"\n[2] Loading reference file to get complete dong list...")
    print(f"    Reference: {reference_file}")
    ref_df = pd.read_csv(reference_file, encoding='utf-8-sig')

    # Get all dongs from reference (this is the complete list)
    all_dongs = sorted(ref_df['Dong_name'].unique())
    all_months = sorted(df['Reporting Month'].unique())

    print(f"    Reference has {len(all_dongs)} dongs")
    print(f"    Labels have {len(df['Dong_name'].unique())} dongs")
    print(f"    Will create labels for all {len(all_dongs)} dongs")

    print(f"\n[3] Creating complete dong-month grid...")
    print(f"    {len(all_dongs)} dongs × {len(all_months)} months = {len(all_dongs) * len(all_months)} combinations")

    # Create complete index
    complete_index = pd.DataFrame(
        list(product(all_months, all_dongs)),
        columns=['Reporting Month', 'Dong_name']
    )

    print(f"    Created {len(complete_index)} combinations")

    # Merge with original data
    print(f"\n[4] Merging with original data...")
    complete_df = complete_index.merge(
        df,
        on=['Reporting Month', 'Dong_name'],
        how='left'
    )

    # Fill missing values with 0
    label_cols = ['Reservation Days', 'Revenue', 'Reservation']
    complete_df[label_cols] = complete_df[label_cols].fillna(0)

    # Count how many values were filled
    filled_records = complete_df[label_cols].isna().any(axis=1).sum()
    print(f"    Added {len(complete_df) - len(df)} new dong-month combinations")
    print(f"    Filled all label columns with 0 for missing combinations")

    # Validate
    print(f"\n[5] Validating complete data...")
    print(f"    Total records: {len(complete_df)}")
    print(f"    Expected: {len(all_dongs)} × {len(all_months)} = {len(all_dongs) * len(all_months)}")
    print(f"    Match: {len(complete_df) == len(all_dongs) * len(all_months)}")

    # Check for any remaining NaNs
    nan_count = complete_df.isna().sum().sum()
    if nan_count > 0:
        print(f"    [WARN] Still have {nan_count} NaN values!")
    else:
        print(f"    [OK] No NaN values")

    # Summary
    print(f"\n[6] Summary statistics:")
    print(complete_df[label_cols].describe())

    # Save
    print(f"\n[7] Saving to {output_file}...")
    complete_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"    [OK] Saved {len(complete_df)} records")

    print("\n" + "=" * 70)
    print("LABELS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nOutput: {output_file}")
    print(f"Records: {len(complete_df)}")
    print(f"Format: Complete dense matrix (all dongs × all months)")
    print("\nNext: Update Model/main.py to use this file as --label_path")
    print("=" * 70)

    return complete_df

if __name__ == "__main__":
    complete_labels = fill_missing_labels()
