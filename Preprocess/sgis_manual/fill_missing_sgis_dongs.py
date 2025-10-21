"""
Fill missing dongs in SGIS data to match the complete 424-dong list from labels
"""
import pandas as pd
import sys
from itertools import product

sys.stdout.reconfigure(encoding='utf-8')

def fill_missing_sgis_dongs(
    sgis_file='sgis_monthly_embedding_aligned.csv',
    reference_file='../../Data/Preprocessed_data/Dong/AirBnB_raw_embedding.csv',
    output_file='sgis_monthly_embedding_complete.csv'
):
    """Fill missing dongs in SGIS data with zeros."""

    print("=" * 70)
    print("FILLING MISSING DONGS IN SGIS DATA")
    print("=" * 70)

    # Load SGIS data
    print(f"\n[1] Loading SGIS data from {sgis_file}...")
    sgis = pd.read_csv(sgis_file, encoding='utf-8-sig')

    print(f"    SGIS records: {len(sgis)}")
    print(f"    Unique dongs: {sgis['Dong_name'].nunique()}")
    print(f"    Unique months: {sgis['Reporting Month'].nunique()}")

    # Load reference to get complete dong list
    print(f"\n[2] Loading reference file to get complete dong list...")
    print(f"    Reference: {reference_file}")
    ref_df = pd.read_csv(reference_file, encoding='utf-8-sig')

    # Get complete dong list from reference
    all_dongs = sorted(ref_df['Dong_name'].unique())
    all_months = sorted(sgis['Reporting Month'].unique())

    print(f"    Reference has {len(all_dongs)} dongs")
    print(f"    SGIS has {sgis['Dong_name'].nunique()} dongs")
    print(f"    Will create SGIS data for all {len(all_dongs)} dongs")

    # Find missing dongs
    sgis_dongs = set(sgis['Dong_name'].unique())
    missing_dongs = set(all_dongs) - sgis_dongs

    if missing_dongs:
        print(f"\n    Missing {len(missing_dongs)} dongs in SGIS:")
        for dong in sorted(missing_dongs):
            print(f"      - {dong}")

    print(f"\n[3] Creating complete dong-month grid...")
    print(f"    {len(all_dongs)} dongs × {len(all_months)} months = {len(all_dongs) * len(all_months)} combinations")

    # Create complete index
    complete_index = pd.DataFrame(
        list(product(all_months, all_dongs)),
        columns=['Reporting Month', 'Dong_name']
    )

    print(f"    Created {len(complete_index)} combinations")

    # Merge with SGIS data
    print(f"\n[4] Merging with SGIS data...")
    complete_sgis = complete_index.merge(
        sgis,
        on=['Reporting Month', 'Dong_name'],
        how='left'
    )

    # Fill missing values with 0 (reasonable for census data)
    feature_cols = ['housing_units', 'total_companies', 'retail_count',
                   'accommodation_count', 'restaurant_count']
    complete_sgis[feature_cols] = complete_sgis[feature_cols].fillna(0)

    print(f"    Added {len(complete_sgis) - len(sgis)} new dong-month combinations")
    print(f"    Filled all feature columns with 0 for missing dongs")

    # Validate
    print(f"\n[5] Validating complete data...")
    print(f"    Total records: {len(complete_sgis)}")
    print(f"    Expected: {len(all_dongs)} × {len(all_months)} = {len(all_dongs) * len(all_months)}")
    print(f"    Match: {len(complete_sgis) == len(all_dongs) * len(all_months)}")

    # Check for any remaining NaNs
    nan_count = complete_sgis.isna().sum().sum()
    if nan_count > 0:
        print(f"    [WARN] Still have {nan_count} NaN values!")
        print(complete_sgis.isna().sum())
    else:
        print(f"    [OK] No NaN values")

    # Summary
    print(f"\n[6] Summary statistics:")
    print(complete_sgis[feature_cols].describe())

    # Fix Reporting Month format to match labels (YYYY-MM-DD instead of YYYY-MM)
    print(f"\n[7] Fixing Reporting Month format to match labels...")
    # Convert YYYY-MM to YYYY-MM-01
    complete_sgis['Reporting Month'] = complete_sgis['Reporting Month'].apply(
        lambda x: f"{x}-01" if len(str(x)) == 7 else x
    )
    print(f"    Converted to YYYY-MM-DD format")
    print(f"    Sample: {complete_sgis['Reporting Month'].iloc[0]}")

    # Save
    print(f"\n[8] Saving to {output_file}...")
    complete_sgis.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"    [OK] Saved {len(complete_sgis)} records")

    print("\n" + "=" * 70)
    print("SGIS DATA COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nOutput: {output_file}")
    print(f"Records: {len(complete_sgis)}")
    print(f"Dongs: {complete_sgis['Dong_name'].nunique()}")
    print(f"Months: {complete_sgis['Reporting Month'].nunique()}")
    print(f"Format: Complete dense matrix (all dongs × all months)")
    print("\nNext: Update main.py embedding_paths_dict to use this file")
    print("=" * 70)

    return complete_sgis

if __name__ == "__main__":
    complete_sgis = fill_missing_sgis_dongs()
