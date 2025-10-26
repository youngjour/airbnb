"""
Align SGIS dates with labels date range
Fills missing months (2017-01 to 2017-06) with forward fill from 2017-07
"""
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

def align_sgis_dates():
    print("=" * 70)
    print("ALIGNING SGIS DATES WITH LABELS")
    print("=" * 70)

    # Load files
    print("\n[1] Loading files...")
    sgis = pd.read_csv('sgis_monthly_embedding_complete.csv', encoding='utf-8-sig')
    labels = pd.read_csv('../../Data/Preprocessed_data/AirBnB_labels_dong.csv', encoding='utf-8-sig')

    print(f"    SGIS shape: {sgis.shape}")
    print(f"    Labels shape: {labels.shape}")

    # Get date ranges
    sgis_months = sorted(sgis['Reporting Month'].unique())
    labels_months = sorted(labels['Reporting Month'].unique())

    print(f"\n[2] Date ranges:")
    print(f"    Labels: {labels_months[0]} to {labels_months[-1]} ({len(labels_months)} months)")
    print(f"    SGIS: {sgis_months[0]} to {sgis_months[-1]} ({len(sgis_months)} months)")

    # Find missing months in SGIS
    sgis_set = set(sgis_months)
    labels_set = set(labels_months)

    missing_in_sgis = sorted(labels_set - sgis_set)
    extra_in_sgis = sorted(sgis_set - labels_set)

    if missing_in_sgis:
        print(f"\n[3] Missing months in SGIS ({len(missing_in_sgis)}):")
        for month in missing_in_sgis:
            print(f"    - {month}")

    if extra_in_sgis:
        print(f"\n[4] Extra months in SGIS (will be trimmed): ({len(extra_in_sgis)}):")
        for month in extra_in_sgis:
            print(f"    - {month}")

    # Trim SGIS to only include labels months
    print(f"\n[5] Trimming SGIS to labels date range...")
    sgis_trimmed = sgis[sgis['Reporting Month'].isin(labels_set)].copy()
    print(f"    Trimmed to {len(sgis_trimmed)} records")

    # Get all dongs and all labels months
    all_dongs = sorted(sgis_trimmed['Dong_name'].unique())
    all_months = labels_months

    print(f"\n[6] Creating complete grid for {len(all_dongs)} dongs × {len(all_months)} months...")

    # Create complete combinations
    from itertools import product
    complete_combinations = []
    for month in all_months:
        for dong in all_dongs:
            complete_combinations.append({
                'Reporting Month': month,
                'Dong_name': dong
            })

    complete_df = pd.DataFrame(complete_combinations)
    print(f"    Created {len(complete_df)} combinations")

    # Merge with SGIS data
    print(f"\n[7] Merging with SGIS data...")
    merged = complete_df.merge(
        sgis_trimmed,
        on=['Reporting Month', 'Dong_name'],
        how='left'
    )

    # Check for NaN values
    feature_cols = ['housing_units', 'total_companies', 'retail_count',
                   'accommodation_count', 'restaurant_count']

    nan_count_before = merged[feature_cols].isna().sum().sum()
    print(f"    NaN count before filling: {nan_count_before}")

    # For missing months (2017-01 to 2017-06), use the earliest available month (2017-07) values
    # This is forward fill by dong
    print(f"\n[8] Filling missing months with forward fill by dong...")

    merged = merged.sort_values(['Dong_name', 'Reporting Month'])

    # Forward fill within each dong group
    merged[feature_cols] = merged.groupby('Dong_name')[feature_cols].ffill()

    # If still NaN (for dongs that have no data), backfill
    merged[feature_cols] = merged.groupby('Dong_name')[feature_cols].bfill()

    # If still NaN, fill with 0
    merged[feature_cols] = merged[feature_cols].fillna(0)

    nan_count_after = merged[feature_cols].isna().sum().sum()
    print(f"    NaN count after filling: {nan_count_after}")

    # Validate
    print(f"\n[9] Validation:")
    print(f"    Total records: {len(merged)}")
    print(f"    Expected: {len(all_dongs)} × {len(all_months)} = {len(all_dongs) * len(all_months)}")
    print(f"    Match: {len(merged) == len(all_dongs) * len(all_months)}")
    print(f"    No NaN: {nan_count_after == 0}")

    # Summary
    print(f"\n[10] Summary statistics:")
    print(merged[feature_cols].describe())

    # Save
    output_file = 'sgis_monthly_embedding_aligned_dates.csv'
    print(f"\n[11] Saving to {output_file}...")
    merged.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"    [OK] Saved {len(merged)} records")

    print("\n" + "=" * 70)
    print("DATE ALIGNMENT COMPLETE!")
    print("=" * 70)
    print(f"\nOutput: {output_file}")
    print(f"Date range: {sorted(merged['Reporting Month'].unique())[0]} to {sorted(merged['Reporting Month'].unique())[-1]}")
    print(f"Matches labels: {set(merged['Reporting Month'].unique()) == set(labels['Reporting Month'].unique())}")
    print("=" * 70)

    return merged

if __name__ == "__main__":
    aligned = align_sgis_dates()
