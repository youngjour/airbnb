"""
Add Airbnb per Housing Unit Feature

This script calculates the number of Airbnb listings per housing unit for each dong-month.

Calculation:
- Count unique listings per dong-month from raw AirBnB data
- Merge with SGIS housing units data
- Calculate: airbnb_per_housing_unit = listing_count / housing_units

This feature represents the penetration rate of Airbnb in each dong's housing market.

Author: Airbnb Prediction Model Enhancement
Date: 2025-10-22
"""

import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

def count_airbnb_listings(airbnb_file='../DATA/AirBnB_data.csv'):
    """Count number of Airbnb listings per dong-month."""
    print("=" * 70)
    print("COUNTING AIRBNB LISTINGS PER DONG-MONTH")
    print("=" * 70)

    print("\n[1] Loading raw AirBnB data...")
    df = pd.read_csv(airbnb_file, low_memory=False, encoding='utf-8-sig')
    print(f"    Total records: {len(df):,}")
    print(f"    Columns: {len(df.columns)}")

    print("\n[2] Counting listings per dong-month...")
    # Each row is a listing, so count rows per dong-month
    listing_counts = df.groupby(['Reporting Month', 'Dong_name']).size().reset_index(name='airbnb_listing_count')

    print(f"    Unique dong-month combinations: {len(listing_counts)}")
    print(f"    Listings per dong-month (sample):")
    print(listing_counts.head(10))

    # Summary statistics
    print(f"\n[3] Summary statistics:")
    print(f"    Min listings: {listing_counts['airbnb_listing_count'].min()}")
    print(f"    Max listings: {listing_counts['airbnb_listing_count'].max()}")
    print(f"    Mean listings: {listing_counts['airbnb_listing_count'].mean():.2f}")
    print(f"    Median listings: {listing_counts['airbnb_listing_count'].median():.2f}")

    return listing_counts


def add_to_sgis_features(
    listing_counts,
    sgis_file='sgis_manual/sgis_ratios_latest.csv',
    output_file='sgis_manual/sgis_improved_final.csv'
):
    """Add Airbnb per housing unit feature to SGIS data."""
    print("\n" + "=" * 70)
    print("ADDING AIRBNB PER HOUSING UNIT FEATURE")
    print("=" * 70)

    print("\n[1] Loading SGIS features...")
    sgis_df = pd.read_csv(sgis_file, encoding='utf-8-sig')
    print(f"    SGIS records: {len(sgis_df):,}")
    print(f"    Features: {sgis_df.columns.tolist()}")

    print("\n[2] Merging with listing counts...")
    # Merge on Reporting Month and Dong_name
    merged = sgis_df.merge(
        listing_counts,
        left_on=['Reporting Month', 'Dong_name'],
        right_on=['Reporting Month', 'Dong_name'],
        how='left'
    )

    # Fill missing listing counts with 0 (dongs with no Airbnb)
    merged['airbnb_listing_count'] = merged['airbnb_listing_count'].fillna(0)

    print(f"    Merged records: {len(merged):,}")
    print(f"    Missing listings (filled with 0): {(merged['airbnb_listing_count'] == 0).sum()}")

    print("\n[3] Calculating Airbnb per housing unit...")
    # Calculate ratio: listing_count / housing_units * 1000 (per 1000 housing units)
    # Use * 1000 to avoid very small numbers
    merged['airbnb_per_1k_housing'] = (
        merged['airbnb_listing_count'] / merged['housing_units'] * 1000
    )

    # Handle division by zero
    merged['airbnb_per_1k_housing'] = merged['airbnb_per_1k_housing'].fillna(0)

    print(f"    Summary of airbnb_per_1k_housing:")
    print(f"    Min: {merged['airbnb_per_1k_housing'].min():.4f}")
    print(f"    Max: {merged['airbnb_per_1k_housing'].max():.4f}")
    print(f"    Mean: {merged['airbnb_per_1k_housing'].mean():.4f}")
    print(f"    Median: {merged['airbnb_per_1k_housing'].median():.4f}")

    print("\n[4] Sample data:")
    print(merged[['Reporting Month', 'Dong_name', 'airbnb_listing_count',
                  'housing_units', 'airbnb_per_1k_housing']].head(10))

    print(f"\n[5] Saving to {output_file}...")
    # Reorder columns - put new features at the end
    feature_cols = ['retail_ratio', 'accommodation_ratio', 'restaurant_ratio',
                   'housing_units', 'airbnb_listing_count', 'airbnb_per_1k_housing']

    final_cols = ['Reporting Month', 'Dong_name'] + feature_cols
    merged = merged[final_cols]

    merged.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"    [OK] Saved {len(merged):,} records")

    print("\n" + "=" * 70)
    print("FINAL FEATURE SET")
    print("=" * 70)
    print("\nFeatures (6 total):")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i}. {col}")

    print("\n" + "=" * 70)
    print("AIRBNB PER HOUSING UNIT COMPLETE!")
    print("=" * 70)

    return merged


def main():
    """Main execution."""
    try:
        # Count Airbnb listings
        listing_counts = count_airbnb_listings()

        # Add to SGIS features
        final_df = add_to_sgis_features(listing_counts)

        print(f"\nFinal dataset summary:")
        print(final_df.describe())

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
