"""
Collect SGIS Business Ratios (Simplified)

The business ratio API only provides CURRENT/LATEST data, not historical.
This script:
1. Collects latest business ratios (once per dong)
2. Collects housing units from existing data
3. Broadcasts ratios across all months in our date range

Features:
- retail_ratio: % of retail businesses
- accommodation_ratio: % of accommodation businesses (competitor indicator)
- restaurant_ratio: % of restaurant/bar businesses (attractiveness indicator)

Author: Airbnb Prediction Model Enhancement
Date: 2025-10-22
"""

import pandas as pd
import sys
from pathlib import Path
import time
import logging

sys.path.append(str(Path(__file__).parent))

from sgis_api_client import SGISAPIClient, BusinessRatioCollector, load_seoul_dong_codes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Credentials
CONSUMER_KEY = "fbf9612b73e54fac8545"
CONSUMER_SECRET = "0543b74f9984418da672"

DONG_CODES_FILE = "한국행정구역분류_행정동코드(7자리)_20210701기준_extracted.csv"
OUTPUT_FILE = "sgis_ratios_latest.csv"


def collect_latest_ratios(client, dong_codes):
    """Collect latest business ratios for all Seoul dongs."""
    logger.info("=" * 70)
    logger.info("COLLECTING LATEST BUSINESS RATIOS")
    logger.info("=" * 70)

    ratio_collector = BusinessRatioCollector(client)

    all_ratios = []
    total = len(dong_codes)

    logger.info(f"\nCollecting business ratios for {total} dongs...")
    logger.info(f"This will take approximately {total * 0.5 / 60:.1f} minutes\n")

    for idx, dong_row in dong_codes.iterrows():
        dong_code = dong_row['dong_code']
        dong_name = dong_row['dong_name']

        logger.info(f"[{idx+1}/{total}] {dong_name} ({dong_code})")

        try:
            ratio_df = ratio_collector.get_business_ratios(dong_code, year=None)

            if not ratio_df.empty:
                ratio_data = {
                    'dong_code': dong_code,
                    'dong_name': dong_name,
                    'retail_ratio': ratio_df['retail_ratio'].iloc[0],
                    'accommodation_ratio': ratio_df['accommodation_ratio'].iloc[0],
                    'restaurant_ratio': ratio_df['restaurant_ratio'].iloc[0]
                }

                all_ratios.append(ratio_data)
                logger.info(f"  ✓ Retail={ratio_data['retail_ratio']:.2f}%, "
                          f"Accom={ratio_data['accommodation_ratio']:.2f}%, "
                          f"Rest={ratio_data['restaurant_ratio']:.2f}%")
            else:
                logger.warning(f"  No data available")

            # Rate limiting
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"  Error - {e}")
            continue

        if (idx + 1) % 50 == 0:
            logger.info(f"\nProgress: {idx+1}/{total} ({100*(idx+1)/total:.1f}%)\n")

    logger.info(f"\n{'=' * 70}")
    logger.info(f"Collected ratios for {len(all_ratios)}/{total} dongs")
    logger.info(f"{'=' * 70}\n")

    return pd.DataFrame(all_ratios)


def broadcast_to_monthly(ratios_df, start_date='2017-01-01', end_date='2022-07-01'):
    """Broadcast ratios across all months in the date range."""
    logger.info("Broad casting ratios to monthly time series...")

    # Create monthly date range
    monthly_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    logger.info(f"Date range: {monthly_dates[0]} to {monthly_dates[-1]} ({len(monthly_dates)} months)")

    all_monthly = []

    for _, dong_row in ratios_df.iterrows():
        for date in monthly_dates:
            monthly_data = {
                'Reporting Month': date.strftime('%Y-%m-01'),
                'Dong_name': dong_row['dong_name'],
                'retail_ratio': dong_row['retail_ratio'],
                'accommodation_ratio': dong_row['accommodation_ratio'],
                'restaurant_ratio': dong_row['restaurant_ratio']
            }
            all_monthly.append(monthly_data)

    monthly_df = pd.DataFrame(all_monthly)

    logger.info(f"Created {len(monthly_df)} monthly records")
    logger.info(f"Unique dongs: {monthly_df['Dong_name'].nunique()}")

    return monthly_df


def add_housing_units(monthly_ratios, housing_file='../Data/Preprocessed_data/Dong/AirBnB_raw_embedding.csv'):
    """
    Add housing units from existing preprocessed data.

    The raw embedding file contains housing units (if available) or we can merge from SGIS data.
    For simplicity, we'll use the existing monthly embedding data approach.
    """
    logger.info("\nAdding housing units from existing data...")

    # For now, use existing SGIS data which has housing units
    try:
        sgis_old = pd.read_csv('sgis_monthly_embedding_aligned_dates.csv', encoding='utf-8-sig')

        if 'housing_units' in sgis_old.columns:
            logger.info("Found housing units in existing SGIS data")

            # Merge
            merged = monthly_ratios.merge(
                sgis_old[['Reporting Month', 'Dong_name', 'housing_units']],
                on=['Reporting Month', 'Dong_name'],
                how='left'
            )

            # Fill missing with 0
            merged['housing_units'] = merged['housing_units'].fillna(0)

            logger.info(f"Merged housing units: {(merged['housing_units'] > 0).sum()} non-zero values")

            return merged
        else:
            logger.warning("No housing units found in existing data, adding placeholder")
            monthly_ratios['housing_units'] = 0
            return monthly_ratios

    except FileNotFoundError:
        logger.warning("Existing SGIS file not found, adding placeholder housing units")
        monthly_ratios['housing_units'] = 0
        return monthly_ratios


def main():
    """Main execution."""
    try:
        # Authenticate
        logger.info("\n[1] Authenticating with SGIS API...")
        client = SGISAPIClient(CONSUMER_KEY, CONSUMER_SECRET)
        client.authenticate()
        logger.info("✓ Authentication successful\n")

        # Load dong codes
        logger.info("[2] Loading Seoul dong codes...")
        dong_codes = load_seoul_dong_codes(DONG_CODES_FILE)
        logger.info(f"✓ Loaded {len(dong_codes)} Seoul dong codes\n")

        # Collect latest ratios
        logger.info("[3] Collecting latest business ratios...")
        ratios_df = collect_latest_ratios(client, dong_codes)

        # Save latest ratios
        ratios_df.to_csv('sgis_ratios_snapshot.csv', index=False, encoding='utf-8-sig')
        logger.info(f"✓ Saved latest ratios to sgis_ratios_snapshot.csv\n")

        # Broadcast to monthly
        logger.info("[4] Broadcasting to monthly time series...")
        monthly_df = broadcast_to_monthly(ratios_df)

        # Add housing units
        logger.info("[5] Adding housing units...")
        final_df = add_housing_units(monthly_df)

        # Save final data
        final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        logger.info(f"✓ Saved monthly data to {OUTPUT_FILE}\n")

        # Summary
        logger.info("=" * 70)
        logger.info("SUMMARY")
        logger.info("=" * 70)
        logger.info(f"\nFinal data shape: {final_df.shape}")
        logger.info(f"Features: {final_df.columns.tolist()}")
        logger.info(f"\nSample statistics:")
        print(final_df[['retail_ratio', 'accommodation_ratio', 'restaurant_ratio', 'housing_units']].describe())

        logger.info("\n" + "=" * 70)
        logger.info("DATA COLLECTION COMPLETE!")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
