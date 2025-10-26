"""
Collect Improved SGIS Features with Business Ratios

This script collects:
1. Business category ratios (retail, accommodation, restaurant) - from corpdistsummary API
2. Housing units - from household API
3. Processes yearly data to monthly with linear interpolation

Features collected:
- retail_ratio: Percentage of retail businesses (category C)
- accommodation_ratio: Percentage of accommodation businesses (category G) - competitor indicator
- restaurant_ratio: Percentage of restaurant/bar businesses (category H) - attractiveness indicator
- housing_units: Number of housing units in each dong

Author: Airbnb Prediction Model Enhancement - Improved Features
Date: 2025-10-22
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from sgis_api_client import (
    SGISAPIClient,
    HouseholdDataCollector,
    BusinessRatioCollector,
    load_seoul_dong_codes
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# API Credentials
CONSUMER_KEY = "fbf9612b73e54fac8545"
CONSUMER_SECRET = "0543b74f9984418da672"

# File paths
DONG_CODES_FILE = "한국행정구역분류_행정동코드(7자리)_20210701기준_extracted.csv"
OUTPUT_FILE = "sgis_improved_features.csv"


def collect_yearly_data(client: SGISAPIClient, dong_codes: pd.DataFrame) -> pd.DataFrame:
    """
    Collect business ratios and housing data for all Seoul dongs across multiple years.

    Args:
        client: Authenticated SGIS API client
        dong_codes: DataFrame with Seoul dong codes

    Returns:
        DataFrame with yearly data for all dongs
    """
    logger.info("=" * 70)
    logger.info("COLLECTING IMPROVED SGIS FEATURES")
    logger.info("=" * 70)

    # Initialize collectors
    ratio_collector = BusinessRatioCollector(client)
    household_collector = HouseholdDataCollector(client)

    # Years to collect (2017-2023 to match our label range 2017-01 to 2022-07)
    years = [str(year) for year in range(2017, 2024)]

    all_data = []
    total_requests = len(dong_codes) * len(years)
    completed = 0

    logger.info(f"\nCollecting data for {len(dong_codes)} dongs × {len(years)} years = {total_requests} requests")
    logger.info(f"This will take approximately {total_requests * 0.5 / 60:.1f} minutes\n")

    for idx, dong_row in dong_codes.iterrows():
        dong_code = dong_row['dong_code']
        dong_name = dong_row['dong_name']

        logger.info(f"[{idx+1}/{len(dong_codes)}] {dong_name} ({dong_code})")

        for year in years:
            try:
                # Collect business ratios (don't pass year - API rejects it!)
                ratio_df = ratio_collector.get_business_ratios(dong_code, year=None)

                # Collect housing data
                housing_df = household_collector.get_household_data(dong_code, year=year)

                # Combine data
                if not ratio_df.empty and not housing_df.empty:
                    # Extract housing units (typically in 'tot_thsnd' field)
                    housing_units = 0
                    if 'tot_thsnd' in housing_df.columns:
                        housing_units = housing_df['tot_thsnd'].iloc[0]

                    combined_data = {
                        'year': year,
                        'dong_code': dong_code,
                        'dong_name': dong_name,
                        'retail_ratio': ratio_df['retail_ratio'].iloc[0],
                        'accommodation_ratio': ratio_df['accommodation_ratio'].iloc[0],
                        'restaurant_ratio': ratio_df['restaurant_ratio'].iloc[0],
                        'housing_units': housing_units
                    }

                    all_data.append(combined_data)
                    logger.info(f"  {year}: ✓ Retail={combined_data['retail_ratio']:.2f}%, "
                              f"Accom={combined_data['accommodation_ratio']:.2f}%, "
                              f"Rest={combined_data['restaurant_ratio']:.2f}%, "
                              f"Housing={combined_data['housing_units']}")
                else:
                    logger.warning(f"  {year}: No data available")

                completed += 1

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"  {year}: Error - {e}")
                completed += 1
                continue

        if (idx + 1) % 50 == 0:
            logger.info(f"\nProgress: {completed}/{total_requests} ({100*completed/total_requests:.1f}%)\n")

    logger.info(f"\n{'=' * 70}")
    logger.info(f"Data collection complete!")
    logger.info(f"Collected {len(all_data)} records from {total_requests} requests")
    logger.info(f"{'=' * 70}\n")

    return pd.DataFrame(all_data)


def convert_yearly_to_monthly(yearly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert yearly census data to monthly time series using linear interpolation.

    Args:
        yearly_df: DataFrame with yearly data

    Returns:
        DataFrame with monthly data (2017-01 to 2022-07)
    """
    logger.info("Converting yearly data to monthly granularity...")

    # Create monthly date range matching labels
    # Labels: 2017-01-01 to 2022-07-01 (67 months)
    start_date = pd.Timestamp('2017-01-01')
    end_date = pd.Timestamp('2022-07-01')
    monthly_dates = pd.date_range(start=start_date, end=end_date, freq='MS')

    logger.info(f"Creating monthly time series: {monthly_dates[0]} to {monthly_dates[-1]} ({len(monthly_dates)} months)")

    all_monthly_data = []

    for dong_name in yearly_df['dong_name'].unique():
        dong_data = yearly_df[yearly_df['dong_name'] == dong_name].copy()
        dong_data = dong_data.sort_values('year')

        if len(dong_data) == 0:
            continue

        # Create yearly timestamps
        dong_data['timestamp'] = pd.to_datetime(dong_data['year'] + '-01-01')

        # Set timestamp as index for resampling
        dong_data = dong_data.set_index('timestamp')

        # Reindex to monthly and interpolate
        monthly_data = dong_data.reindex(
            dong_data.index.union(monthly_dates)
        ).interpolate(method='time')

        # Keep only the monthly dates we need
        monthly_data = monthly_data.loc[monthly_dates]

        # Add dong information and reporting month
        monthly_data['Dong_name'] = dong_name
        monthly_data['Reporting Month'] = monthly_data.index.strftime('%Y-%m-01')

        # Select relevant columns
        monthly_data = monthly_data[[
            'Reporting Month', 'Dong_name',
            'retail_ratio', 'accommodation_ratio', 'restaurant_ratio', 'housing_units'
        ]].reset_index(drop=True)

        all_monthly_data.append(monthly_data)

    monthly_df = pd.concat(all_monthly_data, ignore_index=True)

    logger.info(f"Created {len(monthly_df)} monthly records")
    logger.info(f"Unique dongs: {monthly_df['Dong_name'].nunique()}")
    logger.info(f"Date range: {monthly_df['Reporting Month'].min()} to {monthly_df['Reporting Month'].max()}")

    return monthly_df


def main():
    """Main execution function."""
    try:
        # Initialize API client
        logger.info("\n[1] Authenticating with SGIS API...")
        client = SGISAPIClient(CONSUMER_KEY, CONSUMER_SECRET)
        client.authenticate()
        logger.info("✓ Authentication successful\n")

        # Load Seoul dong codes
        logger.info("[2] Loading Seoul dong codes...")
        dong_codes = load_seoul_dong_codes(DONG_CODES_FILE)
        logger.info(f"✓ Loaded {len(dong_codes)} Seoul dong codes\n")

        # Collect yearly data
        logger.info("[3] Collecting yearly SGIS data...")
        yearly_df = collect_yearly_data(client, dong_codes)

        # Save yearly data
        yearly_output = "sgis_improved_features_yearly.csv"
        yearly_df.to_csv(yearly_output, index=False, encoding='utf-8-sig')
        logger.info(f"✓ Saved yearly data to {yearly_output}\n")

        # Convert to monthly
        logger.info("[4] Converting to monthly time series...")
        monthly_df = convert_yearly_to_monthly(yearly_df)

        # Save monthly data
        monthly_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        logger.info(f"✓ Saved monthly data to {OUTPUT_FILE}\n")

        # Summary statistics
        logger.info("=" * 70)
        logger.info("SUMMARY STATISTICS")
        logger.info("=" * 70)
        logger.info(f"\nYearly data shape: {yearly_df.shape}")
        logger.info(f"Monthly data shape: {monthly_df.shape}")
        logger.info(f"\nFeature summary:")
        logger.info(monthly_df[['retail_ratio', 'accommodation_ratio', 'restaurant_ratio', 'housing_units']].describe())
        logger.info("\n" + "=" * 70)
        logger.info("DATA COLLECTION COMPLETE!")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
