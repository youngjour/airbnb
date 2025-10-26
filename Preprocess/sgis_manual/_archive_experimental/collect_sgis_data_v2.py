"""
SGIS Data Collection Script (Updated)

Collects available data from SGIS API for all Seoul dong districts:
1. Housing Units (household counts)
2. Company Counts (aggregate business establishments)
3. Worker Counts (total employees)

NOTE: Industry-specific breakdowns (accommodations, retail) are NOT available via API.
      The API only provides aggregate counts. For industry-specific data, manual requests
      must be made through the SGIS web portal.

Usage:
    python collect_sgis_data_v2.py --years 2017 2018 2019 2020 2021 2022 2023

Author: Airbnb Prediction Model Enhancement
Date: October 2025
"""

import argparse
import pandas as pd
from pathlib import Path
import logging
import time
from datetime import datetime
from sgis_api_client import (
    SGISAPIClient,
    HouseholdDataCollector,
    CompanyDataCollector,
    load_seoul_dong_codes
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'sgis_collection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SGISDataCollectorV2:
    """Simplified data collector for aggregate SGIS statistics."""

    def __init__(
        self,
        consumer_key: str,
        consumer_secret: str,
        dong_codes_csv: str,
        rate_limit_delay: float = 0.5
    ):
        """Initialize SGIS data collector."""
        self.client = SGISAPIClient(consumer_key, consumer_secret)
        self.household_collector = HouseholdDataCollector(self.client)
        self.company_collector = CompanyDataCollector(self.client)
        self.rate_limit_delay = rate_limit_delay

        # Load Seoul dong codes
        self.dong_codes_df = load_seoul_dong_codes(dong_codes_csv)
        logger.info(f"Loaded {len(self.dong_codes_df)} Seoul dong codes")

    def collect_all_data(self, years: list, output_dir: str) -> dict:
        """
        Collect all available data: housing units and company statistics.

        Args:
            years: List of years to collect (e.g., ['2017', '2018', ...])
            output_dir: Directory to save collected data

        Returns:
            Dictionary with paths to saved files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Collect housing data
        housing_df = self.collect_housing_units(years)
        housing_path = output_path / 'housing_data_raw.csv'
        housing_df.to_csv(housing_path, index=False, encoding='utf-8-sig')
        logger.info(f"\nSaved housing data: {housing_path}")

        # Collect company data
        company_df = self.collect_company_data(years)
        company_path = output_path / 'company_data_raw.csv'
        company_df.to_csv(company_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved company data: {company_path}")

        # Create summary
        self.create_summary(housing_df, company_df, output_path)

        return {
            'housing': housing_path,
            'company': company_path
        }

    def collect_housing_units(self, years: list) -> pd.DataFrame:
        """Collect housing unit data for all Seoul dongs."""
        logger.info("=" * 70)
        logger.info("COLLECTING HOUSING UNITS DATA")
        logger.info("=" * 70)

        all_data = []
        total_requests = len(self.dong_codes_df) * len(years)
        current_request = 0
        success_count = 0
        fail_count = 0

        for year in years:
            logger.info(f"\n>>> Year: {year}")

            for idx, row in self.dong_codes_df.iterrows():
                current_request += 1
                dong_code = str(row['dong_code'])
                dong_name = row['dong_name']
                gu_name = row['gu_name']

                progress_pct = (current_request / total_requests) * 100
                logger.info(
                    f"[{current_request}/{total_requests}] ({progress_pct:.1f}%) "
                    f"{dong_name} ({gu_name}) - {year}"
                )

                try:
                    df = self.household_collector.get_household_data(
                        adm_cd=dong_code,
                        year=year
                    )

                    if not df.empty:
                        # Add metadata
                        df['dong_code'] = row['dong_code_7digit']  # Use 7-digit for consistency
                        df['dong_name'] = dong_name
                        df['gu_name'] = gu_name
                        df['year'] = year
                        all_data.append(df)
                        success_count += 1
                        logger.info(f"  ✓ Retrieved {len(df)} records")
                    else:
                        fail_count += 1
                        logger.warning(f"  ✗ No data returned")

                except Exception as e:
                    fail_count += 1
                    logger.error(f"  ✗ Error: {e}")

                # Rate limiting
                time.sleep(self.rate_limit_delay)

        if all_data:
            result_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"\n{'='*70}")
            logger.info(f"Housing Units Collection Complete:")
            logger.info(f"  Total records: {len(result_df)}")
            logger.info(f"  Success: {success_count}/{total_requests}")
            logger.info(f"  Failed: {fail_count}/{total_requests}")
            logger.info(f"{'='*70}")
            return result_df
        else:
            logger.error("No housing data collected!")
            return pd.DataFrame()

    def collect_company_data(self, years: list) -> pd.DataFrame:
        """Collect aggregate company data for all Seoul dongs."""
        logger.info("\n" + "=" * 70)
        logger.info("COLLECTING COMPANY DATA (AGGREGATE COUNTS)")
        logger.info("=" * 70)
        logger.info("NOTE: API returns total company & worker counts only.")
        logger.info("Industry-specific breakdowns are NOT available via API.")
        logger.info("=" * 70)

        all_data = []
        total_requests = len(self.dong_codes_df) * len(years)
        current_request = 0
        success_count = 0
        fail_count = 0

        for year in years:
            logger.info(f"\n>>> Year: {year}")

            for idx, row in self.dong_codes_df.iterrows():
                current_request += 1
                dong_code = str(row['dong_code'])
                dong_name = row['dong_name']
                gu_name = row['gu_name']

                progress_pct = (current_request / total_requests) * 100
                logger.info(
                    f"[{current_request}/{total_requests}] ({progress_pct:.1f}%) "
                    f"{dong_name} ({gu_name}) - {year}"
                )

                try:
                    df = self.company_collector.get_company_data(
                        adm_cd=dong_code,
                        year=year
                    )

                    if not df.empty:
                        # Add metadata
                        df['dong_code'] = row['dong_code_7digit']
                        df['dong_name'] = dong_name
                        df['gu_name'] = gu_name
                        df['year'] = year
                        all_data.append(df)
                        success_count += 1
                        logger.info(f"  ✓ Retrieved {len(df)} records")
                    else:
                        fail_count += 1
                        logger.warning(f"  ✗ No data returned")

                except Exception as e:
                    fail_count += 1
                    logger.error(f"  ✗ Error: {e}")

                # Rate limiting
                time.sleep(self.rate_limit_delay)

        if all_data:
            result_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"\n{'='*70}")
            logger.info(f"Company Data Collection Complete:")
            logger.info(f"  Total records: {len(result_df)}")
            logger.info(f"  Success: {success_count}/{total_requests}")
            logger.info(f"  Failed: {fail_count}/{total_requests}")
            logger.info(f"{'='*70}")
            return result_df
        else:
            logger.error("No company data collected!")
            return pd.DataFrame()

    def create_summary(self, housing_df: pd.DataFrame, company_df: pd.DataFrame, output_path: Path):
        """Create summary statistics of collected data."""
        summary_path = output_path / 'collection_summary.txt'

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("SGIS DATA COLLECTION SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Collection Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Housing data summary
            f.write("HOUSING UNITS DATA:\n")
            f.write("-" * 70 + "\n")
            if not housing_df.empty:
                f.write(f"Total Records: {len(housing_df)}\n")
                f.write(f"Unique Dongs: {housing_df['dong_code'].nunique()}\n")
                f.write(f"Years Covered: {sorted(housing_df['year'].unique())}\n")
                f.write(f"Columns: {list(housing_df.columns)}\n")

                # Aggregate by dong-year
                agg_housing = housing_df.groupby(['dong_code', 'dong_name', 'year']).agg({
                    'household_cnt': 'sum',
                    'family_member_cnt': 'sum'
                }).reset_index()

                f.write(f"\nSample aggregated data (first 10 dong-years):\n")
                f.write(agg_housing.head(10).to_string())
            else:
                f.write("No housing data collected.\n")

            f.write("\n\n")

            # Company data summary
            f.write("COMPANY DATA (AGGREGATE):\n")
            f.write("-" * 70 + "\n")
            if not company_df.empty:
                f.write(f"Total Records: {len(company_df)}\n")
                f.write(f"Unique Dongs: {company_df['dong_code'].nunique()}\n")
                f.write(f"Years Covered: {sorted(company_df['year'].unique())}\n")
                f.write(f"Columns: {list(company_df.columns)}\n")

                # Aggregate by dong-year
                agg_company = company_df.groupby(['dong_code', 'dong_name', 'year']).agg({
                    'corp_cnt': 'sum',
                    'tot_worker': 'sum'
                }).reset_index()

                f.write(f"\nSample aggregated data (first 10 dong-years):\n")
                f.write(agg_company.head(10).to_string())
            else:
                f.write("No company data collected.\n")

            f.write("\n\n")
            f.write("=" * 70 + "\n")
            f.write("IMPORTANT NOTES:\n")
            f.write("=" * 70 + "\n")
            f.write("1. Housing data is at enumeration district level (more granular than dong)\n")
            f.write("2. Company data is aggregate counts only (no industry breakdowns)\n")
            f.write("3. For industry-specific data, manual requests must be made via SGIS portal\n")
            f.write("4. Data needs to be aggregated to dong level and interpolated to monthly\n")

        logger.info(f"\nSaved summary: {summary_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Collect SGIS census data for Seoul dongs'
    )
    parser.add_argument(
        '--years',
        nargs='+',
        required=True,
        help='Years to collect data for (e.g., 2017 2018 2019 2020 2021 2022 2023)'
    )
    parser.add_argument(
        '--dong-codes-csv',
        default='한국행정구역분류_행정동코드(7자리)_20210701기준_extracted.csv',
        help='Path to dong codes CSV file'
    )
    parser.add_argument(
        '--output-dir',
        default='./collected_data',
        help='Output directory for collected data'
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=0.5,
        help='Delay between API calls in seconds (default: 0.5)'
    )

    args = parser.parse_args()

    # API credentials
    CONSUMER_KEY = "fbf9612b73e54fac8545"
    CONSUMER_SECRET = "0543b74f9984418da672"

    logger.info("=" * 70)
    logger.info("SGIS DATA COLLECTION - STARTED")
    logger.info("=" * 70)
    logger.info(f"Years: {args.years}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Rate limit: {args.rate_limit}s between requests")
    logger.info("=" * 70)

    # Initialize collector
    collector = SGISDataCollectorV2(
        consumer_key=CONSUMER_KEY,
        consumer_secret=CONSUMER_SECRET,
        dong_codes_csv=args.dong_codes_csv,
        rate_limit_delay=args.rate_limit
    )

    # Collect all data
    start_time = datetime.now()
    output_files = collector.collect_all_data(args.years, args.output_dir)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("\n" + "=" * 70)
    logger.info("COLLECTION COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    logger.info(f"\nOutput files:")
    for key, path in output_files.items():
        logger.info(f"  {key}: {path}")
    logger.info("\nNext steps:")
    logger.info("  1. Run preprocessing script to aggregate to dong level")
    logger.info("  2. Interpolate yearly data to monthly time series")
    logger.info("  3. Integrate with existing Airbnb data")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
