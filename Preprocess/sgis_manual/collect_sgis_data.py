"""
SGIS Data Collection Script

Collects housing units, accommodations, and retail store data from SGIS API
for all Seoul dong districts to enhance Airbnb prediction model.

Usage:
    python collect_sgis_data.py --years 2017 2018 2019 2020 2021 2022 2023

Author: Airbnb Prediction Model Enhancement
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
    IndustryCodeHelper,
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


class SGISDataCollector:
    """Main data collector for SGIS statistics."""

    def __init__(
        self,
        consumer_key: str,
        consumer_secret: str,
        dong_codes_csv: str,
        rate_limit_delay: float = 0.5
    ):
        """
        Initialize SGIS data collector.

        Args:
            consumer_key: SGIS API consumer key
            consumer_secret: SGIS API consumer secret
            dong_codes_csv: Path to CSV file with Seoul dong codes
            rate_limit_delay: Delay between API calls in seconds
        """
        self.client = SGISAPIClient(consumer_key, consumer_secret)
        self.household_collector = HouseholdDataCollector(self.client)
        self.company_collector = CompanyDataCollector(self.client)
        self.industry_helper = IndustryCodeHelper(self.client)
        self.rate_limit_delay = rate_limit_delay

        # Load Seoul dong codes
        self.dong_codes_df = load_seoul_dong_codes(dong_codes_csv)
        logger.info(f"Loaded {len(self.dong_codes_df)} Seoul dong codes")

    def collect_housing_units(self, years: list) -> pd.DataFrame:
        """
        Collect housing unit data for all Seoul dongs across multiple years.

        Args:
            years: List of years to collect data for (e.g., ['2017', '2018', ...])

        Returns:
            Consolidated DataFrame with housing unit data
        """
        logger.info("=" * 60)
        logger.info("Starting housing units data collection")
        logger.info("=" * 60)

        all_data = []
        total_requests = len(self.dong_codes_df) * len(years)
        current_request = 0

        for year in years:
            logger.info(f"\nCollecting housing data for year: {year}")

            for idx, row in self.dong_codes_df.iterrows():
                current_request += 1
                dong_code = str(row['dong_code'])
                dong_name = row['dong_name']

                logger.info(
                    f"[{current_request}/{total_requests}] "
                    f"Fetching: {dong_name} ({dong_code}), Year: {year}"
                )

                df = self.household_collector.get_household_data(
                    adm_cd=dong_code,
                    year=year
                )

                if not df.empty:
                    df['dong_name'] = dong_name
                    df['gu_name'] = row['gu_name']
                    all_data.append(df)

                time.sleep(self.rate_limit_delay)

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            logger.info(f"\n✓ Collected {len(result)} total housing records")
            return result
        else:
            logger.warning("No housing data collected")
            return pd.DataFrame()

    def collect_company_data(self, years: list) -> pd.DataFrame:
        """
        Collect company/business data for all Seoul dongs across multiple years.

        Args:
            years: List of years to collect data for

        Returns:
            Consolidated DataFrame with company data (accommodations + retail)
        """
        logger.info("=" * 60)
        logger.info("Starting company data collection")
        logger.info("=" * 60)

        all_data = []
        total_requests = len(self.dong_codes_df) * len(years)
        current_request = 0

        for year in years:
            logger.info(f"\nCollecting company data for year: {year}")

            for idx, row in self.dong_codes_df.iterrows():
                current_request += 1
                dong_code = str(row['dong_code'])
                dong_name = row['dong_name']

                logger.info(
                    f"[{current_request}/{total_requests}] "
                    f"Fetching: {dong_name} ({dong_code}), Year: {year}"
                )

                df = self.company_collector.get_company_data(
                    adm_cd=dong_code,
                    year=year
                )

                if not df.empty:
                    df['dong_name'] = dong_name
                    df['gu_name'] = row['gu_name']
                    all_data.append(df)

                time.sleep(self.rate_limit_delay)

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            logger.info(f"\n✓ Collected {len(result)} total company records")
            return result
        else:
            logger.warning("No company data collected")
            return pd.DataFrame()

    def filter_accommodations(self, company_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter accommodation businesses from company data.

        Common accommodation industry codes (you may need to adjust based on actual data):
        - I: Accommodation and food service activities
        - 55: Accommodation
        - I551: Short-term accommodation
        - I552: Other accommodation

        Args:
            company_df: DataFrame with company data

        Returns:
            Filtered DataFrame with accommodation businesses
        """
        if company_df.empty:
            return pd.DataFrame()

        # Keywords for accommodation filtering
        accommodation_keywords = [
            '숙박',  # Accommodation
            '호텔',  # Hotel
            '모텔',  # Motel
            '게스트하우스',  # Guesthouse
            '펜션',  # Pension
            '리조트',  # Resort
        ]

        # Try to filter by industry code if available
        if 'induty_cl_code' in company_df.columns:
            # Filter by accommodation-related codes
            acc_mask = (
                company_df['induty_cl_code'].astype(str).str.startswith('I55') |
                company_df['induty_cl_code'].astype(str).str.startswith('55')
            )
            accommodations = company_df[acc_mask].copy()
        else:
            accommodations = company_df.copy()

        # Additional filtering by name if available
        if 'induty_cl_nm' in company_df.columns:
            name_mask = company_df['induty_cl_nm'].astype(str).apply(
                lambda x: any(kw in x for kw in accommodation_keywords)
            )
            accommodations = company_df[name_mask].copy()

        logger.info(f"Filtered {len(accommodations)} accommodation records")
        return accommodations

    def filter_retail_stores(self, company_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter retail store businesses from company data.

        Common retail industry codes:
        - G: Wholesale and retail trade
        - 47: Retail trade (except motor vehicles and motorcycles)
        - G471: Retail sale in non-specialized stores
        - G472: Retail sale of food, beverages and tobacco
        - G474: Retail sale of information and communications equipment
        - G475: Retail sale of other household equipment
        - G476: Retail sale of cultural and recreation goods
        - G477: Retail sale of other goods

        Args:
            company_df: DataFrame with company data

        Returns:
            Filtered DataFrame with retail store businesses
        """
        if company_df.empty:
            return pd.DataFrame()

        # Keywords for retail filtering
        retail_keywords = [
            '소매',  # Retail
            '상점',  # Store
            '마트',  # Mart
            '슈퍼마켓',  # Supermarket
            '편의점',  # Convenience store
            '백화점',  # Department store
        ]

        # Try to filter by industry code if available
        if 'induty_cl_code' in company_df.columns:
            # Filter by retail-related codes
            retail_mask = (
                company_df['induty_cl_code'].astype(str).str.startswith('G47') |
                company_df['induty_cl_code'].astype(str).str.startswith('47')
            )
            retail_stores = company_df[retail_mask].copy()
        else:
            retail_stores = company_df.copy()

        # Additional filtering by name if available
        if 'induty_cl_nm' in company_df.columns:
            name_mask = company_df['induty_cl_nm'].astype(str).apply(
                lambda x: any(kw in x for kw in retail_keywords)
            )
            retail_stores = company_df[name_mask].copy()

        logger.info(f"Filtered {len(retail_stores)} retail store records")
        return retail_stores

    def save_data(self, df: pd.DataFrame, filename: str, output_dir: str):
        """
        Save collected data to CSV file.

        Args:
            df: DataFrame to save
            filename: Output filename
            output_dir: Output directory path
        """
        if not df.empty:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            filepath = output_path / filename
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            logger.info(f"✓ Saved data to {filepath}")
            logger.info(f"  - Shape: {df.shape}")
            logger.info(f"  - Columns: {list(df.columns)}")
        else:
            logger.warning(f"No data to save for {filename}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Collect SGIS data for Airbnb prediction model enhancement'
    )
    parser.add_argument(
        '--years',
        nargs='+',
        default=['2017', '2018', '2019', '2020', '2021', '2022', '2023'],
        help='Years to collect data for (default: 2017-2023)'
    )
    parser.add_argument(
        '--dong-codes-csv',
        type=str,
        default='한국행정구역분류_행정동코드(7자리)_20210701기준_extracted.csv',
        help='Path to dong codes CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
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

    # SGIS API credentials
    CONSUMER_KEY = "fbf9612b73e54fac8545"
    CONSUMER_SECRET = "0543b74f9984418da672"

    logger.info("=" * 60)
    logger.info("SGIS Data Collection for Airbnb Prediction Model")
    logger.info("=" * 60)
    logger.info(f"Years: {args.years}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Rate limit delay: {args.rate_limit}s")
    logger.info("=" * 60)

    try:
        # Initialize collector
        collector = SGISDataCollector(
            consumer_key=CONSUMER_KEY,
            consumer_secret=CONSUMER_SECRET,
            dong_codes_csv=args.dong_codes_csv,
            rate_limit_delay=args.rate_limit
        )

        # Collect housing units data
        logger.info("\n[STEP 1/3] Collecting housing units data...")
        housing_df = collector.collect_housing_units(args.years)
        collector.save_data(housing_df, 'housing_units_raw.csv', args.output_dir)

        # Collect company data
        logger.info("\n[STEP 2/3] Collecting company data...")
        company_df = collector.collect_company_data(args.years)
        collector.save_data(company_df, 'company_data_raw.csv', args.output_dir)

        # Filter and save specific business types
        logger.info("\n[STEP 3/3] Filtering and saving specific business types...")

        # Filter accommodations
        accommodations_df = collector.filter_accommodations(company_df)
        collector.save_data(accommodations_df, 'accommodations_raw.csv', args.output_dir)

        # Filter retail stores
        retail_df = collector.filter_retail_stores(company_df)
        collector.save_data(retail_df, 'retail_stores_raw.csv', args.output_dir)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Data Collection Complete!")
        logger.info("=" * 60)
        logger.info(f"Housing units records: {len(housing_df)}")
        logger.info(f"Total company records: {len(company_df)}")
        logger.info(f"Accommodation records: {len(accommodations_df)}")
        logger.info(f"Retail store records: {len(retail_df)}")
        logger.info(f"\nAll data saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"Error during data collection: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
