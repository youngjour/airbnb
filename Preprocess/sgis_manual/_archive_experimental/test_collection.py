"""
Test Script for SGIS Data Collection

This script tests the data collection pipeline with a small sample of dongs
before running the full collection.

Usage:
    python test_collection.py
"""

import pandas as pd
from pathlib import Path
import logging
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
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_authentication():
    """Test 1: Verify API authentication works."""
    logger.info("=" * 60)
    logger.info("TEST 1: Authentication")
    logger.info("=" * 60)

    try:
        client = SGISAPIClient(
            consumer_key="fbf9612b73e54fac8545",
            consumer_secret="0543b74f9984418da672"
        )

        token = client.authenticate()
        logger.info(f"✓ Authentication successful!")
        logger.info(f"  Access Token (first 20 chars): {token[:20]}...")
        logger.info(f"  Token expires at: {client.token_expiry}")
        return client

    except Exception as e:
        logger.error(f"✗ Authentication failed: {e}")
        return None


def test_industry_codes(client):
    """Test 2: Fetch and display industry classification codes."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Industry Classification Codes")
    logger.info("=" * 60)

    try:
        helper = IndustryCodeHelper(client)
        codes_df = helper.fetch_industry_codes()

        if not codes_df.empty:
            logger.info(f"✓ Fetched {len(codes_df)} industry codes")
            logger.info(f"\nColumns: {list(codes_df.columns)}")
            logger.info(f"\nFirst 10 codes:")
            print(codes_df.head(10).to_string())

            # Search for accommodation codes
            logger.info("\n" + "-" * 60)
            logger.info("Searching for accommodation-related codes (숙박):")
            acc_codes = helper.find_codes_by_keyword('숙박')
            if not acc_codes.empty:
                print(acc_codes.to_string())
            else:
                logger.info("No codes found with keyword '숙박'")

            # Search for retail codes
            logger.info("\n" + "-" * 60)
            logger.info("Searching for retail-related codes (소매):")
            retail_codes = helper.find_codes_by_keyword('소매')
            if not retail_codes.empty:
                print(retail_codes.to_string())
            else:
                logger.info("No codes found with keyword '소매'")

            return codes_df
        else:
            logger.warning("✗ No industry codes retrieved")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"✗ Failed to fetch industry codes: {e}")
        return pd.DataFrame()


def test_household_data(client):
    """Test 3: Collect household data for a few sample dongs."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Household Data Collection")
    logger.info("=" * 60)

    # Test dongs (popular tourist areas in Seoul)
    # NOTE: SGIS API requires 8-digit codes (7-digit + trailing "0")
    test_dongs = [
        ('11230510', '신사동', 'Gangnam-gu'),     # Gangnam - Sinsa (Garosu-gil)
        ('11140660', '서교동', 'Mapo-gu'),        # Mapo - Seogyo (Hongdae)
        ('11030650', '이태원1동', 'Yongsan-gu'),  # Yongsan - Itaewon
        ('11020550', '명동', 'Jung-gu'),          # Jung - Myeongdong
        ('11240710', '잠실본동', 'Songpa-gu')     # Songpa - Jamsil
    ]

    collector = HouseholdDataCollector(client)
    test_years = ['2020', '2021']

    all_results = []

    for dong_code, dong_name, gu_name in test_dongs:
        logger.info(f"\n{dong_name} ({gu_name}) - Code: {dong_code}")
        logger.info("-" * 40)

        for year in test_years:
            try:
                df = collector.get_household_data(
                    adm_cd=dong_code,
                    year=year
                )

                if not df.empty:
                    df['dong_name'] = dong_name
                    df['gu_name'] = gu_name
                    all_results.append(df)

                    logger.info(f"  {year}: ✓ Retrieved {len(df)} records")

                    # Show sample data
                    if len(df) > 0:
                        logger.info(f"    Columns: {list(df.columns)}")
                        logger.info(f"    Sample record:")
                        for col in df.columns[:5]:  # Show first 5 columns
                            logger.info(f"      {col}: {df[col].iloc[0]}")
                else:
                    logger.info(f"  {year}: ✗ No data")

            except Exception as e:
                logger.error(f"  {year}: ✗ Error - {e}")

    if all_results:
        result_df = pd.concat(all_results, ignore_index=True)
        logger.info(f"\n✓ Total household records collected: {len(result_df)}")

        # Save test data
        output_path = Path('test_output')
        output_path.mkdir(exist_ok=True)
        result_df.to_csv(output_path / 'test_household_data.csv', index=False, encoding='utf-8-sig')
        logger.info(f"  Saved to: test_output/test_household_data.csv")

        return result_df
    else:
        logger.warning("✗ No household data collected")
        return pd.DataFrame()


def test_company_data(client):
    """Test 4: Collect company data for a few sample dongs."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Company Data Collection")
    logger.info("=" * 60)

    # Same test dongs (8-digit codes)
    test_dongs = [
        ('11230510', '신사동', 'Gangnam-gu'),
        ('11140660', '서교동', 'Mapo-gu'),
        ('11030650', '이태원1동', 'Yongsan-gu'),
    ]

    collector = CompanyDataCollector(client)
    test_years = ['2020', '2021']

    all_results = []

    for dong_code, dong_name, gu_name in test_dongs:
        logger.info(f"\n{dong_name} ({gu_name}) - Code: {dong_code}")
        logger.info("-" * 40)

        for year in test_years:
            try:
                df = collector.get_company_data(
                    adm_cd=dong_code,
                    year=year
                )

                if not df.empty:
                    df['dong_name'] = dong_name
                    df['gu_name'] = gu_name
                    all_results.append(df)

                    logger.info(f"  {year}: ✓ Retrieved {len(df)} company records")

                    # Show sample data
                    if len(df) > 0:
                        logger.info(f"    Columns: {list(df.columns)}")
                        logger.info(f"    Sample record:")
                        for col in df.columns[:5]:  # Show first 5 columns
                            logger.info(f"      {col}: {df[col].iloc[0]}")

                        # Count by industry if available
                        if 'induty_cl_nm' in df.columns:
                            top_industries = df['induty_cl_nm'].value_counts().head(5)
                            logger.info(f"    Top 5 industries:")
                            for industry, count in top_industries.items():
                                logger.info(f"      - {industry}: {count}")
                else:
                    logger.info(f"  {year}: ✗ No data")

            except Exception as e:
                logger.error(f"  {year}: ✗ Error - {e}")

    if all_results:
        result_df = pd.concat(all_results, ignore_index=True)
        logger.info(f"\n✓ Total company records collected: {len(result_df)}")

        # Save test data
        output_path = Path('test_output')
        output_path.mkdir(exist_ok=True)
        result_df.to_csv(output_path / 'test_company_data.csv', index=False, encoding='utf-8-sig')
        logger.info(f"  Saved to: test_output/test_company_data.csv")

        return result_df
    else:
        logger.warning("✗ No company data collected")
        return pd.DataFrame()


def analyze_test_results(household_df, company_df):
    """Test 5: Analyze collected test data."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Data Analysis")
    logger.info("=" * 60)

    # Analyze household data
    if not household_df.empty:
        logger.info("\nHousehold Data Summary:")
        logger.info(f"  Shape: {household_df.shape}")
        logger.info(f"  Unique dongs: {household_df['dong_name'].nunique()}")
        logger.info(f"  Years covered: {sorted(household_df['year'].unique())}")
        logger.info(f"  Columns: {list(household_df.columns)}")

        # Check for numeric columns
        numeric_cols = household_df.select_dtypes(include=['number']).columns
        logger.info(f"  Numeric columns: {list(numeric_cols)}")

        if len(numeric_cols) > 0:
            logger.info(f"\n  Statistics for first numeric column ({numeric_cols[0]}):")
            logger.info(f"    Mean: {household_df[numeric_cols[0]].mean():.2f}")
            logger.info(f"    Min: {household_df[numeric_cols[0]].min():.2f}")
            logger.info(f"    Max: {household_df[numeric_cols[0]].max():.2f}")

    # Analyze company data
    if not company_df.empty:
        logger.info("\n" + "-" * 60)
        logger.info("Company Data Summary:")
        logger.info(f"  Shape: {company_df.shape}")
        logger.info(f"  Unique dongs: {company_df['dong_name'].nunique()}")
        logger.info(f"  Years covered: {sorted(company_df['year'].unique())}")
        logger.info(f"  Columns: {list(company_df.columns)}")

        # Analyze industry distribution
        if 'induty_cl_code' in company_df.columns:
            logger.info(f"\n  Industry Code Distribution:")
            code_dist = company_df['induty_cl_code'].value_counts().head(10)
            for code, count in code_dist.items():
                logger.info(f"    {code}: {count} businesses")

        if 'induty_cl_nm' in company_df.columns:
            logger.info(f"\n  Top 10 Industry Types:")
            industry_dist = company_df['induty_cl_nm'].value_counts().head(10)
            for industry, count in industry_dist.items():
                logger.info(f"    {industry}: {count} businesses")

        # Try to identify accommodations
        logger.info(f"\n  Accommodation-related businesses:")
        accommodation_keywords = ['숙박', '호텔', '모텔', '게스트', '펜션']
        if 'induty_cl_nm' in company_df.columns:
            for keyword in accommodation_keywords:
                count = company_df['induty_cl_nm'].str.contains(keyword, na=False).sum()
                if count > 0:
                    logger.info(f"    '{keyword}': {count} businesses")

        # Try to identify retail
        logger.info(f"\n  Retail-related businesses:")
        retail_keywords = ['소매', '상점', '마트', '편의점']
        if 'induty_cl_nm' in company_df.columns:
            for keyword in retail_keywords:
                count = company_df['induty_cl_nm'].str.contains(keyword, na=False).sum()
                if count > 0:
                    logger.info(f"    '{keyword}': {count} businesses")


def main():
    """Run all tests."""
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║" + " " * 10 + "SGIS DATA COLLECTION TEST SUITE" + " " * 16 + "║")
    logger.info("╚" + "═" * 58 + "╝")

    start_time = datetime.now()

    # Test 1: Authentication
    client = test_authentication()
    if client is None:
        logger.error("\n❌ Authentication failed. Cannot proceed with tests.")
        return

    # Test 2: Industry codes
    industry_codes_df = test_industry_codes(client)

    # Test 3: Household data
    household_df = test_household_data(client)

    # Test 4: Company data
    company_df = test_company_data(client)

    # Test 5: Analysis
    analyze_test_results(household_df, company_df)

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("\n" + "=" * 60)
    logger.info("TEST SUITE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info(f"\nResults:")
    logger.info(f"  ✓ Authentication: {'Success' if client else 'Failed'}")
    logger.info(f"  ✓ Industry codes: {len(industry_codes_df)} codes retrieved")
    logger.info(f"  ✓ Household data: {len(household_df)} records collected")
    logger.info(f"  ✓ Company data: {len(company_df)} records collected")
    logger.info(f"\nTest output saved to: test_output/")

    if not household_df.empty and not company_df.empty:
        logger.info("\n✅ All tests passed! Ready for full data collection.")
        logger.info("\nNext step: Run full collection with:")
        logger.info("  python collect_sgis_data.py --years 2017 2018 2019 2020 2021 2022 2023")
    else:
        logger.warning("\n⚠️  Some tests failed. Please review errors above.")


if __name__ == "__main__":
    main()
