"""
Complete SGIS Data Collection Script
Combines three APIs to collect all required variables for Airbnb model:
1. Housing units (from household API)
2. Accommodations (calculated from company total × ratio)
3. Retail/Restaurants (calculated from company total × ratio)

Author: Claude Code
Date: 2025-10-21
"""

import requests
import pandas as pd
import time
import sys
from typing import Dict, List, Optional
from datetime import datetime
import json

# Configure UTF-8 output for Windows console
sys.stdout.reconfigure(encoding='utf-8')

class CompleteSGISCollector:
    """
    Integrated collector using three SGIS APIs:
    1. Household API - housing units
    2. Company API - total company counts
    3. Startup Business API - business category ratios
    """

    BASE_URL = "https://sgisapi.kostat.go.kr"
    AUTH_ENDPOINT = "/OpenAPI3/auth/authentication.json"
    HOUSEHOLD_ENDPOINT = "/OpenAPI3/stats/household.json"
    COMPANY_ENDPOINT = "/OpenAPI3/stats/company.json"
    BUSINESS_RATIO_ENDPOINT = "/OpenAPI3/startupbiz/corpdistsummary.json"

    def __init__(self, consumer_key: str, consumer_secret: str, rate_limit_delay: float = 0.2):
        """
        Initialize the SGIS collector.

        Args:
            consumer_key: SGIS API consumer key
            consumer_secret: SGIS API consumer secret
            rate_limit_delay: Delay between API calls in seconds (default 0.2)
        """
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = None
        self.rate_limit_delay = rate_limit_delay

    def authenticate(self) -> str:
        """Get access token (valid for 4 hours)"""
        auth_url = f"{self.BASE_URL}{self.AUTH_ENDPOINT}"
        params = {
            "consumer_key": self.consumer_key,
            "consumer_secret": self.consumer_secret
        }

        print(f"[AUTH] Authenticating with SGIS API...")
        response = requests.get(auth_url, params=params, timeout=30)

        if response.status_code == 200:
            data = response.json()
            if data.get("errMsg") == "Success":
                self.access_token = data["result"]["accessToken"]
                print(f"[OK] Access token obtained: {self.access_token[:20]}...")
                return self.access_token
            else:
                raise Exception(f"Authentication failed: {data.get('errMsg', 'Unknown error')}")
        else:
            raise Exception(f"HTTP error during authentication: {response.status_code}")

    def make_request(self, endpoint: str, params: Dict) -> Dict:
        """
        Make an authenticated API request with automatic token refresh.

        Args:
            endpoint: API endpoint path
            params: Request parameters (accessToken will be added automatically)

        Returns:
            API response as dictionary
        """
        if not self.access_token:
            self.authenticate()

        params["accessToken"] = self.access_token
        url = f"{self.BASE_URL}{endpoint}"

        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 401:
            # Token expired, refresh and retry
            print("[WARN] Access token expired, refreshing...")
            self.authenticate()
            params["accessToken"] = self.access_token
            response = requests.get(url, params=params, timeout=30)

        if response.status_code == 200:
            data = response.json()
            if data.get("errMsg") == "Success":
                return data
            else:
                raise Exception(f"API error: {data.get('errMsg', 'Unknown error')}")
        else:
            raise Exception(f"HTTP error: {response.status_code} - {response.text[:200]}")

    def get_housing_units(self, dong_code: str, year: str) -> int:
        """
        Get total housing units for a dong.

        Args:
            dong_code: 8-digit administrative code
            year: Year (e.g., "2020")

        Returns:
            Total household count
        """
        params = {
            "adm_cd": dong_code,
            "year": year,
            "low_search": "1"  # Get enumeration district level data
        }

        try:
            data = self.make_request(self.HOUSEHOLD_ENDPOINT, params)

            if data.get("result"):
                # Sum up all enumeration districts
                total = sum(int(r.get('household_cnt', 0)) for r in data['result'])
                return total
            else:
                print(f"[WARN] No household data for {dong_code} in {year}")
                return 0

        except Exception as e:
            print(f"[ERROR] Failed to get housing units for {dong_code}/{year}: {e}")
            return 0

    def get_total_companies(self, dong_code: str, year: str) -> int:
        """
        Get total company count for a dong.

        Args:
            dong_code: 8-digit administrative code
            year: Year (e.g., "2020")

        Returns:
            Total company count
        """
        params = {
            "adm_cd": dong_code,
            "year": year,
            "low_search": "1"
        }

        try:
            data = self.make_request(self.COMPANY_ENDPOINT, params)

            if data.get("result"):
                # Sum up all enumeration districts
                # Handle 'N/A' or non-numeric values gracefully
                total = 0
                for r in data['result']:
                    corp_cnt = r.get('corp_cnt', '0')
                    if corp_cnt == 'N/A' or corp_cnt is None:
                        continue
                    try:
                        total += int(corp_cnt)
                    except (ValueError, TypeError):
                        # Skip invalid values
                        continue
                return total
            else:
                print(f"[WARN] No company data for {dong_code} in {year}")
                return 0

        except Exception as e:
            print(f"[ERROR] Failed to get company count for {dong_code}/{year}: {e}")
            return 0

    def get_business_ratios(self, dong_code: str) -> Dict[str, float]:
        """
        Get business category distribution ratios for a dong.
        Note: This API does not take a year parameter - it returns current data.

        Args:
            dong_code: 8-digit administrative code

        Returns:
            Dictionary with major category codes and their total percentages
            Example: {'C': 15.90, 'G': 0.13, 'H': 13.84, ...}
        """
        params = {
            "adm_cd": dong_code
            # No year parameter - API doesn't accept it
        }

        try:
            data = self.make_request(self.BUSINESS_RATIO_ENDPOINT, params)

            if data.get("result") and len(data["result"]) > 0:
                # First item is dong-level data
                dong_data = data["result"][0]

                # Sum up detailed ratios by major category
                category_totals = {}
                for item in dong_data.get('theme_list', []):
                    b_theme_cd = item.get('b_theme_cd')
                    dist_per = float(item.get('dist_per', 0))

                    if b_theme_cd not in category_totals:
                        category_totals[b_theme_cd] = 0.0

                    category_totals[b_theme_cd] += dist_per

                return category_totals
            else:
                print(f"[WARN] No business ratio data for {dong_code}")
                return {}

        except Exception as e:
            print(f"[ERROR] Failed to get business ratios for {dong_code}: {e}")
            return {}

    def collect_dong_data(self, dong_code: str, dong_name: str, year: str) -> Dict:
        """
        Collect all data for a single dong-year combination.

        Args:
            dong_code: 8-digit administrative code
            dong_name: Dong name (for reference)
            year: Year string (e.g., "2020")

        Returns:
            Dictionary with all collected metrics
        """
        print(f"\n[{year}] Collecting data for {dong_name} ({dong_code})...")

        # 1. Get housing units
        housing_units = self.get_housing_units(dong_code, year)
        print(f"  Housing units: {housing_units:,}")
        time.sleep(self.rate_limit_delay)

        # 2. Get total company count
        total_companies = self.get_total_companies(dong_code, year)
        print(f"  Total companies: {total_companies:,}")
        time.sleep(self.rate_limit_delay)

        # 3. Get business category ratios
        ratios = self.get_business_ratios(dong_code)
        retail_ratio = ratios.get('C', 0.0)
        accommodation_ratio = ratios.get('G', 0.0)
        restaurant_ratio = ratios.get('H', 0.0)
        print(f"  Retail ratio: {retail_ratio:.2f}%")
        print(f"  Accommodation ratio: {accommodation_ratio:.2f}%")
        print(f"  Restaurant ratio: {restaurant_ratio:.2f}%")
        time.sleep(self.rate_limit_delay)

        # 4. Calculate specific business counts
        retail_count = total_companies * (retail_ratio / 100.0)
        accommodation_count = total_companies * (accommodation_ratio / 100.0)
        restaurant_count = total_companies * (restaurant_ratio / 100.0)

        print(f"  Estimated retail stores: {retail_count:.0f}")
        print(f"  Estimated accommodations: {accommodation_count:.0f}")
        print(f"  Estimated restaurants: {restaurant_count:.0f}")

        return {
            'dong_code': dong_code,
            'dong_name': dong_name,
            'year': year,
            'housing_units': housing_units,
            'total_companies': total_companies,
            'retail_ratio': retail_ratio,
            'accommodation_ratio': accommodation_ratio,
            'restaurant_ratio': restaurant_ratio,
            'retail_count': round(retail_count),
            'accommodation_count': round(accommodation_count),
            'restaurant_count': round(restaurant_count)
        }

    def collect_all_data(
        self,
        dong_codes_df: pd.DataFrame,
        years: List[str],
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Collect data for all dongs across all years.

        Args:
            dong_codes_df: DataFrame with columns ['dong_code', 'dong_name']
                          dong_code should be 8-digit format
            years: List of year strings (e.g., ["2017", "2018", ...])
            output_file: Optional path to save CSV as collection progresses

        Returns:
            DataFrame with all collected data
        """
        print("=" * 70)
        print("COMPLETE SGIS DATA COLLECTION")
        print("=" * 70)
        print(f"Dongs to collect: {len(dong_codes_df)}")
        print(f"Years: {years}")
        print(f"Total API calls: {len(dong_codes_df) * len(years) * 3}")
        print(f"Estimated time: {len(dong_codes_df) * len(years) * 3 * self.rate_limit_delay / 60:.1f} minutes")
        print("=" * 70)

        # Authenticate once at the start
        self.authenticate()

        all_results = []
        total_iterations = len(dong_codes_df) * len(years)
        current_iteration = 0

        start_time = time.time()

        for year in years:
            print(f"\n\n{'=' * 70}")
            print(f"YEAR: {year}")
            print(f"{'=' * 70}")

            for idx, row in dong_codes_df.iterrows():
                current_iteration += 1
                dong_code = str(row['dong_code'])
                dong_name = str(row['dong_name'])

                progress = (current_iteration / total_iterations) * 100
                print(f"\n[Progress: {progress:.1f}%] ({current_iteration}/{total_iterations})")

                try:
                    result = self.collect_dong_data(dong_code, dong_name, year)
                    all_results.append(result)

                    # Save incrementally every 10 records
                    if output_file and len(all_results) % 10 == 0:
                        temp_df = pd.DataFrame(all_results)
                        temp_df.to_csv(output_file, index=False, encoding='utf-8-sig')
                        print(f"  [SAVED] Progress saved to {output_file}")

                except Exception as e:
                    print(f"[ERROR] Failed to collect data for {dong_name}: {e}")
                    # Add empty record to maintain consistency
                    all_results.append({
                        'dong_code': dong_code,
                        'dong_name': dong_name,
                        'year': year,
                        'housing_units': 0,
                        'total_companies': 0,
                        'retail_ratio': 0.0,
                        'accommodation_ratio': 0.0,
                        'restaurant_ratio': 0.0,
                        'retail_count': 0,
                        'accommodation_count': 0,
                        'restaurant_count': 0
                    })

        end_time = time.time()
        duration = end_time - start_time

        print("\n" + "=" * 70)
        print("COLLECTION COMPLETE!")
        print("=" * 70)
        print(f"Total records collected: {len(all_results)}")
        print(f"Total time: {duration / 60:.1f} minutes")
        if total_iterations > 0:
            print(f"Average time per dong-year: {duration / total_iterations:.2f} seconds")

        df = pd.DataFrame(all_results)

        if output_file:
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"[SAVED] Final data saved to {output_file}")

        return df


def load_seoul_dong_codes(csv_path: str) -> pd.DataFrame:
    """
    Load Seoul dong codes from CSV and convert to 8-digit format.

    Args:
        csv_path: Path to CSV with 7-digit dong codes

    Returns:
        DataFrame with columns ['dong_code', 'dong_name']
        dong_code is 8-digit format
    """
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # Extract Seoul dongs only - filter by '시도' column which contains '서울특별시'
    seoul_dongs = df[df['시도'] == '서울특별시'].copy()

    # Keep only rows with actual dong codes (not NaN in 소분류)
    seoul_dongs = seoul_dongs[seoul_dongs['소분류'].notna()].copy()

    # Rename columns
    seoul_dongs = seoul_dongs.rename(columns={
        '소분류': 'dong_code_7digit',
        '읍면동': 'dong_name'
    })

    # Convert 7-digit to 8-digit format
    # IMPORTANT: Convert to int first to remove decimal point!
    seoul_dongs['dong_code'] = seoul_dongs['dong_code_7digit'].astype(int).astype(str) + '0'

    # Select only needed columns
    result = seoul_dongs[['dong_code', 'dong_name']].copy()

    print(f"[OK] Loaded {len(result)} Seoul dong codes")
    print(f"Sample codes: {result['dong_code'].head().tolist()}")

    return result


if __name__ == "__main__":
    # Configuration
    CONSUMER_KEY = "fbf9612b73e54fac8545"
    CONSUMER_SECRET = "0543b74f9984418da672"

    DONG_CODES_CSV = "한국행정구역분류_행정동코드(7자리)_20210701기준_extracted.csv"
    OUTPUT_FILE = "sgis_complete_data.csv"

    # Years to collect (Airbnb data covers 2017-2023)
    YEARS = ["2017", "2018", "2019", "2020", "2021", "2022", "2023"]

    # For testing, use only recent years
    # YEARS = ["2020", "2021", "2022"]

    print("=" * 70)
    print("SGIS COMPLETE DATA COLLECTION SCRIPT")
    print("=" * 70)
    print(f"Consumer Key: {CONSUMER_KEY}")
    print(f"Years: {YEARS}")
    print(f"Output file: {OUTPUT_FILE}")
    print("=" * 70)

    # Load dong codes
    print("\n[1] Loading Seoul dong codes...")
    dong_codes = load_seoul_dong_codes(DONG_CODES_CSV)

    # Initialize collector
    print("\n[2] Initializing collector...")
    collector = CompleteSGISCollector(
        consumer_key=CONSUMER_KEY,
        consumer_secret=CONSUMER_SECRET,
        rate_limit_delay=0.2  # 200ms between API calls
    )

    # Collect all data
    print("\n[3] Starting data collection...")
    df = collector.collect_all_data(
        dong_codes_df=dong_codes,
        years=YEARS,
        output_file=OUTPUT_FILE
    )

    # Display summary statistics
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(df.describe())

    print("\n" + "=" * 70)
    print("SAMPLE DATA (first 10 records)")
    print("=" * 70)
    print(df.head(10).to_string())

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. Review the collected data in sgis_complete_data.csv
2. Run preprocessing to interpolate yearly to monthly data
3. Integrate with Airbnb model embedding pipeline
4. Test model performance with new features

Files created:
- sgis_complete_data.csv: Raw collected data
""")
