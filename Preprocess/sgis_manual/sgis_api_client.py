"""
SGIS API Client for Korean Census Data Collection

This module provides a client for accessing SGIS (Statistical Geographic Information Service)
APIs to collect regional statistics including household and company data for Seoul districts.

Author: Airbnb Prediction Model Enhancement
Date: 2025
"""

import requests
import pandas as pd
import time
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SGISAPIClient:
    """Client for SGIS API access and authentication."""

    BASE_URL = "https://sgisapi.kostat.go.kr"
    AUTH_ENDPOINT = "/OpenAPI3/auth/authentication.json"

    def __init__(self, consumer_key: str, consumer_secret: str):
        """
        Initialize SGIS API client.

        Args:
            consumer_key: SGIS API consumer key (Service ID)
            consumer_secret: SGIS API consumer secret (Security Key)
        """
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = None
        self.token_expiry = None

    def authenticate(self) -> str:
        """
        Authenticate with SGIS API and obtain access token.

        Returns:
            Access token string
        """
        logger.info("Authenticating with SGIS API...")

        auth_url = f"{self.BASE_URL}{self.AUTH_ENDPOINT}"
        params = {
            "consumer_key": self.consumer_key,
            "consumer_secret": self.consumer_secret
        }

        try:
            response = requests.get(auth_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data.get("errMsg") == "Success":
                self.access_token = data["result"]["accessToken"]
                self.token_expiry = data["result"]["accessTimeout"]
                logger.info(f"Authentication successful. Token expires at: {self.token_expiry}")
                return self.access_token
            else:
                raise Exception(f"Authentication failed: {data.get('errMsg', 'Unknown error')}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Authentication request failed: {e}")
            raise

    def ensure_authenticated(self):
        """Ensure we have a valid access token."""
        if self.access_token is None:
            self.authenticate()

    def make_request(self, endpoint: str, params: Dict) -> Dict:
        """
        Make authenticated request to SGIS API.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            Response data dictionary
        """
        self.ensure_authenticated()

        url = f"{self.BASE_URL}{endpoint}"
        params["accessToken"] = self.access_token

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Check for token expiry or authentication errors
            if data.get("errCd") in ["401", "403"]:
                logger.info("Token expired or invalid, re-authenticating...")
                self.authenticate()
                params["accessToken"] = self.access_token
                response = requests.get(url, params=params, timeout=30)
                data = response.json()

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise


class HouseholdDataCollector:
    """Collector for household statistics from SGIS API."""

    HOUSEHOLD_ENDPOINT = "/OpenAPI3/stats/household.json"

    def __init__(self, client: SGISAPIClient):
        """
        Initialize household data collector.

        Args:
            client: Authenticated SGISAPIClient instance
        """
        self.client = client

    def get_household_data(
        self,
        adm_cd: str,
        low_search: str = "1",
        year: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve household statistics for a specific administrative region.

        Args:
            adm_cd: Administrative division code (7-digit dong code)
            low_search: Search level ("1" for detailed level, "0" for summary)
            year: Census year (format: "YYYY"). If None, uses latest available

        Returns:
            DataFrame with household statistics including housing units
        """
        params = {
            "adm_cd": adm_cd,
            "low_search": low_search
        }

        if year:
            params["year"] = year

        try:
            data = self.client.make_request(self.HOUSEHOLD_ENDPOINT, params)

            if data.get("errMsg") == "Success":
                results = data.get("result", [])
                if results:
                    df = pd.DataFrame(results)
                    df['adm_cd'] = adm_cd
                    df['year'] = year if year else 'latest'
                    return df
                else:
                    logger.warning(f"No household data for region: {adm_cd}, year: {year}")
                    return pd.DataFrame()
            else:
                logger.error(f"Failed to retrieve household data: {data.get('errMsg')}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching household data for {adm_cd}: {e}")
            return pd.DataFrame()


class CompanyDataCollector:
    """Collector for company/business statistics from SGIS API."""

    COMPANY_ENDPOINT = "/OpenAPI3/stats/company.json"

    def __init__(self, client: SGISAPIClient):
        """
        Initialize company data collector.

        Args:
            client: Authenticated SGISAPIClient instance
        """
        self.client = client

    def get_company_data(
        self,
        adm_cd: str,
        low_search: str = "1",
        year: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve company/business statistics for a specific administrative region.

        Args:
            adm_cd: Administrative division code (7-digit dong code)
            low_search: Search level ("1" for detailed level, "0" for summary)
            year: Census year (format: "YYYY"). If None, uses latest available

        Returns:
            DataFrame with company statistics including accommodations and retail stores
        """
        params = {
            "adm_cd": adm_cd,
            "low_search": low_search
        }

        if year:
            params["year"] = year

        try:
            data = self.client.make_request(self.COMPANY_ENDPOINT, params)

            if data.get("errMsg") == "Success":
                results = data.get("result", [])
                if results:
                    df = pd.DataFrame(results)
                    df['adm_cd'] = adm_cd
                    df['year'] = year if year else 'latest'
                    return df
                else:
                    logger.warning(f"No company data for region: {adm_cd}, year: {year}")
                    return pd.DataFrame()
            else:
                logger.error(f"Failed to retrieve company data: {data.get('errMsg')}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching company data for {adm_cd}: {e}")
            return pd.DataFrame()


class IndustryCodeHelper:
    """Helper to fetch and manage industry classification codes."""

    INDUSTRY_CODE_ENDPOINT = "/OpenAPI3/stats/industrycode.json"

    def __init__(self, client: SGISAPIClient):
        """Initialize industry code helper."""
        self.client = client
        self.industry_codes = None

    def fetch_industry_codes(self) -> pd.DataFrame:
        """
        Fetch available industry classification codes.

        Returns:
            DataFrame with industry codes and descriptions
        """
        try:
            data = self.client.make_request(self.INDUSTRY_CODE_ENDPOINT, {})

            if data.get("errMsg") == "Success":
                results = data.get("result", [])
                self.industry_codes = pd.DataFrame(results)
                logger.info(f"Fetched {len(self.industry_codes)} industry codes")
                return self.industry_codes
            else:
                logger.error(f"Failed to fetch industry codes: {data.get('errMsg')}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching industry codes: {e}")
            return pd.DataFrame()

    def find_codes_by_keyword(self, keyword: str) -> pd.DataFrame:
        """
        Find industry codes matching a keyword.

        Args:
            keyword: Search keyword (e.g., "숙박", "소매")

        Returns:
            DataFrame with matching industry codes
        """
        if self.industry_codes is None:
            self.fetch_industry_codes()

        if self.industry_codes is not None and not self.industry_codes.empty:
            # Search in code name or description
            mask = self.industry_codes.astype(str).apply(
                lambda x: x.str.contains(keyword, case=False, na=False)
            ).any(axis=1)
            return self.industry_codes[mask]
        else:
            return pd.DataFrame()


def load_seoul_dong_codes(csv_path: str) -> pd.DataFrame:
    """
    Load Seoul dong codes from the administrative division CSV file.

    Args:
        csv_path: Path to the administrative division CSV file

    Returns:
        DataFrame with dong codes and names for Seoul
    """
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')

        # Filter for Seoul (대분류 == 11) and dong level (소분류 is not empty)
        seoul_dongs = df[
            (df['대분류'] == 11) &
            (df['소분류'].notna())
        ].copy()

        # Rename columns for easier access
        seoul_dongs = seoul_dongs.rename(columns={
            '소분류': 'dong_code_7digit',
            '읍면동': 'dong_name',
            '중분류': 'gu_code',
            '시군구': 'gu_name',
            '영문 표기': 'dong_name_en'
        })

        # Convert 7-digit codes to 8-digit codes for SGIS API
        # SGIS API requires 8-digit codes (7-digit + trailing "0")
        # First convert to int to remove decimal point, then to string
        seoul_dongs['dong_code'] = seoul_dongs['dong_code_7digit'].astype(int).astype(str) + '0'

        # Select relevant columns
        seoul_dongs = seoul_dongs[['dong_code', 'dong_name', 'gu_code', 'gu_name', 'dong_name_en', 'dong_code_7digit']]

        logger.info(f"Loaded {len(seoul_dongs)} Seoul dong codes (converted to 8-digit format)")
        return seoul_dongs

    except Exception as e:
        logger.error(f"Error loading dong codes: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    print("SGIS API Client Module")
    print("=" * 50)
    print("This module provides classes for collecting data from SGIS API.")
    print("\nMain classes:")
    print("- SGISAPIClient: Base API client with authentication")
    print("- HouseholdDataCollector: Collect household/housing data")
    print("- CompanyDataCollector: Collect company/business data")
    print("- IndustryCodeHelper: Helper for industry classification codes")
    print("\nSee collect_sgis_data.py for usage examples.")
