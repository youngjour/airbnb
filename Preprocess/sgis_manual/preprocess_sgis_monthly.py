"""
Preprocess SGIS Data for Airbnb Model
Converts yearly SGIS data (2017-2023) to monthly time series (67 months: 2017-07 to 2023-01)

Required output format:
- Columns: Dong_name, Reporting Month, [feature columns]
- 67 months of data for each dong
- Monthly granularity with linear interpolation
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import sys

# Configure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

class SGISMonthlyPreprocessor:
    """Convert yearly SGIS data to monthly time series for Airbnb model."""

    def __init__(
        self,
        start_year_month: str = "2017-07",
        end_year_month: str = "2023-01",
        input_file: str = "sgis_complete_data.csv",
        output_file: str = "sgis_monthly_embedding.csv"
    ):
        """
        Initialize preprocessor.

        Args:
            start_year_month: Start month in YYYY-MM format
            end_year_month: End month in YYYY-MM format
            input_file: Path to yearly SGIS data CSV
            output_file: Path to output monthly CSV
        """
        self.start_date = datetime.strptime(start_year_month, "%Y-%m")
        self.end_date = datetime.strptime(end_year_month, "%Y-%m")
        self.input_file = input_file
        self.output_file = output_file

        # Generate all months in range
        self.months = self._generate_months()
        print(f"[INFO] Total months to generate: {len(self.months)}")
        print(f"[INFO] Period: {self.months[0]} to {self.months[-1]}")

    def _generate_months(self):
        """Generate list of all months in range."""
        months = []
        current = self.start_date
        while current <= self.end_date:
            months.append(current.strftime("%Y-%m"))
            current += relativedelta(months=1)
        return months

    def load_yearly_data(self):
        """Load yearly SGIS data."""
        print(f"\n[1] Loading yearly data from {self.input_file}...")
        df = pd.read_csv(self.input_file, encoding='utf-8-sig')

        print(f"    Shape: {df.shape}")
        print(f"    Columns: {df.columns.tolist()}")
        print(f"    Years: {sorted(df['year'].unique())}")
        print(f"    Dongs: {df['dong_code'].nunique()}")

        # Check for duplicate dong names
        dup_names = df.groupby('dong_name')['dong_code'].nunique()
        duplicates = dup_names[dup_names > 1]

        if len(duplicates) > 0:
            print(f"\n    [WARN] Found {len(duplicates)} dong names with multiple codes:")
            for dong_name in duplicates.index:
                codes = df[df['dong_name'] == dong_name]['dong_code'].unique()
                print(f"           {dong_name}: {codes}")

                # Disambiguate by appending dong_code to duplicate names
                for code in codes:
                    mask = df['dong_code'] == code
                    df.loc[mask, 'dong_name'] = f"{dong_name} ({code})"

            print(f"    [OK] Disambiguated duplicate dong names using dong codes")

        return df

    def interpolate_monthly(self, yearly_df):
        """
        Interpolate yearly data to monthly using linear interpolation.

        Strategy:
        - For each dong, interpolate between consecutive years
        - Use forward fill for months before first year
        - Use backward fill for months after last year
        """
        print(f"\n[2] Interpolating yearly data to monthly...")

        all_monthly_data = []

        dongs = yearly_df['dong_code'].unique()
        print(f"    Processing {len(dongs)} dongs...")

        # Feature columns to interpolate
        feature_cols = [
            'housing_units',
            'total_companies',
            'retail_count',
            'accommodation_count',
            'restaurant_count'
        ]

        for idx, dong_code in enumerate(dongs):
            if (idx + 1) % 50 == 0:
                print(f"    Progress: {idx + 1}/{len(dongs)} dongs")

            # Get yearly data for this dong
            dong_yearly = yearly_df[yearly_df['dong_code'] == dong_code].sort_values('year')

            if dong_yearly.empty:
                print(f"    [WARN] No data for dong {dong_code}")
                continue

            dong_name = dong_yearly.iloc[0]['dong_name']

            # Create monthly series for each feature
            monthly_features = {}

            for feature in feature_cols:
                # Get yearly values
                years = dong_yearly['year'].values
                values = dong_yearly[feature].values

                # Create a mapping from year to value
                year_to_value = dict(zip(years, values))

                # Interpolate for each month
                monthly_values = []
                for month_str in self.months:
                    year = int(month_str.split('-')[0])
                    month = int(month_str.split('-')[1])

                    # Calculate fractional year (e.g., 2020-06 = 2020.5)
                    fractional_year = year + (month - 1) / 12.0

                    # Find surrounding years in data
                    available_years = sorted(year_to_value.keys())

                    if year in year_to_value:
                        # Exact year match
                        value = year_to_value[year]
                    elif year < min(available_years):
                        # Before first year - use first year value
                        value = year_to_value[min(available_years)]
                    elif year > max(available_years):
                        # After last year - use last year value
                        value = year_to_value[max(available_years)]
                    else:
                        # Interpolate between two years
                        year_before = max([y for y in available_years if y <= year])
                        year_after = min([y for y in available_years if y > year])

                        value_before = year_to_value[year_before]
                        value_after = year_to_value[year_after]

                        # Linear interpolation
                        if year_after != year_before:
                            weight = (year - year_before) / (year_after - year_before)
                            value = value_before + weight * (value_after - value_before)
                        else:
                            value = value_before

                    monthly_values.append(value)

                monthly_features[feature] = monthly_values

            # Create records for this dong
            for i, month_str in enumerate(self.months):
                record = {
                    'Dong_name': dong_name,
                    'Reporting Month': month_str,
                    'housing_units': monthly_features['housing_units'][i],
                    'total_companies': monthly_features['total_companies'][i],
                    'retail_count': monthly_features['retail_count'][i],
                    'accommodation_count': monthly_features['accommodation_count'][i],
                    'restaurant_count': monthly_features['restaurant_count'][i]
                }
                all_monthly_data.append(record)

        monthly_df = pd.DataFrame(all_monthly_data)
        print(f"    Generated {len(monthly_df)} monthly records")

        return monthly_df

    def validate_output(self, monthly_df):
        """Validate the output monthly data."""
        print(f"\n[3] Validating output data...")

        # Check dimensions
        expected_months = len(self.months)
        unique_dongs = monthly_df['Dong_name'].nunique()
        unique_months = monthly_df['Reporting Month'].nunique()

        print(f"    Unique dongs: {unique_dongs}")
        print(f"    Unique months: {unique_months} (expected: {expected_months})")
        print(f"    Total records: {len(monthly_df)} (expected: {unique_dongs} × {expected_months})")

        # Check for missing combinations
        expected_records = unique_dongs * expected_months
        if len(monthly_df) != expected_records:
            print(f"    [WARN] Record count mismatch!")
            print(f"           Expected: {expected_records}")
            print(f"           Got: {len(monthly_df)}")
        else:
            print(f"    [OK] All dong-month combinations present")

        # Check for NaN values
        feature_cols = ['housing_units', 'total_companies', 'retail_count',
                       'accommodation_count', 'restaurant_count']
        for col in feature_cols:
            nan_count = monthly_df[col].isna().sum()
            if nan_count > 0:
                print(f"    [WARN] {col}: {nan_count} NaN values")
            else:
                print(f"    [OK] {col}: No NaN values")

        # Summary statistics
        print(f"\n    Summary Statistics:")
        print(monthly_df[feature_cols].describe())

        return True

    def save_output(self, monthly_df):
        """Save monthly data to CSV."""
        print(f"\n[4] Saving output to {self.output_file}...")

        # Ensure proper column order
        column_order = [
            'Dong_name',
            'Reporting Month',
            'housing_units',
            'total_companies',
            'retail_count',
            'accommodation_count',
            'restaurant_count'
        ]

        monthly_df = monthly_df[column_order]
        monthly_df.to_csv(self.output_file, index=False, encoding='utf-8-sig')

        print(f"    [OK] Saved {len(monthly_df)} records")
        print(f"    [OK] File: {self.output_file}")

    def run(self):
        """Run the complete preprocessing pipeline."""
        print("=" * 70)
        print("SGIS MONTHLY DATA PREPROCESSING")
        print("=" * 70)

        # Load yearly data
        yearly_df = self.load_yearly_data()

        # Interpolate to monthly
        monthly_df = self.interpolate_monthly(yearly_df)

        # Validate output
        self.validate_output(monthly_df)

        # Save output
        self.save_output(monthly_df)

        print("\n" + "=" * 70)
        print("PREPROCESSING COMPLETE!")
        print("=" * 70)
        print(f"\nOutput file: {self.output_file}")
        print(f"Total records: {len(monthly_df)}")
        print(f"Format: Dong_name, Reporting Month, [features]")
        print("\nNext steps:")
        print("1. Use this file as --embed1, --embed2, or --embed3 in Model/main.py")
        print("2. Example: python main.py --embed1 ../Preprocess/sgis_manual/sgis_monthly_embedding.csv")
        print("=" * 70)


if __name__ == "__main__":
    # Configuration
    preprocessor = SGISMonthlyPreprocessor(
        start_year_month="2017-07",
        end_year_month="2023-01",
        input_file="sgis_complete_data.csv",
        output_file="sgis_monthly_embedding.csv"
    )

    # Run preprocessing
    preprocessor.run()
