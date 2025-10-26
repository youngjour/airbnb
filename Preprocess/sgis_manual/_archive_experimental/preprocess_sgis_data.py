"""
SGIS Data Preprocessing for Airbnb Model Integration

Converts collected SGIS data (housing units, accommodations, retail stores)
into monthly time-series embeddings that can be integrated with the existing
Airbnb prediction model.

Author: Airbnb Prediction Model Enhancement
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SGISDataPreprocessor:
    """Preprocessor for SGIS data to create model-ready embeddings."""

    def __init__(self, start_month: str = '2017-07', end_month: str = '2023-01'):
        """
        Initialize SGIS data preprocessor.

        Args:
            start_month: Start month in format 'YYYY-MM' (matching Airbnb data)
            end_month: End month in format 'YYYY-MM' (matching Airbnb data)
        """
        self.start_month = start_month
        self.end_month = end_month

        # Generate monthly time range
        self.months = pd.date_range(
            start=start_month,
            end=end_month,
            freq='MS'
        ).strftime('%Y-%m').tolist()

        logger.info(f"Initialized preprocessor for {len(self.months)} months: {self.start_month} to {self.end_month}")

    def interpolate_yearly_to_monthly(
        self,
        yearly_df: pd.DataFrame,
        value_column: str,
        group_by_cols: List[str] = ['dong_code', 'dong_name']
    ) -> pd.DataFrame:
        """
        Interpolate yearly data to monthly data using linear interpolation.

        Since SGIS census data is typically yearly, we need to interpolate
        to match the monthly granularity of the Airbnb data.

        Args:
            yearly_df: DataFrame with yearly data
            value_column: Column name containing the values to interpolate
            group_by_cols: Columns to group by (typically dong identifiers)

        Returns:
            DataFrame with monthly interpolated data
        """
        logger.info(f"Interpolating yearly data to monthly for column: {value_column}")

        monthly_data = []

        # Get unique dongs
        unique_dongs = yearly_df[group_by_cols].drop_duplicates()

        for _, dong_row in unique_dongs.iterrows():
            # Filter data for this dong
            mask = True
            for col in group_by_cols:
                mask = mask & (yearly_df[col] == dong_row[col])

            dong_data = yearly_df[mask].copy()

            if dong_data.empty:
                continue

            # Sort by year
            dong_data = dong_data.sort_values('year')

            # Create year-value mapping
            year_values = dict(zip(dong_data['year'].astype(int), dong_data[value_column]))

            # Generate monthly data
            for month in self.months:
                year = int(month.split('-')[0])

                # Interpolate value
                if year in year_values:
                    value = year_values[year]
                else:
                    # Linear interpolation between available years
                    available_years = sorted(year_values.keys())
                    if year < min(available_years):
                        value = year_values[min(available_years)]
                    elif year > max(available_years):
                        value = year_values[max(available_years)]
                    else:
                        # Find surrounding years
                        lower_year = max([y for y in available_years if y < year])
                        upper_year = min([y for y in available_years if y > year])

                        # Linear interpolation
                        lower_value = year_values[lower_year]
                        upper_value = year_values[upper_year]
                        ratio = (year - lower_year) / (upper_year - lower_year)
                        value = lower_value + ratio * (upper_value - lower_value)

                # Create monthly record
                monthly_record = {col: dong_row[col] for col in group_by_cols}
                monthly_record['Reporting Month'] = month
                monthly_record[value_column] = value
                monthly_data.append(monthly_record)

        result_df = pd.DataFrame(monthly_data)
        logger.info(f"Generated {len(result_df)} monthly records")
        return result_df

    def process_housing_units(
        self,
        housing_raw_path: str,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Process housing units data into monthly time series.

        Args:
            housing_raw_path: Path to raw housing units CSV
            output_path: Optional path to save processed data

        Returns:
            Processed DataFrame with monthly housing unit counts by dong
        """
        logger.info("=" * 60)
        logger.info("Processing housing units data")
        logger.info("=" * 60)

        # Load raw data
        df = pd.read_csv(housing_raw_path, encoding='utf-8-sig')
        logger.info(f"Loaded {len(df)} raw housing records")
        logger.info(f"Columns: {list(df.columns)}")

        # Identify the housing unit count column
        # Common column names: 'household_cnt', 'cnt', 'val', 'value'
        value_col = None
        for col in ['household_cnt', 'cnt', 'val', 'value', 'count']:
            if col in df.columns:
                value_col = col
                break

        if value_col is None:
            # Try to find numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                value_col = numeric_cols[0]
                logger.warning(f"Using first numeric column as value: {value_col}")

        if value_col is None:
            raise ValueError("Could not identify housing unit count column")

        logger.info(f"Using value column: {value_col}")

        # Aggregate by dong and year (sum all housing types if multiple records)
        agg_df = df.groupby(['dong_code', 'dong_name', 'year'], as_index=False)[value_col].sum()

        # Interpolate to monthly
        monthly_df = self.interpolate_yearly_to_monthly(
            agg_df,
            value_column=value_col,
            group_by_cols=['dong_code', 'dong_name']
        )

        # Rename column
        monthly_df = monthly_df.rename(columns={value_col: 'housing_units'})

        # Save if output path provided
        if output_path:
            monthly_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"✓ Saved processed housing units to {output_path}")

        return monthly_df

    def process_company_counts(
        self,
        company_raw_path: str,
        output_path: Optional[str] = None,
        data_type: str = 'company'
    ) -> pd.DataFrame:
        """
        Process company/accommodation/retail data into monthly time series.

        Args:
            company_raw_path: Path to raw company CSV
            output_path: Optional path to save processed data
            data_type: Type of data ('accommodations' or 'retail')

        Returns:
            Processed DataFrame with monthly business counts by dong
        """
        logger.info("=" * 60)
        logger.info(f"Processing {data_type} data")
        logger.info("=" * 60)

        # Load raw data
        df = pd.read_csv(company_raw_path, encoding='utf-8-sig')
        logger.info(f"Loaded {len(df)} raw {data_type} records")

        # Count businesses by dong, year
        count_df = df.groupby(['dong_code', 'dong_name', 'year'], as_index=False).size()
        count_df = count_df.rename(columns={'size': f'{data_type}_count'})

        logger.info(f"Aggregated to {len(count_df)} dong-year combinations")

        # Interpolate to monthly
        monthly_df = self.interpolate_yearly_to_monthly(
            count_df,
            value_column=f'{data_type}_count',
            group_by_cols=['dong_code', 'dong_name']
        )

        # Save if output path provided
        if output_path:
            monthly_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"✓ Saved processed {data_type} to {output_path}")

        return monthly_df

    def create_combined_embedding(
        self,
        housing_df: pd.DataFrame,
        accommodations_df: pd.DataFrame,
        retail_df: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Combine all SGIS features into a single embedding DataFrame.

        Args:
            housing_df: Processed housing units DataFrame
            accommodations_df: Processed accommodations DataFrame
            retail_df: Processed retail stores DataFrame
            output_path: Optional path to save combined embedding

        Returns:
            Combined DataFrame with all SGIS features
        """
        logger.info("=" * 60)
        logger.info("Creating combined SGIS embedding")
        logger.info("=" * 60)

        # Merge dataframes
        combined = housing_df.merge(
            accommodations_df,
            on=['dong_code', 'dong_name', 'Reporting Month'],
            how='outer'
        )

        combined = combined.merge(
            retail_df,
            on=['dong_code', 'dong_name', 'Reporting Month'],
            how='outer'
        )

        # Fill missing values with 0
        combined = combined.fillna(0)

        # Sort by month and dong
        combined = combined.sort_values(['Reporting Month', 'dong_name'])

        # Normalize features (optional but recommended for neural networks)
        feature_cols = ['housing_units', 'accommodations_count', 'retail_count']

        for col in feature_cols:
            if col in combined.columns:
                combined[f'{col}_normalized'] = (
                    (combined[col] - combined[col].mean()) / combined[col].std()
                )

        logger.info(f"Combined embedding shape: {combined.shape}")
        logger.info(f"Columns: {list(combined.columns)}")
        logger.info(f"Date range: {combined['Reporting Month'].min()} to {combined['Reporting Month'].max()}")
        logger.info(f"Number of dongs: {combined['dong_name'].nunique()}")

        # Save if output path provided
        if output_path:
            combined.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"✓ Saved combined embedding to {output_path}")

        return combined

    def align_with_existing_data(
        self,
        sgis_df: pd.DataFrame,
        reference_path: str,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Align SGIS embedding with existing Airbnb data structure.

        This ensures that the dong names and time periods match exactly
        with your existing preprocessed data.

        Args:
            sgis_df: SGIS embedding DataFrame
            reference_path: Path to reference Airbnb data CSV (e.g., labels)
            output_path: Optional path to save aligned data

        Returns:
            Aligned DataFrame matching the reference structure
        """
        logger.info("=" * 60)
        logger.info("Aligning SGIS data with existing Airbnb data")
        logger.info("=" * 60)

        # Load reference data
        reference_df = pd.read_csv(reference_path)
        logger.info(f"Reference data shape: {reference_df.shape}")

        # Get unique dong names and months from reference
        ref_dongs = reference_df['Dong_name'].unique()
        ref_months = reference_df['Reporting Month'].unique()

        logger.info(f"Reference dongs: {len(ref_dongs)}")
        logger.info(f"Reference months: {len(ref_months)}")

        # Filter SGIS data to match reference
        aligned = sgis_df[
            (sgis_df['dong_name'].isin(ref_dongs)) &
            (sgis_df['Reporting Month'].isin(ref_months))
        ].copy()

        # Rename dong_name column to Dong_name for consistency
        aligned = aligned.rename(columns={'dong_name': 'Dong_name'})

        # Ensure all dong-month combinations exist
        full_index = pd.MultiIndex.from_product(
            [ref_months, ref_dongs],
            names=['Reporting Month', 'Dong_name']
        )

        aligned = aligned.set_index(['Reporting Month', 'Dong_name'])
        aligned = aligned.reindex(full_index, fill_value=0)
        aligned = aligned.reset_index()

        logger.info(f"Aligned data shape: {aligned.shape}")

        # Save if output path provided
        if output_path:
            aligned.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"✓ Saved aligned SGIS embedding to {output_path}")

        return aligned


def main():
    """Main preprocessing workflow."""
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess SGIS data for model integration')
    parser.add_argument(
        '--input-dir',
        type=str,
        default='./collected_data',
        help='Directory containing raw SGIS data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../Data/Preprocessed_data/Dong',
        help='Output directory for processed embeddings'
    )
    parser.add_argument(
        '--reference-data',
        type=str,
        default='../Data/Preprocessed_data/AirBnB_labels_dong.csv',
        help='Path to reference Airbnb data for alignment'
    )
    parser.add_argument(
        '--start-month',
        type=str,
        default='2017-07',
        help='Start month (YYYY-MM)'
    )
    parser.add_argument(
        '--end-month',
        type=str,
        default='2023-01',
        help='End month (YYYY-MM)'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SGIS Data Preprocessing Pipeline")
    logger.info("=" * 60)

    # Initialize preprocessor
    preprocessor = SGISDataPreprocessor(
        start_month=args.start_month,
        end_month=args.end_month
    )

    # Define paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Process housing units
        logger.info("\n[STEP 1/4] Processing housing units...")
        housing_df = preprocessor.process_housing_units(
            housing_raw_path=str(input_dir / 'housing_units_raw.csv'),
            output_path=str(output_dir / 'SGIS_housing_units_monthly.csv')
        )

        # Process accommodations
        logger.info("\n[STEP 2/4] Processing accommodations...")
        accommodations_df = preprocessor.process_company_counts(
            company_raw_path=str(input_dir / 'accommodations_raw.csv'),
            output_path=str(output_dir / 'SGIS_accommodations_monthly.csv'),
            data_type='accommodations'
        )

        # Process retail stores
        logger.info("\n[STEP 3/4] Processing retail stores...")
        retail_df = preprocessor.process_company_counts(
            company_raw_path=str(input_dir / 'retail_stores_raw.csv'),
            output_path=str(output_dir / 'SGIS_retail_monthly.csv'),
            data_type='retail'
        )

        # Create combined embedding
        logger.info("\n[STEP 4/4] Creating combined embedding...")
        combined_df = preprocessor.create_combined_embedding(
            housing_df=housing_df,
            accommodations_df=accommodations_df,
            retail_df=retail_df,
            output_path=str(output_dir / 'SGIS_combined_embedding.csv')
        )

        # Align with existing Airbnb data
        logger.info("\n[BONUS] Aligning with existing Airbnb data...")
        aligned_df = preprocessor.align_with_existing_data(
            sgis_df=combined_df,
            reference_path=args.reference_data,
            output_path=str(output_dir / 'SGIS_embedding_aligned.csv')
        )

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Preprocessing Complete!")
        logger.info("=" * 60)
        logger.info(f"Processed data saved to: {output_dir}")
        logger.info(f"\nGenerated files:")
        logger.info(f"  - SGIS_housing_units_monthly.csv: {housing_df.shape}")
        logger.info(f"  - SGIS_accommodations_monthly.csv: {accommodations_df.shape}")
        logger.info(f"  - SGIS_retail_monthly.csv: {retail_df.shape}")
        logger.info(f"  - SGIS_combined_embedding.csv: {combined_df.shape}")
        logger.info(f"  - SGIS_embedding_aligned.csv: {aligned_df.shape}")

        logger.info("\n📊 You can now use 'sgis' as a new embedding option in main.py!")

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
