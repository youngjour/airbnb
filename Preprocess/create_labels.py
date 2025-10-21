"""
Create Labels File for Airbnb Prediction Model
Aggregates raw AirBnB data to dong-month level for the three target variables:
- Reservation Days
- Revenue
- Reservation (Number of Reservations)
"""

import pandas as pd
import sys

# Configure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

def create_labels(input_file='../DATA/AirBnB_data.csv',
                  output_file='../DATA/AirBnB_labels_dong.csv'):
    """
    Create labels file by aggregating raw AirBnB data to dong-month level.

    Args:
        input_file: Path to raw AirBnB CSV
        output_file: Path to output labels CSV
    """
    print("=" * 70)
    print("CREATING AIRBNB LABELS FILE")
    print("=" * 70)

    # Load raw data
    print(f"\n[1] Loading raw data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"    Loaded {len(df)} records")
    print(f"    Columns: {df.columns.tolist()}")

    # Check required columns
    required_cols = ['Reporting Month', 'Dong_name', 'Reservation Days',
                     'Revenue (USD)', 'Number of Reservations']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Select and rename columns
    print(f"\n[2] Preparing label columns...")
    labels_df = df[required_cols].copy()

    # Rename to match model expectations
    labels_df = labels_df.rename(columns={
        'Revenue (USD)': 'Revenue',
        'Number of Reservations': 'Reservation'
    })

    print(f"    Label columns: {labels_df.columns.tolist()}")

    # Aggregate to dong-month level (sum all listings in each dong-month)
    print(f"\n[3] Aggregating to dong-month level...")
    aggregated = labels_df.groupby(['Reporting Month', 'Dong_name']).agg({
        'Reservation Days': 'sum',
        'Revenue': 'sum',
        'Reservation': 'sum'
    }).reset_index()

    print(f"    Aggregated to {len(aggregated)} dong-month combinations")
    print(f"    Unique dongs: {aggregated['Dong_name'].nunique()}")
    print(f"    Unique months: {aggregated['Reporting Month'].nunique()}")

    # Sort by month and dong
    aggregated = aggregated.sort_values(['Reporting Month', 'Dong_name'])

    # Display sample
    print(f"\n[4] Sample data:")
    print(aggregated.head(10).to_string(index=False))

    # Summary statistics
    print(f"\n[5] Summary statistics:")
    print(aggregated[['Reservation Days', 'Revenue', 'Reservation']].describe())

    # Save to CSV
    print(f"\n[6] Saving to {output_file}...")
    aggregated.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"    [OK] Saved {len(aggregated)} records")

    print("\n" + "=" * 70)
    print("LABELS FILE CREATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nOutput: {output_file}")
    print(f"Format: Dong_name, Reporting Month, Reservation Days, Revenue, Reservation")
    print("\nNext step: Use this file with --label_path in Model/main.py")
    print("=" * 70)

    return aggregated

if __name__ == "__main__":
    labels_df = create_labels()
