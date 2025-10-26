"""
Create Custom SGIS Feature Subsets - User Requested Combinations
"""

import pandas as pd

# Load full SGIS improved dataset
df = pd.read_csv('sgis_improved_final.csv', encoding='utf-8-sig')

print(f"Original data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# Base columns
base_cols = ['Reporting Month', 'Dong_name']

# Custom subsets as requested by user
custom_subsets = {
    # Experiment 3: RAW + two ratios (accommodation + retail)
    'two_ratios': {
        'features': ['accommodation_ratio', 'retail_ratio'],
        'description': 'Competition (accommodation) + Attractiveness (retail)'
    },

    # Experiment 5: RAW + housing + 3 ratios
    'housing_plus_ratios': {
        'features': ['housing_units', 'retail_ratio', 'accommodation_ratio',
                    'restaurant_ratio'],
        'description': 'Market size + business mix ratios'
    }
}

# Create subset files
print("Creating custom feature subsets...\n")
print("=" * 70)

for subset_name, subset_info in custom_subsets.items():
    features = subset_info['features']
    description = subset_info['description']

    # Select columns
    cols = base_cols + features
    subset_df = df[cols].copy()

    # Save
    output_file = f'sgis_improved_subset_{subset_name}.csv'
    subset_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"[{subset_name.upper()}]")
    print(f"  Description: {description}")
    print(f"  Features ({len(features)}): {', '.join(features)}")
    print(f"  Output: {output_file}")
    print(f"  Shape: {subset_df.shape}")
    print()

print("=" * 70)
print(f"\nCreated {len(custom_subsets)} custom subset files")
