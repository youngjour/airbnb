"""
Create SGIS Feature Subsets for Systematic Testing

Generates multiple CSV files with different feature combinations
to test which features improve model performance.
"""

import pandas as pd

# Load full SGIS improved dataset
df = pd.read_csv('sgis_improved_final.csv', encoding='utf-8-sig')

print(f"Original data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# Base columns (always included)
base_cols = ['Reporting Month', 'Dong_name']

# Feature definitions
all_features = {
    'retail_ratio': 'Retail business ratio (%)',
    'accommodation_ratio': 'Accommodation business ratio (%)',
    'restaurant_ratio': 'Restaurant/bar business ratio (%)',
    'housing_units': 'Number of housing units',
    'airbnb_listing_count': 'Number of Airbnb listings',
    'airbnb_per_1k_housing': 'Airbnb per 1000 housing units'
}

# Define subsets
subsets = {
    # Experiment 2: Competition/Saturation
    'competition': {
        'features': ['accommodation_ratio', 'airbnb_per_1k_housing'],
        'description': 'Market competition and saturation indicators'
    },

    # Experiment 3: Attractiveness
    'attractiveness': {
        'features': ['retail_ratio', 'restaurant_ratio'],
        'description': 'Neighborhood attractiveness for tourists'
    },

    # Experiment 4: Ratios only
    'ratios': {
        'features': ['retail_ratio', 'accommodation_ratio', 'restaurant_ratio'],
        'description': 'Business mix ratios (scale-invariant)'
    },

    # Experiment 5: Penetration only
    'penetration': {
        'features': ['airbnb_per_1k_housing'],
        'description': 'Market saturation metric only'
    },

    # Experiment 6: No redundancy
    'no_redundancy': {
        'features': ['retail_ratio', 'accommodation_ratio', 'restaurant_ratio',
                    'airbnb_per_1k_housing'],
        'description': 'Remove features that overlap with raw data'
    }
}

# Create subset files
print("Creating feature subsets...\n")
print("=" * 70)

for subset_name, subset_info in subsets.items():
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
print(f"\nCreated {len(subsets)} subset files")
print("\nNext steps:")
print("1. Test SGIS-only: Use full sgis_improved_final.csv as embed1")
print("2. Test subsets: Use raw as embed1, subset as embed2")
print("3. Compare results to find optimal feature combination")
