"""
Generate Airbnb LLM prompts from AirBnB_data.csv
Creates two versions:
- AirBnB_SSP_wo_prompts.csv (without listing titles)
- AirBnB_SSP_w_prompts.csv (with listing titles)

Based on Hongju's airbnb_prompt_preprocess.ipynb
"""

import pandas as pd
from collections import Counter
import ast
import json
from tqdm import tqdm
import os

print("="*80)
print("AIRBNB PROMPT GENERATION")
print("="*80)

# Load Airbnb data
file_path = '../../../DATA/AirBnB_data.csv'
print(f"\nLoading AirBnB data from: {file_path}")
df = pd.read_csv(file_path, low_memory=False)
print(f"[OK] Loaded {len(df)} rows x {len(df.columns)} columns")

# Define feature categories
category_columns = ['Property Type', 'Listing Type', 'Airbnb Response Time (Text)',
                    'Cancellation Policy', 'Check-in Time', 'Checkout Time']
binary_columns = ['Airbnb Superhost', 'Instantbook Enabled', 'Pets Allowed']
numerical_columns = ['Bedrooms', 'Bathrooms', 'Max Guests', 'Available Days', 'Blocked Days',
                     'Response Rate', 'Number of Reviews', 'Minimum Stay', 'Number of Photos',
                     'Overall Rating']

print(f"\nFeature categories:")
print(f"  Category columns: {len(category_columns)}")
print(f"  Binary columns: {len(binary_columns)}")
print(f"  Numerical columns: {len(numerical_columns)}")

# Amenities preprocessing
print("\nProcessing amenities...")
def parse_amenities(amenities):
    try:
        if isinstance(amenities, str):
            if amenities.startswith("{"):  # JSON format
                amenities_dict = json.loads(amenities)
                return [item.lower() for sublist in amenities_dict.values() for item in sublist]
            elif amenities.startswith("["):  # List string format
                return [item.lower() for item in json.loads(amenities)]
        return []
    except:
        return []

df["Amenities"] = df["Amenities"].apply(parse_amenities)

# Get top 50 amenities
all_amenities = [amenity for sublist in df["Amenities"] for amenity in sublist]
amenity_counts = Counter(all_amenities)
top_amenities = [amenity for amenity, count in amenity_counts.most_common(50)]
print(f"[OK] Found {len(amenity_counts)} unique amenities, using top 50")

# Create full date-dong index (28,408 combinations)
dong_names = list(df['Dong_name'].unique())
dong_names.append('상계8동')  # Add missing dong
date_range = sorted(df['Reporting Month'].unique())

print(f"\nCreating full index:")
print(f"  Dates: {len(date_range)} months ({date_range[0]} to {date_range[-1]})")
print(f"  Dongs: {len(dong_names)} districts")
print(f"  Total combinations: {len(date_range) * len(dong_names)}")

full_index = pd.MultiIndex.from_product([date_range, dong_names],
                                        names=['Reporting Month', 'Dong_name'])
df_keys = pd.DataFrame(index=full_index).reset_index()

# Generate prompts
print("\nGenerating prompts for all dong-months...")
SSP_prompt_wo_listing = []
SSP_prompt_w_listing = []

for idx, row in tqdm(df_keys.iterrows(), total=len(df_keys), desc="Generating prompts"):
    month = row['Reporting Month']
    dong = row['Dong_name']

    ex_df = df.loc[(df['Reporting Month']==month) & (df['Dong_name']==dong), ]
    prompt = [f"[{month} | {dong}] AirBnB Feature Summary:"
              f"Total number of AirBnB: {len(ex_df)}"]

    if ex_df.empty:
        prompt.append('There is no AirBnB')
        SSP_prompt_wo_listing.append('\n'.join(prompt))
        SSP_prompt_w_listing.append('\n'.join(prompt))
        continue

    # 1. Category columns processing
    category_prompt = [f"Category Column Attributes"]
    for col in category_columns:
        category = [f"\nCategory: {col} Information: Total number with data: {len(ex_df[col])}"]
        vc = ex_df[col].value_counts()
        for val, count in vc.items():
            category.append(f"   {val}: {count}")
        category_prompt.extend(category)

    # Amenities processing - only top 50
    amenities_series = ex_df['Amenities']
    total_amenities = []
    for val in amenities_series:
        try:
            items = ast.literal_eval(val) if isinstance(val, str) else val
            if isinstance(items, list):
                total_amenities.extend(items)
        except:
            continue
    counts = Counter(total_amenities)
    top_amenity_counts = {amenity: counts[amenity] for amenity in top_amenities if amenity in counts}

    category = [f"\nCategory: Amenities Information: Total number with data: {len(total_amenities)}"]
    for k, v in top_amenity_counts.items():
        category.append(f"   {k}: {v}")
    category_prompt.extend(category)
    category_prompt.append('---------------------------------------------------------------------------\n')

    # 2. Binary columns processing
    binary_prompt = [f"Binary Column Attributes\n"]
    for col in binary_columns:
        col_data = ex_df[col].dropna()
        total = len(col_data)

        if total == 0:
            binary_prompt.append(f"{col} Information: No data available")
            continue

        binary = [f"{col} Information: Total number with data: {total}"]
        count = ex_df[col].sum()
        binary.append(f"number of {col}: {count}")
        binary_prompt.extend(binary)
    binary_prompt.append('---------------------------------------------------------------------------\n')

    # 3. Numerical columns processing
    numerical_prompt = [f"Numerical Column Attributes"]
    for col in numerical_columns:
        col_data = ex_df[col].dropna()
        total = len(col_data)

        if total == 0:
            numerical_prompt.append(f"{col} Information: No data available")
            continue

        numerical = [f"{col} Information: Total number with data: {total}"]
        stats = col_data.describe()
        numerical.append(f"   Mean: {stats['mean']:.2f}")
        numerical.append(f"   Std Dev: {stats['std']:.2f}")
        numerical.append(f"   Median: {col_data.median():.2f}")
        numerical.append(f"   Min: {stats['min']:.2f}")
        numerical.append(f"   Max: {stats['max']:.2f}")
        numerical_prompt.extend(numerical)
    numerical_prompt.append('---------------------------------------------------------------------------\n')

    # 4. Listing title processing (optional)
    listing_series = ex_df['Listing Title']
    total = len(listing_series)

    listing_prompt = [f"AirBnB Listing Information: Total number with data: {total}"]
    for val in listing_series:
        listing_prompt.append(f"'{val}', ")
    listing_prompt.append('---------------------------------------------------------------------------\n')

    last_prompt = [f'Assume you are a data analyst that is familiar with AirBnB market. Give me the embedding of this {dong} at {month}']

    # Combine prompts
    full_prompt_wo_listing = "\n".join(
        prompt + category_prompt + binary_prompt + numerical_prompt + last_prompt
    )
    full_prompt_w_listing = "\n".join(
        prompt + category_prompt + binary_prompt + numerical_prompt + listing_prompt + last_prompt
    )

    SSP_prompt_wo_listing.append(full_prompt_wo_listing)
    SSP_prompt_w_listing.append(full_prompt_w_listing)

# Create dataframes
print("\nCreating output dataframes...")
airbnb_SSP_wo_listing = df_keys.copy()
airbnb_SSP_w_listing = df_keys.copy()

airbnb_SSP_wo_listing['prompt'] = SSP_prompt_wo_listing
airbnb_SSP_w_listing['prompt'] = SSP_prompt_w_listing

print(f"[OK] Without listings shape: {airbnb_SSP_wo_listing.shape}")
print(f"[OK] With listings shape: {airbnb_SSP_w_listing.shape}")

# Save to CSV
os.makedirs('../dong_prompts_new', exist_ok=True)

output_wo = '../dong_prompts_new/AirBnB_SSP_wo_prompts.csv'
output_w = '../dong_prompts_new/AirBnB_SSP_w_prompts.csv'

print(f"\nSaving prompts...")
airbnb_SSP_wo_listing.to_csv(output_wo, index=False, encoding='utf-8')
airbnb_SSP_w_listing.to_csv(output_w, index=False, encoding='utf-8')

print(f"[OK] Saved without listings to: {output_wo}")
print(f"[OK] Saved with listings to: {output_w}")

print("\n" + "="*80)
print("AIRBNB PROMPT GENERATION COMPLETE!")
print("="*80)
print(f"\nOutput files:")
print(f"  1. {output_wo} ({len(airbnb_SSP_wo_listing)} prompts)")
print(f"  2. {output_w} ({len(airbnb_SSP_w_listing)} prompts)")
print(f"\nNext step: Generate LLM embeddings using these prompts")
