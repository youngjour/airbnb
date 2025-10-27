"""
Generate LLM prompts from dong-level tourism data

Combines:
1. Korean credit card sales (domestic tourism spending)
2. Foreign credit card sales (international tourism spending)
3. Navigation searches by category (tourism interest signals)

Output: Natural language prompts describing tourism characteristics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print(f"{'='*100}")
print(f"TOURISM LLM PROMPT GENERATION")
print(f"{'='*100}\n")

# Paths
TOURISM_DIR = Path(r'C:\Users\jour\Documents\GitHub\airbnb\Preprocess\tmap_tourism')
TOURISM_DONG_FILE = TOURISM_DIR / 'tourism_dong_level_distributed.csv'

# Step 1: Load dong-level tourism data
print(f"STEP 1: LOADING DONG-LEVEL TOURISM DATA")
print(f"{'-'*100}")

df = pd.read_csv(TOURISM_DONG_FILE, encoding='utf-8-sig')

print(f"[OK] Loaded: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Time range: {df['month'].min()} to {df['month'].max()}")
print(f"Unique dongs: {df['Dong_name'].nunique()}")

# Step 2: Engineer tourism features
print(f"\nSTEP 2: ENGINEERING TOURISM FEATURES")
print(f"{'-'*100}")

# 2a: Total tourism spending
df['total_cc_sales'] = df['korean_cc_sales'] + df['foreign_cc_sales']

# 2b: International tourism ratio
df['foreign_ratio'] = df['foreign_cc_sales'] / (df['total_cc_sales'] + 1e-6)  # Avoid division by zero

# 2c: Per-capita tourism intensity (normalize by navigation searches as proxy for popularity)
df['cc_per_search'] = df['total_cc_sales'] / (df['navigation_searches'] + 1)

# 2d: Navigation category ratios (excluding '전체' total)
# Get navigation columns (those starting with 'nav_' but not '전체')
nav_cols = [col for col in df.columns if col.startswith('nav_')]

# Remove the 'total' column if exists
nav_cols = [col for col in nav_cols if '전체' not in col]

print(f"Navigation categories: {len(nav_cols)}")
for col in nav_cols:
    # Calculate ratio of each category to total navigation searches
    df[f'{col}_ratio'] = df[col] / (df['navigation_searches'] + 1)

# Step 3: Calculate rankings and percentiles
print(f"\nSTEP 3: CALCULATING RANKINGS")
print(f"{'-'*100}")

# For each month, calculate ranking by tourism metrics
df['cc_rank'] = df.groupby('month')['total_cc_sales'].rank(ascending=False, method='min')
df['cc_percentile'] = df.groupby('month')['total_cc_sales'].rank(pct=True)

df['nav_rank'] = df.groupby('month')['navigation_searches'].rank(ascending=False, method='min')
df['nav_percentile'] = df.groupby('month')['navigation_searches'].rank(pct=True)

# Step 4: Calculate growth trends
print(f"\nSTEP 4: CALCULATING GROWTH TRENDS")
print(f"{'-'*100}")

# Sort by dong and month
df = df.sort_values(['Dong_name', 'month'])

# Calculate month-over-month growth
df['cc_mom_growth'] = df.groupby('Dong_name')['total_cc_sales'].pct_change()
df['nav_mom_growth'] = df.groupby('Dong_name')['navigation_searches'].pct_change()

# Calculate 12-month rolling average for seasonal baseline
df['cc_seasonal_avg'] = df.groupby('Dong_name')['total_cc_sales'].transform(
    lambda x: x.rolling(window=12, min_periods=1).mean()
)
df['cc_seasonal_index'] = df['total_cc_sales'] / (df['cc_seasonal_avg'] + 1)

# Step 5: Generate natural language prompts
print(f"\nSTEP 5: GENERATING NATURAL LANGUAGE PROMPTS")
print(f"{'-'*100}")

prompts = []

# Category name mapping (Korean to English for better interpretability)
category_names = {
    'nav_자연관광_ratio': ('Natural tourism', 'parks and nature sites'),
    'nav_역사관광_ratio': ('Historical tourism', 'palaces and heritage sites'),
    'nav_체험관광_ratio': ('Experience tourism', 'cultural experiences'),
    'nav_문화관광_ratio': ('Cultural tourism', 'museums and galleries'),
    'nav_레저스포츠_ratio': ('Leisure/Sports', 'sports and recreation'),
    'nav_쇼핑_ratio': ('Shopping', 'markets and shopping centers'),
    'nav_음식_ratio': ('Dining', 'restaurants and cafes'),
    'nav_숙박_ratio': ('Accommodation', 'hotels and guesthouses'),
    'nav_기타관광_ratio': ('Other tourism', 'miscellaneous attractions')
}

print(f"Generating prompts for {len(df)} records...")

for idx, row in df.iterrows():
    if idx % 5000 == 0:
        print(f"  Processing {idx}/{len(df)}...")

    # Extract key metrics
    dong = row['Dong_name']
    gu = row['Gu_name']
    month = str(row['month'])
    year = month[:4]
    month_num = month[4:6]

    total_cc = row['total_cc_sales']
    korean_cc = row['korean_cc_sales']
    foreign_cc = row['foreign_cc_sales']
    foreign_pct = row['foreign_ratio'] * 100
    nav_searches = row['navigation_searches']

    cc_rank = int(row['cc_rank'])
    cc_percentile = row['cc_percentile']
    nav_rank = int(row['nav_rank'])

    mom_growth = row['cc_mom_growth']
    seasonal_index = row['cc_seasonal_index']

    # Find dominant tourism categories (top 3)
    cat_ratios = {k: row[k] for k in category_names.keys() if k in row.index}
    top_categories = sorted(cat_ratios.items(), key=lambda x: x[1], reverse=True)[:3]

    # Build prompt
    prompt = f"In {year}-{month_num}, {dong} in {gu} district recorded tourism activity worth {total_cc:,.0f} won in credit card transactions, ranking {cc_rank} among Seoul neighborhoods."

    # Add international tourism context
    if foreign_pct > 10:
        prompt += f" International tourists contributed {foreign_pct:.1f}% of spending ({foreign_cc:,.0f} won), indicating strong appeal to foreign visitors."
    elif foreign_pct > 5:
        prompt += f" International tourism accounted for {foreign_pct:.1f}% of spending, showing moderate foreign visitor interest."
    else:
        prompt += f" Tourism spending was primarily domestic ({100-foreign_pct:.1f}%), suggesting local or national appeal."

    # Add navigation search context
    prompt += f" The area received {nav_searches:,.0f} navigation searches for tourist destinations, ranking {nav_rank} in search interest."

    # Add dominant tourism types
    if len(top_categories) > 0 and top_categories[0][1] > 0.1:
        cat1_name, cat1_desc = category_names.get(top_categories[0][0], ('Tourism', 'attractions'))
        cat1_pct = top_categories[0][1] * 100
        prompt += f" The dominant tourism interest was {cat1_name} ({cat1_pct:.1f}% of searches), particularly {cat1_desc}."

        if len(top_categories) > 1 and top_categories[1][1] > 0.05:
            cat2_name, cat2_desc = category_names.get(top_categories[1][0], ('Tourism', 'attractions'))
            cat2_pct = top_categories[1][1] * 100
            prompt += f" Secondary interest focused on {cat2_name} ({cat2_pct:.1f}%), including {cat2_desc}."

    # Add growth trend interpretation
    if pd.notna(mom_growth):
        if abs(mom_growth) > 0.20:
            direction = "surged" if mom_growth > 0 else "dropped"
            prompt += f" Tourism activity {direction} {abs(mom_growth)*100:.1f}% from the previous month, indicating significant seasonal or event-driven change."
        elif abs(mom_growth) > 0.05:
            direction = "increased" if mom_growth > 0 else "decreased"
            prompt += f" Activity {direction} {abs(mom_growth)*100:.1f}% month-over-month, showing {direction} tourism momentum."

    # Add seasonal context
    if pd.notna(seasonal_index):
        if seasonal_index > 1.2:
            prompt += f" This represents peak season activity, {(seasonal_index-1)*100:.0f}% above the annual average for this neighborhood."
        elif seasonal_index < 0.8:
            prompt += f" This was an off-season period, {(1-seasonal_index)*100:.0f}% below the typical annual average."

    # Add Airbnb relevance
    if cc_percentile > 0.8:
        prompt += f" High tourism spending and visitor interest make this area particularly attractive for Airbnb hosts targeting tourists."
    elif cc_percentile > 0.5:
        prompt += f" Moderate tourism activity suggests potential for Airbnb accommodation, especially during peak periods."
    else:
        prompt += f" While tourism activity is present, the area may appeal more to niche markets or local experiences for Airbnb guests."

    prompts.append({
        'Dong_name': dong,
        'Gu_name': gu,
        'month': month,
        'prompt': prompt
    })

# Create DataFrame
prompts_df = pd.DataFrame(prompts)

print(f"\n[OK] Generated {len(prompts_df)} prompts")

# Step 6: Save prompts
print(f"\nSTEP 6: SAVING PROMPTS")
print(f"{'-'*100}")

output_file = TOURISM_DIR / 'tourism_prompts.csv'
prompts_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"[OK] Saved: {output_file}")
print(f"Shape: {prompts_df.shape}")

# Show sample prompts
print(f"\n{'='*100}")
print(f"SAMPLE PROMPTS")
print(f"{'='*100}\n")

# Show 5 random samples
samples = prompts_df.sample(n=min(5, len(prompts_df)))
for idx, (_, row) in enumerate(samples.iterrows(), 1):
    print(f"Sample {idx}:")
    print(f"  Dong: {row['Dong_name']}")
    print(f"  Month: {row['month']}")
    print(f"  Prompt length: {len(row['prompt'])} chars")
    print(f"  Prompt: {row['prompt'][:300]}...")
    print()

# Statistics
print(f"{'='*100}")
print(f"PROMPT STATISTICS")
print(f"{'='*100}\n")

prompt_lengths = prompts_df['prompt'].str.len()
print(f"Prompt length statistics:")
print(f"  Mean: {prompt_lengths.mean():.0f} chars")
print(f"  Median: {prompt_lengths.median():.0f} chars")
print(f"  Min: {prompt_lengths.min():.0f} chars")
print(f"  Max: {prompt_lengths.max():.0f} chars")

print(f"\n{'='*100}")
print(f"NEXT STEP: Generate LLM embeddings using Llama 3.2")
print(f"{'='*100}")
