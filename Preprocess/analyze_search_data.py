"""
Analyze web search trends data structure
"""
import pandas as pd
import sys
import os

# Navigate to correct directory
file_path = r'C:\Users\jour\Documents\GitHub\airbnb\Data\20251026174807_지역별 검색건수.csv'

# Try different encodings
encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin1']
df = None

for enc in encodings:
    try:
        df = pd.read_csv(file_path, encoding=enc)
        print(f"[OK] Successfully read with encoding: {enc}")
        break
    except Exception as e:
        print(f"[FAIL] Failed with {enc}")
        continue

if df is not None:
    print(f"\n{'='*80}")
    print(f"DATA STRUCTURE ANALYSIS")
    print(f"{'='*80}")

    print(f"\nShape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    print(f"\nFirst 20 rows:")
    print(df.head(20).to_string())

    print(f"\n{'='*80}")
    print(f"DATA COVERAGE")
    print(f"{'='*80}")

    # Get column names
    col_month = df.columns[0]
    col_region = df.columns[1]
    col_category = df.columns[2]
    col_count = df.columns[3]

    print(f"\nUnique months: {df[col_month].nunique()}")
    print(f"Month range: {df[col_month].min()} to {df[col_month].max()}")
    print(f"\nUnique regions: {df[col_region].nunique()}")
    print(f"Sample regions:")
    print(df[col_region].unique()[:10])

    print(f"\nUnique categories: {df[col_category].nunique()}")
    print(f"Categories:")
    print(df[col_category].unique())

    # Check for Seoul data
    print(f"\n{'='*80}")
    print(f"SEOUL DATA CHECK")
    print(f"{'='*80}")

    seoul_data = df[df[col_region].str.contains('서울', na=False)]
    print(f"\nSeoul records: {len(seoul_data)}")
    print(f"Seoul regions: {seoul_data[col_region].nunique()}")
    print(f"\nSample Seoul data:")
    print(seoul_data.head(30).to_string())

    # Check specific gu (district)
    print(f"\n{'='*80}")
    print(f"SAMPLE GU DATA (강남구)")
    print(f"{'='*80}")

    gangnam_data = df[df[col_region].str.contains('강남', na=False)]
    if len(gangnam_data) > 0:
        print(f"\nGangnam records: {len(gangnam_data)}")
        print(gangnam_data.head(20).to_string())
    else:
        print("No Gangnam data found")
