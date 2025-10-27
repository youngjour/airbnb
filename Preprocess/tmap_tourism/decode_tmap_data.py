"""
Decode and analyze Tmap navigation search data
"""
import pandas as pd
import sys

file_path = r'C:\Users\jour\Documents\GitHub\airbnb\Data\20251026174807_지역별 검색건수.csv'
output_file = r'C:\Users\jour\Documents\GitHub\airbnb\Preprocess\tmap_tourism\tmap_data_analysis.txt'

# Open output file
outf = open(output_file, 'w', encoding='utf-8')

def log(msg):
    print(msg)
    outf.write(msg + '\n')

# Try to read with proper encoding
try:
    df = pd.read_csv(file_path, encoding='euc-kr')
    log("[OK] Successfully read with EUC-KR encoding")
except:
    try:
        df = pd.read_csv(file_path, encoding='cp949')
        log("[OK] Successfully read with CP949 encoding")
    except:
        df = pd.read_csv(file_path, encoding='latin1')
        log("[OK] Successfully read with Latin1 encoding")

print(f"\n{'='*100}")
print(f"TMAP NAVIGATION SEARCH DATA STRUCTURE")
print(f"{'='*100}")

print(f"\nShape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Rename columns for clarity
df.columns = ['reporting_month', 'region', 'category', 'search_count']

print(f"\n{'='*100}")
print(f"DATA SAMPLE (First 30 rows)")
print(f"{'='*100}")
print(df.head(30).to_string(index=False))

print(f"\n{'='*100}")
print(f"DATA COVERAGE")
print(f"{'='*100}")

print(f"\nTemporal coverage:")
print(f"  Months: {df['reporting_month'].nunique()}")
print(f"  Range: {df['reporting_month'].min()} to {df['reporting_month'].max()}")
print(f"  Month list: {sorted(df['reporting_month'].unique())}")

print(f"\nGeographic coverage:")
print(f"  Unique regions (gu): {df['region'].nunique()}")
print(f"  Region list:")
for region in sorted(df['region'].unique()):
    count = len(df[df['region'] == region])
    print(f"    - {region}: {count} records")

print(f"\nCategory coverage:")
print(f"  Unique categories: {df['category'].nunique()}")
print(f"  Category list:")
for cat in sorted(df['category'].unique()):
    count = len(df[df['category'] == cat])
    avg_searches = df[df['category'] == cat]['search_count'].mean()
    print(f"    - {cat}: {count} records, avg {avg_searches:,.0f} searches/month")

print(f"\n{'='*100}")
print(f"SEOUL GU BREAKDOWN")
print(f"{'='*100}")

# Analyze specific gu
sample_gu = df['region'].unique()[0]
gu_data = df[df['region'] == sample_gu]
print(f"\nSample GU: {sample_gu}")
print(f"Records: {len(gu_data)}")
print(f"Categories per month: {len(gu_data) / gu_data['reporting_month'].nunique():.0f}")
print(f"\nCategory breakdown:")
print(gu_data.groupby('category')['search_count'].agg(['count', 'mean', 'min', 'max']))

print(f"\n{'='*100}")
print(f"MONTHLY TRENDS (Gangnam-gu if available)")
print(f"{'='*100}")

gangnam = df[df['region'].str.contains('강남', na=False)]
if len(gangnam) > 0:
    gangnam_total = gangnam[gangnam['category'] == '전체']
    if len(gangnam_total) > 0:
        print(f"\nGangnam-gu total searches by month:")
        print(gangnam_total[['reporting_month', 'search_count']].to_string(index=False))

print(f"\n{'='*100}")
print(f"STATISTICS")
print(f"{'='*100}")

print(f"\nSearch count statistics:")
print(df['search_count'].describe())

print(f"\nTop 10 gu by total searches (전체 category):")
total_cat = df[df['category'] == '전체']
top_gu = total_cat.groupby('region')['search_count'].sum().sort_values(ascending=False).head(10)
for region, count in top_gu.items():
    print(f"  {region}: {count:,}")

print(f"\n{'='*100}")
print(f"READY FOR PREPROCESSING")
print(f"{'='*100}")
print(f"\nThis data can be processed to dong-level using:")
print(f"1. Dong-to-Gu mapping from SGIS")
print(f"2. Population-weighted or Airbnb-density-weighted distribution")
print(f"3. Feature engineering from 9 destination categories")
print(f"4. LLM prompt generation for tourism characteristics")
