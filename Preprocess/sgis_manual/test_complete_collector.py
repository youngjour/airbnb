"""
Test the complete SGIS collector with sample dongs
Validates all three APIs work together correctly
"""

import pandas as pd
from collect_sgis_complete import CompleteSGISCollector, load_seoul_dong_codes

# Configuration
CONSUMER_KEY = "fbf9612b73e54fac8545"
CONSUMER_SECRET = "0543b74f9984418da672"

# Test with a few popular tourist areas
TEST_DONGS = pd.DataFrame([
    {'dong_code': '11230510', 'dong_name': '신사동'},      # Gangnam - trendy shopping
    {'dong_code': '11140660', 'dong_name': '서교동'},      # Hongdae - nightlife
    {'dong_code': '11030650', 'dong_name': '이태원1동'},   # Itaewon - international
    {'dong_code': '11020550', 'dong_name': '명동'},        # Myeongdong - shopping
    {'dong_code': '11240710', 'dong_name': '잠실본동'}     # Jamsil - business/sports
])

# Test with recent years only
TEST_YEARS = ["2020", "2021", "2022"]

print("=" * 70)
print("TESTING COMPLETE SGIS COLLECTOR")
print("=" * 70)
print(f"Test dongs: {len(TEST_DONGS)}")
print(f"Test years: {TEST_YEARS}")
print(f"Expected records: {len(TEST_DONGS) * len(TEST_YEARS)}")
print("=" * 70)

# Initialize collector
collector = CompleteSGISCollector(
    consumer_key=CONSUMER_KEY,
    consumer_secret=CONSUMER_SECRET,
    rate_limit_delay=0.2
)

# Run test collection
df = collector.collect_all_data(
    dong_codes_df=TEST_DONGS,
    years=TEST_YEARS,
    output_file="test_sgis_complete_data.csv"
)

print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)
print(f"Total records: {len(df)}")
print(f"Missing data records: {df[df['housing_units'] == 0].shape[0]}")
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
print(df[['housing_units', 'total_companies', 'retail_count',
          'accommodation_count', 'restaurant_count']].describe())

print("\n" + "=" * 70)
print("SAMPLE RECORDS")
print("=" * 70)
print(df.head(10).to_string())

print("\n" + "=" * 70)
print("BUSINESS RATIOS BY DONG (2022)")
print("=" * 70)
df_2022 = df[df['year'] == '2022']
for idx, row in df_2022.iterrows():
    print(f"\n{row['dong_name']} ({row['dong_code']}):")
    print(f"  Retail: {row['retail_ratio']:.2f}% ({row['retail_count']:.0f} stores)")
    print(f"  Accommodation: {row['accommodation_ratio']:.2f}% ({row['accommodation_count']:.0f} businesses)")
    print(f"  Restaurant: {row['restaurant_ratio']:.2f}% ({row['restaurant_count']:.0f} restaurants)")

print("\n" + "=" * 70)
print("TEST COMPLETE!")
print("=" * 70)
print("If test passed, you can run the full collection with:")
print("  python collect_sgis_complete.py")
print("=" * 70)
