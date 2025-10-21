import pandas as pd
import sys

# Set stdout encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

csv_path = "한국행정구역분류_행정동코드(7자리)_20210701기준_extracted.csv"

# Try different encodings
encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr', 'latin1']

for enc in encodings:
    print(f"\n{'='*70}")
    print(f"Trying encoding: {enc}")
    print('='*70)
    try:
        df = pd.read_csv(csv_path, encoding=enc)
        print("Shape:", df.shape)
        print("\nColumns:", df.columns.tolist())
        print("\nFirst 3 rows:")
        print(df.head(3).to_string())

        # Check if Korean characters are readable
        if '대분류' in df.columns:
            print("\n>>> SUCCESS! Correct encoding found!")
            seoul_dongs = df[df['대분류'] == '서울특별시']
            print(f"Seoul dongs: {len(seoul_dongs)}")
            break
    except Exception as e:
        print(f"Failed with {enc}: {e}")
