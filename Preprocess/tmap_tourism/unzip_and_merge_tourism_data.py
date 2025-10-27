"""
Unzip and merge tourism data from Korean Tourism Data Portal

Three datasets:
1. Korean credit card sales by gu
2. Foreign credit card sales by gu
3. Navigation search records by gu

Time period: 2018.01 - 2024.12
"""

import zipfile
import pandas as pd
import os
import glob
from pathlib import Path

# Paths
DATA_DIR = Path(r'C:\Users\jour\Documents\GitHub\airbnb\Data')
OUTPUT_DIR = Path(r'C:\Users\jour\Documents\GitHub\airbnb\Preprocess\tmap_tourism')
EXTRACT_DIR = OUTPUT_DIR / 'extracted'

# Create directories
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

print(f"{'='*100}")
print(f"TOURISM DATA EXTRACTION AND MERGING")
print(f"{'='*100}\n")

# Step 1: Find all zip files
zip_files = sorted(DATA_DIR.glob('*2025102618*.zip'))
print(f"Found {len(zip_files)} zip files:\n")

for i, zf in enumerate(zip_files, 1):
    print(f"  {i}. {zf.name}")

# Step 2: Unzip all files
print(f"\n{'='*100}")
print(f"STEP 1: UNZIPPING FILES")
print(f"{'='*100}\n")

all_extracted_files = []

for zip_file in zip_files:
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Extract to a subdirectory named after the zip file
            extract_subdir = EXTRACT_DIR / zip_file.stem
            extract_subdir.mkdir(exist_ok=True)
            zip_ref.extractall(extract_subdir)

            # Get list of extracted CSV files
            csv_files = list(extract_subdir.glob('**/*.csv'))
            all_extracted_files.extend(csv_files)

            print(f"[OK] {zip_file.name}")
            print(f"     Extracted {len(csv_files)} CSV files")
    except Exception as e:
        print(f"[ERROR] {zip_file.name}: {str(e)}")

print(f"\nTotal extracted CSV files: {len(all_extracted_files)}")

# Step 3: Categorize files by examining content
print(f"\n{'='*100}")
print(f"STEP 2: CATEGORIZING FILES")
print(f"{'='*100}\n")

korean_cc_files = []
foreign_cc_files = []
navigation_files = []
unknown_files = []

for csv_file in all_extracted_files:
    try:
        # Read first few rows to identify file type
        df_sample = pd.read_csv(csv_file, nrows=5, encoding='utf-8-sig')

        # Identify file type by column names or filename
        filename = csv_file.name.lower()
        columns = [str(col).lower() for col in df_sample.columns]

        # Check file type
        if '검색' in csv_file.name or 'search' in filename or '검색건수' in str(df_sample.columns):
            navigation_files.append(csv_file)
            category = "NAVIGATION"
        elif '외국인' in csv_file.name or 'foreign' in filename or '외국인' in str(df_sample.columns):
            foreign_cc_files.append(csv_file)
            category = "FOREIGN_CC"
        elif '내국인' in csv_file.name or '매출' in csv_file.name or 'korean' in filename:
            korean_cc_files.append(csv_file)
            category = "KOREAN_CC"
        else:
            # Try to infer from content
            if len(korean_cc_files) == 0:
                korean_cc_files.append(csv_file)
                category = "KOREAN_CC (inferred)"
            elif len(foreign_cc_files) == 0:
                foreign_cc_files.append(csv_file)
                category = "FOREIGN_CC (inferred)"
            elif len(navigation_files) == 0:
                navigation_files.append(csv_file)
                category = "NAVIGATION (inferred)"
            else:
                unknown_files.append(csv_file)
                category = "UNKNOWN"

        print(f"[{category}] {csv_file.relative_to(EXTRACT_DIR)}")

    except Exception as e:
        print(f"[ERROR] {csv_file.name}: {str(e)}")
        unknown_files.append(csv_file)

print(f"\nCategorization summary:")
print(f"  Korean CC files: {len(korean_cc_files)}")
print(f"  Foreign CC files: {len(foreign_cc_files)}")
print(f"  Navigation files: {len(navigation_files)}")
print(f"  Unknown files: {len(unknown_files)}")

# Step 4: Merge each category
print(f"\n{'='*100}")
print(f"STEP 3: MERGING FILES BY CATEGORY")
print(f"{'='*100}\n")

def merge_files(file_list, output_name, description):
    """Merge multiple CSV files into one"""
    if not file_list:
        print(f"[SKIP] {description}: No files to merge")
        return None

    print(f"\n{description}")
    print(f"{'-'*100}")

    all_dfs = []

    for csv_file in sorted(file_list):
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            all_dfs.append(df)
            print(f"  [OK] {csv_file.name}: {df.shape}")
        except Exception as e:
            try:
                df = pd.read_csv(csv_file, encoding='cp949')
                all_dfs.append(df)
                print(f"  [OK] {csv_file.name}: {df.shape} (CP949)")
            except Exception as e2:
                print(f"  [ERROR] {csv_file.name}: {str(e2)}")

    if not all_dfs:
        print(f"[ERROR] No data could be loaded")
        return None

    # Concatenate all dataframes
    merged_df = pd.concat(all_dfs, ignore_index=True)

    # Remove duplicates if any
    initial_rows = len(merged_df)
    merged_df = merged_df.drop_duplicates()
    final_rows = len(merged_df)

    if initial_rows > final_rows:
        print(f"\n  Removed {initial_rows - final_rows} duplicate rows")

    # Save merged file
    output_path = OUTPUT_DIR / output_name
    merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n  [SAVED] {output_path.name}")
    print(f"  Final shape: {merged_df.shape}")
    print(f"  Columns: {merged_df.columns.tolist()}")

    # Show sample
    print(f"\n  Sample data (first 3 rows):")
    print(merged_df.head(3).to_string())

    return merged_df

# Merge each category
korean_cc_df = merge_files(korean_cc_files, 'korean_cc_sales_2018_2024.csv',
                           'KOREAN CREDIT CARD SALES')

foreign_cc_df = merge_files(foreign_cc_files, 'foreign_cc_sales_2018_2024.csv',
                            'FOREIGN CREDIT CARD SALES')

navigation_df = merge_files(navigation_files, 'navigation_searches_2018_2024.csv',
                            'NAVIGATION SEARCH RECORDS')

# Summary
print(f"\n{'='*100}")
print(f"EXTRACTION AND MERGING COMPLETE")
print(f"{'='*100}\n")

print(f"Output files created in: {OUTPUT_DIR}\n")

if korean_cc_df is not None:
    print(f"1. korean_cc_sales_2018_2024.csv: {korean_cc_df.shape}")
    if '기준년월' in korean_cc_df.columns or 'reporting_month' in [c.lower() for c in korean_cc_df.columns]:
        month_col = '기준년월' if '기준년월' in korean_cc_df.columns else [c for c in korean_cc_df.columns if 'month' in c.lower()][0]
        print(f"   Time range: {korean_cc_df[month_col].min()} to {korean_cc_df[month_col].max()}")

if foreign_cc_df is not None:
    print(f"\n2. foreign_cc_sales_2018_2024.csv: {foreign_cc_df.shape}")
    if '기준년월' in foreign_cc_df.columns or 'reporting_month' in [c.lower() for c in foreign_cc_df.columns]:
        month_col = '기준년월' if '기준년월' in foreign_cc_df.columns else [c for c in foreign_cc_df.columns if 'month' in c.lower()][0]
        print(f"   Time range: {foreign_cc_df[month_col].min()} to {foreign_cc_df[month_col].max()}")

if navigation_df is not None:
    print(f"\n3. navigation_searches_2018_2024.csv: {navigation_df.shape}")
    if '기준년월' in navigation_df.columns or 'reporting_month' in [c.lower() for c in navigation_df.columns]:
        month_col = '기준년월' if '기준년월' in navigation_df.columns else [c for c in navigation_df.columns if 'month' in c.lower()][0]
        print(f"   Time range: {navigation_df[month_col].min()} to {navigation_df[month_col].max()}")

print(f"\nNext steps:")
print(f"  1. Distribute gu-level data to dong level")
print(f"  2. Engineer tourism features")
print(f"  3. Generate LLM prompts")
print(f"  4. Create tourism embeddings")
