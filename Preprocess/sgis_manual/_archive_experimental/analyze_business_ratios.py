"""
Analyze business ratio data from the startup business API
Extract and sum ratios by major category
"""

import requests
import json
import pandas as pd

# Your credentials
CONSUMER_KEY = "fbf9612b73e54fac8545"
CONSUMER_SECRET = "0543b74f9984418da672"

print("=" * 70)
print("Analyzing Business Category Ratios")
print("=" * 70)

# Authenticate
auth_url = "https://sgisapi.kostat.go.kr/OpenAPI3/auth/authentication.json"
response = requests.get(auth_url, params={
    "consumer_key": CONSUMER_KEY,
    "consumer_secret": CONSUMER_SECRET
})

access_token = response.json()["result"]["accessToken"]
print(f"[OK] Authenticated\n")

# Get data for Sinsa-dong
url = "https://sgisapi.kostat.go.kr/OpenAPI3/startupbiz/corpdistsummary.json"
test_code = "11230510"  # Sinsa-dong, Gangnam

params = {
    "accessToken": access_token,
    "adm_cd": test_code
}

response = requests.get(url, params=params)
result = response.json()

# Extract dong-level data (first item)
dong_data = result['result'][0]

print(f"Dong: {dong_data['adm_nm']} ({dong_data['adm_cd']})")
print(f"Major business categories: {dong_data['b_theme_list']}")
print("\n" + "=" * 70)

# Sum up ratios by major category
category_totals = {}

for item in dong_data['theme_list']:
    b_theme_cd = item['b_theme_cd']
    dist_per = float(item['dist_per'])

    if b_theme_cd not in category_totals:
        category_totals[b_theme_cd] = 0.0

    category_totals[b_theme_cd] += dist_per

# Define category names
category_names = {
    'C': 'Retail (소매)',
    'D': 'Personal Services (개인서비스)',
    'F': 'Entertainment (여가/오락)',
    'G': 'Accommodation (숙박)',
    'H': 'Restaurant (음식점)',
    'I': 'Education (교육)',
    'J': 'Medical/Health (의료/건강)',
    'K': 'Real Estate (부동산)'
}

print("TOTAL RATIOS BY MAJOR CATEGORY:")
print("=" * 70)

for code in sorted(category_totals.keys()):
    name = category_names.get(code, 'Unknown')
    ratio = category_totals[code]
    print(f"{code}: {ratio:6.2f}% - {name}")

print("\n" + "=" * 70)
print("KEY CATEGORIES FOR AIRBNB MODEL:")
print("=" * 70)

# Focus on our target categories
target_categories = {
    'C': 'Retail',
    'G': 'Accommodation',
    'H': 'Restaurant'
}

for code, name in target_categories.items():
    ratio = category_totals.get(code, 0.0)
    print(f"{name:15s} ({code}): {ratio:6.2f}%")

print("\n" + "=" * 70)
print("HOW TO USE THIS DATA:")
print("=" * 70)
print("""
1. Collect business ratios for each dong (this API)
2. Collect total company counts for each dong (company.json API we already tested)
3. Calculate estimated counts:
   - Retail count = Total companies × (Retail ratio / 100)
   - Accommodation count = Total companies × (Accommodation ratio / 100)
   - Restaurant count = Total companies × (Restaurant ratio / 100)

Example for Sinsa-dong with 6,550 total companies:
   - Retail: 6,550 × (15.90 / 100) = 1,041 stores
   - Accommodation: 6,550 × (0.13 / 100) = 9 businesses
   - Restaurant: 6,550 × (13.84 / 100) = 906 restaurants
""")

print("\n" + "=" * 70)
print("DETAILED BREAKDOWN - ACCOMMODATION (G):")
print("=" * 70)

for item in dong_data['theme_list']:
    if item['b_theme_cd'] == 'G':
        print(f"  {item['s_theme_cd_nm']:30s}: {float(item['dist_per']):5.2f}%")

print("\n" + "=" * 70)
print("DETAILED BREAKDOWN - RESTAURANT (H):")
print("=" * 70)

for item in dong_data['theme_list']:
    if item['b_theme_cd'] == 'H':
        print(f"  {item['s_theme_cd_nm']:30s}: {float(item['dist_per']):5.2f}%")

print("\n" + "=" * 70)
print("DETAILED BREAKDOWN - RETAIL (C):")
print("=" * 70)

for item in dong_data['theme_list']:
    if item['b_theme_cd'] == 'C':
        print(f"  {item['s_theme_cd_nm']:30s}: {float(item['dist_per']):5.2f}%")
