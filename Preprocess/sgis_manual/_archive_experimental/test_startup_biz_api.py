"""
Test the Startup Business API - Corporation Distribution Summary
This should return ratios of business categories!

Business category codes:
- 'C': Retail (소매)
- 'G': Accommodation (숙박)
- 'H': Restaurant (음식점)
"""

import requests
import json

# Your credentials
CONSUMER_KEY = "fbf9612b73e54fac8545"
CONSUMER_SECRET = "0543b74f9984418da672"

print("=" * 70)
print("Testing SGIS Startup Business API - Corp Distribution Summary")
print("=" * 70)

# Step 1: Get access token
print("\n[1] Authenticating...")
auth_url = "https://sgisapi.kostat.go.kr/OpenAPI3/auth/authentication.json"
response = requests.get(auth_url, params={
    "consumer_key": CONSUMER_KEY,
    "consumer_secret": CONSUMER_SECRET
})

if response.status_code == 200:
    data = response.json()
    access_token = data["result"]["accessToken"]
    print(f"[OK] Got access token: {access_token[:20]}...")
else:
    print("[FAIL] Authentication failed")
    exit(1)

# Step 2: Test the correct endpoint
print("\n[2] Testing /OpenAPI3/startupbiz/corpdistsummary.json")
print("=" * 70)

url = "https://sgisapi.kostat.go.kr/OpenAPI3/startupbiz/corpdistsummary.json"

# Test with Sinsa-dong, Gangnam (popular tourist area)
test_code = "11230510"  # 8-digit dong code
test_year = "2020"

params = {
    "accessToken": access_token,
    "adm_cd": test_code,
    "year": test_year
}

print(f"\nTest parameters:")
print(f"  Dong code: {test_code} (신사동, Gangnam)")
print(f"  Year: {test_year}")

try:
    response = requests.get(url, params=params, timeout=10)
    print(f"\nHTTP Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Error code: {result.get('errCd', 'N/A')}")
        print(f"Error message: {result.get('errMsg', 'N/A')}")

        if result.get('errMsg') == 'Success':
            print("\n" + "!" * 70)
            print("SUCCESS! API IS WORKING!")
            print("!" * 70)

            print("\nFull API Response:")
            print(json.dumps(result, indent=2, ensure_ascii=False))

            if result.get('result'):
                print("\n" + "=" * 70)
                print("ANALYZING RESULT STRUCTURE")
                print("=" * 70)

                res = result['result']

                if isinstance(res, list):
                    print(f"\nResult is a LIST with {len(res)} items")

                    if res:
                        print(f"\nFirst item structure:")
                        print(json.dumps(res[0], indent=2, ensure_ascii=False))

                        print(f"\nAll keys in first item:")
                        print(list(res[0].keys()))

                        # Check for business category codes
                        print("\n" + "-" * 70)
                        print("Looking for business category ratios:")
                        print("-" * 70)

                        for key, value in res[0].items():
                            if key in ['C', 'G', 'H']:
                                category_name = {
                                    'C': 'Retail (소매)',
                                    'G': 'Accommodation (숙박)',
                                    'H': 'Restaurant (음식점)'
                                }[key]
                                print(f"  {key}: {value} - {category_name}")

                        print("\n" + "-" * 70)
                        print("ALL data fields:")
                        print("-" * 70)
                        for key, value in res[0].items():
                            print(f"  {key}: {value}")

                elif isinstance(res, dict):
                    print(f"\nResult is a DICTIONARY")
                    print(f"Keys: {list(res.keys())}")
                    print(json.dumps(res, indent=2, ensure_ascii=False))

                    # Check for business category codes
                    print("\n" + "-" * 70)
                    print("Looking for business category ratios:")
                    print("-" * 70)

                    for key, value in res.items():
                        if key in ['C', 'G', 'H']:
                            category_name = {
                                'C': 'Retail (소매)',
                                'G': 'Accommodation (숙박)',
                                'H': 'Restaurant (음식점)'
                            }[key]
                            print(f"  {key}: {value} - {category_name}")

        else:
            print(f"\nAPI call failed: {result.get('errMsg')}")
            print(f"Full response: {json.dumps(result, indent=2, ensure_ascii=False)}")

    else:
        print(f"HTTP error: {response.status_code}")
        print(f"Response text: {response.text[:500]}")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

# Step 3: Test with multiple dongs to understand data structure
print("\n\n[3] Testing with multiple popular tourist areas")
print("=" * 70)

test_dongs = [
    ('11230510', '신사동', 'Gangnam'),
    ('11140660', '서교동', 'Hongdae'),
    ('11030650', '이태원1동', 'Itaewon'),
    ('11020550', '명동', 'Myeongdong'),
]

for dong_code, dong_name, area_name in test_dongs:
    print(f"\n{dong_name} ({area_name}) - Code: {dong_code}")
    print("-" * 70)

    params = {
        "accessToken": access_token,
        "adm_cd": dong_code,
        "year": test_year
    }

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            result = response.json()

            if result.get('errMsg') == 'Success' and result.get('result'):
                res = result['result']

                if isinstance(res, list) and res:
                    data = res[0]
                elif isinstance(res, dict):
                    data = res
                else:
                    print("  No data")
                    continue

                # Extract business category ratios
                retail = data.get('C', 'N/A')
                accommodation = data.get('G', 'N/A')
                restaurant = data.get('H', 'N/A')

                print(f"  Retail (C): {retail}")
                print(f"  Accommodation (G): {accommodation}")
                print(f"  Restaurant (H): {restaurant}")

            else:
                print(f"  Failed: {result.get('errMsg', 'Unknown error')}")
        else:
            print(f"  HTTP error: {response.status_code}")

    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "=" * 70)
print("Test Complete!")
print("=" * 70)
