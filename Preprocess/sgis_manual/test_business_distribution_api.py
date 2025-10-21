"""
Test the Business Distribution Summary API
This API may provide industry ratios/percentages by business category
"""

import requests
import json

# Your credentials
CONSUMER_KEY = "fbf9612b73e54fac8545"
CONSUMER_SECRET = "0543b74f9984418da672"

print("=" * 70)
print("Testing SGIS Business Distribution Summary API")
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

# Step 2: Test Business Distribution Summary API
print("\n[2] Testing Business Distribution Summary API...")
print("=" * 70)

# Test endpoint based on the documentation URL pattern
test_endpoints = [
    "/OpenAPI3/stats/corpdistsummary.json",
    "/OpenAPI3/analysis/corpdistsummary.json",
    "/OpenAPI3/lifebiz/corpdistsummary.json",
]

test_code = "11230510"  # Sinsa-dong, Gangnam
test_year = "2020"

for endpoint in test_endpoints:
    print(f"\nTesting endpoint: {endpoint}")

    url = f"https://sgisapi.kostat.go.kr{endpoint}"

    params = {
        "accessToken": access_token,
        "adm_cd": test_code,
        "year": test_year
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"  HTTP Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"  Response keys: {list(result.keys())}")
            print(f"  Error code: {result.get('errCd', 'N/A')}")
            print(f"  Error message: {result.get('errMsg', 'N/A')}")

            if result.get('errMsg') == 'Success':
                print(f"\n  >>> SUCCESS! Found the correct endpoint!")
                print(f"\n  Full response:")
                print(json.dumps(result, indent=2, ensure_ascii=False))

                if result.get('result'):
                    print(f"\n  Result structure:")
                    if isinstance(result['result'], list):
                        print(f"    - Type: List with {len(result['result'])} items")
                        if result['result']:
                            print(f"    - First item keys: {list(result['result'][0].keys())}")
                            print(f"\n    First 3 records:")
                            for i, item in enumerate(result['result'][:3]):
                                print(f"\n    Record {i+1}:")
                                print(json.dumps(item, indent=6, ensure_ascii=False))
                    elif isinstance(result['result'], dict):
                        print(f"    - Type: Dictionary")
                        print(f"    - Keys: {list(result['result'].keys())}")
                        print(json.dumps(result['result'], indent=4, ensure_ascii=False))

                break
        else:
            print(f"  HTTP error: {response.status_code}")

    except Exception as e:
        print(f"  Error: {e}")

# Step 3: Test with different parameters
print("\n\n[3] Testing with additional parameters...")
print("=" * 70)

# Try adding industry classification code parameter
param_variations = [
    {"accessToken": access_token, "adm_cd": test_code, "year": test_year},
    {"accessToken": access_token, "adm_cd": test_code, "year": test_year, "corp_cls_se": "55"},
    {"accessToken": access_token, "adm_cd": test_code, "year": test_year, "induty_cl_code": "55"},
    {"accessToken": access_token, "adm_cd": test_code, "year": test_year, "low_search": "1"},
]

for i, params in enumerate(param_variations, 1):
    print(f"\nVariation {i}: {params}")

    url = "https://sgisapi.kostat.go.kr/OpenAPI3/stats/corpdistsummary.json"

    try:
        response = requests.get(url, params=params, timeout=10)
        result = response.json()

        if result.get('errMsg') == 'Success':
            print(f"  >>> SUCCESS!")
            if result.get('result'):
                print(f"  Result count: {len(result['result']) if isinstance(result['result'], list) else 'N/A'}")
                print(f"  Sample data:")
                if isinstance(result['result'], list) and result['result']:
                    print(json.dumps(result['result'][0], indent=4, ensure_ascii=False))
                else:
                    print(json.dumps(result['result'], indent=4, ensure_ascii=False))
        else:
            print(f"  Failed: {result.get('errMsg')}")
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "=" * 70)
print("Test Complete!")
print("=" * 70)
