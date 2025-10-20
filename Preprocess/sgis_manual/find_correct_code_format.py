"""
Find the correct administrative code format for SGIS API
Tests different code formats to see which one works
"""

import requests
import json

# Your credentials
CONSUMER_KEY = "fbf9612b73e54fac8545"
CONSUMER_SECRET = "0543b74f9984418da672"

print("=" * 60)
print("Finding Correct Administrative Code Format")
print("=" * 60)

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

# Test codes for 신사동 (Gangnam)
test_codes = {
    "7-digit original": "1123051",
    "8-digit with 0": "11230510",
    "10-digit BD (법정동)": "1123010100",  # Based on typical format
    "10-digit with trailing 0s": "1123051000",
    "5-digit (Gu level)": "11230",  # Gangnam-gu
}

print("\n[2] Testing household API with different code formats...")
print("=" * 60)

household_url = "https://sgisapi.kostat.go.kr/OpenAPI3/stats/household.json"

for code_name, code in test_codes.items():
    print(f"\nTesting: {code_name} = '{code}'")

    params = {
        "accessToken": access_token,
        "adm_cd": code,
        "low_search": "1",
        "year": "2020"
    }

    response = requests.get(household_url, params=params)
    result = response.json()

    print(f"  Status: {response.status_code}")
    print(f"  Error Code: {result.get('errCd', 'N/A')}")
    print(f"  Error Message: {result.get('errMsg', 'N/A')}")

    if result.get("errMsg") == "Success":
        print(f"  >>> SUCCESS! Correct format: {code_name} = '{code}'")
        print(f"  Result count: {len(result.get('result', []))}")
        if result.get('result'):
            print(f"  Sample data keys: {list(result['result'][0].keys())}")
            print(f"  Sample record:")
            print(json.dumps(result['result'][0], indent=4, ensure_ascii=False))
        break
    else:
        print(f"  [X] Failed")

print("\n" + "=" * 60)
print("[3] Testing company API with the same formats...")
print("=" * 60)

company_url = "https://sgisapi.kostat.go.kr/OpenAPI3/stats/company.json"

for code_name, code in test_codes.items():
    print(f"\nTesting: {code_name} = '{code}'")

    params = {
        "accessToken": access_token,
        "adm_cd": code,
        "low_search": "1",
        "year": "2020"
    }

    response = requests.get(company_url, params=params)
    result = response.json()

    print(f"  Status: {response.status_code}")
    print(f"  Error Code: {result.get('errCd', 'N/A')}")
    print(f"  Error Message: {result.get('errMsg', 'N/A')}")

    if result.get("errMsg") == "Success":
        print(f"  >>> SUCCESS! Correct format: {code_name} = '{code}'")
        print(f"  Result count: {len(result.get('result', []))}")
        if result.get('result'):
            print(f"  Sample data keys: {list(result['result'][0].keys())}")
            print(f"  Sample record:")
            print(json.dumps(result['result'][0], indent=4, ensure_ascii=False))
        break
    else:
        print(f"  [X] Failed")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
