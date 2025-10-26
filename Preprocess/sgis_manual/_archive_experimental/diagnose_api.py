"""
Diagnostic script to inspect raw SGIS API responses
"""

import requests
import json

# Your credentials
CONSUMER_KEY = "fbf9612b73e54fac8545"
CONSUMER_SECRET = "0543b74f9984418da672"

print("=" * 60)
print("SGIS API Diagnostic Tool")
print("=" * 60)

# Step 1: Get access token
print("\n[1] Getting access token...")
auth_url = "https://sgisapi.kostat.go.kr/OpenAPI3/auth/authentication.json"
auth_params = {
    "consumer_key": CONSUMER_KEY,
    "consumer_secret": CONSUMER_SECRET
}

response = requests.get(auth_url, params=auth_params)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

if response.status_code == 200:
    data = response.json()
    if data.get("errMsg") == "Success":
        access_token = data["result"]["accessToken"]
        print(f"\n✓ Access token obtained: {access_token[:20]}...")
    else:
        print(f"\n✗ Authentication failed")
        exit(1)
else:
    print(f"\n✗ HTTP error")
    exit(1)

# Step 2: Try household API with a sample dong code
print("\n" + "=" * 60)
print("[2] Testing household API...")
print("=" * 60)

test_code = "1123051"  # 신사동 (Gangnam)
household_url = "https://sgisapi.kostat.go.kr/OpenAPI3/stats/household.json"
household_params = {
    "accessToken": access_token,
    "adm_cd": test_code,
    "low_search": "1",
    "year": "2020"
}

print(f"\nURL: {household_url}")
print(f"Parameters: {json.dumps(household_params, indent=2)}")

response = requests.get(household_url, params=household_params)
print(f"\nStatus: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

# Step 3: Try company API
print("\n" + "=" * 60)
print("[3] Testing company API...")
print("=" * 60)

company_url = "https://sgisapi.kostat.go.kr/OpenAPI3/stats/company.json"
company_params = {
    "accessToken": access_token,
    "adm_cd": test_code,
    "low_search": "1",
    "year": "2020"
}

print(f"\nURL: {company_url}")
print(f"Parameters: {json.dumps(company_params, indent=2)}")

response = requests.get(company_url, params=company_params)
print(f"\nStatus: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

# Step 4: Try industry code API
print("\n" + "=" * 60)
print("[4] Testing industry code API...")
print("=" * 60)

industry_url = "https://sgisapi.kostat.go.kr/OpenAPI3/stats/industrycode.json"
industry_params = {
    "accessToken": access_token
}

print(f"\nURL: {industry_url}")
print(f"Parameters: {json.dumps(industry_params, indent=2)}")

response = requests.get(industry_url, params=industry_params)
print(f"\nStatus: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

# Step 5: Try with different code formats
print("\n" + "=" * 60)
print("[5] Testing different code formats...")
print("=" * 60)

test_codes = [
    "1123051",     # 7-digit (from CSV)
    "11230",       # 5-digit (gu level)
    "11",          # 2-digit (si level)
]

for code in test_codes:
    print(f"\nTrying code: {code}")
    params = {
        "accessToken": access_token,
        "adm_cd": code,
        "low_search": "1",
        "year": "2020"
    }
    response = requests.get(household_url, params=params)
    result = response.json()
    print(f"  Status: {response.status_code}")
    print(f"  Error code: {result.get('errCd', 'N/A')}")
    print(f"  Error message: {result.get('errMsg', 'N/A')}")
    if result.get("errMsg") == "Success":
        print(f"  ✓ SUCCESS! Use code format: {code}")
        print(f"  Result count: {len(result.get('result', []))}")
        if result.get('result'):
            print(f"  Sample data: {json.dumps(result['result'][0], indent=4, ensure_ascii=False)}")
        break

print("\n" + "=" * 60)
print("Diagnostic complete!")
print("=" * 60)
