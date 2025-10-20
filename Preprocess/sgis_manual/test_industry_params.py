"""
Test different parameters for Company API to get industry-specific data
"""

import requests
import json

# Your credentials
CONSUMER_KEY = "fbf9612b73e54fac8545"
CONSUMER_SECRET = "0543b74f9984418da672"

print("=" * 60)
print("Testing Company API Parameters for Industry Data")
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

# Test dong code (Sinsa-dong, Gangnam)
test_code = "11230510"
company_url = "https://sgisapi.kostat.go.kr/OpenAPI3/stats/company.json"

print("\n[2] Testing different parameter combinations...")
print("=" * 60)

# Test 1: Default parameters (what we've been using)
print("\nTest 1: Default parameters")
params = {
    "accessToken": access_token,
    "adm_cd": test_code,
    "low_search": "1",
    "year": "2020"
}
response = requests.get(company_url, params=params)
result = response.json()
print(f"  Status: {result.get('errMsg')}")
if result.get('result'):
    print(f"  Result count: {len(result['result'])}")
    print(f"  Sample keys: {list(result['result'][0].keys())}")

# Test 2: Try low_search = 0 (summary level)
print("\nTest 2: low_search=0 (summary level)")
params = {
    "accessToken": access_token,
    "adm_cd": test_code,
    "low_search": "0",
    "year": "2020"
}
response = requests.get(company_url, params=params)
result = response.json()
print(f"  Status: {result.get('errMsg')}")
if result.get('result'):
    print(f"  Result count: {len(result['result'])}")
    print(f"  Sample keys: {list(result['result'][0].keys())}")
    print(f"  Sample record:")
    print(json.dumps(result['result'][0], indent=4, ensure_ascii=False))

# Test 3: Try without year parameter
print("\nTest 3: Without year parameter (latest data)")
params = {
    "accessToken": access_token,
    "adm_cd": test_code,
    "low_search": "1"
}
response = requests.get(company_url, params=params)
result = response.json()
print(f"  Status: {result.get('errMsg')}")
if result.get('result'):
    print(f"  Result count: {len(result['result'])}")
    print(f"  Sample keys: {list(result['result'][0].keys())}")

# Test 4: Try industry_cl parameter (if it exists)
print("\nTest 4: Try with industry classification parameter")
params = {
    "accessToken": access_token,
    "adm_cd": test_code,
    "low_search": "1",
    "year": "2020",
    "induty_cl_code": "55"  # Try accommodation industry
}
response = requests.get(company_url, params=params)
result = response.json()
print(f"  Status: {result.get('errMsg')}")
print(f"  Error code: {result.get('errCd')}")
if result.get('result'):
    print(f"  Result count: {len(result['result'])}")
    print(f"  Sample keys: {list(result['result'][0].keys())}")

# Test 5: Check if there's a different key in the result that has industry data
print("\nTest 5: Detailed inspection of a single record")
params = {
    "accessToken": access_token,
    "adm_cd": test_code,
    "low_search": "1",
    "year": "2020"
}
response = requests.get(company_url, params=params)
result = response.json()
if result.get('result'):
    print(f"  Full first record:")
    print(json.dumps(result['result'][0], indent=4, ensure_ascii=False))
    print(f"\n  All unique keys across all records:")
    all_keys = set()
    for record in result['result']:
        all_keys.update(record.keys())
    print(f"  {sorted(all_keys)}")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
