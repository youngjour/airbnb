"""
Explore alternative SGIS API endpoints for industry-specific company data
Based on the manual, we need to find if there's a different endpoint
"""

import requests
import json

# Your credentials
CONSUMER_KEY = "fbf9612b73e54fac8545"
CONSUMER_SECRET = "0543b74f9984418da672"

print("=" * 70)
print("Exploring SGIS API Endpoints for Industry-Specific Data")
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
    print(f"[OK] Got access token")
else:
    print("[FAIL] Authentication failed")
    exit(1)

# Test dong code
test_code = "11230510"
base_url = "https://sgisapi.kostat.go.kr"

# Try different potential endpoints based on common API patterns
potential_endpoints = [
    "/OpenAPI3/stats/companydetail.json",
    "/OpenAPI3/stats/companybyindustry.json",
    "/OpenAPI3/stats/business.json",
    "/OpenAPI3/stats/industrybusiness.json",
    "/OpenAPI3/stats/establishment.json",
    "/OpenAPI3/boundary/hadmarea.json",  # Administrative boundary - might have data
]

print("\n[2] Testing potential endpoints...")
print("=" * 70)

for endpoint in potential_endpoints:
    print(f"\nTesting: {endpoint}")

    params = {
        "accessToken": access_token,
        "adm_cd": test_code,
        "year": "2020"
    }

    try:
        url = base_url + endpoint
        response = requests.get(url, params=params, timeout=10)
        result = response.json()

        print(f"  HTTP Status: {response.status_code}")
        print(f"  Error Code: {result.get('errCd', 'N/A')}")
        print(f"  Error Message: {result.get('errMsg', 'N/A')}")

        if result.get('errMsg') == 'Success' and result.get('result'):
            print(f"  >>> SUCCESS! Found data!")
            print(f"  Result count: {len(result.get('result', []))}")
            if result.get('result'):
                print(f"  Sample keys: {list(result['result'][0].keys())}")
                print(f"  Sample record:")
                print(json.dumps(result['result'][0], indent=4, ensure_ascii=False))
                print("\n" + "!" * 70)
                print("FOUND A WORKING ENDPOINT WITH DATA!")
                print("!" * 70)
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "=" * 70)
print("[3] Checking if we can add industry filter to company API...")
print("=" * 70)

# Try adding different industry-related parameters
industry_params_to_test = [
    {"induty_cl_code": "55"},
    {"industry_code": "55"},
    {"biz_code": "55"},
    {"class_code": "55"},
    {"cp2_bnu_55": "1"},
    {"detail_yn": "Y"},
    {"detail": "1"},
    {"industry": "55"},
]

company_url = "https://sgisapi.kostat.go.kr/OpenAPI3/stats/company.json"

for extra_params in industry_params_to_test:
    print(f"\nTesting params: {extra_params}")

    params = {
        "accessToken": access_token,
        "adm_cd": test_code,
        "low_search": "1",
        "year": "2020"
    }
    params.update(extra_params)

    try:
        response = requests.get(company_url, params=params, timeout=10)
        result = response.json()

        if result.get('errMsg') == 'Success':
            print(f"  Success! Checking if data differs...")
            if result.get('result'):
                print(f"  Result count: {len(result['result'])}")
                print(f"  Sample keys: {list(result['result'][0].keys())}")
                # Check if we got more columns
                if len(result['result'][0].keys()) > 3:  # More than corp_cnt, tot_worker, adm_cd
                    print(f"  >>> FOUND ADDITIONAL COLUMNS!")
                    print(json.dumps(result['result'][0], indent=4, ensure_ascii=False))
        else:
            print(f"  Failed: {result.get('errMsg')}")
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "=" * 70)
print("Exploration Complete!")
print("=" * 70)

print("\n[CONCLUSION]")
print("If no alternative endpoints were found, the industry-specific data")
print("is likely ONLY available through:")
print("  1. SGIS Portal web interface (manual data request)")
print("  2. Statistics Data Center (SDC) bulk downloads")
print("  3. Pre-made statistical tables")
