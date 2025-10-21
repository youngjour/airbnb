"""
Test different parameter combinations for the startup business API
"""

import requests
import json

# Your credentials
CONSUMER_KEY = "fbf9612b73e54fac8545"
CONSUMER_SECRET = "0543b74f9984418da672"

print("=" * 70)
print("Testing Different Parameter Combinations")
print("=" * 70)

# Authenticate
auth_url = "https://sgisapi.kostat.go.kr/OpenAPI3/auth/authentication.json"
response = requests.get(auth_url, params={
    "consumer_key": CONSUMER_KEY,
    "consumer_secret": CONSUMER_SECRET
})

access_token = response.json()["result"]["accessToken"]
print(f"[OK] Access token: {access_token[:20]}...")

url = "https://sgisapi.kostat.go.kr/OpenAPI3/startupbiz/corpdistsummary.json"
test_code = "11230510"  # Sinsa-dong, Gangnam

# Try different parameter combinations
param_combinations = [
    {"accessToken": access_token, "adm_cd": test_code},
    {"accessToken": access_token, "adm_cd": test_code, "year": "2020"},
    {"accessToken": access_token, "adm_cd": test_code, "year": "2021"},
    {"accessToken": access_token, "adm_cd": test_code, "year": "2022"},
    {"accessToken": access_token, "cd": test_code},
    {"accessToken": access_token, "cd": test_code, "year": "2020"},
    {"accessToken": access_token, "admcode": test_code},
    {"accessToken": access_token, "dong_cd": test_code},
]

for i, params in enumerate(param_combinations, 1):
    print(f"\n[Test {i}] Parameters: {params}")
    print("-" * 70)

    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"HTTP Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Error Code: {result.get('errCd', 'N/A')}")
            print(f"Error Message: {result.get('errMsg', 'N/A')}")

            if result.get('errMsg') == 'Success':
                print("\n>>> SUCCESS!")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                break
            else:
                if result.get('errMsg'):
                    print(f"API Error: {result.get('errMsg')}")
        else:
            print(f"HTTP Error: {response.text[:200]}")

    except Exception as e:
        print(f"Exception: {e}")

print("\n" + "=" * 70)
print("Test Complete")
print("=" * 70)
