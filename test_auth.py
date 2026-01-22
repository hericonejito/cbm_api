#!/usr/bin/env python3
"""
Quick test script for authentication functionality.
Run with: python test_auth.py
"""

import requests
import sys

API_BASE_URL = "http://localhost:8000"

def test_auth():
    print("=" * 50)
    print("Testing CBM API Authentication")
    print("=" * 50)

    # Test 1: Check API is running
    print("\n1. Checking API connectivity...")
    try:
        response = requests.get(f"{API_BASE_URL}/docs", timeout=5)
        if response.status_code == 200:
            print("   ✅ API is running")
        else:
            print(f"   ❌ API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to API. Is it running on localhost:8000?")
        return False

    # Test 2: Verify protected endpoint requires auth
    print("\n2. Testing protected endpoint without token...")
    response = requests.get(f"{API_BASE_URL}/videos")
    if response.status_code == 401:
        print("   ✅ Protected endpoint correctly requires authentication")
    else:
        print(f"   ❌ Expected 401, got {response.status_code}")
        return False

    # Test 3: Login with valid credentials
    print("\n3. Testing login with valid credentials...")
    response = requests.post(
        f"{API_BASE_URL}/token",
        data={"username": "stergios", "password": "changeme123"},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    if response.status_code == 200:
        token_data = response.json()
        token = token_data.get("access_token")
        print("   ✅ Login successful!")
        print(f"   Token: {token[:50]}...")
    else:
        print(f"   ❌ Login failed: {response.text}")
        return False

    # Test 4: Login with invalid credentials
    print("\n4. Testing login with invalid credentials...")
    response = requests.post(
        f"{API_BASE_URL}/token",
        data={"username": "stergios", "password": "wrongpassword"},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    if response.status_code == 401:
        print("   ✅ Invalid credentials correctly rejected")
    else:
        print(f"   ❌ Expected 401, got {response.status_code}")
        return False

    # Test 5: Access protected endpoint with token
    print("\n5. Testing protected endpoint with token...")
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_BASE_URL}/videos", headers=headers)

    if response.status_code == 200:
        print("   ✅ Protected endpoint accessible with valid token")
        print(f"   Response: {response.json()}")
    else:
        print(f"   ❌ Expected 200, got {response.status_code}: {response.text}")
        return False

    # Test 6: Get current user info
    print("\n6. Testing /users/me endpoint...")
    response = requests.get(f"{API_BASE_URL}/users/me", headers=headers)

    if response.status_code == 200:
        user_data = response.json()
        print("   ✅ User info retrieved successfully")
        print(f"   Username: {user_data.get('username')}")
        print(f"   Full Name: {user_data.get('full_name')}")
        print(f"   Permissions: {user_data.get('permissions')}")
    else:
        print(f"   ❌ Expected 200, got {response.status_code}")
        return False

    # Test 7: Test feedback stats (requires 'read' permission)
    print("\n7. Testing /feedback/stats endpoint...")
    response = requests.get(f"{API_BASE_URL}/feedback/stats", headers=headers)

    if response.status_code == 200:
        print("   ✅ Feedback stats retrieved successfully")
    else:
        print(f"   ❌ Expected 200, got {response.status_code}: {response.text}")
        return False

    # Test 8: Test with second user
    print("\n8. Testing login with second user (dimitris)...")
    response = requests.post(
        f"{API_BASE_URL}/token",
        data={"username": "dimitris", "password": "changeme123"},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    if response.status_code == 200:
        print("   ✅ Second user login successful!")
    else:
        print(f"   ❌ Login failed: {response.text}")
        return False

    print("\n" + "=" * 50)
    print("✅ All authentication tests passed!")
    print("=" * 50)

    print("\n📋 Quick Reference:")
    print("-" * 50)
    print("Users available:")
    print("  - stergios / changeme123")
    print("  - dimitris / changeme123")
    print("\nTo get a token:")
    print(f'  curl -X POST "{API_BASE_URL}/token" \\')
    print('    -d "username=stergios&password=changeme123"')
    print("\nTo use the token:")
    print(f'  curl -H "Authorization: Bearer <TOKEN>" "{API_BASE_URL}/videos"')

    return True


if __name__ == "__main__":
    success = test_auth()
    sys.exit(0 if success else 1)
