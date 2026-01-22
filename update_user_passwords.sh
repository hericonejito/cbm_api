#!/bin/bash
# Script to regenerate password hashes in users.json
# Run this inside your Docker container or locally after installing dependencies

echo "Regenerating password hashes..."
echo "This will update all users to use password: changeme123"
echo ""

python3 << 'EOF'
import os
import sys
import json

# Add app directory to path
sys.path.insert(0, '/app')

from app.auth import get_password_hash, load_users_db, save_users_db

default_password = "changeme123"

db = load_users_db()
users = db.get("users", {})

if not users:
    print("No users found in database")
    sys.exit(1)

print(f"Regenerating password hashes for {len(users)} users...")
print(f"Default password: {default_password}")
print()

for username, user_data in users.items():
    print(f"Regenerating hash for user: {username}")
    new_hash = get_password_hash(default_password)
    user_data["hashed_password"] = new_hash
    print(f"  New hash: {new_hash[:50]}...")

db["users"] = users
save_users_db(db)

print()
print("✅ Password hashes regenerated successfully!")
print(f"All users now have password: {default_password}")
print()
print("⚠️  IMPORTANT: Change passwords after first login!")
EOF
