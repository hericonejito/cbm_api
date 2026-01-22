#!/usr/bin/env python3
"""
Script to regenerate password hashes for users.json
This is needed when bcrypt version changes or password hashes become incompatible.
"""

import os
import json
import sys

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.auth import get_password_hash, load_users_db, save_users_db

def regenerate_passwords():
    """Regenerate password hashes for all users with the default password"""
    default_password = "changeme123"
    
    db = load_users_db()
    users = db.get("users", {})
    
    if not users:
        print("No users found in database")
        return
    
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

if __name__ == "__main__":
    regenerate_passwords()
