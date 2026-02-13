import json
import os
from pathlib import Path
from typing import Optional, Dict

# Path to users JSON
USERS_FILE = Path(__file__).parent / "users.json"

def load_users() -> Dict:
    """Load users from JSON file"""
    if not USERS_FILE.exists():
        # Default users if file doesn't exist for some reason
        return {
            "users": [
                {"username": "admin", "password": "password123", "name": "Administrator"}
            ]
        }
    
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def authenticate(username, password) -> Optional[Dict]:
    """
    Validate credentials and return user info if successful
    """
    data = load_users()
    for user in data.get("users", []):
        if user.get("username") == username and user.get("password") == password:
            return {
                "username": user.get("username"),
                "name": user.get("name")
            }
    return None
