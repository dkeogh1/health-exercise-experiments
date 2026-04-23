#!/usr/bin/env python3
"""
Manual Strava OAuth flow for headless environments.
You'll need to:
1. Run this script
2. Copy the authorization URL
3. Open it in a browser on YOUR machine
4. Paste the authorization code back here
"""

import os
import sys
import json
import urllib.parse
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

# Load existing .env if present
ENV_PATH = Path(__file__).parent.parent / "config" / ".env"
load_dotenv(ENV_PATH)

CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
REDIRECT_URI = "http://localhost:8000/auth/callback"

if not CLIENT_ID or not CLIENT_SECRET:
    print("❌ Error: STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET not found in config/.env")
    print("Fill in config/.env first, then run again.")
    sys.exit(1)

def get_auth_url():
    """Generate the Strava authorization URL"""
    auth_url = f"https://www.strava.com/oauth/authorize?client_id={CLIENT_ID}&response_type=code&redirect_uri={REDIRECT_URI}&scope=activity:read_all"
    return auth_url

def exchange_code_for_token(code):
    """Exchange auth code for access token"""
    url = "https://www.strava.com/oauth/token"
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
    }
    
    req = urllib.request.Request(
        url,
        data=urllib.parse.urlencode(data).encode("utf-8"),
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result
    except Exception as e:
        print(f"❌ Error exchanging code: {e}")
        sys.exit(1)

def save_tokens(token_data):
    """Save tokens to config/.env"""
    with open(ENV_PATH, "r") as f:
        lines = f.readlines()
    
    # Update or add token fields
    fields = {
        "STRAVA_ACCESS_TOKEN": token_data["access_token"],
        "STRAVA_REFRESH_TOKEN": token_data["refresh_token"],
    }
    
    new_lines = []
    for line in lines:
        added = False
        for key, value in fields.items():
            if line.startswith(key + "="):
                new_lines.append(f"{key}={value}\n")
                fields.pop(key, None)
                added = True
                break
        if not added:
            new_lines.append(line)
    
    # Add any remaining fields
    for key, value in fields.items():
        new_lines.append(f"{key}={value}\n")
    
    with open(ENV_PATH, "w") as f:
        f.writelines(new_lines)
    
    print(f"✅ Tokens saved to {ENV_PATH}")

def main():
    print("🔐 Strava Manual OAuth Flow (Headless)\n")
    print(f"Client ID: {CLIENT_ID}\n")
    
    # Step 1: Show authorization URL
    auth_url = get_auth_url()
    print("📋 Authorization URL:")
    print(f"\n{auth_url}\n")
    print("⚠️  Copy the URL above and paste it into a browser on your local machine.")
    print("You'll be asked to authorize the app. After authorizing, you'll be redirected")
    print("to a localhost URL (it might show an error, that's OK).\n")
    
    # Step 2: Get auth code from user
    code = input("🔑 Paste the authorization code here: ").strip()
    
    if not code:
        print("❌ No code provided")
        sys.exit(1)
    
    # Step 3: Exchange code for tokens
    print("\n🔄 Exchanging code for tokens...")
    token_data = exchange_code_for_token(code)
    
    # Step 4: Save tokens
    save_tokens(token_data)
    
    print("\n✨ Authentication complete!")
    print(f"Access token expires in: {token_data.get('expires_in')} seconds")
    print("\nNext step: python scripts/strava_export.py")

if __name__ == "__main__":
    main()
