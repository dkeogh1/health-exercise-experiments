#!/usr/bin/env python3
"""
Strava OAuth authentication flow.
Saves access token + refresh token to config/.env
"""

import os
import sys
import json
import urllib.parse
import http.server
import socketserver
import webbrowser
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

auth_code = None
server = None

class AuthHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        global auth_code, server
        
        if "/auth/callback" in self.path:
            # Extract auth code from query params
            parsed_url = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed_url.query)
            
            if "code" in params:
                auth_code = params["code"][0]
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(b"<h1>Success!</h1><p>You can close this window.</p>")
                print(f"✅ Authorization code received: {auth_code[:10]}...")
                # Shutdown server after handling
                server.shutdown()
            else:
                self.send_response(400)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(b"<h1>Error</h1><p>No auth code found.</p>")
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress request logs
        pass

def get_auth_code():
    global server
    
    # Start local server
    PORT = 8000
    Handler = AuthHandler
    server = socketserver.TCPServer(("", PORT), Handler)
    
    print(f"🌐 Starting OAuth callback server on http://localhost:{PORT}")
    print("Opening browser for authorization...")
    
    # Build authorization URL
    auth_url = f"https://www.strava.com/oauth/authorize?client_id={CLIENT_ID}&response_type=code&redirect_uri={REDIRECT_URI}&scope=activity:read_all"
    
    # Open browser
    webbrowser.open(auth_url)
    
    # Wait for callback
    server.handle_request()
    
    return auth_code

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
    
    updated = False
    new_lines = []
    for line in lines:
        added = False
        for key, value in fields.items():
            if line.startswith(key + "="):
                new_lines.append(f"{key}={value}\n")
                fields.pop(key)
                added = True
                updated = True
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
    print("🔐 Strava OAuth Flow")
    print(f"Client ID: {CLIENT_ID}")
    print()
    
    # Step 1: Get authorization code
    code = get_auth_code()
    if not code:
        print("❌ No authorization code received")
        sys.exit(1)
    
    # Step 2: Exchange code for tokens
    print("\n🔄 Exchanging code for tokens...")
    token_data = exchange_code_for_token(code)
    
    # Step 3: Save tokens
    save_tokens(token_data)
    
    print("\n✨ Authentication complete!")
    print(f"Access token expires in: {token_data.get('expires_in')} seconds")
    print("\nNext step: python scripts/strava_export.py")

if __name__ == "__main__":
    main()
