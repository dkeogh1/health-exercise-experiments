#!/usr/bin/env python3
"""
Export all Strava activities to CSV/Parquet.
Handles pagination and token refresh.
"""

import os
import sys
import json
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load .env
ENV_PATH = Path(__file__).parent.parent / "config" / ".env"
load_dotenv(ENV_PATH)

ACCESS_TOKEN = os.getenv("STRAVA_ACCESS_TOKEN")
REFRESH_TOKEN = os.getenv("STRAVA_REFRESH_TOKEN")
CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")

if not all([ACCESS_TOKEN, REFRESH_TOKEN, CLIENT_ID, CLIENT_SECRET]):
    print("❌ Missing credentials in config/.env")
    print("Run: python scripts/strava_auth.py")
    sys.exit(1)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def refresh_access_token():
    """Refresh expired access token using refresh token"""
    global ACCESS_TOKEN
    
    url = "https://www.strava.com/oauth/token"
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": REFRESH_TOKEN,
    }
    
    req = urllib.request.Request(
        url,
        data=urllib.parse.urlencode(data).encode("utf-8"),
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            ACCESS_TOKEN = result["access_token"]
            
            # Update .env
            with open(ENV_PATH, "r") as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                if line.startswith("STRAVA_ACCESS_TOKEN="):
                    new_lines.append(f"STRAVA_ACCESS_TOKEN={ACCESS_TOKEN}\n")
                else:
                    new_lines.append(line)
            
            with open(ENV_PATH, "w") as f:
                f.writelines(new_lines)
            
            print("✅ Access token refreshed")
            return True
    except urllib.error.HTTPError as e:
        print(f"❌ Failed to refresh token: {e}")
        return False

def fetch_activities(page=1, per_page=200):
    """Fetch activities from Strava API"""
    url = f"https://www.strava.com/api/v3/athlete/activities?page={page}&per_page={per_page}"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 401:  # Unauthorized
            print("⚠️  Access token expired, refreshing...")
            if refresh_access_token():
                return fetch_activities(page, per_page)
        raise

def flatten_activity(activity):
    """Flatten nested activity object for CSV"""
    return {
        "id": activity.get("id"),
        "name": activity.get("name"),
        "type": activity.get("type"),
        "sport_type": activity.get("sport_type"),
        "start_date": activity.get("start_date"),
        "distance_m": activity.get("distance"),
        "duration_s": activity.get("moving_time"),
        "elevation_m": activity.get("total_elevation_gain"),
        "average_speed_ms": activity.get("average_speed"),
        "max_speed_ms": activity.get("max_speed"),
        "average_hr": activity.get("average_heartrate"),
        "max_hr": activity.get("max_heartrate"),
        "average_watts": activity.get("average_watts"),
        "kilojoules": activity.get("kilojoules"),
        "calories": activity.get("calories"),
        "device_name": activity.get("device_name"),
        "visibility": activity.get("visibility"),
    }

def export_to_csv(activities):
    """Export activities to CSV"""
    try:
        import csv
    except ImportError:
        print("❌ csv module not found")
        return False
    
    output_file = DATA_DIR / f"activities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    if not activities:
        print("⚠️  No activities to export")
        return False
    
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=activities[0].keys())
        writer.writeheader()
        writer.writerows(activities)
    
    print(f"✅ Exported {len(activities)} activities to {output_file}")
    return True

def export_to_json(activities):
    """Export activities to JSON"""
    output_file = DATA_DIR / f"activities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, "w") as f:
        json.dump(activities, f, indent=2)
    
    print(f"✅ Exported {len(activities)} activities to {output_file}")
    return True

def main():
    print("📊 Strava Data Export")
    print(f"Data directory: {DATA_DIR}\n")
    
    activities = []
    page = 1
    per_page = 200
    
    try:
        while True:
            print(f"📥 Fetching page {page}...")
            batch = fetch_activities(page, per_page)
            
            if not batch:
                print(f"✅ Reached end of data at page {page}")
                break
            
            # Flatten each activity
            flattened = [flatten_activity(a) for a in batch]
            activities.extend(flattened)
            print(f"   Got {len(batch)} activities (total: {len(activities)})")
            
            page += 1
    
    except KeyboardInterrupt:
        print("\n⏸️  Export interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    if not activities:
        print("❌ No activities exported")
        sys.exit(1)
    
    # Export formats
    export_to_json(activities)
    export_to_csv(activities)
    
    print(f"\n✨ Export complete: {len(activities)} total activities")

if __name__ == "__main__":
    main()
