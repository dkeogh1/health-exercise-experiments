#!/usr/bin/env python3
"""
Strava Data Export (no external deps except built-ins)

Fetches all activities from Strava API and exports to CSV.
Requires a valid refresh token.
"""

import os
import sys
import json
import csv
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlencode

# Load .env manually
env_path = Path(__file__).parent.parent / "config" / ".env"
env_vars = {}
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                env_vars[key] = val

CLIENT_ID = env_vars.get("STRAVA_CLIENT_ID")
CLIENT_SECRET = env_vars.get("STRAVA_CLIENT_SECRET")
REFRESH_TOKEN = env_vars.get("STRAVA_REFRESH_TOKEN")

TOKEN_URL = "https://www.strava.com/oauth/token"
ACTIVITIES_URL = "https://www.strava.com/api/v3/athlete/activities"

DATA_DIR = Path(__file__).parent.parent / "data"


def api_request(url, method="GET", headers=None, data=None):
    """Make a simple HTTP request."""
    if headers is None:
        headers = {}
    
    if data is not None:
        data = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    
    req = Request(url, data=data, headers=headers, method=method)
    with urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))


def refresh_access_token():
    """Get a new access token using the refresh token."""
    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": REFRESH_TOKEN,
    }
    result = api_request(TOKEN_URL, method="POST", data=payload)
    return result["access_token"]


def fetch_all_activities(access_token):
    """Fetch all activities with pagination."""
    activities = []
    page = 1
    per_page = 200

    headers = {"Authorization": f"Bearer {access_token}"}

    print(f"📥 Fetching activities...")

    while True:
        params = urlencode({"page": page, "per_page": per_page})
        url = f"{ACTIVITIES_URL}?{params}"
        
        batch = api_request(url, headers=headers)
        
        if not batch:
            break

        activities.extend(batch)
        print(f"   Page {page}: {len(batch)} activities ({len(activities)} total)")
        page += 1

    return activities


def export_to_csv(activities):
    """Export activities to CSV."""
    if not activities:
        print("❌ No activities to export.")
        return

    # Flatten structure: keep useful fields
    rows = []
    for act in activities:
        rows.append({
            "id": act.get("id"),
            "date": act.get("start_date"),
            "name": act.get("name"),
            "type": act.get("type"),
            "distance_m": act.get("distance"),
            "duration_s": act.get("moving_time"),
            "elevation_m": act.get("total_elevation_gain"),
            "avg_speed_ms": act.get("average_speed"),
            "max_speed_ms": act.get("max_speed"),
            "avg_hr": act.get("average_heartrate"),
            "max_hr": act.get("max_heartrate"),
            "calories": act.get("calories"),
            "gear_id": act.get("gear_id"),
            "commute": act.get("commute"),
            "trainer": act.get("trainer"),
            "kudos": act.get("kudos_count"),
        })

    output_path = DATA_DIR / "activities.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Exported {len(rows)} activities to {output_path}")


def main():
    """Run the export."""
    if not all([CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN]):
        print("❌ Missing credentials in config/.env")
        sys.exit(1)

    print("\n📊 Strava Data Export")
    print("=" * 50)

    try:
        print("\n🔑 Refreshing access token...")
        access_token = refresh_access_token()
        print(f"   ✓ Token refreshed")

        activities = fetch_all_activities(access_token)
        export_to_csv(activities)

        print("\n🎉 Done!")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
