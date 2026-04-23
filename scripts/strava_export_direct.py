#!/usr/bin/env python3
"""
Strava Data Export (using direct access token)

Fetches all activities from Strava API and exports to CSV.
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

# For direct auth, read from .env
ACCESS_TOKEN = os.getenv("STRAVA_ACCESS_TOKEN", "15ef2f1d5b6dcd9608e6e4bd789cc3f186ce02fe")

ACTIVITIES_URL = "https://www.strava.com/api/v3/athlete/activities"

DATA_DIR = Path(__file__).parent.parent / "data"


def api_request(url, headers=None):
    """Make a GET request."""
    if headers is None:
        headers = {}
    
    req = Request(url, headers=headers)
    try:
        with urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"Request failed: {e}")
        raise


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
    if not ACCESS_TOKEN:
        print("❌ Missing access token")
        sys.exit(1)

    print("\n📊 Strava Data Export")
    print("=" * 50)

    try:
        activities = fetch_all_activities(ACCESS_TOKEN)
        export_to_csv(activities)

        print("\n🎉 Done!")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
