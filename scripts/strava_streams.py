#!/usr/bin/env python3
"""Pull per-second streams (HR, power, cadence, lat/lng, altitude, distance,
temperature) for every Strava activity ID and save as Parquet.

Output layout:
    data/processed/streams/strava_<id>.parquet   # one-row-per-sample

Writes progress to ``data/processed/streams/_strava_progress.json`` so the
pull is resumable. Skips activities whose Parquet already exists.

Strava rate limits (as of early 2026): 100 requests / 15 min, 1000 / day per
access token. We watch for HTTP 429 and back off, and also self-throttle so a
single session doesn't burn the daily budget in one shot. A full pull of
~1,650 activities takes roughly 2 h wall time at 100 requests / 15 min; a
daily pull of the last ~200 activities takes a few minutes.

Usage:
    python scripts/strava_streams.py                    # pull everything missing
    python scripts/strava_streams.py --limit 100        # only the first 100 missing
    python scripts/strava_streams.py --since 2024-01-01 # only activities on/after
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / "config" / ".env"
RAW_DIR = ROOT / "data" / "raw"
STREAMS_DIR = ROOT / "data" / "processed" / "streams"
PROGRESS_FILE = STREAMS_DIR / "_strava_progress.json"

load_dotenv(ENV_PATH)

CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
ACCESS_TOKEN = os.getenv("STRAVA_ACCESS_TOKEN")
REFRESH_TOKEN = os.getenv("STRAVA_REFRESH_TOKEN")

STREAM_KEYS = (
    "time,latlng,altitude,distance,velocity_smooth,"
    "heartrate,cadence,watts,temp,moving,grade_smooth"
)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def refresh_token() -> None:
    """Exchange the refresh token for a new access token; update .env."""
    global ACCESS_TOKEN
    if not all([CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN]):
        sys.exit("Missing STRAVA_* creds in config/.env")

    data = urllib.parse.urlencode({
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": REFRESH_TOKEN,
    }).encode()
    req = urllib.request.Request("https://www.strava.com/oauth/token", data=data, method="POST")
    with urllib.request.urlopen(req) as resp:
        j = json.loads(resp.read())
    ACCESS_TOKEN = j["access_token"]

    # Persist back to .env so subsequent script runs pick it up.
    lines = ENV_PATH.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("STRAVA_ACCESS_TOKEN="):
            lines[i] = f"STRAVA_ACCESS_TOKEN={ACCESS_TOKEN}"
            break
    ENV_PATH.write_text("\n".join(lines) + "\n")
    print("  (refreshed STRAVA_ACCESS_TOKEN)")


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def get(url: str) -> tuple[int, dict | list | None, dict]:
    """GET with the current access token. Returns (status, body, headers)."""
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {ACCESS_TOKEN}"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read()), dict(resp.headers)
    except urllib.error.HTTPError as e:
        body = None
        try:
            body = json.loads(e.read())
        except Exception:
            pass
        return e.code, body, dict(e.headers or {})


def fetch_streams(activity_id: int) -> list[dict] | None:
    """Return the raw streams JSON for an activity, or None on hard failure."""
    url = (
        f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
        f"?keys={STREAM_KEYS}&key_by_type=false"
    )
    attempts = 0
    while attempts < 3:
        status, body, headers = get(url)
        if status == 200:
            # Strava returns a list when ?key_by_type=false is honoured, or a
            # dict keyed by stream type otherwise. Accept both.
            if isinstance(body, list):
                return body
            if isinstance(body, dict):
                return [{"type": k, **v} for k, v in body.items()]
            return None
        if status == 401:
            refresh_token()
            attempts += 1
            continue
        if status == 429:
            # Respect Strava's retry-after hint, then fall back to 15 min.
            retry = int(headers.get("Retry-After", "900"))
            print(f"  429: sleeping {retry}s before retry")
            time.sleep(retry)
            attempts += 1
            continue
        if status == 404:
            # Activity was deleted / private / missing streams.
            return []
        # Short one-liner; never dump the response body (it can be 100s of KB).
        err = (body or {}).get("message", "") if isinstance(body, dict) else ""
        print(f"  activity {activity_id}: HTTP {status} {err}")
        return None
    return None


# ---------------------------------------------------------------------------
# Stream → DataFrame
# ---------------------------------------------------------------------------


def streams_to_df(streams_json: list[dict], activity_id: int, start_iso: str) -> pd.DataFrame:
    """Strava streams are a list of {type, data, ...}. Convert to tidy DataFrame."""
    if not streams_json:
        return pd.DataFrame()
    by_type = {s["type"]: s["data"] for s in streams_json if "type" in s and "data" in s}
    n = max(len(v) for v in by_type.values())

    out = pd.DataFrame(index=range(n))
    if "time" in by_type:
        start = pd.Timestamp(start_iso).tz_convert("UTC") if pd.Timestamp(start_iso).tzinfo \
            else pd.Timestamp(start_iso, tz="UTC")
        out["timestamp"] = start + pd.to_timedelta(by_type["time"], unit="s")
    out["activity_id"] = activity_id
    rename = {
        "latlng": None,  # split into lat/lng
        "altitude": "altitude",
        "distance": "distance",
        "velocity_smooth": "speed",
        "heartrate": "heart_rate",
        "cadence": "cadence",
        "watts": "power",
        "temp": "temperature",
        "moving": "moving",
        "grade_smooth": "grade",
    }
    for src, dst in rename.items():
        if src == "latlng" and "latlng" in by_type:
            arr = by_type["latlng"]
            out["lat"] = [p[0] if p else None for p in arr]
            out["lng"] = [p[1] if p else None for p in arr]
        elif dst and src in by_type:
            out[dst] = by_type[src]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {}


def save_progress(state: dict) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(state, indent=2, default=str))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="Max activities to process in this run")
    ap.add_argument("--since", type=str, default=None,
                    help="Only activities on/after YYYY-MM-DD")
    ap.add_argument("--throttle-s", type=float, default=9.5,
                    help="Self-throttle between requests (sec). Default 9.5 stays under 100/15min.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-fetch even if the Parquet already exists")
    args = ap.parse_args()

    # Pick up the latest Strava activity summary file.
    strava_jsons = sorted(RAW_DIR.glob("activities_*.json"), reverse=True)
    if not strava_jsons:
        sys.exit("No activities_*.json in data/raw/. Run scripts/strava_export.py first.")
    with open(strava_jsons[0]) as f:
        acts = json.load(f)

    if args.since:
        cutoff = pd.Timestamp(args.since, tz="UTC")
        acts = [a for a in acts if pd.Timestamp(a["start_date"]) >= cutoff]

    # Order oldest-first so an interruption leaves the newer tail intact.
    acts.sort(key=lambda a: a["start_date"])

    STREAMS_DIR.mkdir(parents=True, exist_ok=True)
    state = load_progress()
    pulled = 0
    skipped = 0
    failures: list[int] = []

    for a in acts:
        aid = a["id"]
        path = STREAMS_DIR / f"strava_{aid}.parquet"
        if path.exists() and not args.overwrite:
            skipped += 1
            continue
        if args.limit is not None and pulled >= args.limit:
            break

        streams = fetch_streams(aid)
        if streams is None:
            failures.append(aid)
            continue
        df = streams_to_df(streams, aid, a["start_date"])
        if df.empty:
            # Mark explicitly so we don't re-probe every run.
            state.setdefault("empty", []).append(aid)
            state["empty"] = sorted(set(state["empty"]))
            save_progress(state)
            continue
        df.to_parquet(path, index=False)
        pulled += 1
        if pulled % 10 == 0:
            print(f"  pulled {pulled}  skipped {skipped}  failures {len(failures)}  "
                  f"(last: {a['name'][:40]!r} @ {a['start_date'][:10]})")
        state["last_pulled_id"] = aid
        state["last_run"] = datetime.now(timezone.utc).isoformat()
        save_progress(state)
        time.sleep(args.throttle_s)

    print(f"\n✓ Done.  pulled={pulled}  skipped={skipped}  failures={len(failures)}")
    if failures:
        print(f"  failure ids: {failures[:10]}{'...' if len(failures) > 10 else ''}")


if __name__ == "__main__":
    main()
