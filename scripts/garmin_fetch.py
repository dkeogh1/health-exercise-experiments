#!/usr/bin/env python3
"""Fetch Garmin Connect activities via ``python-garminconnect``.

Writes one row per activity to ``data/raw/garmin_activities_<stamp>.parquet``.
Session/token cache lives in ``~/.garminconnect`` (library default) so
subsequent runs skip the full login flow.

Usage:
    python scripts/garmin_fetch.py                  # resume or first-time login
    python scripts/garmin_fetch.py --limit 500      # latest N activities
    python scripts/garmin_fetch.py --refresh-token  # force re-login

Library choice: we originally used ``garth`` but it was upstream-deprecated in
2025 when Garmin deployed Cloudflare/mobile-client anti-bot measures that now
return 429 on ``sso.garmin.com/mobile/api/login``. ``garminconnect`` bundles
``curl_cffi`` for TLS fingerprint impersonation and falls back from the
mobile login to a web login, which currently succeeds. Expect this to be
fragile; if the API breaks again, fall back to Garmin's GDPR data archive.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

try:
    from garminconnect import Garmin
except ImportError:
    sys.exit(
        "garminconnect not installed. Run: pip install garminconnect"
    )

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / "config" / ".env"
DATA_DIR = ROOT / "data" / "raw"
TOKEN_DIR = Path.home() / ".garminconnect"  # library default; respected by Garmin()

load_dotenv(ENV_PATH)
GARMIN_EMAIL = os.getenv("GARMIN_EMAIL")
GARMIN_PASSWORD = os.getenv("GARMIN_PASSWORD")


def build_client(refresh: bool = False) -> Garmin:
    """Resume cached session if present; otherwise log in fresh.

    ``Garmin.login(tokenstore=path)`` auto-saves a new session and
    auto-resumes an existing one when the same path is passed. Its
    return value is ``(None, None)`` on success and an MFA state tuple
    when MFA is required; we opt-in to ``return_on_mfa=True`` so we can
    prompt for the code instead of blocking on ``input()`` inside the
    library.
    """
    if not GARMIN_EMAIL or not GARMIN_PASSWORD:
        sys.exit(
            "GARMIN_EMAIL/GARMIN_PASSWORD not in config/.env; cannot log in."
        )

    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    if refresh:
        # Force a fresh login by wiping the token dir.
        for f in TOKEN_DIR.iterdir():
            f.unlink()

    client = Garmin(
        email=GARMIN_EMAIL,
        password=GARMIN_PASSWORD,
        is_cn=False,
        return_on_mfa=True,
    )
    result = client.login(tokenstore=str(TOKEN_DIR))

    # MFA handshake: `return_on_mfa=True` yields ("needs_mfa", client_state).
    if isinstance(result, tuple) and result and result[0] == "needs_mfa":
        code = input("Garmin MFA code: ").strip()
        client.resume_login(result[1], code)

    try:
        who = client.display_name or client.full_name or "<unknown>"
    except Exception:
        who = "<unknown>"
    print(f"✓ Garmin session ready (user={who}); token dir={TOKEN_DIR}")
    return client


def fetch_activities(client: Garmin, limit: int, page_size: int = 100) -> list[dict]:
    """Pull activity summaries, newest first. Stops early if Garmin returns < page."""
    activities: list[dict] = []
    start = 0
    while len(activities) < limit:
        batch_size = min(page_size, limit - len(activities))
        page = client.get_activities(start, batch_size)
        if not page:
            break
        activities.extend(page)
        print(f"  page start={start}: +{len(page)} (total {len(activities)})")
        if len(page) < batch_size:
            break
        start += batch_size
    return activities[:limit]


def _first(d: dict, *keys):
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def flatten(a: dict) -> dict:
    """Flatten a Garmin activity summary to a column-friendly dict.

    Dict-valued fields (activityType, hrTimeInZone) are JSON-encoded so the
    row fits cleanly into Parquet.
    """
    atype = a.get("activityType") or {}
    hr_zones = a.get("hrTimeInZone") or {}
    return {
        "id": a.get("activityId"),
        "start_time_local": a.get("startTimeLocal"),
        "start_time_gmt": a.get("startTimeGMT"),
        "activity_name": a.get("activityName"),
        "activity_type": atype.get("typeKey"),
        "parent_type": atype.get("parentTypeId"),
        "distance_m": a.get("distance"),
        "duration_s": a.get("duration"),
        "elapsed_s": a.get("elapsedDuration"),
        "moving_s": a.get("movingDuration"),
        "elevation_gain_m": a.get("elevationGain"),
        "elevation_loss_m": a.get("elevationLoss"),
        "avg_speed_ms": a.get("averageSpeed"),
        "max_speed_ms": a.get("maxSpeed"),
        "avg_hr": a.get("averageHR"),
        "max_hr": a.get("maxHR"),
        "avg_power": a.get("avgPower"),
        "max_power": a.get("maxPower"),
        "normalized_power": a.get("normPower"),
        "training_stress_score": a.get("trainingStressScore"),
        "intensity_factor": a.get("intensityFactor"),
        "avg_cadence": _first(
            a,
            "averageRunningCadenceInStepsPerMinute",
            "averageBikingCadenceInRevPerMinute",
            "averageCadence",
        ),
        "max_cadence": _first(
            a,
            "maxRunningCadenceInStepsPerMinute",
            "maxBikingCadenceInRevPerMinute",
            "maxCadence",
        ),
        "calories": a.get("calories"),
        "aerobic_training_effect": a.get("aerobicTrainingEffect"),
        "anaerobic_training_effect": a.get("anaerobicTrainingEffect"),
        "training_effect_label": a.get("trainingEffectLabel"),
        "vo2_max": a.get("vO2MaxValue"),
        "device_id": a.get("deviceId"),
        "hr_time_in_zones_json": json.dumps(hr_zones) if hr_zones else None,
        "activity_type_json": json.dumps(atype) if atype else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=10_000,
                    help="Max activities to fetch (newest first). Default 10k.")
    ap.add_argument("--refresh-token", action="store_true",
                    help="Force re-login even if a cached session exists.")
    args = ap.parse_args()

    client = build_client(refresh=args.refresh_token)

    print(f"Fetching up to {args.limit:,} activities...")
    raw = fetch_activities(client, limit=args.limit)
    if not raw:
        sys.exit("No activities returned")

    df = pd.DataFrame([flatten(a) for a in raw])
    df["start_time_local"] = pd.to_datetime(df["start_time_local"], errors="coerce")
    df["start_time_gmt"] = pd.to_datetime(df["start_time_gmt"], errors="coerce", utc=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_parquet = DATA_DIR / f"garmin_activities_{stamp}.parquet"
    df.to_parquet(out_parquet, index=False)
    size_mb = out_parquet.stat().st_size / 1e6
    print(f"✓ Wrote {len(df):,} rows → {out_parquet} ({size_mb:.1f} MB)")

    # Side-load a CSV too, so the pipeline stays CSV-compatible during transition.
    out_csv = DATA_DIR / f"garmin_activities_{stamp}.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Also wrote CSV → {out_csv}")

    print("\nActivity-type counts (top 10):")
    print(df["activity_type"].value_counts().head(10).to_string())
    pct_power = df["avg_power"].notna().mean() * 100
    pct_tss = df["training_stress_score"].notna().mean() * 100
    print(f"\nCoverage: avg_power={pct_power:.1f}%  TSS={pct_tss:.1f}%")


if __name__ == "__main__":
    main()
