#!/usr/bin/env python3
"""Build data/processed/sessions.parquet from the latest raw sources."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import sessions  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "sessions.parquet"


def _latest(pattern: str) -> Path | None:
    hits = sorted(RAW.glob(pattern), reverse=True)
    return hits[0] if hits else None


def main() -> None:
    strava_json = _latest("activities_*.json")
    garmin_parquet = _latest("garmin_activities_*.parquet")
    garmin_fit_parquet = RAW / "garmin_gdpr_summaries.parquet"
    if not garmin_fit_parquet.exists():
        garmin_fit_parquet = None
    apple_db = ROOT / "data" / "processed" / "apple_health.db"
    if not apple_db.exists():
        apple_db = None
    streams_dir = ROOT / "data" / "processed" / "streams"

    print(f"Strava     : {strava_json}")
    print(f"Garmin API : {garmin_parquet}")
    print(f"Garmin FIT : {garmin_fit_parquet}")
    print(f"Apple      : {apple_db}")
    print(f"Streams    : {streams_dir}")

    if strava_json is None:
        sys.exit("No Strava activities_*.json in data/raw/")

    t0 = time.time()
    df = sessions.build(
        strava_json, garmin_parquet, apple_db,
        garmin_fit_parquet=garmin_fit_parquet,
        streams_dir=streams_dir,
    )
    sessions.write(df, OUT)

    dt = time.time() - t0
    print(f"\n✓ Wrote {len(df):,} sessions → {OUT} ({OUT.stat().st_size/1024:.1f} KB, {dt:.1f}s)")
    print(f"\nPrimary-source breakdown:")
    print(df["primary_source"].value_counts().to_string())
    print(f"\nActivity-type breakdown:")
    print(df["activity_type"].value_counts().to_string())
    print(f"\nMulti-source coverage:")
    multi = df[df["sources"].apply(lambda x: isinstance(x, list) and len(x) > 1)]
    print(f"  rows in ≥2 sources: {len(multi)} / {len(df)} ({len(multi)/len(df)*100:.1f}%)")
    strava_w_garmin = df[
        df["strava_id"].notna() & df["garmin_id"].notna()
    ]
    print(f"  Strava ∩ Garmin: {len(strava_w_garmin)}")
    apple_tagged = df[df["apple_workout_key"].notna()]
    print(f"  Strava/Garmin ∩ Apple: {len(apple_tagged)}")
    print(f"\nCoverage on key enrichments:")
    for c in ["avg_power", "tss", "normalized_power", "aerobic_te",
              "vo2max_estimate", "streams_path"]:
        pct = df[c].notna().mean() * 100
        print(f"  {c:20s}  {pct:5.1f}%")


if __name__ == "__main__":
    main()
