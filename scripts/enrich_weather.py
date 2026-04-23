#!/usr/bin/env python3
"""Enrich data/processed/sessions.parquet with historical weather from Open-Meteo.

Reads every Strava stream file on disk, picks up its first lat/lng, and
fetches hourly weather for the activity's UTC window. Results are cached in
``data/processed/weather_cache.sqlite`` so incremental re-runs (after new
streams land) don't re-hit the API.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import weather  # noqa: E402

SESSIONS = ROOT / "data" / "processed" / "sessions.parquet"
STREAMS_DIR = ROOT / "data" / "processed" / "streams"
CACHE = ROOT / "data" / "processed" / "weather_cache.sqlite"


def main() -> None:
    sessions = pd.read_parquet(SESSIONS)
    sessions["start_time_utc"] = pd.to_datetime(sessions["start_time_utc"], utc=True)

    t0 = time.time()
    enriched = weather.enrich(sessions, STREAMS_DIR, CACHE)
    enriched.to_parquet(SESSIONS, index=False)
    dt = time.time() - t0

    pct_weather = enriched["air_temp_c"].notna().mean() * 100
    print(
        f"\n✓ Wrote {len(enriched):,} sessions back to {SESSIONS} "
        f"({enriched['air_temp_c'].notna().sum()} with weather = "
        f"{pct_weather:.1f}%)   {dt:.1f}s"
    )
    if enriched["air_temp_c"].notna().sum():
        print("\nWeather summary (enriched rows):")
        for c in ("air_temp_c", "apparent_temp_c", "humidity_pct",
                  "wind_ms", "precip_mm", "cloud_pct"):
            s = enriched[c].dropna()
            if not s.empty:
                print(f"  {c:20s}  n={len(s):4d}  "
                      f"mean={s.mean():6.1f}  min={s.min():6.1f}  max={s.max():6.1f}")


if __name__ == "__main__":
    main()
