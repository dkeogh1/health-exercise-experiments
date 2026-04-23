"""Historical weather enrichment via the Open-Meteo archive API.

Open-Meteo's historical reanalysis (ERA5 underneath) is free, keyless, and
covers ~1980-present at ~10 km resolution. We fetch hourly weather for each
activity's location/date, then aggregate to the activity window.

Cache:
    data/processed/weather_cache.sqlite
        weather_days(lat_r REAL, lng_r REAL, date TEXT, payload TEXT,
                     fetched_at TEXT, PRIMARY KEY (lat_r, lng_r, date))

Location is rounded to 2 decimal places (~1 km) so nearby activities on the
same day share a cache row. The ``payload`` column stores the raw JSON.

Public API:
    fetch_day(lat, lng, date)      → list[dict]       # 24 hourly rows, cached
    weather_at(lat, lng, start, end) → dict           # aggregates across window
    enrich(sessions, streams_dir)  → DataFrame        # bulk-enrich sessions.parquet
"""

from __future__ import annotations

import json
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

API_BASE = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_FIELDS = (
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _open_cache(cache_path: Path) -> sqlite3.Connection:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cache_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS weather_days (
            lat_r      REAL,
            lng_r      REAL,
            date       TEXT,
            payload    TEXT,
            fetched_at TEXT,
            PRIMARY KEY (lat_r, lng_r, date)
        )
        """
    )
    return conn


def _round_coord(x: float) -> float:
    return round(float(x), 2)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def _http_get(url: str, retries: int = 3, backoff_s: float = 2.0) -> dict:
    """GET a URL with basic retry on transient errors."""
    last_err: Exception | None = None
    for i in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "strava-analysis/0.1"}
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(backoff_s * (i + 1))
                last_err = e
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            time.sleep(backoff_s * (i + 1))
    raise RuntimeError(f"Open-Meteo request failed after {retries} retries: {last_err}")


def fetch_day(
    lat: float, lng: float, date: str,
    cache_path: Path,
) -> list[dict]:
    """Hourly weather for a single (lat, lng, date) — cached.

    Returns a list of 24 dicts (one per hour UTC) with fields:
    ``hour`` (ISO), ``temperature_2m``, ``apparent_temperature``,
    ``relative_humidity_2m``, ``dew_point_2m``, ``wind_speed_10m``,
    ``wind_direction_10m``, ``precipitation``, ``cloud_cover``.
    """
    lat_r = _round_coord(lat)
    lng_r = _round_coord(lng)
    conn = _open_cache(cache_path)
    cur = conn.execute(
        "SELECT payload FROM weather_days WHERE lat_r=? AND lng_r=? AND date=?",
        (lat_r, lng_r, date),
    )
    row = cur.fetchone()
    if row:
        conn.close()
        return json.loads(row[0])

    params = {
        "latitude": lat_r,
        "longitude": lng_r,
        "start_date": date,
        "end_date": date,
        "hourly": ",".join(HOURLY_FIELDS),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }
    url = f"{API_BASE}?{urllib.parse.urlencode(params)}"
    body = _http_get(url)

    hourly = body.get("hourly", {})
    times = hourly.get("time") or []
    rows = []
    for i, t in enumerate(times):
        rows.append(
            {
                "hour": t,
                **{f: (hourly.get(f) or [None] * len(times))[i] for f in HOURLY_FIELDS},
            }
        )

    conn.execute(
        "INSERT OR REPLACE INTO weather_days (lat_r, lng_r, date, payload, fetched_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (lat_r, lng_r, date, json.dumps(rows),
         datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Aggregate over an activity window
# ---------------------------------------------------------------------------


def weather_at(
    lat: float, lng: float,
    start_utc: pd.Timestamp, end_utc: pd.Timestamp,
    cache_path: Path,
) -> dict:
    """Mean weather over the [start, end] window for a single lat/lng.

    If the window straddles two dates we concat both and slice by hour.
    Returns a flat dict with ``air_temp_c``, ``apparent_temp_c``,
    ``humidity_pct``, ``dew_point_c``, ``wind_ms``, ``wind_dir_deg``,
    ``precip_mm``, ``cloud_pct``.
    """
    start_utc = pd.Timestamp(start_utc, tz="UTC") if pd.Timestamp(start_utc).tzinfo is None else pd.Timestamp(start_utc).tz_convert("UTC")
    end_utc = pd.Timestamp(end_utc, tz="UTC") if pd.Timestamp(end_utc).tzinfo is None else pd.Timestamp(end_utc).tz_convert("UTC")

    dates = []
    d = start_utc.normalize()
    while d <= end_utc.normalize():
        dates.append(d.strftime("%Y-%m-%d"))
        d = d + pd.Timedelta(days=1)

    all_rows: list[dict] = []
    for day in dates:
        all_rows.extend(fetch_day(lat, lng, day, cache_path))

    if not all_rows:
        return {}
    df = pd.DataFrame(all_rows)
    df["hour"] = pd.to_datetime(df["hour"], utc=True)
    in_window = df[(df["hour"] >= start_utc.floor("h"))
                   & (df["hour"] <= end_utc.ceil("h"))]
    if in_window.empty:
        return {}
    out = {
        "air_temp_c":       float(in_window["temperature_2m"].mean(skipna=True)),
        "apparent_temp_c":  float(in_window["apparent_temperature"].mean(skipna=True)),
        "humidity_pct":     float(in_window["relative_humidity_2m"].mean(skipna=True)),
        "dew_point_c":      float(in_window["dew_point_2m"].mean(skipna=True)),
        "wind_ms":          float(in_window["wind_speed_10m"].mean(skipna=True)),
        "wind_dir_deg":     float(in_window["wind_direction_10m"].mean(skipna=True)),
        "precip_mm":        float(in_window["precipitation"].sum(skipna=True)),
        "cloud_pct":        float(in_window["cloud_cover"].mean(skipna=True)),
    }
    return out


# ---------------------------------------------------------------------------
# Bulk enrichment
# ---------------------------------------------------------------------------


def session_lat_lng_from_stream(stream_df: pd.DataFrame) -> tuple[float, float] | None:
    """First non-null (lat, lng) in a stream DataFrame."""
    if "lat" not in stream_df.columns or "lng" not in stream_df.columns:
        return None
    ll = stream_df[["lat", "lng"]].dropna()
    if ll.empty:
        return None
    return float(ll["lat"].iloc[0]), float(ll["lng"].iloc[0])


def enrich(
    sessions: pd.DataFrame,
    streams_dir: Path,
    cache_path: Path,
    skip_indoor: bool = True,
    throttle_s: float = 0.05,
) -> pd.DataFrame:
    """For each session with a stream file on disk, look up lat/lng from the
    stream, pull weather for the activity window, attach columns to the row.

    Mutates nothing; returns a new DataFrame.
    """
    out = sessions.copy()
    for col in ("air_temp_c", "apparent_temp_c", "humidity_pct", "dew_point_c",
                "wind_ms", "wind_dir_deg", "precip_mm", "cloud_pct",
                "start_lat", "start_lng"):
        if col not in out.columns:
            out[col] = pd.NA

    n_enriched = 0
    n_skipped_indoor = 0
    n_no_stream = 0
    n_no_latlng = 0
    n_failed = 0

    for i, row in out.iterrows():
        if skip_indoor and bool(row.get("is_indoor", False)):
            n_skipped_indoor += 1
            continue
        sid = row.get("strava_id")
        if pd.isna(sid):
            continue
        stream_path = streams_dir / f"strava_{int(sid)}.parquet"
        if not stream_path.exists():
            n_no_stream += 1
            continue

        try:
            stream = pd.read_parquet(stream_path)
            latlng = session_lat_lng_from_stream(stream)
            if latlng is None:
                n_no_latlng += 1
                continue
            lat, lng = latlng
            start_utc = pd.Timestamp(row["start_time_utc"])
            end_utc = start_utc + pd.Timedelta(seconds=float(row["duration_s"] or 0))
            w = weather_at(lat, lng, start_utc, end_utc, cache_path)
            if not w:
                n_failed += 1
                continue
            out.at[i, "start_lat"] = lat
            out.at[i, "start_lng"] = lng
            for k, v in w.items():
                out.at[i, k] = v
            n_enriched += 1
            if throttle_s:
                time.sleep(throttle_s)
        except Exception as e:
            print(f"  session {sid}: {type(e).__name__}: {e}")
            n_failed += 1
            continue

    print(
        f"weather enrichment summary: "
        f"enriched={n_enriched}  indoor_skip={n_skipped_indoor}  "
        f"no_stream={n_no_stream}  no_latlng={n_no_latlng}  failed={n_failed}"
    )
    return out
