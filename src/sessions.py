"""Build the canonical ``sessions.parquet`` — one row per unique physical activity.

Reconciles Strava (the primary source, because it has names and the longest
history) with Garmin Connect (richer metrics: power, TSS, NP, IF, TE) and
Apple Health workouts (for devices that never hit Strava). Matching uses a
fuzzy rule: same start time within ±120 s AND same duration within 5 %.

Output schema:

    session_id            str   — canonical; strava_id > garmin_id > apple_key
    strava_id             Int64 — nullable
    garmin_id             Int64 — nullable
    apple_workout_key     str   — nullable; "type|start|end|source"
    sources               list[str]
    primary_source        str
    start_time_utc        datetime64[ns, UTC]
    start_time_local      datetime64[ns]          — from device/app
    duration_s            float
    moving_s              float                   — Garmin-only
    elapsed_s             float                   — Garmin-only
    distance_m            float
    elevation_gain_m      float
    elevation_loss_m      float                   — Garmin-only
    activity_type         str                     — unified taxonomy
    sport_type            str                     — source-specific
    is_indoor             bool
    avg_hr                float
    max_hr                float
    avg_power             float
    max_power             float
    normalized_power      float                   — Garmin-only
    avg_cadence           float
    max_cadence           float
    tss                   float                   — Garmin-only
    intensity_factor      float                   — Garmin-only
    aerobic_te            float                   — Garmin-only
    anaerobic_te          float                   — Garmin-only
    calories              float
    kilojoules            float                   — Strava-only
    vo2max_estimate       float                   — Garmin per-activity
    device_name           str
    activity_name         str
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Unified activity-type taxonomy
# ---------------------------------------------------------------------------

# Key is "source:raw_value" (lowercased), value is (unified_type, is_indoor).
_TYPE_MAP: dict[str, tuple[str, bool]] = {
    # Strava "type" and "sport_type" (lowercased)
    "strava:ride": ("ride", False),
    "strava:virtualride": ("virtual_ride", True),
    "strava:run": ("run", False),
    "strava:trailrun": ("trail_run", False),
    "strava:walk": ("walk", False),
    "strava:hike": ("hike", False),
    "strava:workout": ("workout", True),
    "strava:backcountryski": ("ski", False),
    "strava:mountainbikeride": ("ride", False),
    "strava:gravelride": ("ride", False),
    # Garmin activity_type (typeKey)
    "garmin:road_biking": ("ride", False),
    "garmin:cycling": ("ride", False),
    "garmin:mountain_biking": ("ride", False),
    "garmin:gravel_cycling": ("ride", False),
    "garmin:indoor_cycling": ("virtual_ride", True),
    "garmin:virtual_ride": ("virtual_ride", True),
    "garmin:running": ("run", False),
    "garmin:trail_running": ("trail_run", False),
    "garmin:treadmill_running": ("run", True),
    "garmin:walking": ("walk", False),
    "garmin:hiking": ("hike", False),
    "garmin:strength_training": ("workout", True),
    "garmin:training": ("workout", True),  # FIT uses bare "training"
    # Apple HKWorkoutActivityType
    "apple:hkworkoutactivitytypecycling": ("ride", False),
    "apple:hkworkoutactivitytyperunning": ("run", False),
    "apple:hkworkoutactivitytypewalking": ("walk", False),
    "apple:hkworkoutactivitytypehiking": ("hike", False),
    "apple:hkworkoutactivitytypedownhillskiing": ("ski", False),
    "apple:hkworkoutactivitytypeother": ("other", False),
}


def _unify_type(source: str, raw: Any) -> tuple[str, bool]:
    if raw is None:
        return ("other", False)
    key = f"{source}:{str(raw).strip().lower()}"
    return _TYPE_MAP.get(key, ("other", False))


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------


def load_strava(path: Path) -> pd.DataFrame:
    with open(path) as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    df = df.rename(columns={
        "id": "strava_id",
        "name": "activity_name",
        "type": "strava_type",
        "sport_type": "strava_sport_type",
        "average_hr": "avg_hr",
        "max_hr": "max_hr",
        "average_watts": "avg_power",
        "average_speed_ms": "avg_speed_ms",
        "max_speed_ms": "max_speed_ms",
        "elevation_m": "elevation_gain_m",
        "calories": "calories",
        "kilojoules": "kilojoules",
    })
    df["strava_id"] = pd.to_numeric(df["strava_id"]).astype("Int64")
    df["start_time_utc"] = pd.to_datetime(df["start_date"], utc=True)
    df["start_time_local"] = df["start_time_utc"].dt.tz_convert(None)
    # Infer type
    types = df["strava_sport_type"].fillna(df["strava_type"]).map(
        lambda v: _unify_type("strava", v)
    )
    df["activity_type"] = [t[0] for t in types]
    df["is_indoor"] = [t[1] for t in types]
    df["sport_type"] = df["strava_sport_type"].fillna(df["strava_type"])
    df["source"] = "strava"
    return df


def load_garmin(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.rename(columns={
        "id": "garmin_id",
        "training_stress_score": "tss",
        "aerobic_training_effect": "aerobic_te",
        "anaerobic_training_effect": "anaerobic_te",
        "vo2_max": "vo2max_estimate",
    })
    df["garmin_id"] = pd.to_numeric(df["garmin_id"]).astype("Int64")
    df["start_time_utc"] = pd.to_datetime(df["start_time_gmt"], utc=True, errors="coerce")
    df["start_time_local"] = pd.to_datetime(df["start_time_local"], errors="coerce")
    types = df["activity_type"].map(lambda v: _unify_type("garmin", v))
    df["activity_type_unified"] = [t[0] for t in types]
    df["is_indoor"] = [t[1] for t in types]
    df["sport_type"] = df["activity_type"]  # preserve raw
    df["activity_type"] = df["activity_type_unified"]
    df = df.drop(columns=["activity_type_unified"])
    df["source"] = "garmin"
    return df


_FIT_FILENAME_ID = re.compile(r"_(\d+)\.fit$")


def load_garmin_fit(path: Path) -> pd.DataFrame:
    """Load Garmin GDPR FIT session summaries (one row per activity FIT)."""
    df = pd.read_parquet(path)

    # The FIT filename is <email>_<garmin_activity_id>.fit — recover the id.
    def _extract_id(p: Any) -> Any:
        if not isinstance(p, str):
            return None
        m = _FIT_FILENAME_ID.search(p)
        return int(m.group(1)) if m else None

    df["garmin_id"] = df["source_path"].map(_extract_id).astype("Int64")

    df = df.rename(columns={
        "total_elapsed_time": "duration_s",
        "total_timer_time": "moving_s",
        "total_distance": "distance_m",
        "total_ascent": "elevation_gain_m",
        "total_descent": "elevation_loss_m",
        "avg_heart_rate": "avg_hr",
        "max_heart_rate": "max_hr",
        "training_stress_score": "tss",
        "total_calories": "calories",
    })
    df["start_time_utc"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
    df["start_time_local"] = df["start_time_utc"].dt.tz_convert(None)
    types = df["sport"].map(lambda v: _unify_type("garmin", v))
    df["activity_type"] = [t[0] for t in types]
    df["is_indoor"] = [t[1] for t in types]
    df["sport_type"] = df["sport"]
    df["device_name"] = df["device_product"]
    df["source"] = "garmin_fit"
    return df


def load_apple_workouts(db_path: Path) -> pd.DataFrame:
    """Apple Workout summaries with a composite natural key."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM workouts", conn)
    df["start_time_utc"] = pd.to_datetime(df["startDate_ts"], unit="s", utc=True)
    df["start_time_local"] = df["start_time_utc"].dt.tz_convert(None)
    df["apple_workout_key"] = (
        df["workoutActivityType"].fillna("")
        + "|" + df["startDate"].fillna("")
        + "|" + df["endDate"].fillna("")
        + "|" + df["sourceName"].fillna("")
    )
    df["duration_s"] = pd.to_numeric(df["duration"], errors="coerce") * 60  # Apple logs minutes
    # Apple distance unit is 'mi' or 'km' depending on user; convert to metres.
    dist = pd.to_numeric(df["totalDistance"], errors="coerce")
    unit = df["totalDistanceUnit"].fillna("")
    df["distance_m"] = (
        dist.where(unit != "mi", dist * 1609.344)
            .where(unit != "km", dist * 1000.0)
    )
    df["calories"] = pd.to_numeric(df["totalEnergyBurned"], errors="coerce")
    types = df["workoutActivityType"].map(lambda v: _unify_type("apple", v))
    df["activity_type"] = [t[0] for t in types]
    df["is_indoor"] = [t[1] for t in types]
    df["sport_type"] = df["workoutActivityType"]
    df["device_name"] = df["sourceName"]
    df["source"] = "apple"
    return df


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def _fuzzy_match(
    base: pd.DataFrame,
    other: pd.DataFrame,
    start_tol_s: float = 120,
    duration_tol_frac: float = 0.05,
) -> pd.Series:
    """For each row in ``base``, return the matching index in ``other`` (or NA).

    Match: |Δstart| ≤ start_tol_s AND |Δduration| ≤ duration_tol_frac · base_duration.
    If multiple candidates match, the closest start-time wins.
    """
    # Sort other by start; use searchsorted for candidate shortlist.
    o = other[["start_time_utc", "duration_s"]].reset_index(drop=False).rename(
        columns={"index": "other_idx"}
    ).sort_values("start_time_utc").reset_index(drop=True)

    o_times = o["start_time_utc"].values.astype("datetime64[ns]")
    matches = pd.Series([pd.NA] * len(base), index=base.index, dtype="Int64")

    for i, row in base[["start_time_utc", "duration_s"]].iterrows():
        ts = pd.Timestamp(row["start_time_utc"])
        if pd.isna(ts):
            continue
        ts64 = ts.to_datetime64()
        lo_ts = ts64 - pd.Timedelta(seconds=start_tol_s).to_timedelta64()
        hi_ts = ts64 + pd.Timedelta(seconds=start_tol_s).to_timedelta64()

        lo = o_times.searchsorted(lo_ts, side="left")
        hi = o_times.searchsorted(hi_ts, side="right")
        if lo == hi:
            continue

        base_dur = row["duration_s"]
        cand = o.iloc[lo:hi]
        if pd.notna(base_dur):
            dur_ok = (
                (cand["duration_s"] - base_dur).abs()
                <= max(1.0, duration_tol_frac * base_dur)
            )
            cand = cand[dur_ok]
        if cand.empty:
            continue
        # Closest start time wins.
        cand = cand.assign(dt=(cand["start_time_utc"] - ts).abs())
        winner = cand.sort_values("dt").iloc[0]
        matches.at[i] = int(winner["other_idx"])

    return matches


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


_OUTPUT_COLS = [
    "session_id", "strava_id", "garmin_id", "apple_workout_key",
    "sources", "primary_source",
    "start_time_utc", "start_time_local",
    "duration_s", "moving_s", "elapsed_s",
    "distance_m", "elevation_gain_m", "elevation_loss_m",
    "activity_type", "sport_type", "is_indoor",
    "avg_hr", "max_hr",
    "avg_power", "max_power", "normalized_power",
    "avg_cadence", "max_cadence",
    "tss", "intensity_factor", "aerobic_te", "anaerobic_te",
    "calories", "kilojoules", "vo2max_estimate",
    "device_name", "activity_name",
    "streams_path",
]


def build(
    strava_json: Path,
    garmin_parquet: Path | None,
    apple_db: Path | None,
    garmin_fit_parquet: Path | None = None,
    streams_dir: Path | None = None,
) -> pd.DataFrame:
    """Return the unified sessions DataFrame."""
    strava = load_strava(strava_json)
    strava["sources"] = [["strava"] for _ in range(len(strava))]

    garmin = load_garmin(garmin_parquet) if garmin_parquet else None
    garmin_fit = load_garmin_fit(garmin_fit_parquet) if garmin_fit_parquet else None
    apple = load_apple_workouts(apple_db) if apple_db else None

    # ---- Merge Garmin into Strava via fuzzy match --------------------------
    if garmin is not None and not garmin.empty:
        garmin_match = _fuzzy_match(strava, garmin)
        for i, gi in garmin_match.items():
            if pd.isna(gi):
                continue
            g = garmin.iloc[int(gi)]
            s = strava.iloc[i]
            # Copy Garmin-only fields into the Strava row.
            strava.at[i, "garmin_id"] = g["garmin_id"]
            strava.at[i, "moving_s"] = g.get("moving_s")
            strava.at[i, "elapsed_s"] = g.get("elapsed_s")
            strava.at[i, "elevation_loss_m"] = g.get("elevation_loss_m")
            strava.at[i, "normalized_power"] = g.get("normalized_power")
            strava.at[i, "max_power"] = g.get("max_power")
            strava.at[i, "avg_cadence"] = g.get("avg_cadence")
            strava.at[i, "max_cadence"] = g.get("max_cadence")
            strava.at[i, "tss"] = g.get("tss")
            strava.at[i, "intensity_factor"] = g.get("intensity_factor")
            strava.at[i, "aerobic_te"] = g.get("aerobic_te")
            strava.at[i, "anaerobic_te"] = g.get("anaerobic_te")
            strava.at[i, "vo2max_estimate"] = g.get("vo2max_estimate")
            strava.at[i, "sources"] = list(set(s["sources"] + ["garmin"]))
            # Prefer Garmin's power if Strava was null.
            if pd.isna(s.get("avg_power")) and pd.notna(g.get("avg_power")):
                strava.at[i, "avg_power"] = g["avg_power"]

        # Tack on Garmin activities that had no Strava match.
        garmin_unmatched_mask = ~pd.Index(range(len(garmin))).isin(
            garmin_match.dropna().astype(int)
        )
        garmin_extra = garmin[garmin_unmatched_mask].copy()
        if not garmin_extra.empty:
            garmin_extra["sources"] = [["garmin"] for _ in range(len(garmin_extra))]
            garmin_extra["strava_id"] = pd.NA
            garmin_extra["kilojoules"] = pd.NA
            strava = pd.concat([strava, garmin_extra], ignore_index=True, sort=False)

    # ---- Merge Garmin GDPR FIT (per-activity FIT summaries + streams_path) -
    if garmin_fit is not None and not garmin_fit.empty:
        fit_match = _fuzzy_match(strava, garmin_fit)
        matched_fit_idx: set[int] = set()
        for i, fi in fit_match.items():
            if pd.isna(fi):
                continue
            f = garmin_fit.iloc[int(fi)]
            s = strava.iloc[i]
            matched_fit_idx.add(int(fi))
            # The headline addition: pointer to the per-second parquet.
            strava.at[i, "streams_path"] = f.get("streams_path")
            if pd.isna(s.get("garmin_id")) and pd.notna(f.get("garmin_id")):
                strava.at[i, "garmin_id"] = f["garmin_id"]
            # Backfill any session-summary fields the Strava+Connect merge missed.
            for col in ("avg_power", "max_power", "normalized_power",
                        "intensity_factor", "tss", "avg_cadence", "max_cadence",
                        "elevation_loss_m", "moving_s"):
                if pd.isna(s.get(col)) and pd.notna(f.get(col)):
                    strava.at[i, col] = f[col]
            srcs = strava.at[i, "sources"]
            strava.at[i, "sources"] = (
                list(set(srcs + ["garmin_fit"])) if isinstance(srcs, list)
                else ["garmin_fit"]
            )

        # FIT-only sessions (e.g. pre-Connect-API-pull history, Garmin-only workouts).
        fit_unmatched_mask = ~pd.Index(range(len(garmin_fit))).isin(matched_fit_idx)
        fit_extra = garmin_fit[fit_unmatched_mask].copy()
        if not fit_extra.empty:
            fit_extra["sources"] = [["garmin_fit"] for _ in range(len(fit_extra))]
            fit_extra["strava_id"] = pd.NA
            fit_extra["kilojoules"] = pd.NA
            strava = pd.concat([strava, fit_extra], ignore_index=True, sort=False)

    # ---- Merge Apple into current unified via fuzzy match ------------------
    # Apple overlaps heavily with Strava/Garmin, so it's mostly for surfacing
    # the ``apple_workout_key`` as a pointer into per-workout streams.
    if apple is not None and not apple.empty:
        apple_match = _fuzzy_match(strava, apple)
        for i, ai in apple_match.items():
            if pd.isna(ai):
                continue
            a = apple.iloc[int(ai)]
            strava.at[i, "apple_workout_key"] = a["apple_workout_key"]
            srcs = strava.at[i, "sources"]
            strava.at[i, "sources"] = list(set(srcs + ["apple"])) if isinstance(srcs, list) else ["apple"]
        # We don't pull Apple-only workouts into sessions — Apple's workout
        # table heavily overlaps Strava/Garmin and we already have the
        # per-sample data via ``apple_health.db``.

    # ---- Canonical session_id + primary_source ----------------------------
    def _canon_id(row):
        if pd.notna(row.get("strava_id")):
            return f"strava_{int(row['strava_id'])}"
        if pd.notna(row.get("garmin_id")):
            return f"garmin_{int(row['garmin_id'])}"
        if isinstance(row.get("apple_workout_key"), str) and row["apple_workout_key"]:
            return f"apple_{abs(hash(row['apple_workout_key'])) % 10**12}"
        return None

    strava["session_id"] = strava.apply(_canon_id, axis=1)
    strava["primary_source"] = strava["session_id"].str.split("_").str[0]

    # ---- Backfill streams_path from on-disk Strava parquets ---------------
    # Strava streams live at <streams_dir>/strava_<id>.parquet. Only set when
    # the file actually exists, and don't overwrite a FIT-derived path
    # (FIT streams are richer where available).
    if streams_dir is not None:
        streams_dir = Path(streams_dir)
        if "streams_path" not in strava.columns:
            strava["streams_path"] = pd.NA
        need_fill = strava["streams_path"].isna() & strava["strava_id"].notna()
        for i in strava.index[need_fill]:
            sid = strava.at[i, "strava_id"]
            p = streams_dir / f"strava_{int(sid)}.parquet"
            if p.exists():
                strava.at[i, "streams_path"] = str(p)

    # Ensure all output columns exist.
    for col in _OUTPUT_COLS:
        if col not in strava.columns:
            strava[col] = pd.NA

    # Order + types.
    out = strava[_OUTPUT_COLS].copy()
    out = out.sort_values("start_time_utc").reset_index(drop=True)
    return out


def write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
