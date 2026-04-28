"""Parse Garmin/Strava FIT files with ``fitdecode`` into Parquet.

For each FIT file we emit two things:

1. A ``session_summary`` dict (one entry, matches the FIT's `session` message):
   start_time, duration, distance, avg/max HR, avg/max power, normalized power
   (if present), TSS, sport, total ascent, etc. This lines up with the
   ``sessions.parquet`` schema and can be fuzzy-merged with the existing
   Strava/Garmin records.

2. A ``streams`` DataFrame (one row per `record` message, typically 1 Hz):
   timestamp, lat, lng, altitude, distance, speed, heart_rate, cadence,
   power, temperature — plus any developer fields (e.g. Stryd running power,
   leg-spring stiffness, left/right balance).

Streams are written as Parquet to ``<out_dir>/<hash>.parquet`` where the hash
is a stable SHA-1 of (start_time + duration + device_id) so repeat ingests
of the same activity land in the same file.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

try:
    import fitdecode
except ImportError as e:  # pragma: no cover
    raise ImportError("fitdecode is required: pip install fitdecode") from e

# Fields on the `record` message we want to extract. Additional fields
# (developer or manufacturer-specific) are still picked up dynamically below.
_RECORD_FIELDS = (
    "timestamp",
    "position_lat",
    "position_long",
    "altitude",
    "enhanced_altitude",
    "distance",
    "speed",
    "enhanced_speed",
    "heart_rate",
    "cadence",
    "fractional_cadence",
    "power",
    "accumulated_power",
    "temperature",
    "grade",
    "left_right_balance",
    "left_torque_effectiveness",
    "right_torque_effectiveness",
    "left_pedal_smoothness",
    "right_pedal_smoothness",
    "vertical_oscillation",
    "vertical_ratio",
    "stance_time",
    "stance_time_percent",
    "stance_time_balance",
    "step_length",
)

_SESSION_FIELDS = (
    "timestamp",
    "start_time",
    "total_elapsed_time",
    "total_timer_time",
    "total_distance",
    "total_ascent",
    "total_descent",
    "avg_heart_rate",
    "max_heart_rate",
    "avg_power",
    "max_power",
    "normalized_power",
    "training_stress_score",
    "intensity_factor",
    "avg_cadence",
    "max_cadence",
    "avg_speed",
    "max_speed",
    "enhanced_avg_speed",
    "enhanced_max_speed",
    "total_calories",
    "total_work",
    "sport",
    "sub_sport",
    "avg_temperature",
    "max_temperature",
    "avg_running_cadence",
    "avg_stance_time",
    "avg_step_length",
    "avg_vertical_oscillation",
    "avg_vertical_ratio",
)


def _frame_dict(frame) -> dict[str, Any]:
    """Flatten a fitdecode FitDataMessage into {field_name: value}.

    Captures both native and developer fields. Semicircle lat/lng are
    converted to degrees.
    """
    out: dict[str, Any] = {}
    for field in frame.fields:
        name = field.name
        v = field.value
        if name in ("position_lat", "position_long") and isinstance(v, (int, float)):
            v = v * (180.0 / 2**31)  # FIT semicircles → degrees
        elif name == "left_right_balance":
            # Polymorphic across FITs: int percentage for dual-sided meters,
            # string ('left'/'right') when only one side is present. Force str
            # so pyarrow gets a consistent column.
            v = str(v) if v is not None else None
        if v is not None:
            out[name] = v
    # Developer fields (e.g. Stryd running power, form power)
    for devfield in getattr(frame, "dev_fields", []) or []:
        try:
            out[f"dev_{devfield.name}"] = devfield.value
        except Exception:
            pass
    return out


def iter_frames(fit_path: Path) -> Iterator:
    """Yield every FitDataMessage in the file."""
    with fitdecode.FitReader(str(fit_path)) as fr:
        for frame in fr:
            if isinstance(frame, fitdecode.FitDataMessage):
                yield frame


def parse_fit(fit_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return (session_summary, streams_df) for a single FIT file."""
    records: list[dict[str, Any]] = []
    session: dict[str, Any] = {}
    sport: dict[str, Any] = {}
    file_id: dict[str, Any] = {}

    for frame in iter_frames(fit_path):
        if frame.name == "record":
            records.append(_frame_dict(frame))
        elif frame.name == "session" and not session:
            session = _frame_dict(frame)
        elif frame.name == "sport" and not sport:
            sport = _frame_dict(frame)
        elif frame.name == "file_id" and not file_id:
            file_id = _frame_dict(frame)

    streams = pd.DataFrame.from_records(records)
    if "timestamp" in streams.columns:
        streams["timestamp"] = pd.to_datetime(streams["timestamp"], utc=True, errors="coerce")
        streams = streams.sort_values("timestamp").reset_index(drop=True)

    # Pack a compact session summary (with whatever fields we asked for).
    summary = {k: session.get(k) for k in _SESSION_FIELDS if k in session}
    # Add sport/sub_sport from the `sport` message if the `session` didn't have it.
    summary.setdefault("sport", sport.get("sport"))
    summary.setdefault("sub_sport", sport.get("sub_sport"))
    summary["device_manufacturer"] = file_id.get("manufacturer")
    # garmin_product is a str for known products, raw int for unknown codes.
    prod = file_id.get("garmin_product") or file_id.get("product")
    summary["device_product"] = str(prod) if prod is not None else None
    summary["file_serial_number"] = file_id.get("serial_number")
    summary["source_path"] = str(fit_path)
    return summary, streams


def activity_hash(summary: dict[str, Any]) -> str:
    """Stable ID for a FIT file: sha1 of (start + elapsed + serial)."""
    start = summary.get("start_time") or summary.get("timestamp")
    dur = summary.get("total_elapsed_time") or summary.get("total_timer_time") or 0
    serial = summary.get("file_serial_number") or ""
    key = f"{start}|{dur}|{serial}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def fit_to_parquet(
    fit_path: Path, out_dir: Path, overwrite: bool = False
) -> dict[str, Any] | None:
    """Parse one FIT file and write its streams to ``out_dir/<hash>.parquet``.

    Returns the enriched session summary (with ``activity_hash`` and
    ``streams_path`` fields), or ``None`` if the FIT had no usable data.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, streams = parse_fit(fit_path)
    if streams.empty:
        return None

    h = activity_hash(summary)
    streams_path = out_dir / f"{h}.parquet"
    if streams_path.exists() and not overwrite:
        # Keep existing file; still return the summary so callers can index.
        summary["activity_hash"] = h
        summary["streams_path"] = str(streams_path)
        summary["skipped"] = True
        return summary

    streams.to_parquet(streams_path, index=False)
    summary["activity_hash"] = h
    summary["streams_path"] = str(streams_path)
    summary["streams_rows"] = len(streams)
    summary["skipped"] = False
    return summary


def ingest_directory(
    fit_dir: Path, out_dir: Path, overwrite: bool = False
) -> pd.DataFrame:
    """Parse every ``*.fit`` / ``*.fit.gz`` in ``fit_dir`` and return a
    DataFrame of session summaries (one row per FIT)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    fit_files = sorted(
        list(fit_dir.rglob("*.fit")) + list(fit_dir.rglob("*.fit.gz"))
    )
    print(f"Found {len(fit_files)} FIT files under {fit_dir}")
    for i, fp in enumerate(fit_files, 1):
        try:
            s = fit_to_parquet(fp, out_dir, overwrite=overwrite)
            if s is not None:
                summaries.append(s)
        except Exception as e:
            print(f"  [{i}/{len(fit_files)}] {fp.name}: {type(e).__name__}: {e}")
            continue
        if i % 50 == 0:
            print(f"  [{i}/{len(fit_files)}] {fp.name}")
    return pd.DataFrame(summaries)
