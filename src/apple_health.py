"""Apple Health ingest + query layer.

Converts the Apple Health XML export into a SQLite database with one table
per record type (plus a workouts table and optional workout_points table),
then provides pandas-returning query helpers for downstream analysis.

Why a custom ingest (rather than healthkit-to-sqlite)?
The upstream library hashes workout attrs for the primary key and crashes
with UNIQUE-constraint errors when the same workout is logged by multiple
sources (common on exports that include both Apple Watch and third-party
apps). We use INSERT OR IGNORE instead.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator
from xml.etree import ElementTree as ET

import pandas as pd


def _apple_ts_to_epoch(s: str | None) -> int | None:
    """Parse an Apple Health timestamp like '2026-03-25 12:14:09 -0400' to Unix seconds.

    Returns None for missing or malformed input. Accepts space-separated offsets
    (``-0400``) which fromisoformat can't parse directly on older Pythons.
    """
    if not s:
        return None
    try:
        # Turn "2026-03-25 12:14:09 -0400" into "2026-03-25T12:14:09-04:00".
        datepart, timepart, tz = s.split(" ")
        tz_fmt = f"{tz[:3]}:{tz[3:]}" if len(tz) == 5 else tz
        iso = f"{datepart}T{timepart}{tz_fmt}"
        return int(datetime.fromisoformat(iso).astimezone(timezone.utc).timestamp())
    except Exception:
        return None

RECORD_TAG = "Record"
WORKOUT_TAG = "Workout"
ACTIVITY_SUMMARY_TAG = "ActivitySummary"


def _short_type(type_id: str) -> str:
    """HKQuantityTypeIdentifierHeartRate -> HeartRate"""
    return (
        type_id.replace("HKQuantityTypeIdentifier", "")
        .replace("HKCategoryTypeIdentifier", "")
        .replace("HKDataTypeIdentifier", "")
    )


def _iter_top_level(xml_path: Path, tags: set[str]) -> Iterator[tuple[str, ET.Element]]:
    """Stream the XML and yield (tag, element) for each matching top-level tag.

    Uses iterparse and clears each element after yielding to keep memory flat.
    """
    context = ET.iterparse(str(xml_path), events=("end",))
    for _event, elem in context:
        if elem.tag in tags:
            yield elem.tag, elem
            elem.clear()


def xml_to_sqlite(xml_path: Path, db_path: Path, batch_size: int = 5000) -> dict:
    """Convert Apple Health export.xml to a SQLite DB.

    Returns a dict of {table: row_count} summarising what was ingested.
    """
    xml_path = Path(xml_path)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")

    known_record_tables: dict[str, list[str]] = {}
    buffered: dict[str, list[dict]] = {}
    workout_buffer: list[dict] = []
    summary_buffer: list[dict] = []
    counts: dict[str, int] = {}

    def flush_record_table(table: str) -> None:
        rows = buffered.get(table)
        if not rows:
            return
        cols = known_record_tables[table]
        col_list = ",".join(f'"{c}"' for c in cols)
        placeholders = ",".join("?" for _ in cols)
        conn.executemany(
            f'INSERT INTO "{table}" ({col_list}) VALUES ({placeholders})',
            [[r.get(c) for c in cols] for r in rows],
        )
        counts[table] = counts.get(table, 0) + len(rows)
        buffered[table] = []

    def ensure_record_table(table: str, row: dict) -> None:
        if table in known_record_tables:
            # Widen if we discovered new keys.
            new_cols = [k for k in row if k not in known_record_tables[table]]
            for c in new_cols:
                conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{c}" TEXT')
                known_record_tables[table].append(c)
            return
        cols = list(row.keys())
        col_sql = ",".join(f'"{c}" TEXT' for c in cols)
        conn.execute(f'CREATE TABLE "{table}" ({col_sql})')
        # Helpful indexes for our common access patterns.
        if "startDate_ts" in cols:
            conn.execute(
                f'CREATE INDEX "ix_{table}_startDate_ts" ON "{table}" ("startDate_ts")'
            )
        known_record_tables[table] = cols

    # Workouts schema (fixed). We use a natural key to ignore dupes from
    # multiple sources recording the same workout.
    conn.execute(
        """
        CREATE TABLE workouts (
            workoutActivityType TEXT,
            duration            REAL,
            durationUnit        TEXT,
            totalDistance       REAL,
            totalDistanceUnit   TEXT,
            totalEnergyBurned   REAL,
            totalEnergyBurnedUnit TEXT,
            sourceName          TEXT,
            sourceVersion       TEXT,
            device              TEXT,
            creationDate        TEXT,
            startDate           TEXT,
            endDate             TEXT,
            startDate_ts        INTEGER,
            endDate_ts          INTEGER,
            metadata            TEXT,
            PRIMARY KEY (workoutActivityType, startDate, endDate, sourceName)
        )
        """
    )
    conn.execute("CREATE INDEX ix_workouts_startDate_ts ON workouts (startDate_ts)")

    # Activity summaries schema (fixed).
    conn.execute(
        """
        CREATE TABLE activity_summary (
            dateComponents             TEXT PRIMARY KEY,
            activeEnergyBurned         REAL,
            activeEnergyBurnedGoal     REAL,
            activeEnergyBurnedUnit     TEXT,
            appleMoveTime              REAL,
            appleMoveTimeGoal          REAL,
            appleExerciseTime          REAL,
            appleExerciseTimeGoal      REAL,
            appleStandHours            REAL,
            appleStandHoursGoal        REAL
        )
        """
    )

    for tag, el in _iter_top_level(
        xml_path, {RECORD_TAG, WORKOUT_TAG, ACTIVITY_SUMMARY_TAG}
    ):
        if tag == RECORD_TAG:
            row = dict(el.attrib)
            type_id = row.pop("type", None)
            if not type_id:
                continue
            # Merge any MetadataEntry children into meta_<key> columns.
            for child in el.findall("MetadataEntry"):
                key = child.attrib.get("key", "")
                # Strip anything that would confuse SQL identifiers; keep it
                # readable but safe. Underscores only.
                safe_key = "".join(c if c.isalnum() else "_" for c in key)
                row[f"meta_{safe_key}"] = child.attrib.get("value")
            # Derived Unix-epoch columns for fast, timezone-safe comparisons.
            row["startDate_ts"] = _apple_ts_to_epoch(row.get("startDate"))
            row["endDate_ts"] = _apple_ts_to_epoch(row.get("endDate"))
            table = f"r_{_short_type(type_id)}"
            ensure_record_table(table, row)
            buffered.setdefault(table, []).append(row)
            if len(buffered[table]) >= batch_size:
                flush_record_table(table)
        elif tag == WORKOUT_TAG:
            start_raw = el.attrib.get("startDate")
            end_raw = el.attrib.get("endDate")
            row = {
                "workoutActivityType": el.attrib.get("workoutActivityType"),
                "duration": _try_float(el.attrib.get("duration")),
                "durationUnit": el.attrib.get("durationUnit"),
                "totalDistance": _try_float(el.attrib.get("totalDistance")),
                "totalDistanceUnit": el.attrib.get("totalDistanceUnit"),
                "totalEnergyBurned": _try_float(el.attrib.get("totalEnergyBurned")),
                "totalEnergyBurnedUnit": el.attrib.get("totalEnergyBurnedUnit"),
                "sourceName": el.attrib.get("sourceName"),
                "sourceVersion": el.attrib.get("sourceVersion"),
                "device": el.attrib.get("device"),
                "creationDate": el.attrib.get("creationDate"),
                "startDate": start_raw,
                "endDate": end_raw,
                "startDate_ts": _apple_ts_to_epoch(start_raw),
                "endDate_ts": _apple_ts_to_epoch(end_raw),
                "metadata": _format_metadata(el),
            }
            workout_buffer.append(row)
            if len(workout_buffer) >= batch_size:
                _flush_workouts(conn, workout_buffer, counts)
                workout_buffer = []
        elif tag == ACTIVITY_SUMMARY_TAG:
            row = {k: _try_float(v) if k != "dateComponents" else v
                   for k, v in el.attrib.items()}
            row.setdefault("activeEnergyBurnedUnit", el.attrib.get("activeEnergyBurnedUnit"))
            summary_buffer.append(row)
            if len(summary_buffer) >= batch_size:
                _flush_summaries(conn, summary_buffer, counts)
                summary_buffer = []

    for table in list(buffered.keys()):
        flush_record_table(table)
    if workout_buffer:
        _flush_workouts(conn, workout_buffer, counts)
    if summary_buffer:
        _flush_summaries(conn, summary_buffer, counts)

    conn.commit()
    conn.close()
    return counts


def _try_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _format_metadata(el: ET.Element) -> str | None:
    pairs = []
    for child in el.findall("MetadataEntry"):
        pairs.append(f"{child.attrib.get('key','')}={child.attrib.get('value','')}")
    return "; ".join(pairs) if pairs else None


def _flush_workouts(conn, rows, counts):
    cols = list(rows[0].keys())
    col_list = ",".join(f'"{c}"' for c in cols)
    placeholders = ",".join("?" for _ in cols)
    conn.executemany(
        f'INSERT OR IGNORE INTO workouts ({col_list}) VALUES ({placeholders})',
        [[r.get(c) for c in cols] for r in rows],
    )
    counts["workouts"] = counts.get("workouts", 0) + len(rows)


def _flush_summaries(conn, rows, counts):
    # Be forgiving about which columns a given export uses.
    known = [
        "dateComponents",
        "activeEnergyBurned",
        "activeEnergyBurnedGoal",
        "activeEnergyBurnedUnit",
        "appleMoveTime",
        "appleMoveTimeGoal",
        "appleExerciseTime",
        "appleExerciseTimeGoal",
        "appleStandHours",
        "appleStandHoursGoal",
    ]
    col_list = ",".join(f'"{c}"' for c in known)
    placeholders = ",".join("?" for _ in known)
    conn.executemany(
        f'INSERT OR IGNORE INTO activity_summary ({col_list}) VALUES ({placeholders})',
        [[r.get(c) for c in known] for r in rows],
    )
    counts["activity_summary"] = counts.get("activity_summary", 0) + len(rows)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def _read_sql(db_path: Path, sql: str, params: Iterable = ()) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=list(params))


def list_tables(db_path: Path) -> pd.DataFrame:
    return _read_sql(
        db_path,
        "SELECT name, (SELECT COUNT(*) FROM pg) AS n "
        "FROM sqlite_master AS m "
        "LEFT JOIN (SELECT 1 AS pg) WHERE m.type='table' ORDER BY name",
    )


def table_counts(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn
        )["name"].tolist()
        rows = []
        for t in tables:
            n = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            rows.append({"table": t, "n": n})
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def records(db_path: Path, type_short: str) -> pd.DataFrame:
    """Read one record-type table (e.g. 'HeartRate', 'HeartRateVariabilitySDNN').

    Parses startDate/endDate as tz-aware UTC datetimes and casts value to float.
    """
    table = f"r_{type_short}"
    df = _read_sql(db_path, f'SELECT * FROM "{table}"')
    for col in ("startDate", "endDate", "creationDate"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def workouts(db_path: Path) -> pd.DataFrame:
    df = _read_sql(db_path, "SELECT * FROM workouts")
    for col in ("startDate", "endDate", "creationDate"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def daily_metrics(db_path: Path) -> pd.DataFrame:
    """Aggregate one row per local-day, with the signals we care about.

    Uses the US/Pacific-neutral approach of dating by the startDate's date
    component (in UTC); good enough for multi-year trend analysis.
    """
    queries = {
        "heart_rate_avg": (
            "HeartRate", "AVG(CAST(value AS REAL))"
        ),
        "heart_rate_min": (
            "HeartRate", "MIN(CAST(value AS REAL))"
        ),
        "heart_rate_max": (
            "HeartRate", "MAX(CAST(value AS REAL))"
        ),
        "resting_hr": (
            "RestingHeartRate", "AVG(CAST(value AS REAL))"
        ),
        "walking_hr_avg": (
            "WalkingHeartRateAverage", "AVG(CAST(value AS REAL))"
        ),
        "hrv_sdnn_ms": (
            "HeartRateVariabilitySDNN", "AVG(CAST(value AS REAL))"
        ),
        "vo2max": (
            "VO2Max", "AVG(CAST(value AS REAL))"
        ),
        "oxygen_saturation": (
            "OxygenSaturation", "AVG(CAST(value AS REAL))"
        ),
        "respiratory_rate": (
            "RespiratoryRate", "AVG(CAST(value AS REAL))"
        ),
        "steps": ("StepCount", "SUM(CAST(value AS REAL))"),
        "active_energy": ("ActiveEnergyBurned", "SUM(CAST(value AS REAL))"),
        "basal_energy": ("BasalEnergyBurned", "SUM(CAST(value AS REAL))"),
        "flights_climbed": ("FlightsClimbed", "SUM(CAST(value AS REAL))"),
        "exercise_min": ("AppleExerciseTime", "SUM(CAST(value AS REAL))"),
        "stand_min": ("AppleStandTime", "SUM(CAST(value AS REAL))"),
        "hrr1": ("HeartRateRecoveryOneMinute", "MAX(CAST(value AS REAL))"),
        "wrist_temp_sleep": (
            "AppleSleepingWristTemperature", "AVG(CAST(value AS REAL))"
        ),
    }

    with sqlite3.connect(db_path) as conn:
        existing = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'r_%'"
            )
        }
        frames = []
        for metric, (table_short, agg) in queries.items():
            tbl = f"r_{table_short}"
            if tbl not in existing:
                continue
            sql = (
                f"SELECT substr(startDate, 1, 10) AS date, {agg} AS val "
                f'FROM "{tbl}" WHERE value IS NOT NULL '
                "GROUP BY substr(startDate, 1, 10)"
            )
            df = pd.read_sql_query(sql, conn)
            df["date"] = pd.to_datetime(df["date"])
            frames.append(df.rename(columns={"val": metric}).set_index("date"))

        # Sleep needs special handling (category values + duration math).
        if "r_SleepAnalysis" in existing:
            sleep_df = pd.read_sql_query(
                """
                SELECT substr(endDate, 1, 10) AS date,
                       SUM((julianday(substr(endDate,1,19)) - julianday(substr(startDate,1,19))) * 24 * 60) AS sleep_minutes
                  FROM r_SleepAnalysis
                 WHERE value IN (
                    'HKCategoryValueSleepAnalysisAsleepUnspecified',
                    'HKCategoryValueSleepAnalysisAsleepCore',
                    'HKCategoryValueSleepAnalysisAsleepDeep',
                    'HKCategoryValueSleepAnalysisAsleepREM',
                    'HKCategoryValueSleepAnalysisAsleep'
                 )
                 GROUP BY substr(endDate, 1, 10)
                """,
                conn,
            )
            sleep_df["date"] = pd.to_datetime(sleep_df["date"])
            frames.append(sleep_df.set_index("date"))

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).sort_index().reset_index()
    return out


def rhr_by_source(db_path: Path) -> pd.DataFrame:
    """Daily mean resting HR, *segmented by source device*.

    Use this instead of ``daily_metrics`` when the trend over multi-year
    spans might straddle a device switch (e.g. Garmin → Apple Watch).
    Optical sensors have well-known inter-device biases (~5–10 bpm at rest
    is typical) and merging sources blindly will produce artefactual step-
    changes on the switch date.

    Returns columns: date, source, n, rhr_bpm.
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT substr(startDate, 1, 10) AS date,
                   sourceName AS source,
                   AVG(CAST(value AS REAL)) AS rhr_bpm,
                   COUNT(*) AS n
              FROM r_RestingHeartRate
             WHERE value IS NOT NULL
             GROUP BY substr(startDate, 1, 10), sourceName
             ORDER BY date
            """,
            conn,
        )
    df["date"] = pd.to_datetime(df["date"])
    return df


def workout_records(
    db_path: Path, workout_start: pd.Timestamp, workout_end: pd.Timestamp,
    type_short: str,
) -> pd.DataFrame:
    """All records of `type_short` whose start falls inside the workout window.

    Uses the indexed ``startDate_ts`` (Unix seconds) column for a fast,
    timezone-safe range scan.
    """
    table = f"r_{type_short}"
    # Accept any timezone-aware or naive pandas Timestamp.
    start_ts = int(pd.Timestamp(workout_start).tz_convert("UTC").timestamp()) \
        if pd.Timestamp(workout_start).tzinfo else int(pd.Timestamp(workout_start).timestamp())
    end_ts = int(pd.Timestamp(workout_end).tz_convert("UTC").timestamp()) \
        if pd.Timestamp(workout_end).tzinfo else int(pd.Timestamp(workout_end).timestamp())
    sql = (
        f'SELECT startDate, endDate, value, unit, sourceName, startDate_ts '
        f'FROM "{table}" '
        f"WHERE startDate_ts BETWEEN ? AND ? "
        f"ORDER BY startDate_ts"
    )
    df = _read_sql(db_path, sql, params=(start_ts, end_ts))
    if not df.empty:
        ts = pd.to_numeric(df["startDate_ts"], errors="coerce")
        df["startDate"] = pd.to_datetime(ts, unit="s", utc=True)
        df["endDate"] = pd.to_datetime(df["endDate"], utc=True, errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df
