#!/usr/bin/env python3
"""Parse every FIT file in the Garmin GDPR archive into per-second Parquet streams.

Input:  data/raw/garmin_gdpr/fit/**/*.fit     (extracted from UploadedFiles_Part{1..4}.zip)
Output: data/processed/streams/<hash>.parquet (one per activity with record data)
        data/raw/garmin_gdpr_summaries.parquet (session summary per ingested FIT)

Idempotent — `src.fit.fit_to_parquet` skips any FIT whose output parquet already
exists. Non-activity FITs (daily monitoring, settings, etc.) have no `record`
messages and are naturally skipped with no output.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fit import ingest_directory  # noqa: E402

FIT_DIR = ROOT / "data" / "raw" / "garmin_gdpr" / "fit"
STREAMS_DIR = ROOT / "data" / "processed" / "streams"
SUMMARY_PATH = ROOT / "data" / "raw" / "garmin_gdpr_summaries.parquet"


def main() -> None:
    if not FIT_DIR.exists():
        sys.exit(f"{FIT_DIR} not found — extract UploadedFiles_Part*.zip first")
    summaries = ingest_directory(FIT_DIR, STREAMS_DIR)
    if summaries.empty:
        print("No session summaries produced.")
        return
    summaries.to_parquet(SUMMARY_PATH, index=False)
    print(f"\n✓ Wrote {len(summaries)} session summaries → {SUMMARY_PATH}")
    # Quick breakdown
    if "sport" in summaries.columns:
        print("\nBy sport:")
        print(summaries["sport"].value_counts().head(20).to_string())
    if "start_time" in summaries.columns:
        import pandas as pd
        st = pd.to_datetime(summaries["start_time"], utc=True, errors="coerce")
        print(f"\nDate range: {st.min()} → {st.max()}  ({st.notna().sum()} with timestamps)")


if __name__ == "__main__":
    main()
