#!/usr/bin/env python3
"""Convert data/raw/apple_health_export/export.xml -> data/processed/apple_health.db."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make `src` importable when running this as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import apple_health  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
XML = ROOT / "data" / "raw" / "apple_health_export" / "export.xml"
DB = ROOT / "data" / "processed" / "apple_health.db"


def main() -> None:
    if not XML.exists():
        sys.exit(f"Not found: {XML}")
    print(f"Ingesting {XML} ({XML.stat().st_size/1e9:.2f} GB)...")
    t0 = time.time()
    counts = apple_health.xml_to_sqlite(XML, DB)
    dt = time.time() - t0
    print(f"Done in {dt:.1f}s. DB at {DB} ({DB.stat().st_size/1e6:.1f} MB)")
    print()
    print("Top 25 tables by row count:")
    for table, n in sorted(counts.items(), key=lambda kv: -kv[1])[:25]:
        print(f"  {n:>10,}  {table}")


if __name__ == "__main__":
    main()
