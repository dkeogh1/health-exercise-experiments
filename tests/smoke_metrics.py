#!/usr/bin/env python3
"""Smoke-test every major function in ``src.metrics`` against real data.

Runs:
- Observed HRmax / HRrest
- Banister + Edwards TRIMP on a recent session
- hrTSS + efficiency factor
- daily_load + ctl_atl_tsb across all 1,710 sessions
- ACWR
- MMP + Monod-Scherrer CP/W' on whatever streams are on disk
- Time-in-zone + aerobic decoupling on one stream
- Rolling-z on HRV
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import apple_health as ah  # noqa: E402
from src import metrics as m  # noqa: E402

sessions = pd.read_parquet(ROOT / "data" / "processed" / "sessions.parquet")
sessions["start_time_utc"] = pd.to_datetime(sessions["start_time_utc"], utc=True)
sessions["year"] = sessions["start_time_utc"].dt.year

apple_db = ROOT / "data" / "processed" / "apple_health.db"
apple_daily = ah.daily_metrics(apple_db) if apple_db.exists() else None

print("=" * 70)
print("ANCHORS")
print("=" * 70)
hrmax = m.observed_hrmax(sessions)
hrrest = m.observed_hrrest(apple_daily)
lthr = m.lthr_from_hrmax(hrmax)
print(f"Observed HRmax (99.5 pct of max_hr): {hrmax:.0f} bpm")
print(f"Observed HRrest (5th pct of daily RHR): {hrrest:.1f} bpm")
print(f"LTHR proxy (0.91 · HRmax): {lthr:.0f} bpm")

# Per-year anchors for the last 6 years.
years = sorted(sessions["year"].dropna().unique())[-6:]
print(f"\nPer-year HRmax + LTHR + observed HRrest ({years[0]}–{years[-1]}):")
for y in years:
    a = m.anchors_for_year(int(y), sessions, apple_daily)
    print(f"  {int(y)}  HRmax={a.hrmax!s:>6}  HRrest={a.hrrest!s:>6}  LTHR={a.lthr!s:>6}")

print("\n" + "=" * 70)
print("SESSION-LEVEL LOAD (latest Run)")
print("=" * 70)
latest_run = (
    sessions[sessions["activity_type"] == "run"]
    .sort_values("start_time_utc", ascending=False)
    .iloc[0]
)
print(
    f"Session: {latest_run['activity_name']!r} "
    f"@ {latest_run['start_time_utc'].date()}  "
    f"dur={latest_run['duration_s']/60:.1f} min  "
    f"avg_hr={latest_run['avg_hr']}  max_hr={latest_run['max_hr']}"
)
banister = m.banister_trimp(
    latest_run["duration_s"], latest_run["avg_hr"], hrmax, hrrest,
)
edwards = m.edwards_trimp_from_avg(
    latest_run["duration_s"], latest_run["avg_hr"], hrmax,
)
hrtss_v = m.hrtss(latest_run["duration_s"], latest_run["avg_hr"], lthr)
ef = m.efficiency_factor(latest_run.get("avg_power"), latest_run["avg_hr"])
print(f"  Banister TRIMP       = {banister:.1f}")
print(f"  Edwards TRIMP (avg)  = {edwards:.1f}")
print(f"  hrTSS                = {hrtss_v:.1f}")
print(f"  Efficiency factor    = {ef!s}")

print("\n" + "=" * 70)
print("DAILY LOAD + PMC (CTL/ATL/TSB) ACROSS FULL HISTORY")
print("=" * 70)
# Build per-year LTHR dict so hrTSS uses the right anchor for each year.
lthr_by_year: dict[int, float] = {}
for y in sorted(sessions["year"].dropna().unique()):
    a = m.anchors_for_year(int(y), sessions)
    if a.lthr is not None:
        lthr_by_year[int(y)] = a.lthr

daily = m.daily_load(sessions, lthr_by_year=lthr_by_year)
pmc = m.ctl_atl_tsb(daily)
print(f"Daily rows: {len(pmc)}   "
      f"({pmc['date'].min().date()} → {pmc['date'].max().date()})")
print(f"Peak CTL: {pmc['ctl'].max():.0f} on {pmc.loc[pmc['ctl'].idxmax(), 'date'].date()}")
print(f"Current CTL: {pmc['ctl'].iloc[-1]:.0f}  ATL: {pmc['atl'].iloc[-1]:.0f}  "
      f"TSB: {pmc['tsb'].iloc[-1]:+.0f}")
# Spot-check coverage vs. Garmin-native TSS.
tss_cov = sessions["tss"].notna().mean() * 100
print(f"\nSession-level TSS coverage: {tss_cov:.1f}%  "
      f"(rest use hrTSS fallback)")

print("\n" + "=" * 70)
print("ACWR (Gabbett) — compute but interpret skeptically (Impellizzeri 2020)")
print("=" * 70)
acwr = m.acwr(pmc)
print(f"Recent ACWR (last 7 d): {acwr.tail(7).round(2).tolist()}")
print(f"All-time distribution: {acwr.describe().round(2).to_dict()}")

print("\n" + "=" * 70)
print("STREAM-LEVEL METRICS")
print("=" * 70)
stream_dir = ROOT / "data" / "processed" / "streams"
stream_files = sorted(stream_dir.glob("strava_*.parquet"))
print(f"Streams on disk: {len(stream_files)}")

if stream_files:
    # All-time MMP curve (across however many streams are ready).
    print("\nMean-maximal power (across all available streams):")
    streams = [pd.read_parquet(f) for f in stream_files]
    mmp = m.mmp_across_streams(streams)
    for dur in [1, 5, 60, 300, 600, 1200, 3600]:
        if dur in mmp.index and pd.notna(mmp[dur]):
            print(f"  {dur:>5}s  {mmp[dur]:5.0f} W")

    cp, w_prime = m.critical_power(mmp)
    if pd.notna(cp):
        print(f"\nMonod-Scherrer CP fit (180–1200s):  CP={cp:.0f} W   W'={w_prime:.0f} J")

    # Per-stream demo: pick one with power.
    for st, f in zip(streams, stream_files):
        if "power" in st.columns and st["power"].notna().sum() > 300:
            print(f"\nPicked stream {f.name} for per-session demo ({len(st)} rows)")
            tss_from_p = m.tss_from_power_stream(st, ftp=max(200, cp or 200))
            dec = m.aerobic_decoupling(st)
            drift = m.hr_drift(st)
            hr_zones = m.time_in_zone(st, hrmax, zones="pct_hrmax")
            print(f"  TSS (from power stream, FTP guess): {tss_from_p:.1f}")
            print(f"  Aerobic decoupling: {dec!s}%")
            print(f"  HR drift: {drift:+.2f} bpm/min" if drift is not None else "  HR drift: n/a")
            print(f"  Seconds in HR zone:")
            for zone, secs in hr_zones.items():
                print(f"    {zone}  {int(secs):>5} s  ({secs/60:>5.1f} min)")
            break

print("\n" + "=" * 70)
print("HRV rolling z-score (Plews/Altini style) — proof of life")
print("=" * 70)
if apple_daily is not None and "hrv_sdnn_ms" in apple_daily.columns:
    z = m.rolling_z(apple_daily["hrv_sdnn_ms"], window=28)
    non_null = z.dropna()
    if not non_null.empty:
        print(f"Non-null z-scores: {len(non_null)}")
        print(f"Latest 5: {non_null.tail(5).round(2).tolist()}")

print("\n✓ smoke test complete")
