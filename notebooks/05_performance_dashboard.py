# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Performance dashboard
#
# Reads `data/processed/sessions.parquet`, the streams dir, and
# `data/raw/apple_health.csv` / `data/processed/apple_health.db`, and renders
# the canonical endurance-analytics views through `src.metrics`:
#
# 1. Anchors snapshot — HRmax / HRrest / LTHR / FTP per year
# 2. Performance Management Chart (CTL / ATL / TSB over the full history)
# 3. TSS calendar heatmap
# 4. Weekly training volume by activity type
# 5. HR-vs-pace per year (runs)
# 6. Mean-maximal power curve (all-time + per year, from streams)
# 7. Aerobic decoupling trend (Pw:HR split-half, from streams)
# 8. Time-in-zone stacked weekly bars + 80:20 polarization index (from streams)
# 9. RHR + HRV + CTL three-panel overlay
#
# Stream-dependent sections (6, 7, 8) degrade gracefully when the background
# Strava streams pull isn't finished — they only use whatever Parquet files
# are on disk.

# %%
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 5)
# %matplotlib inline

import sys
ROOT = Path("..").resolve()
sys.path.insert(0, str(ROOT))

from src import apple_health as ah
from src import metrics as m

SESSIONS_PATH = ROOT / "data" / "processed" / "sessions.parquet"
STREAMS_DIR   = ROOT / "data" / "processed" / "streams"
APPLE_DB      = ROOT / "data" / "processed" / "apple_health.db"
APPLE_CSV     = ROOT / "data" / "raw" / "apple_health.csv"

sessions = pd.read_parquet(SESSIONS_PATH)
sessions["start_time_utc"] = pd.to_datetime(sessions["start_time_utc"], utc=True)
sessions["date"] = sessions["start_time_utc"].dt.tz_convert(None).dt.date
sessions["date"] = pd.to_datetime(sessions["date"])
sessions["year"] = sessions["start_time_utc"].dt.year
apple_daily = pd.read_csv(APPLE_CSV, parse_dates=["date"])

stream_files = sorted(STREAMS_DIR.glob("strava_*.parquet"))
print(f"sessions.parquet : {len(sessions):,} rows "
      f"({sessions['start_time_utc'].min().date()} → "
      f"{sessions['start_time_utc'].max().date()})")
print(f"streams available: {len(stream_files)}")
print(f"apple_health.csv : {len(apple_daily):,} daily rows")

# %% [markdown]
# ## 1. Anchors snapshot
#
# Per-year HRmax (99.5th percentile of session max HR), HRrest (5th pct of
# daily RHR from Apple Health), and the derived LTHR (0.91 · HRmax).
# These anchors drive the hrTSS / TRIMP / zone calculations below.

# %%
years = sorted(sessions["year"].dropna().astype(int).unique())
rows = []
for y in years:
    a = m.anchors_for_year(y, sessions, apple_daily)
    rows.append({
        "year": y,
        "HRmax": round(a.hrmax, 1) if a.hrmax else None,
        "HRrest": round(a.hrrest, 1) if a.hrrest else None,
        "LTHR": round(a.lthr, 1) if a.lthr else None,
    })
anchors_df = pd.DataFrame(rows)
print(anchors_df.to_string(index=False))

fig = go.Figure()
fig.add_trace(go.Scatter(x=anchors_df["year"], y=anchors_df["HRmax"], name="HRmax",
                         mode="lines+markers", line_color="#d62728"))
fig.add_trace(go.Scatter(x=anchors_df["year"], y=anchors_df["LTHR"], name="LTHR",
                         mode="lines+markers", line_color="#ff7f0e"))
fig.add_trace(go.Scatter(x=anchors_df["year"], y=anchors_df["HRrest"], name="HRrest",
                         mode="lines+markers", line_color="#2ca02c"))
fig.update_layout(title="Physiological anchors by year",
                  yaxis_title="bpm", xaxis_title="Year", height=350)
fig.show()

# %% [markdown]
# ## 2. Performance Management Chart (PMC)
#
# The canonical Coggan/Banister chart: grey TSS bars per day, blue CTL (42-day
# EWMA, "fitness"), red ATL (7-day EWMA, "fatigue"), black TSB line (CTL−ATL,
# "form"). Load = Garmin TSS where present (last 2 years), otherwise hrTSS
# computed from avg_hr + per-year LTHR.

# %%
lthr_by_year = {int(r["year"]): float(r["LTHR"])
                for _, r in anchors_df.iterrows() if pd.notna(r["LTHR"])}
daily = m.daily_load(sessions, lthr_by_year=lthr_by_year)
pmc = m.ctl_atl_tsb(daily)
print(f"Current  CTL={pmc['ctl'].iloc[-1]:.0f}   "
      f"ATL={pmc['atl'].iloc[-1]:.0f}   "
      f"TSB={pmc['tsb'].iloc[-1]:+.0f}")
print(f"All-time peak CTL = {pmc['ctl'].max():.0f} on "
      f"{pmc.loc[pmc['ctl'].idxmax(), 'date'].date()}")

fig = go.Figure()
fig.add_trace(go.Bar(x=pmc["date"], y=pmc["load"], name="TSS",
                     marker_color="lightgray", opacity=0.5))
fig.add_trace(go.Scatter(x=pmc["date"], y=pmc["ctl"], name="CTL (fitness)",
                         line=dict(color="#1f77b4", width=2)))
fig.add_trace(go.Scatter(x=pmc["date"], y=pmc["atl"], name="ATL (fatigue)",
                         line=dict(color="#d62728", width=2)))
fig.add_trace(go.Scatter(x=pmc["date"], y=pmc["tsb"], name="TSB (form)",
                         line=dict(color="black", width=1.5, dash="dash"),
                         yaxis="y2"))
fig.update_layout(
    title="Performance Management Chart (10-year)",
    height=500, hovermode="x unified",
    yaxis=dict(title="TSS / CTL / ATL"),
    yaxis2=dict(title="TSB", overlaying="y", side="right", showgrid=False),
    legend=dict(orientation="h", y=-0.15),
)
fig.show()

# %% [markdown]
# ## 3. TSS calendar heatmap
#
# Each cell = a single day's training load. Dark clusters flag heavy blocks;
# white stretches = rest or off-season.

# %%
# Simple hand-rolled calendar heatmap (52 weeks × 7 days, one block per year).
# `july` is broken against recent matplotlib; not worth the monkeypatch.
def calendar_heatmap(series: pd.Series, ax, year: int, vmax: float) -> None:
    """Plot one calendar year as a 7×53 grid."""
    year_series = series[series.index.year == year]
    # Map each date → (week_of_year, day_of_week).
    grid = np.full((7, 54), np.nan)  # 00–53 inclusive
    for d, v in year_series.items():
        week = int(d.strftime("%U"))  # 00–53
        dow = int(d.strftime("%w"))   # 0=Sun
        grid[dow, week] = v
    ax.imshow(grid, cmap="YlOrRd", aspect="auto",
              vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"], fontsize=8)
    ax.set_xticks([])
    ax.set_title(f"{year}", fontsize=10, loc="left")
    ax.set_facecolor("whitesmoke")


cutoff = pmc["date"].max() - pd.DateOffset(years=5)
pmc_recent = pmc[pmc["date"] >= cutoff].set_index("date")["load"]
years_to_show = sorted(pmc_recent.index.year.unique())
vmax = float(pmc_recent.quantile(0.95))  # clip top 5% so heavy days don't wash out

fig, axes = plt.subplots(len(years_to_show), 1, figsize=(16, 1.2 * len(years_to_show)))
for ax, y in zip(np.atleast_1d(axes), years_to_show):
    calendar_heatmap(pmc_recent, ax, y, vmax)
plt.suptitle("Daily training load (TSS) — each cell = 1 day", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Weekly volume by activity type
#
# Stacked bar of hours per week, split by sport. Lets you spot multi-sport
# blocks and injury-driven single-sport periods at a glance.

# %%
weekly = (
    sessions.assign(week=sessions["date"].dt.to_period("W-SUN").apply(lambda p: p.start_time))
    .groupby(["week", "activity_type"])["duration_s"].sum()
    .unstack(fill_value=0) / 3600.0
)
weekly.plot(kind="bar", stacked=True, width=1.0, figsize=(18, 5),
            cmap="tab10")
plt.title("Weekly training hours by activity type")
plt.ylabel("Hours")
plt.xlabel("Week")
plt.xticks([])  # too dense to label; hovercontrol not needed in static
plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. HR-vs-pace per year (runs only)
#
# Fitness drift over time: at the same HR, pace should get faster (curves
# move up). Each dot is one run; colored by year. Runs only because cycling
# pace is dominated by terrain/wind.

# %%
runs = sessions[
    sessions["activity_type"].isin(["run", "trail_run"])
    & sessions["avg_hr"].notna()
    & sessions["distance_m"].notna()
    & (sessions["duration_s"] > 0)
].copy()
runs["pace_kmh"] = runs["distance_m"] / 1000 / (runs["duration_s"] / 3600)
# Clean obvious outliers (pace > 25 km/h on a run = treadmill bug / mis-tag).
runs = runs[(runs["pace_kmh"] > 4) & (runs["pace_kmh"] < 25)]

fig = px.scatter(
    runs, x="avg_hr", y="pace_kmh", color="year",
    opacity=0.6,
    title=f"HR vs pace (runs, n={len(runs):,})",
    labels={"avg_hr": "Avg HR (bpm)", "pace_kmh": "Pace (km/h)"},
    color_continuous_scale="Viridis",
)
fig.update_layout(height=450)
fig.show()

# Year-by-year pace-at-moderate-HR: restrict to 140–160 bpm band for apples-to-apples.
band = runs[(runs["avg_hr"] >= 140) & (runs["avg_hr"] <= 160)]
yearly_pace = band.groupby("year")["pace_kmh"].agg(["count", "median", "mean"]).round(2)
print("Median pace at moderate HR (140–160 bpm):")
print(yearly_pace.to_string())

# %% [markdown]
# ## 6. Mean-maximal power curve (from Strava streams)
#
# Log-duration on x, best power on y. The all-time envelope is the outer hull
# of every year's curve. Useful for: tracking fitness year-over-year, seeing
# which durations you've actually pushed hard at, and fitting CP/W' from the
# "sustainable" portion (3–20 min).
#
# Limited by the number of streams on disk — currently **{N}** of 1,651.

# %%
print(f"Streams available for MMP: {len(stream_files)}")
if stream_files:
    # All-time curve
    streams_by_year: dict[int, list[pd.DataFrame]] = {}
    for f in stream_files:
        df = pd.read_parquet(f)
        if "power" not in df.columns or df["power"].notna().sum() < 60:
            continue
        ts = pd.to_datetime(df["timestamp"], utc=True).min() if "timestamp" in df else None
        y = ts.year if ts is not None else 0
        streams_by_year.setdefault(y, []).append(df)

    durations = [1, 5, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600]
    curves = {}
    if streams_by_year:
        all_streams = [s for lst in streams_by_year.values() for s in lst]
        curves["all-time"] = m.mmp_across_streams(all_streams, durations_s=durations)
        for y in sorted(streams_by_year):
            curves[str(y)] = m.mmp_across_streams(streams_by_year[y], durations_s=durations)

        fig = go.Figure()
        for name, curve in curves.items():
            vis = curve.dropna()
            if vis.empty:
                continue
            width = 3 if name == "all-time" else 1.3
            color = "black" if name == "all-time" else None
            fig.add_trace(go.Scatter(
                x=vis.index, y=vis.values, mode="lines+markers", name=name,
                line=dict(width=width, color=color),
            ))
        fig.update_xaxes(type="log", title="Duration (s, log)")
        fig.update_yaxes(title="Best mean power (W)")
        fig.update_layout(
            title=f"Mean-maximal power curve ({len(all_streams)} power streams)",
            height=450,
        )
        fig.show()

        all_time = curves["all-time"]
        cp, w_prime = m.critical_power(all_time, start_s=180, end_s=1200)
        if pd.notna(cp):
            print(f"Monod-Scherrer fit (180–1200s):  CP ≈ {cp:.0f} W   W' ≈ {w_prime:.0f} J")
            print("(Note: as the streams pull completes, CP will revise upward for")
            print(" hard short efforts pulled from later activities.)")
    else:
        print("No power-carrying streams yet; chart skipped.")
else:
    print("No streams pulled yet; chart skipped.")

# %% [markdown]
# ## 7. Aerobic decoupling trend
#
# For each long steady-state session (≥30 min with HR + power), compute Friel's
# Pw:HR split-half decoupling. Values > 5% = HR crept up while power stayed
# flat (aerobic-base shortfall or under-fueling). Trend down over time =
# aerobic-base gains.

# %%
decoupling_points = []
for f in stream_files:
    df = pd.read_parquet(f)
    if "power" not in df.columns or "heart_rate" not in df.columns:
        continue
    ts = pd.to_datetime(df["timestamp"], utc=True).min() if "timestamp" in df.columns else None
    if ts is None:
        continue
    dec = m.aerobic_decoupling(df, min_duration_s=1800)
    if dec is None:
        continue
    aid = int(f.stem.removeprefix("strava_"))
    decoupling_points.append({
        "date": ts, "decoupling_pct": dec, "activity_id": aid,
    })

if decoupling_points:
    dp = pd.DataFrame(decoupling_points).sort_values("date")
    print(f"Decoupling measurements: {len(dp)}")
    fig = px.scatter(dp, x="date", y="decoupling_pct",
                     title="Aerobic decoupling (Pw:HR split-half) trend",
                     labels={"decoupling_pct": "Decoupling %"},
                     trendline="lowess", opacity=0.6)
    fig.add_hline(y=5, line_dash="dash", line_color="red",
                  annotation_text="5% threshold (aerobic base red flag)")
    fig.update_layout(height=400)
    fig.show()
else:
    print("No ≥30-min HR+power streams yet; decoupling chart skipped.")

# %% [markdown]
# ## 8. Time-in-zone + 80:20 polarization index
#
# Sums HR-zone seconds across all available streams, bucketed by week.
# The **polarization index** = (Z1+Z2 time) / (Z4+Z5 time); Seiler's 80:20
# rule suggests endurance athletes do ~80% easy (Z1/Z2) and ~20% hard
# (Z4/Z5), with very little in the middle.

# %%
if stream_files and anchors_df["HRmax"].notna().any():
    hrmax_global = float(anchors_df["HRmax"].max())
    week_zone = []
    for f in stream_files:
        df = pd.read_parquet(f)
        if "heart_rate" not in df.columns:
            continue
        ts = pd.to_datetime(df["timestamp"], utc=True).min() if "timestamp" in df.columns else None
        if ts is None:
            continue
        z = m.time_in_zone(df, hrmax_global, zones="pct_hrmax")
        if z.empty:
            continue
        week = pd.Timestamp(ts).tz_convert(None).to_period("W-SUN").start_time
        rec = {"week": week}
        for zone in z.index:
            rec[zone] = int(z[zone])
        week_zone.append(rec)

    if week_zone:
        zdf = pd.DataFrame(week_zone).fillna(0).groupby("week").sum()
        zdf_h = zdf / 3600  # hours
        # Polarization index per week.
        pol = (zdf["z1"] + zdf["z2"]) / (zdf["z4"] + zdf["z5"]).replace(0, np.nan)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(16, 7), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        zdf_h.plot(kind="bar", stacked=True, ax=ax1, width=1.0,
                   color=["#1b9e77", "#7fcdbb", "#fee08b", "#fc8d59", "#d73027"])
        ax1.set_ylabel("Hours"); ax1.set_title("Weekly time-in-zone (from available streams)")
        ax1.legend(title="HR zone", bbox_to_anchor=(1.01, 1), loc="upper left")
        pol.plot(kind="line", ax=ax2, color="black", marker=".")
        ax2.axhline(4, color="red", linestyle="--", linewidth=0.8)
        ax2.set_ylabel("(Z1+Z2) / (Z4+Z5)")
        ax2.set_title("Polarization index (Seiler 80:20 ≈ 4.0)")
        ax2.set_xticks([])
        plt.tight_layout()
        plt.show()
else:
    print("No streams yet, or no HRmax anchor; zone-distribution chart skipped.")

# %% [markdown]
# ## 9. RHR + HRV + CTL three-panel overlay
#
# The classic overtraining dashboard: rising RHR + falling HRV + persistently
# high CTL = overreach warning. All on the same x-axis so signals align
# visually.
#
# **Important**: RHR is plotted *per source device* rather than as a merged
# series, because optical-HR sensors have systematic inter-device bias
# (~5–10 bpm at rest). Merging Garmin + Apple Watch RHR into one line
# produced a spurious step-up on the mid-2025 device switch; segmenting
# by source reveals the actual within-device trend is fitness-gain.

# %%
rhr_src = ah.rhr_by_source(APPLE_DB)
apple = apple_daily.sort_values("date").set_index("date").copy()
apple["hrv_r7"] = apple["hrv_sdnn_ms"].rolling(7, min_periods=3).mean()

pmc_indexed = pmc.set_index("date").copy()

fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
# RHR — colour by source device
colors = {"Connect": "#1f77b4", "Daniel’s Apple Watch": "#d62728"}
for src, grp in rhr_src.groupby("source"):
    grp = grp.sort_values("date")
    rolled = grp.set_index("date")["rhr_bpm"].rolling(7, min_periods=3).mean()
    axes[0].scatter(grp["date"], grp["rhr_bpm"], s=4,
                    color=colors.get(src, "#888"), alpha=0.3, label=None)
    axes[0].plot(rolled.index, rolled.values,
                 color=colors.get(src, "#888"), linewidth=2,
                 label=f"{src} (7-day avg)")
axes[0].set_ylabel("RHR (bpm)")
axes[0].legend(loc="upper right")
axes[0].set_title("Resting HR — by source device")

# HRV
axes[1].plot(apple.index, apple["hrv_sdnn_ms"], color="#cccccc",
             label="HRV-SDNN (daily)", linewidth=0.5, marker=".")
axes[1].plot(apple.index, apple["hrv_r7"], color="#1f77b4",
             label="HRV 7-day avg", linewidth=2)
axes[1].set_ylabel("HRV-SDNN (ms)")
axes[1].legend(loc="upper right")
axes[1].set_title("HRV (Apple SDNN; sparse by design)")

# CTL
axes[2].plot(pmc_indexed.index, pmc_indexed["ctl"], color="#1f77b4", label="CTL")
axes[2].plot(pmc_indexed.index, pmc_indexed["atl"], color="#d62728", label="ATL")
axes[2].set_ylabel("Training load")
axes[2].legend(loc="upper right")
axes[2].set_title("CTL (fitness) / ATL (fatigue)")

# Focus last 3 years so sparse earlier data doesn't dominate.
axes[-1].set_xlim(pmc_indexed.index.max() - pd.DateOffset(years=3),
                  pmc_indexed.index.max())
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Weather × pace (and HR) for outdoor sessions
#
# Historical weather comes from Open-Meteo's ERA5-backed archive, joined to
# every activity that has a stream file (so we can pull its start lat/lng).
# Indoor/Zwift sessions are deliberately skipped.
#
# Two questions this chart answers:
# 1. Does temperature drag your pace at a fixed HR? (heat stress visibility)
# 2. At what temperature do you run the fastest, HR-for-HR?

# %%
enriched = sessions[sessions["air_temp_c"].notna()].copy()
print(f"Sessions with weather: {len(enriched):,}  "
      f"({enriched['air_temp_c'].notna().sum() / len(sessions) * 100:.1f}% of total)")

if len(enriched):
    # Pace in km/h for outdoor runs + rides
    enriched["pace_kmh"] = (
        enriched["distance_m"] / 1000 /
        (enriched["duration_s"] / 3600)
    )
    outdoor = enriched[
        enriched["activity_type"].isin(["run", "trail_run", "ride"])
        & enriched["avg_hr"].notna()
        & (enriched["pace_kmh"] > 4)
        & (enriched["pace_kmh"] < 45)
    ].copy()

    # Bin temperature into 4-degree buckets for legibility.
    outdoor["temp_bucket"] = pd.cut(
        outdoor["air_temp_c"],
        bins=[-10, 0, 5, 10, 15, 20, 25, 30, 40],
    )

    # Pace vs apparent temperature at moderate HR band (140-160 bpm) — runs.
    run_band = outdoor[
        outdoor["activity_type"].isin(["run", "trail_run"])
        & outdoor["avg_hr"].between(140, 160)
    ]
    if len(run_band) >= 10:
        fig = px.scatter(
            run_band, x="apparent_temp_c", y="pace_kmh",
            color="humidity_pct", size="duration_s",
            trendline="ols",
            color_continuous_scale="RdYlBu_r",
            title=(
                f"Run pace vs. apparent temperature  "
                f"(HR 140–160 bpm only, n={len(run_band)})"
            ),
            labels={
                "apparent_temp_c": "Apparent temperature (°C)",
                "pace_kmh": "Pace (km/h)",
                "humidity_pct": "Humidity %",
            },
        )
        fig.update_layout(height=450)
        fig.show()
    else:
        print("Not enough weather-tagged runs in the 140-160 bpm band yet.")

    # Pace-per-temp-bucket — broken down by activity type.
    summary = (
        outdoor.groupby(["activity_type", "temp_bucket"], observed=True)
        ["pace_kmh"].agg(["count", "median"]).round(2)
    )
    print("Median pace by temp bucket × activity type:")
    print(summary.to_string())

    # HR vs temp at constant effort — are you running 'hotter' in heat?
    fig = px.scatter(
        outdoor, x="air_temp_c", y="avg_hr",
        color="activity_type",
        opacity=0.6, trendline="lowess",
        title="Avg HR vs air temperature (higher = heat-driven drift)",
        labels={"air_temp_c": "Air temperature (°C)", "avg_hr": "Avg HR (bpm)"},
    )
    fig.update_layout(height=400)
    fig.show()

# %% [markdown]
# ## Notes & next steps
#
# - Stream-dependent charts (6, 7, 8) and the weather overlay (10) fill out
#   automatically as the background streams pull progresses — just re-run.
# - HRV trend coverage is intentionally sparse: Apple only samples HRV during
#   Breathe sessions (~once a day), so expect ~10 % day-coverage.
# - CP, FTP, and the temperature-vs-pace regression will all revise upward
#   in confidence once more streams are on disk; current numbers are a
#   *preview*, not a final answer.
# - **Always segment RHR / HRV / walking-HR by `sourceName`** if your query
#   spans the mid-2025 Garmin → Apple Watch switch — wrist optical sensors
#   have ~8 bpm inter-device bias at rest.
