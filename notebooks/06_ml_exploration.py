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
# # Machine learning exploration
#
# Three interpretable ML methods applied to the personal training archive:
#
# 1. **Change-point detection on CTL** (`ruptures` PELT, L2 cost) —
#    auto-segments the 10-year fitness history into blocks whose mean CTL is
#    roughly constant. Replaces eyeballing the PMC with a data-driven
#    structure.
# 2. **UMAP + KMeans workout clustering** — discovers your "workout
#    vocabulary" (recovery spins, long rides, VO₂max intervals, easy runs,
#    long runs, indoor Zwift, hikes…) from per-session features, without any
#    manual label.
# 3. **IsolationForest daily anomaly detection** — joins daily training load
#    with RHR / HRV / sleep / steps and flags the ~2 % most-unusual days.
#    Good for surfacing illness, data-quality bugs, and outlier events.
#
# All three are deliberately *interpretable* — no opaque deep learning on a
# dataset of O(3,500) days and O(1,700) sessions.

# %%
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (14, 5)
# %matplotlib inline

ROOT = Path("..").resolve()
sys.path.insert(0, str(ROOT))

from src import apple_health as ah
from src import metrics as m
from src import ml

SESSIONS_PATH = ROOT / "data" / "processed" / "sessions.parquet"
APPLE_DB      = ROOT / "data" / "processed" / "apple_health.db"
APPLE_CSV     = ROOT / "data" / "raw" / "apple_health.csv"

sessions = pd.read_parquet(SESSIONS_PATH)
sessions["start_time_utc"] = pd.to_datetime(sessions["start_time_utc"], utc=True)
sessions["year"] = sessions["start_time_utc"].dt.year

apple_daily = pd.read_csv(APPLE_CSV, parse_dates=["date"])

# Build the PMC we'll reuse.
hrmax_by_year = {
    int(y): m.observed_hrmax(sessions[sessions["year"] == y]) or np.nan
    for y in sorted(sessions["year"].dropna().astype(int).unique())
}
lthr_by_year = {y: 0.91 * v for y, v in hrmax_by_year.items() if pd.notna(v)}
daily = m.daily_load(sessions, lthr_by_year=lthr_by_year)
pmc = m.ctl_atl_tsb(daily)
print(f"Sessions: {len(sessions):,}   PMC days: {len(pmc):,}")

# %% [markdown]
# ## 1. Change-point detection on CTL
#
# ``ruptures.Pelt(model="l2")`` finds the number and location of change
# points that minimise L2 reconstruction cost + ``penalty · n_changepoints``.
# Higher penalty → fewer, longer segments.
#
# We print each segment with its mean CTL and mean daily load; then overlay
# the segment boundaries on the PMC.

# %%
segments = ml.change_points_on_ctl(pmc, penalty=50, min_size=30)
print(f"Detected {len(segments)} training blocks (penalty=50):\n")
print(f"{'start':>10}  {'end':>10}  {'days':>5}  {'mean CTL':>9}  {'load/day':>9}")
print("-" * 60)
for s in segments:
    print(
        f"{s.start.date()!s:>10}  {s.end.date()!s:>10}  "
        f"{s.n_days:>5}  {s.mean_ctl:>9.1f}  {s.mean_load:>9.1f}"
    )

# %%
# Overlay: PMC with vertical lines at each change point, colored bands.
fig = go.Figure()
fig.add_trace(go.Scatter(x=pmc["date"], y=pmc["ctl"], name="CTL",
                         line=dict(color="#1f77b4", width=2)))
for i, seg in enumerate(segments):
    fig.add_vrect(x0=seg.start, x1=seg.end,
                  fillcolor=px.colors.qualitative.Set2[i % 8],
                  opacity=0.15, layer="below", line_width=0,
                  annotation_text=f"μ={seg.mean_ctl:.0f}",
                  annotation_position="top left",
                  annotation_font_size=9)
fig.update_layout(
    title=f"CTL with {len(segments)} auto-detected training blocks",
    height=450, yaxis_title="CTL",
)
fig.show()

# %% [markdown]
# ### Interpretation
#
# Each coloured band is a "fitness phase": a period where the PELT algorithm
# judged the mean CTL as roughly constant. Large CTL-jumps between bands are
# the transitions that merit labels ("base block", "peaking", "detraining",
# "off-season"). The algorithm has no idea about calendar years — it just
# looks at the time series.

# %% [markdown]
# ## 2. Workout clustering (UMAP + KMeans)
#
# Feature vector per session:
# duration, distance, avg_hr, avg_power, elevation/km, intensity ratio
# (avg_hr / year's HRmax), kJ/min, plus two one-hot flags for run / ride.
# UMAP → 2D, KMeans with 6 clusters, colour each point in the embedding.

# %%
feat = ml.workout_feature_matrix(sessions, hrmax_by_year=hrmax_by_year)
# Drop rows where avg_hr and avg_power are both zero (no intensity info at all).
has_intensity = (feat["avg_hr"] > 0) | (feat["avg_power"] > 0)
feat = feat[has_intensity]
print(f"Sessions going into UMAP: {len(feat):,}")
print(f"Features: {feat.columns.tolist()}")

clusters = ml.cluster_workouts(feat, n_clusters=6)
clusters = clusters.join(sessions, how="left")

fig = px.scatter(
    clusters, x="umap_1", y="umap_2", color="cluster",
    hover_data=["activity_name", "activity_type", "distance_m",
                "duration_s", "avg_hr", "avg_power"],
    title="UMAP embedding of all sessions, coloured by KMeans cluster",
    color_continuous_scale="Turbo", opacity=0.6,
)
fig.update_layout(height=500)
fig.show()

# Characterise each cluster by the mean of interpretable columns.
print("\nCluster profiles (means):")
profile = clusters.groupby("cluster").agg(
    n=("cluster", "size"),
    dur_min=("duration_s", lambda s: s.mean() / 60),
    dist_km=("distance_m", lambda s: s.mean() / 1000),
    avg_hr=("avg_hr", "mean"),
    avg_power=("avg_power", "mean"),
    indoor_pct=("is_indoor", lambda s: s.mean() * 100),
    run_pct=("activity_type", lambda s: (s.isin(["run", "trail_run"])).mean() * 100),
    ride_pct=("activity_type", lambda s: (s.isin(["ride", "virtual_ride"])).mean() * 100),
).round(1)
print(profile.to_string())

# %% [markdown]
# ### Interpretation
#
# Each cluster should correspond to a natural workout "type":
# easy recovery, long endurance, tempo/threshold, VO₂max intervals, indoor
# Zwift, hike. Cross-check the ``run_pct`` / ``ride_pct`` / ``indoor_pct``
# columns to see which sport dominates each cluster, then use mean duration
# and intensity to label them by hand (once).
#
# Caveat: without per-session stream features (time-in-zone, power-curve
# percentiles) the clusters can't distinguish a 30-min tempo from a 30-min
# easy run — both are "moderate HR, 30 min run". Re-run once the streams
# pull completes.

# %% [markdown]
# ## 3. Daily anomaly detection (IsolationForest)
#
# Join per-day training load with RHR / HRV / sleep / steps. Fit
# ``IsolationForest(contamination=0.02)`` on standardised features — the
# bottom 2 % most-unusual days get flagged.
#
# RHR is taken from the **majority source device** per day (via
# ``apple_health.rhr_by_source``) so the Garmin→Apple Watch switch doesn't
# drive the anomaly score.

# %%
rhr_src = ah.rhr_by_source(APPLE_DB)
daily_feat = ml.build_daily_feature_matrix(pmc, apple_daily, rhr_by_source=rhr_src)
print(f"Daily feature matrix: {len(daily_feat):,} rows, "
      f"complete (no NaN): {len(daily_feat.dropna()):,}")

anomalies = ml.anomaly_scores(
    daily_feat,
    feature_cols=("load", "ctl", "atl", "tsb",
                  "rhr", "hrv_sdnn_ms", "sleep_minutes"),
    contamination=0.02,
)
print(f"\nAnomaly-scored days: {len(anomalies):,}  "
      f"flagged: {anomalies['is_anomaly'].sum()}")
print("\n10 most anomalous days:")
top = anomalies.sort_values("score").head(10)
print(top[["date", "score", "load", "ctl", "rhr", "hrv_sdnn_ms",
           "sleep_minutes"]].round(1).to_string(index=False))

# %%
# Plot: anomaly score over time, highlight flagged days.
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=anomalies["date"], y=anomalies["score"], mode="lines",
    name="Anomaly score (higher = more normal)", line=dict(color="#888"),
))
flagged = anomalies[anomalies["is_anomaly"]]
fig.add_trace(go.Scatter(
    x=flagged["date"], y=flagged["score"], mode="markers",
    marker=dict(color="red", size=9),
    name=f"Flagged ({len(flagged)} days)",
    hovertext=[
        f"{d.date()}<br>load={l:.0f}  CTL={c:.0f}  RHR={r:.0f}  HRV={h:.0f}"
        for d, l, c, r, h in zip(flagged["date"], flagged["load"],
                                  flagged["ctl"], flagged["rhr"],
                                  flagged["hrv_sdnn_ms"])
    ],
))
fig.update_layout(
    title=f"IsolationForest daily anomaly score ({len(anomalies):,} scored days)",
    height=400, yaxis_title="Decision-function score",
    hovermode="closest",
)
fig.show()

# %% [markdown]
# ### Interpretation
#
# The red dots are the ~2 % most-unusual days in your archive — typically
# they cluster around:
#
# - Sudden very-high training load + low recovery (possible overreach)
# - Abnormally low HRV + elevated RHR (illness onset / bad sleep)
# - Very long sessions on normally-recovery days
# - Data-quality weirdness (Apple Watch not worn + Garmin-only day, etc.)
#
# IsolationForest doesn't tell you *why* a day is anomalous — you have to
# eyeball the feature row to diagnose. Use the hover tooltips on the red
# markers.

# %% [markdown]
# ## Not attempted (on purpose)
#
# - **XGBoost "did I hit target today?"** — target definition is fraught
#   (pace at fixed HR band, NP / avg HR ratio, etc.) and careful time-series
#   CV is needed to avoid leakage through CTL (which includes today's
#   load). A motivated future notebook.
# - **State-space Banister** — fitting personal fitness/fatigue τs via a
#   Kalman filter on TSS → performance markers. Good science; needs well-
#   labelled "performance" observations (FTP tests, race results) that we
#   don't have in the archive yet.
# - **DFA-α₁ threshold detection** — needs continuous R-R intervals during
#   exercise. Apple Watch ECG is only 30-second snippets; Garmin FR645 *can*
#   log R-R but must be explicitly enabled. Skip until we confirm R-R in
#   the GDPR archive FITs.
