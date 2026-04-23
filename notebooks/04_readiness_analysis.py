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
# # Readiness & Biometric Correlation
#
# A daily readiness score driven by **HRV, RHR, and sleep**, computed from the
# rebuilt Apple Health ingest (`src/apple_health.py` → `apple_health.csv`).
#
# Methodology follows Plews & Laursen 2013 ("Training Adaptation and HRV in
# Elite Endurance Athletes") and Altini's HRV4Training: compare each day's
# signal to its own rolling baseline rather than to fixed thresholds. All three
# components are z-scored vs. a trailing window, then combined linearly.
#
# ### Column-name changes vs. the previous version of this notebook
# - `hrv_rmssd_ms` → `hrv_sdnn_ms` — Apple stores **SDNN**, not RMSSD, and only
#   samples during Breathe sessions, so we use what's actually there.
# - `sleep_hours` → `sleep_minutes / 60` — the new ingest writes minutes.
# - `heart_rate_min` is still the per-day minimum of HR samples; we prefer
#   `resting_hr` (Apple's own daily estimate) when available.

# %%
import json
import warnings
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)
# %matplotlib inline

print("✅ Libraries imported")

# %% [markdown]
# ## Load data

# %%
DATA_DIR = Path("../data/raw")

# Latest Strava export (one-row-per-activity).
strava_files = sorted(DATA_DIR.glob("activities_*.json"), reverse=True)
assert strava_files, "No Strava activity JSON found in data/raw/"
with open(strava_files[0]) as f:
    strava_raw = json.load(f)

df_strava = pd.DataFrame(strava_raw)
df_strava["start_date"] = pd.to_datetime(df_strava["start_date"])
df_strava["date"] = df_strava["start_date"].dt.date
df_strava["distance_km"] = df_strava["distance_m"] / 1000
df_strava["duration_hours"] = df_strava["duration_s"] / 3600
df_strava["pace_kmh"] = df_strava["distance_km"] / df_strava["duration_hours"]

print(f"✅ Strava: {len(df_strava):,} activities  "
      f"({df_strava['start_date'].min().date()} → {df_strava['start_date'].max().date()})")

# %%
# Apple Health daily aggregate (from src/apple_health.py's daily_metrics()).
apple_path = DATA_DIR / "apple_health.csv"
df_apple = pd.read_csv(apple_path, parse_dates=["date"])
df_apple = df_apple.sort_values("date").reset_index(drop=True)
df_apple["date_only"] = df_apple["date"].dt.date

# Convenience: hours-of-sleep in the unit the eye expects.
df_apple["sleep_hours"] = df_apple["sleep_minutes"] / 60

# RHR: prefer Apple's own `resting_hr`, fall back to daily HR minimum.
df_apple["rhr"] = df_apple["resting_hr"].fillna(df_apple["heart_rate_min"])

print(f"✅ Apple Health: {len(df_apple):,} days  "
      f"({df_apple['date'].min().date()} → {df_apple['date'].max().date()})")
print("\nCoverage on key signals:")
for col in ("rhr", "hrv_sdnn_ms", "sleep_hours", "vo2max", "hrr1"):
    n = df_apple[col].notna().sum()
    pct = n / len(df_apple) * 100
    print(f"  {col:15s}  {n:>5,} days  ({pct:4.1f}%)")

# %% [markdown]
# ## Readiness score (Plews / Altini style)
#
# Each component is a **z-score vs. its own 28-day rolling baseline**, then we
# combine them with opinionated but interpretable weights:
#
# - **z_hrv**: `(ln_sdnn − rolling_mean) / rolling_sd`  → positive is good.
# - **z_rhr**: `−(rhr − rolling_mean) / rolling_sd`  → lower RHR is good, so we
#   flip the sign.
# - **z_sleep**: `(sleep_hours − rolling_mean) / rolling_sd`  → more sleep vs.
#   your own norm is good.
#
# `readiness_z = 0.4·z_hrv + 0.4·z_rhr + 0.2·z_sleep`
#
# The 0.4/0.4/0.2 split weighs recovery signals (HRV, RHR) above sleep
# quantity, because sleep duration has more noise-per-day than
# cardiovascular markers in most amateur data. A principled next step is to
# *fit* the weights from data (§ H6 below can be extended with a regression).
#
# We also compute a traffic-light mapping (Green / Yellow / Red) from the
# z-score using ±0.5 SD thresholds.

# %%
BASELINE_WINDOW_DAYS = 28


def rolling_z(series: pd.Series, window: int) -> pd.Series:
    """Z-score of each point vs. the trailing `window` days (excluding itself)."""
    mean = series.rolling(window, min_periods=7).mean().shift(1)
    std = series.rolling(window, min_periods=7).std().shift(1)
    return (series - mean) / std


d = df_apple.copy()
# HRV: log-transform (RMSSD/SDNN are log-normal; Plews 2013).
d["ln_sdnn"] = np.log(d["hrv_sdnn_ms"])
d["z_hrv"] = rolling_z(d["ln_sdnn"], BASELINE_WINDOW_DAYS)
d["z_rhr"] = -rolling_z(d["rhr"], BASELINE_WINDOW_DAYS)
d["z_sleep"] = rolling_z(d["sleep_hours"], BASELINE_WINDOW_DAYS)

d["readiness_z"] = 0.4 * d["z_hrv"] + 0.4 * d["z_rhr"] + 0.2 * d["z_sleep"]

# Fall back gracefully on days where one component is missing.
has_any = d[["z_hrv", "z_rhr", "z_sleep"]].notna().any(axis=1)
d.loc[~has_any, "readiness_z"] = np.nan


def classify(z):
    if pd.isna(z):
        return "Unknown"
    if z >= 0.5:
        return "Green (Go Hard)"
    if z >= -0.5:
        return "Yellow (Moderate)"
    return "Red (Recover)"


d["readiness_level"] = d["readiness_z"].apply(classify)

print("Readiness distribution over the full dataset:")
print(d["readiness_level"].value_counts())
print("\nreadiness_z summary:")
print(d["readiness_z"].describe().round(2))

# %%
# Readiness over time (last 2 years, drop unknown days).
recent = d[d["date"] >= d["date"].max() - pd.Timedelta(days=730)].dropna(subset=["readiness_z"])

fig = px.scatter(
    recent, x="date", y="readiness_z",
    color="readiness_level",
    color_discrete_map={
        "Green (Go Hard)": "#2ca02c",
        "Yellow (Moderate)": "#ff7f0e",
        "Red (Recover)": "#d62728",
    },
    title="Daily readiness z-score (last 2 years)",
    labels={"readiness_z": "Readiness z"},
)
fig.add_hline(y=0.5, line_dash="dash", line_color="#2ca02c")
fig.add_hline(y=-0.5, line_dash="dash", line_color="#d62728")
fig.show()

# %% [markdown]
# ## H6: Does prior-day readiness predict workout performance?
#
# Join each Strava run/ride to the *previous day's* readiness score and test
# whether readiness correlates with pace. Replicates the original H6 but
# against the corrected score.

# %%
d["next_date"] = d["date_only"].apply(lambda x: x + timedelta(days=1))
prior = d[["next_date", "readiness_z", "readiness_level"]].rename(
    columns={
        "next_date": "date",
        "readiness_z": "prior_readiness_z",
        "readiness_level": "prior_readiness_level",
    }
)

merged = df_strava.merge(prior, on="date", how="left")
df_r = merged[merged["prior_readiness_z"].notna() & merged["pace_kmh"].notna()].copy()

print(f"Matched workouts (with prior-day readiness): {len(df_r):,}")

if len(df_r) >= 20:
    r, p = stats.pearsonr(df_r["prior_readiness_z"], df_r["pace_kmh"])
    print(f"\nPearson correlation: r = {r:+.3f}   p = {p:.4f}   n = {len(df_r)}")
    if p < 0.05:
        direction = "higher" if r > 0 else "lower"
        print(f"  → statistically significant: better readiness → {direction} pace")
    else:
        print("  → not significant at α=0.05")

    print("\nPace by prior-day readiness level:")
    print(
        df_r.groupby("prior_readiness_level")["pace_kmh"]
            .agg(["count", "mean", "median", "std"])
            .round(2)
    )

# %%
if len(df_r) >= 20:
    fig = px.scatter(
        df_r, x="prior_readiness_z", y="pace_kmh",
        color="prior_readiness_level",
        color_discrete_map={
            "Green (Go Hard)": "#2ca02c",
            "Yellow (Moderate)": "#ff7f0e",
            "Red (Recover)": "#d62728",
        },
        trendline="ols",
        title=f"Prior-day readiness vs. pace  (r={r:+.3f}, p={p:.4f}, n={len(df_r)})",
        labels={"prior_readiness_z": "Prior-day readiness z", "pace_kmh": "Pace (km/h)"},
    )
    fig.show()

# %% [markdown]
# ## RHR trend: fitness gain vs. overtraining
#
# Sustained +5 bpm over a 7-day rolling baseline is a classic overreaching
# marker (Buchheit 2014). Below: the raw daily RHR plus a 28-day rolling mean
# to cut noise.

# %%
d["rhr_r7"] = d["rhr"].rolling(7, min_periods=3).mean()
d["rhr_r28"] = d["rhr"].rolling(28, min_periods=7).mean()

fig = px.line(
    d.dropna(subset=["rhr"]),
    x="date", y=["rhr", "rhr_r7", "rhr_r28"],
    title="Resting HR (raw, 7-day avg, 28-day avg)",
    labels={"value": "RHR (bpm)", "variable": ""},
)
fig.show()

recent_rhr = d[d["date"] >= d["date"].max() - timedelta(days=30)]["rhr"].mean()
old_rhr = d[(d["date"] >= d["date"].max() - timedelta(days=90))
            & (d["date"] < d["date"].max() - timedelta(days=60))]["rhr"].mean()
delta = recent_rhr - old_rhr
print(f"Last 30d mean RHR: {recent_rhr:.1f} bpm")
print(f"60-90d ago mean RHR: {old_rhr:.1f} bpm")
print(f"Δ = {delta:+.1f} bpm  →  "
      + ("fitness gain" if delta < -1 else
         "overreaching flag" if delta > 3 else "stable"))

# %% [markdown]
# ## Sleep vs. performance (next-day)

# %%
sp = merged[
    merged["pace_kmh"].notna()
].merge(
    d[["date_only", "sleep_hours"]].rename(columns={"date_only": "date"}),
    on="date", how="left"
).dropna(subset=["sleep_hours"])

if len(sp) >= 20:
    r_sleep, p_sleep = stats.pearsonr(sp["sleep_hours"], sp["pace_kmh"])
    print(f"Sleep vs. next-day pace: r = {r_sleep:+.3f}  p = {p_sleep:.4f}  n = {len(sp)}")
    fig = px.scatter(
        sp, x="sleep_hours", y="pace_kmh", trendline="ols",
        title=f"Previous night's sleep vs. pace  (r={r_sleep:+.3f}, p={p_sleep:.4f})",
        labels={"sleep_hours": "Sleep (hours)", "pace_kmh": "Pace (km/h)"},
    )
    fig.show()
else:
    print("Not enough overlapping sleep/pace data to test.")

# %% [markdown]
# ## Notes & caveats
#
# - **Apple's HRV is SDNN, not RMSSD**, and is sampled opportunistically
#   (mostly during Breathe sessions). Expect ~10% day-coverage and ignore
#   day-to-day swings under ~10 ms.
# - HRV and sleep coverage only began around 2020 (sleep tracking) and 2024
#   (after the Apple Watch Series 10 upgrade), so the readiness score is
#   meaningful for the recent years only.
# - The 0.4/0.4/0.2 component weights are a reasonable default, *not* fitted
#   to your data. A follow-on experiment: regress pace or NP on each of the
#   three z-scores and use the estimated coefficients as weights.
# - The rolling-28-day baseline is the de-facto HRV4Training default and
#   tolerates travel/illness better than a fixed mean.
