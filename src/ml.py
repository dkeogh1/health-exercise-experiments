"""ML helpers for the strava-analysis project.

Three methods, chosen because each produces *interpretable* output on
personal fitness data rather than a black-box score:

1. **Change-point detection on CTL** (`ruptures` PELT, L2 cost) — segments
   your training history into blocks whose mean fitness is roughly constant.
   Useful for auto-labelling "base 2020", "peak 2022-08", "detraining 2023",
   etc., without having to eyeball the PMC.

2. **Workout clustering** (UMAP → KMeans) — builds a feature vector per
   session (duration, intensity, elevation, HR+power profile, sport type),
   embeds to 2D with UMAP, then clusters with KMeans. Discovers your
   "workout vocabulary" — recovery spins vs. VO₂max intervals vs. long
   weekend rides — without a manually-maintained label.

3. **Daily anomaly scores** (IsolationForest) — joins daily training load
   with RHR / HRV / sleep and flags days whose feature vector is far from
   the rest. Good for surfacing "possible illness / data-quality / unusual-
   circumstance" days. *Segments RHR by source device* before feeding the
   model — never merge Garmin + Apple Watch RHR into one series.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# =============================================================================
# 1. Change-point on CTL
# =============================================================================


@dataclass(frozen=True)
class CTLSegment:
    start: pd.Timestamp
    end: pd.Timestamp
    mean_ctl: float
    mean_load: float
    n_days: int


def change_points_on_ctl(
    pmc: pd.DataFrame, penalty: float = 50, min_size: int = 28,
) -> list[CTLSegment]:
    """Segment the CTL time series with PELT (Truong et al.) and an L2 cost.

    Parameters
    ----------
    pmc : DataFrame from ``metrics.ctl_atl_tsb`` (must have ``date``, ``ctl``,
        ``load``).
    penalty : higher = fewer change points. 50 typically yields ~10–20 blocks
        over a 10-year archive.
    min_size : minimum segment length in days.

    Returns a list of ``CTLSegment``s covering the full series contiguously.
    """
    import ruptures as rpt  # local import — heavy dep

    ctl = pmc["ctl"].astype(float).values
    algo = rpt.Pelt(model="l2", min_size=min_size).fit(ctl)
    breaks = algo.predict(pen=penalty)  # indices, last is len(ctl)
    segments: list[CTLSegment] = []
    prev = 0
    for b in breaks:
        s = pmc.iloc[prev:b]
        if s.empty:
            continue
        segments.append(
            CTLSegment(
                start=s["date"].iloc[0],
                end=s["date"].iloc[-1],
                mean_ctl=float(s["ctl"].mean()),
                mean_load=float(s["load"].mean()),
                n_days=len(s),
            )
        )
        prev = b
    return segments


# =============================================================================
# 2. Workout clustering
# =============================================================================


_DEFAULT_FEATURES = (
    "duration_min",
    "distance_km",
    "avg_hr",
    "avg_power",
    "elevation_per_km",
    "intensity_ratio",  # avg_hr / hrmax_year
    "kj_per_min",
)


def workout_feature_matrix(
    sessions: pd.DataFrame,
    hrmax_by_year: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Build a per-session feature matrix suitable for UMAP / KMeans.

    Returns a DataFrame whose index is the session's position in ``sessions``
    and whose columns are the features in ``_DEFAULT_FEATURES`` plus two
    one-hot columns for sport (``is_run`` and ``is_ride``). Indoor sessions
    get ``elevation_per_km = 0``.
    """
    d = sessions.copy()
    d["start_time_utc"] = pd.to_datetime(d["start_time_utc"], utc=True)
    d["year"] = d["start_time_utc"].dt.year
    d["duration_min"] = d["duration_s"] / 60
    d["distance_km"] = d["distance_m"] / 1000
    d["elevation_per_km"] = (
        (d["elevation_gain_m"].fillna(0) / d["distance_km"].replace(0, np.nan))
        .fillna(0)
        .clip(upper=200)  # absurdly steep → clamp to keep distance numerics stable
    )
    d["kj_per_min"] = (d["kilojoules"].fillna(0) / d["duration_min"]).fillna(0)

    # Intensity ratio relative to year's HRmax.
    if hrmax_by_year is None:
        overall = d["max_hr"].quantile(0.995)
        d["intensity_ratio"] = d["avg_hr"] / overall
    else:
        d["intensity_ratio"] = d.apply(
            lambda r: r["avg_hr"] / hrmax_by_year.get(int(r["year"]), np.nan)
            if pd.notna(r["avg_hr"]) else np.nan,
            axis=1,
        )

    feat = d[list(_DEFAULT_FEATURES)].copy()
    # Fill missing: 0 for additive features, mean for HR-centric ones.
    for c in ("avg_hr", "avg_power", "intensity_ratio"):
        if c in feat:
            feat[c] = feat[c].fillna(feat[c].mean())
    feat = feat.fillna(0)

    # One-hot sport.
    feat["is_run"] = d["activity_type"].isin(["run", "trail_run"]).astype(int)
    feat["is_ride"] = d["activity_type"].isin(["ride", "virtual_ride"]).astype(int)

    return feat


def cluster_workouts(
    feature_matrix: pd.DataFrame,
    n_clusters: int = 6,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """Embed sessions to 2D with UMAP then cluster with KMeans.

    Returns a DataFrame with columns ``umap_1``, ``umap_2``, ``cluster``.
    """
    import umap  # local import — heavy dep
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    if len(feature_matrix) < n_clusters:
        return pd.DataFrame(
            {"umap_1": [], "umap_2": [], "cluster": []},
            index=feature_matrix.index,
        )

    X = StandardScaler().fit_transform(feature_matrix.values)
    reducer = umap.UMAP(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        random_state=random_state,
    )
    emb = reducer.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(emb)
    return pd.DataFrame(
        {"umap_1": emb[:, 0], "umap_2": emb[:, 1], "cluster": labels},
        index=feature_matrix.index,
    )


# =============================================================================
# 3. Daily anomaly detection (IsolationForest)
# =============================================================================


def build_daily_feature_matrix(
    pmc: pd.DataFrame,
    apple_daily: pd.DataFrame,
    rhr_by_source: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join daily training-load (PMC) with Apple Health daily signals.

    ``rhr_by_source`` is the output of ``src.apple_health.rhr_by_source``; if
    given, the RHR column is taken from the majority source on each day
    (preventing the Garmin→Apple Watch bias step from dominating the anomaly
    model).
    """
    pmc_ix = pmc.set_index("date")[["load", "ctl", "atl", "tsb"]]
    apple_ix = apple_daily.set_index("date")[
        ["hrv_sdnn_ms", "sleep_minutes", "steps", "hrr1", "respiratory_rate"]
    ]

    if rhr_by_source is not None and not rhr_by_source.empty:
        # For each date, the row with the highest `n` (most samples) wins.
        rhr_by_source_sorted = rhr_by_source.sort_values(
            ["date", "n"], ascending=[True, False]
        )
        rhr_dom = rhr_by_source_sorted.groupby("date").first()[["rhr_bpm", "source"]]
        rhr_dom = rhr_dom.rename(columns={"rhr_bpm": "rhr"})
    else:
        rhr_dom = apple_daily.set_index("date")[["resting_hr"]].rename(
            columns={"resting_hr": "rhr"}
        )
        rhr_dom["source"] = "unknown"

    joined = pmc_ix.join([apple_ix, rhr_dom], how="outer").sort_index()
    joined.index.name = "date"
    return joined.reset_index()


def anomaly_scores(
    daily: pd.DataFrame,
    feature_cols: Iterable[str] | None = None,
    contamination: float = 0.02,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit IsolationForest on per-day feature rows and return a score per day.

    Days with a feature missing are dropped. ``contamination=0.02`` flags the
    most unusual 2 % of days. Returns a DataFrame with ``date``, ``score``
    (lower = more anomalous), ``is_anomaly`` (bool), and the original features
    that went into the fit.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    cols = list(feature_cols) if feature_cols else [
        "load", "ctl", "atl", "tsb",
        "rhr", "hrv_sdnn_ms", "sleep_minutes", "steps",
    ]
    cols = [c for c in cols if c in daily.columns]

    complete = daily[["date", *cols]].dropna()
    if len(complete) < 30:
        return complete.assign(score=np.nan, is_anomaly=False)

    X = StandardScaler().fit_transform(complete[cols].values)
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
    )
    iso.fit(X)
    complete = complete.copy()
    complete["score"] = iso.decision_function(X)
    complete["is_anomaly"] = iso.predict(X) == -1
    return complete.reset_index(drop=True)
