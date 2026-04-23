"""Endurance-performance metrics.

All functions are pure (no I/O) and work on ``pandas`` structures. The module
is organised top-to-bottom:

1. **Anchors** (HRmax / HRrest / LTHR / FTP) — computed per year from the data
2. **Session-level load** (TRIMP variants, hrTSS, efficiency factor)
3. **Daily aggregation** (best-available load → CTL/ATL/TSB Performance
   Management Chart)
4. **Stream-level** (mean-maximal power, Monod-Scherrer CP/W′, time-in-zone,
   aerobic decoupling, HR drift)
5. **HRV helpers** (rolling z-score, mirroring what notebook 04 uses)

Conventions:

- ``sessions`` is the DataFrame produced by ``src.sessions.build`` — one row
  per activity with ``start_time_utc``, ``duration_s``, ``avg_hr``, ``max_hr``,
  ``avg_power``, ``tss``, ``activity_type``, ``distance_m``, etc.
- ``stream`` is the 1 Hz DataFrame written by ``scripts/strava_streams.py``
  with ``timestamp``, ``heart_rate``, ``power``, ``lat``, ``lng``, ``altitude``,
  ``distance``, ``speed``, ``cadence``, optional ``moving``, ``temperature``.

Opinionated choices called out in the roadmap memory:

- HRmax is the observed 99.5 th percentile across the archive — not 220 − age.
- LTHR defaults to 0.91 · HRmax when no explicit threshold test is available
  (Friel's "approximation from observed max").
- FTP defaults to 0.95 · best-20-min-mean-power (Coggan).
- CTL τ = 42 d and ATL τ = 7 d (Coggan convention; see roadmap for caveats).
- ACWR is implemented but flagged controversial — Impellizzeri 2020.
- Apple Watch continuous-HRV is **SDNN**, not RMSSD; our z-score takes the
  column at face value rather than trying to convert.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

# =============================================================================
# 1. Anchors
# =============================================================================


@dataclass(frozen=True)
class Anchors:
    """Per-year physiological anchors used by the rest of the module."""
    year: int | None
    hrmax: float | None
    hrrest: float | None
    lthr: float | None
    ftp: float | None

    def with_fallbacks(self, fallback: "Anchors") -> "Anchors":
        """Return a copy where any None is filled from `fallback`."""
        return Anchors(
            year=self.year,
            hrmax=self.hrmax if self.hrmax is not None else fallback.hrmax,
            hrrest=self.hrrest if self.hrrest is not None else fallback.hrrest,
            lthr=self.lthr if self.lthr is not None else fallback.lthr,
            ftp=self.ftp if self.ftp is not None else fallback.ftp,
        )


def observed_hrmax(sessions: pd.DataFrame, quantile: float = 0.995) -> float | None:
    """Observed HRmax as the high-quantile of per-activity max_hr.

    Using 99.5th percentile (not the single max) filters out spurious spikes
    from optical-HR misreads while still capturing true peak efforts.
    """
    s = pd.to_numeric(sessions.get("max_hr"), errors="coerce")
    s = s.dropna()
    if s.empty:
        return None
    return float(s.quantile(quantile))


def observed_hrrest(apple_daily: pd.DataFrame | None) -> float | None:
    """Resting HR as the 5th-percentile of Apple's own daily RHR."""
    if apple_daily is None or apple_daily.empty:
        return None
    col = "resting_hr" if "resting_hr" in apple_daily.columns else "heart_rate_min"
    s = pd.to_numeric(apple_daily.get(col), errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.quantile(0.05))


def best_mean_power(
    stream: pd.DataFrame, window_s: int, power_col: str = "power"
) -> float | None:
    """Best rolling-mean power over a given window inside a single stream.

    Returns None if the stream has no power data or is shorter than window_s.
    """
    if stream.empty or power_col not in stream.columns:
        return None
    p = pd.to_numeric(stream[power_col], errors="coerce")
    p = p.dropna()
    if len(p) < window_s:
        return None
    best = p.rolling(window_s).mean().max()
    return float(best) if pd.notna(best) else None


def ftp_from_best_20min(stream_power_values: Iterable[float | None]) -> float | None:
    """Given best-20-min powers across many activities, return 0.95 × max."""
    vals = [v for v in stream_power_values if v is not None]
    if not vals:
        return None
    return float(0.95 * max(vals))


def lthr_from_hrmax(hrmax: float | None, factor: float = 0.91) -> float | None:
    if hrmax is None:
        return None
    return float(factor * hrmax)


def anchors_for_year(
    year: int,
    sessions: pd.DataFrame,
    apple_daily: pd.DataFrame | None = None,
    best_20min_power: float | None = None,
) -> Anchors:
    """Build the anchors for `year` using that year's slice of sessions."""
    year_sessions = sessions[
        pd.to_datetime(sessions["start_time_utc"], utc=True).dt.year == year
    ]
    hrmax = observed_hrmax(year_sessions)
    hrrest = (
        observed_hrrest(
            apple_daily[apple_daily["date"].dt.year == year]
        )
        if apple_daily is not None
        else None
    )
    lthr = lthr_from_hrmax(hrmax)
    ftp = 0.95 * best_20min_power if best_20min_power else None
    return Anchors(year=year, hrmax=hrmax, hrrest=hrrest, lthr=lthr, ftp=ftp)


# =============================================================================
# 2. Session-level load
# =============================================================================


def _hr_reserve_fraction(avg_hr: float, hrmax: float, hrrest: float) -> float:
    """Karvonen HR reserve fraction, clipped to [0, 1]."""
    denom = hrmax - hrrest
    if denom <= 0:
        return 0.0
    return float(np.clip((avg_hr - hrrest) / denom, 0.0, 1.0))


def banister_trimp(
    duration_s: float, avg_hr: float, hrmax: float, hrrest: float,
    sex: str = "M",
) -> float | None:
    """Banister (1991) TRIMP in arbitrary units.

    ``TRIMP = duration_min · HRr · (a · exp(b · HRr))``

    Male:    a = 0.64, b = 1.92
    Female:  a = 0.86, b = 1.67
    """
    if None in (duration_s, avg_hr, hrmax, hrrest):
        return None
    if np.isnan(avg_hr) or np.isnan(hrmax) or np.isnan(hrrest):
        return None
    hrr = _hr_reserve_fraction(avg_hr, hrmax, hrrest)
    a, b = (0.86, 1.67) if sex.upper().startswith("F") else (0.64, 1.92)
    return float((duration_s / 60.0) * hrr * a * np.exp(b * hrr))


def edwards_trimp_from_avg(
    duration_s: float, avg_hr: float, hrmax: float,
) -> float | None:
    """Edwards TRIMP approximated from *average* HR.

    The canonical Edwards TRIMP is time-in-zone weighted 1..5. Without a HR
    stream we use the dominant zone implied by ``avg_hr / hrmax``:

        ≥0.90 → weight 5
        0.80–0.89 → 4
        0.70–0.79 → 3
        0.60–0.69 → 2
        <0.60 → 1

    Call ``edwards_trimp_from_stream`` when you have the actual HR stream —
    it's noticeably more accurate for interval sessions.
    """
    if None in (duration_s, avg_hr, hrmax) or hrmax <= 0:
        return None
    if np.isnan(avg_hr) or np.isnan(hrmax):
        return None
    frac = avg_hr / hrmax
    if frac >= 0.90:
        w = 5
    elif frac >= 0.80:
        w = 4
    elif frac >= 0.70:
        w = 3
    elif frac >= 0.60:
        w = 2
    else:
        w = 1
    return float((duration_s / 60.0) * w)


def edwards_trimp_from_stream(
    stream: pd.DataFrame, hrmax: float, hr_col: str = "heart_rate",
) -> float | None:
    """Edwards TRIMP from a true HR stream, summed by time-in-zone × weight."""
    if stream.empty or hr_col not in stream.columns or hrmax <= 0:
        return None
    hr = pd.to_numeric(stream[hr_col], errors="coerce").dropna()
    if hr.empty:
        return None
    frac = hr / hrmax
    weights = np.select(
        [frac >= 0.90, frac >= 0.80, frac >= 0.70, frac >= 0.60],
        [5, 4, 3, 2],
        default=1,
    )
    # Assume 1 Hz stream; each sample = 1/60 min.
    return float(np.sum(weights) / 60.0)


def hrtss(
    duration_s: float, avg_hr: float, lthr: float,
) -> float | None:
    """Heart-rate TSS approximation.

    hrTSS ≈ 100 · (duration_h) · (avg_hr / LTHR)²

    Matches the Coggan convention of "TSS = 100 per 1 h at threshold."
    """
    if None in (duration_s, avg_hr, lthr) or lthr <= 0:
        return None
    if np.isnan(avg_hr) or np.isnan(lthr):
        return None
    return float(100.0 * (duration_s / 3600.0) * (avg_hr / lthr) ** 2)


def efficiency_factor(
    avg_power: float | None, avg_hr: float | None,
) -> float | None:
    """Friel's Efficiency Factor = NP / avg_HR (bike) or NGP / avg_HR (run).

    We compute ``avg_power / avg_hr`` as a close-enough proxy when NP isn't
    available. Trend it — absolute values aren't comparable across sports.
    """
    if not avg_power or not avg_hr or avg_hr <= 0:
        return None
    if np.isnan(avg_power) or np.isnan(avg_hr):
        return None
    return float(avg_power / avg_hr)


def tss_from_power_stream(
    stream: pd.DataFrame, ftp: float, power_col: str = "power",
) -> float | None:
    """Full Coggan TSS from a 1 Hz power stream:

        NP = (mean(P_rolling30s**4))**0.25
        IF = NP / FTP
        TSS = (duration_s · NP · IF) / (FTP · 3600) · 100
    """
    if stream.empty or power_col not in stream.columns or ftp <= 0:
        return None
    p = pd.to_numeric(stream[power_col], errors="coerce").fillna(0)
    if p.empty:
        return None
    rolled = p.rolling(30, min_periods=1).mean()
    np_ = float(np.mean(rolled ** 4) ** 0.25)
    duration_s = len(p)
    IF = np_ / ftp
    return float((duration_s * np_ * IF) / (ftp * 3600.0) * 100.0)


def best_available_load(
    session_row: pd.Series,
    lthr: float | None = None,
) -> float | None:
    """Pick the best training-load number for one session.

    Priority: Garmin ``tss`` > ``hrTSS`` computed from avg_hr+LTHR. We do
    *not* pool TSS and hrTSS as if they were the same unit, but they're close
    enough for a PMC that wants a single load time series.
    """
    tss = session_row.get("tss")
    if pd.notna(tss):
        return float(tss)
    if lthr is not None:
        return hrtss(
            session_row.get("duration_s"),
            session_row.get("avg_hr"),
            lthr,
        )
    return None


# =============================================================================
# 3. Daily aggregation + Performance Management Chart
# =============================================================================


def daily_load(
    sessions: pd.DataFrame,
    lthr_by_year: dict[int, float] | float | None = None,
    load_col: str | None = None,
) -> pd.DataFrame:
    """Collapse sessions → one row per calendar day with summed training load.

    Parameters
    ----------
    sessions : DataFrame from src.sessions.build
    lthr_by_year : either a constant LTHR, or {year: LTHR}. Used as fallback
        when a session has no ``tss`` column.
    load_col : if given, use this column directly (e.g. "tss" to force Garmin-
        only). If None, falls back to ``best_available_load`` per session.

    Returns a DataFrame indexed by date with columns:
        load            — summed training load
        sessions        — # activities that day
    """
    s = sessions.copy()
    s["date"] = pd.to_datetime(s["start_time_utc"], utc=True).dt.date
    s["date"] = pd.to_datetime(s["date"])
    s["year"] = s["date"].dt.year

    if load_col is not None:
        per_session_load = pd.to_numeric(s[load_col], errors="coerce")
    else:
        def _row_load(row):
            if isinstance(lthr_by_year, dict):
                lthr = lthr_by_year.get(int(row["year"]))
            else:
                lthr = lthr_by_year
            return best_available_load(row, lthr)
        per_session_load = s.apply(_row_load, axis=1)

    s = s.assign(load=per_session_load.astype("float64"))
    daily = s.groupby("date").agg(load=("load", "sum"), sessions=("load", "count"))
    # Reindex to a dense calendar so EWMA below isn't confused by gaps.
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full, fill_value=0)
    daily.index.name = "date"
    return daily.reset_index()


def ctl_atl_tsb(
    daily: pd.DataFrame, ctl_tau: int = 42, atl_tau: int = 7,
) -> pd.DataFrame:
    """Coggan/Banister Performance Management Chart on a dense daily load series.

    CTL = EWMA(load, τ=42)
    ATL = EWMA(load, τ=7)
    TSB = CTL - ATL  (form: positive = fresh, negative = loaded)

    ``alpha = 1 - exp(-1/τ)`` — the canonical decay constant, *not* 2/(τ+1)
    which is pandas' default for ``span=``.
    """
    if daily.empty:
        return daily.assign(ctl=np.nan, atl=np.nan, tsb=np.nan)

    load = daily["load"].astype(float).values
    ctl_alpha = 1.0 - np.exp(-1.0 / ctl_tau)
    atl_alpha = 1.0 - np.exp(-1.0 / atl_tau)

    ctl = np.zeros_like(load)
    atl = np.zeros_like(load)
    for i, x in enumerate(load):
        prev_ctl = ctl[i - 1] if i else 0
        prev_atl = atl[i - 1] if i else 0
        ctl[i] = prev_ctl + ctl_alpha * (x - prev_ctl)
        atl[i] = prev_atl + atl_alpha * (x - prev_atl)
    out = daily.copy()
    out["ctl"] = ctl
    out["atl"] = atl
    out["tsb"] = ctl - atl
    return out


def acwr(daily: pd.DataFrame, acute: int = 7, chronic: int = 28) -> pd.Series:
    """Acute:chronic workload ratio (Gabbett 2016).

    NOTE: Impellizzeri et al. 2020 argue ACWR is largely a statistical artefact
    — report it with skepticism, not as a decision rule.
    """
    ac = daily["load"].rolling(acute, min_periods=1).sum()
    ch = daily["load"].rolling(chronic, min_periods=1).sum() / (chronic / acute)
    return (ac / ch).replace([np.inf, -np.inf], np.nan)


# =============================================================================
# 4. Stream-level metrics
# =============================================================================


def mean_maximal_power(
    stream: pd.DataFrame,
    durations_s: Iterable[int] = (
        1, 5, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600, 7200,
    ),
    power_col: str = "power",
) -> pd.Series:
    """Best rolling-mean power over each duration, from a single stream.

    Returns a Series indexed by duration (seconds) with NaN where the stream
    is shorter than the window.
    """
    out: dict[int, float] = {}
    if stream.empty or power_col not in stream.columns:
        return pd.Series(dtype=float, name="mmp")
    p = pd.to_numeric(stream[power_col], errors="coerce").dropna()
    for w in durations_s:
        if len(p) < w:
            out[w] = np.nan
        else:
            out[w] = float(p.rolling(w).mean().max())
    s = pd.Series(out, name="mmp")
    s.index.name = "duration_s"
    return s


def mmp_across_streams(
    streams: Iterable[pd.DataFrame],
    durations_s: Iterable[int] = (
        1, 5, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600, 7200,
    ),
    power_col: str = "power",
) -> pd.Series:
    """Best power per duration across *many* streams — i.e. your all-time
    mean-maximal power curve."""
    partials = [
        mean_maximal_power(s, durations_s=durations_s, power_col=power_col)
        for s in streams
    ]
    if not partials:
        return pd.Series(dtype=float, name="mmp")
    df = pd.concat(partials, axis=1)
    return df.max(axis=1).rename("mmp")


def critical_power(
    mmp: pd.Series, start_s: int = 180, end_s: int = 1200,
) -> tuple[float, float]:
    """Monod-Scherrer 2-parameter CP from the mean-maximal-power curve.

    Linear fit: ``work = W' + CP · duration`` over durations in [start_s, end_s].
    Returns (CP watts, W' joules).
    """
    s = mmp.dropna()
    s = s[(s.index >= start_s) & (s.index <= end_s)]
    if len(s) < 3:
        return (float("nan"), float("nan"))
    durations = np.array(s.index, dtype=float)
    power = s.values.astype(float)
    work = power * durations
    # Linear regression work = W' + CP · t
    slope, intercept = np.polyfit(durations, work, 1)
    return float(slope), float(intercept)


def time_in_zone(
    stream: pd.DataFrame,
    anchor: float,
    zones: str = "pct_hrmax",
    col: str = "heart_rate",
) -> pd.Series:
    """Seconds spent in each zone, for a single stream.

    zones = "pct_hrmax"  → 5 zones at 0.6/0.7/0.8/0.9 of HRmax
    zones = "pct_lthr"   → Friel 7-zone running/cycling model
    zones = "pct_ftp"    → Coggan 7-zone power model

    ``col`` should be the HR stream column name (or ``power`` for pct_ftp).
    """
    if stream.empty or col not in stream.columns or not anchor or anchor <= 0:
        return pd.Series(dtype=float, name="seconds_in_zone")

    v = pd.to_numeric(stream[col], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float, name="seconds_in_zone")

    if zones == "pct_hrmax":
        edges = np.array([0, 0.60, 0.70, 0.80, 0.90, np.inf]) * anchor
        labels = ["z1", "z2", "z3", "z4", "z5"]
    elif zones == "pct_lthr":
        edges = np.array([0, 0.85, 0.89, 0.94, 1.00, 1.03, 1.06, np.inf]) * anchor
        labels = ["z1", "z2", "z3", "z4", "z5a", "z5b", "z5c"]
    elif zones == "pct_ftp":
        edges = np.array([0, 0.55, 0.75, 0.90, 1.05, 1.20, 1.50, np.inf]) * anchor
        labels = ["z1", "z2", "z3", "z4", "z5", "z6", "z7"]
    else:
        raise ValueError(f"unknown zones={zones!r}")

    bins = pd.cut(v, bins=edges, labels=labels, include_lowest=True, right=False)
    counts = bins.value_counts().reindex(labels, fill_value=0)
    # Assume 1 Hz → seconds.
    out = counts.astype(float)
    out.name = "seconds_in_zone"
    return out


def aerobic_decoupling(
    stream: pd.DataFrame,
    power_col: str = "power",
    hr_col: str = "heart_rate",
    min_duration_s: int = 1800,
) -> float | None:
    """Friel's ``Pw:HR`` aerobic decoupling %.

    Split the session in half, compute mean power/HR for each half, then:

        decoupling = (eff2 - eff1) / eff1 · 100

    where ``eff = mean_power / mean_hr``. Negative = aerobically efficient
    (HR stayed flat). >5 % positive = classic "you went too hard / under-fueled
    / aerobically unfit" signal.

    Falls back to pace:HR when no power column (runs without Stryd/S10).
    """
    if stream.empty or hr_col not in stream.columns:
        return None
    hr = pd.to_numeric(stream[hr_col], errors="coerce")
    effort_col = power_col if power_col in stream.columns else "speed"
    if effort_col not in stream.columns:
        return None
    eff = pd.to_numeric(stream[effort_col], errors="coerce")

    df = pd.DataFrame({"hr": hr, "eff": eff}).dropna()
    if len(df) < min_duration_s:
        return None

    mid = len(df) // 2
    h1 = df.iloc[:mid]
    h2 = df.iloc[mid:]
    eff1 = h1["eff"].mean() / h1["hr"].mean()
    eff2 = h2["eff"].mean() / h2["hr"].mean()
    if eff1 <= 0:
        return None
    return float((eff2 - eff1) / eff1 * 100.0)


def hr_drift(
    stream: pd.DataFrame, hr_col: str = "heart_rate", min_duration_s: int = 600,
) -> float | None:
    """Slope of HR vs time (bpm / min) over a single stream — rough cardiac
    drift proxy when no power is available."""
    if stream.empty or hr_col not in stream.columns:
        return None
    hr = pd.to_numeric(stream[hr_col], errors="coerce").dropna()
    if len(hr) < min_duration_s:
        return None
    x = np.arange(len(hr), dtype=float) / 60.0  # minutes
    slope, _ = np.polyfit(x, hr.values.astype(float), 1)
    return float(slope)


# =============================================================================
# 5. HRV helpers
# =============================================================================


def rolling_z(series: pd.Series, window: int = 28, min_periods: int = 7) -> pd.Series:
    """Z-score of each point vs. the prior `window` (excluding itself)."""
    mean = series.rolling(window, min_periods=min_periods).mean().shift(1)
    std = series.rolling(window, min_periods=min_periods).std().shift(1)
    return (series - mean) / std
