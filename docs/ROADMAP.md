# strava-analysis roadmap

Six-phase plan to turn strava-analysis into a real endurance-analytics
platform, from a deep-research pass on 2026-04-23. The audit behind it
found the raw data far richer (10 yrs Strava, 11 yrs Apple Health with
HRV / sleep / running power / RHR / VO₂max / running dynamics /
body composition) than the original descriptive-stats notebooks used:
the Apple Health parser dropped most of the signal, there was no
training-load model, no streams, and Garmin had never been pulled.

Standing rules (device-bias segmentation, anchors, CTL α, ...) live in
[AGENTS.md](../AGENTS.md) under *Known gotchas*. Don't rebuild what's
built: use `src/metrics.py`, `src/ml.py`, etc.

## Phase status (as of 2026-09-27)

- **P0 (fixes/hygiene) — DONE**: Apple Health ingest rewritten
  (`src/apple_health.py` → SQLite, one table per record type plus
  `workouts`). `scripts/garmin_fetch.py` on `python-garminconnect`.
  Notebooks jupytext-paired `py:percent`. Notebook 04 rewritten with
  Plews/Altini rolling-z readiness.
- **P1 (data lake) — DONE except the Strava GDPR archive**:
  `src/sessions.py` + `scripts/build_sessions.py` → `sessions.parquet`
  (fuzzy match |Δstart| < 120 s & |Δdur| < 5 %). Strava streams pull
  finished 2026-04-24. Garmin GDPR FIT archive ingested
  (`scripts/garmin_gdpr_ingest.py`, commit 0c41e32); `sessions.parquet`
  carries a `streams_path` column pointing at the richest stream (FIT
  preferred). The Strava GDPR archive was requested 2026-04-23 but is not
  in `data/raw/`.
- **P2 (metrics) — DONE**: `src/metrics.py` — anchors (observed, per
  year), session load (Banister/Edwards TRIMP, hrTSS, EF,
  `tss_from_power_stream`), PMC (`daily_load` + `ctl_atl_tsb`),
  stream-level (MMP, Monod-Scherrer CP, TiZ, aerobic decoupling, HR
  drift), `rolling_z` HRV helper. `tests/smoke_metrics.py` exercises all.
- **P3 (viz) — DONE**: `notebooks/05_performance_dashboard`: anchors,
  PMC, calendar heatmap (hand-rolled — `july` and `calmap` both broken),
  weekly volume stacks, HR-vs-pace per year, MMP curves, decoupling
  trend, TiZ + polarization index, RHR/HRV/CTL overlay (RHR segmented by
  `sourceName`), weather × pace.
- **P4 (ML) — PARTIAL (3/6 methods)**: `src/ml.py` +
  `notebooks/06_ml_exploration`: change-point on CTL, UMAP + KMeans
  workout classes, IsolationForest anomaly days. The other three were
  skipped on purpose (below).
- **P5 (novel, optional) — NOT STARTED**: streamlit/panel dashboard, LLM
  weekly narrative, intervals.icu upload-and-benchmark.

## Settled decisions (don't re-litigate)

- **Garmin via `garminconnect`, not `garth`**: garth was deprecated
  upstream after Garmin's 2025 Cloudflare anti-bot change; the GDPR
  archive is the fallback if garminconnect breaks too.
- **Observed HRmax** (99.5th percentile over the archive), per year —
  never 220 − age.
- **Reimplement Edwards TRIMP** instead of using Strava's Suffer Score,
  which is undocumented.
- **Skipped in P4**: XGBoost target prediction (leaks via CTL),
  state-space Banister (no labels), DFA-α₁ (no R-R interval streams).
- **Be skeptical of / skip**: ACWR (Impellizzeri 2020 — statistical
  artefact; computed, never decided on); "continuous" Apple HRV (it is
  sporadic SDNN); deep learning on < 5k days of data; single-day
  readiness as a go/no-go — trend only.

## Next candidate moves (pick one; don't default to building)

1. **Pick a real question** — "Is cross-training helping?", "Realistic
   peaking plan from current CTL?", "Precursors to the IsoForest-flagged
   anomaly days?". The machinery answers these now.
2. **Strava GDPR archive**, if it's available: extract to
   `data/raw/strava_gdpr/`, run `src.fit.ingest_directory()` on it (it reads
   only `*.fit` / `*.fit.gz`; GPX/TCX would need a parser), then
   `build_sessions.py` and `enrich_weather.py`.
3. **P5** — streamlit mini-dashboard or LLM weekly narrative. Low urgency:
   notebook 05 already covers most of a dashboard.
