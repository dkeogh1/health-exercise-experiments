# strava-analysis — project notes

Personal endurance-analytics pipeline. Strava + Garmin + Apple Health →
unified `sessions.parquet` + per-second `streams/*.parquet` → metrics
module (TRIMP / PMC / MMP / CP / decoupling / zones / HRV) → notebooks
(dashboard + ML exploration).

## Environment

Python 3.12 in a micromamba env. Activate with the interpreter path
directly — no shell activation required:

```bash
PY=/home/dk/.local/share/mamba/envs/strava-analysis/bin/python3
JUPYTEXT=/home/dk/.local/share/mamba/envs/strava-analysis/bin/jupytext
```

Requirements live in `scripts/requirements.txt`. The env already has
everything listed plus: `healthkit-to-sqlite`, `garminconnect`,
`fitdecode`, `jupytext`, `july`, `pyarrow`, `statsmodels`, `sklearn`,
`umap-learn`, `ruptures`, `pyod`-equivalent (IsolationForest via sklearn).

## Layout

```
src/                    # pure modules — no I/O except explicit loaders
  apple_health.py         XML → SQLite + per-source RHR query
  sessions.py             unified fuzzy-merge of Strava/Garmin/Apple
  metrics.py              TRIMP, PMC, MMP, CP, zones, decoupling
  weather.py              Open-Meteo historical + SQLite cache
  fit.py                  fitdecode FIT → Parquet (GDPR archive ready)
  ml.py                   ruptures + UMAP + IsolationForest
scripts/                # one-command drivers
  apple_health_ingest.py  → data/processed/apple_health.db
  strava_export.py        → data/raw/activities_<stamp>.json+csv
  strava_streams.py       → data/processed/streams/strava_<id>.parquet
  garmin_fetch.py         → data/raw/garmin_activities_<stamp>.parquet
  build_sessions.py       → data/processed/sessions.parquet
  enrich_weather.py       → weather columns on sessions.parquet
notebooks/              # jupytext-paired .ipynb + .py:percent
  01_exploratory        initial EDA (session-level only)
  02_hypothesis_testing 5 H1-H5 tests, session-level
  03_biometric_corr     early draft, superseded by 04
  04_readiness          Plews/Altini rolling-z readiness, RHR/HRV/sleep
  05_performance_dash   10 sections — PMC, MMP, clusters, zones, weather
  06_ml_exploration     CTL change-point, UMAP clusters, IsoForest
tests/
  smoke_metrics.py        exercises every metric on real data
data/
  raw/                    gitignored: Strava json, Garmin parquet,
                          Apple Health export.xml (1.2 GB), apple_health.csv
  processed/              gitignored: apple_health.db, sessions.parquet,
                          streams/strava_<id>.parquet, weather_cache.sqlite
config/
  .env                    gitignored — Strava + Garmin creds
```

## Common operations

```bash
# Rebuild sessions.parquet after any raw-data change
$PY scripts/build_sessions.py

# Sync notebook pair (run whenever you edit either side)
cd notebooks && $JUPYTEXT --sync 05_performance_dashboard.py

# Smoke-test all metrics against current data
$PY tests/smoke_metrics.py

# Re-enrich weather after new streams land (idempotent, cache-aware)
$PY scripts/enrich_weather.py

# Fresh Apple Health export → SQLite (~35s for 1.2 GB XML)
$PY scripts/apple_health_ingest.py

# Pull new Strava activities
$PY scripts/strava_export.py
```

## In-flight work — resuming after power-down

**Strava streams pull** runs as a long-lived background process. On reboot
it will be dead; to resume:

```bash
cd /home/dk/repos/strava-analysis
nohup $PY scripts/strava_streams.py \
  > data/processed/streams/_strava_stdout.log 2>&1 &
```

The script is idempotent — it skips any activity whose
`data/processed/streams/strava_<id>.parquet` already exists, so restarts
cost nothing. Progress state in
`data/processed/streams/_strava_progress.json`. Last snapshot (2026-04-23
20:33 UTC): 539 of ~1,651 streams pulled, last_pulled_id 5815879873
(oldest-first order; mid-2021 era).

**After streams complete**, re-run (in order):
1. `$PY scripts/enrich_weather.py` — fills weather for the new streams
2. Re-execute notebooks 05 and 06 — MMP, decoupling, zone-distribution,
   weather charts, UMAP clusters all get denser

## Known gotchas (load-bearing — don't forget)

- **RHR / HRV / walking-HR / SpO₂ must be segmented by `sourceName`**
  across multi-year spans. Garmin FR645 was primary 2020–mid-2025; Apple
  Watch S10 after. Inter-device bias ≈ 8 bpm at rest. Never plot a merged
  line. Use `ah.rhr_by_source(db)` instead of
  `ah.daily_metrics(db)["resting_hr"]`.

- **Apple Watch HRV is SDNN, not RMSSD**, and only sampled during Breathe
  sessions (~1 / day). Expect ~10 % day-coverage. Ignore day-to-day
  changes < 10 ms.

- **`garth` is deprecated** (Cloudflare 429s since 2025). We use
  `python-garminconnect` which has `curl_cffi` TLS impersonation and web-
  login fallback. If that breaks too, Garmin's GDPR data archive is the
  fallback.

- **Apple Watch S10 running power ≠ Stryd** — different physical
  constructs. Don't pool for CP fits.

- **Per-year anchors, not archive-wide**. HRmax, LTHR, FTP change over
  10 years; `metrics.anchors_for_year()` returns them per year.

- **CTL/ATL use canonical `α = 1 − exp(−1/τ)`**, not pandas' default
  `span = 2/(τ+1)` for `.ewm()`.

- **ACWR is computed but don't trust it** — Impellizzeri 2020 showed it's
  a statistical artefact. Report with skepticism.

## What's NOT in git

- `data/raw/*` (personal activity data, Apple Health XML 1.2 GB)
- `data/processed/*` (SQLite DB, Parquet files, weather cache)
- `config/.env` (Strava + Garmin credentials)
- `.claude/` (Claude Code local session state)
- `~/.garminconnect/` (cached Garmin session token, outside repo)

## Memory index

Claude-Code project memory at
`~/.claude/projects/-home-dk-repos-strava-analysis/memory/`:
- `project_roadmap.md` — 6-phase plan with current status
- `feedback_device_bias.md` — the source-segmentation rule above

## Pending external events

- **Strava GDPR data archive** requested 2026-04-23 (24–48 h ETA, email
  notification). Extract to `data/raw/strava_gdpr/` and run
  `src.fit.ingest_directory()` to parse every FIT into
  `data/processed/streams/` at fuller resolution than the API streams.
- **Garmin GDPR data archive** also requested 2026-04-23.
