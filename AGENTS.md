# strava-analysis — project notes

Personal endurance-analytics pipeline. Strava + Garmin + Apple Health →
unified `sessions.parquet` + per-second `streams/*.parquet` → metrics
module (TRIMP / PMC / MMP / CP / decoupling / zones / HRV) → notebooks
(dashboard + ML exploration).

## Secrets & privacy (public repo)

This repo is public on GitHub; the data it analyses is personal health
and location data. None of this is in git. Never commit, print, or paste
it into chat or any third-party service:

- `config/.env` — Strava + Garmin credentials and tokens. Scripts read
  credentials only from it; never hardcode a token or fallback value.
- `~/.garminconnect/` — cached Garmin session token (outside the repo).
- `data/raw/*`, `data/processed/*` — activities, GPS streams, Apple
  Health export, GDPR archives and everything derived. Aggregate it in
  code; don't quote rows, coordinates, or personal physiology values in
  commits or docs. On dkbl1 its only backup is the host restic job.
- notebook outputs. Commit `.ipynb` with outputs cleared:
  `${PY%/*}/jupyter nbconvert --clear-output --inplace notebooks/<nb>.ipynb`.

Local agent state dirs such as `.claude/` are gitignored too.

## Setup

Python 3.12. On dkbl1 the env already exists; call its interpreter
directly, no activation needed (`source activate.sh` puts it on `PATH`
if you want that):

```bash
PY=${PY:-$HOME/.local/share/mamba/envs/strava-analysis/bin/python3}
JUPYTEXT=${PY%/*}/jupytext
```

Fresh machine: any 3.12 env works, e.g.

```bash
python3.12 -m venv venv && PY=$PWD/venv/bin/python3
$PY -m pip install -r scripts/requirements.txt \
    fitdecode scikit-learn umap-learn ruptures jupyterlab
cp config/.env.template config/.env
```

`scripts/requirements.txt` lacks the ML/FIT extras on that line; add new
deps there. The user fills in `config/.env` (`STRAVA_CLIENT_ID` /
`STRAVA_CLIENT_SECRET`, `GARMIN_EMAIL` / `GARMIN_PASSWORD`).

**Auth is interactive; hand it to the user.** Strava:
`$PY scripts/strava_auth_manual.py` prints the authorize URL and reads the
pasted code back; tokens go to `config/.env` and refresh automatically
after that. Don't run `strava_auth.py` on dkbl1: it opens a browser and
listens on port 8000 on all interfaces. Garmin: the first
`garmin_fetch.py` login may prompt for an MFA code.
`strava_export.py` is the canonical exporter; `strava_export_direct.py` /
`strava_export_simple.py` are early one-offs whose `data/activities.csv`
nothing reads. More: [config/strava_auth_setup.md](config/strava_auth_setup.md).

## Layout

```
src/                    # pure modules — no I/O except explicit loaders
  apple_health.py         XML → SQLite + per-source RHR query
  sessions.py             unified fuzzy-merge of Strava/Garmin/Apple/FIT
  metrics.py              TRIMP, PMC, MMP, CP, zones, decoupling
  weather.py              Open-Meteo historical + SQLite cache
  fit.py                  fitdecode FIT → Parquet (GDPR archives)
  ml.py                   ruptures + UMAP + IsolationForest
scripts/                # one-command drivers
  apple_health_ingest.py  data/raw/apple_health_export/export.xml → apple_health.db
  strava_export.py        → data/raw/activities_<stamp>.json+csv
  strava_streams.py       → data/processed/streams/strava_<id>.parquet
  garmin_fetch.py         → data/raw/garmin_activities_<stamp>.parquet
  garmin_gdpr_ingest.py   data/raw/garmin_gdpr/fit → streams/<hash>.parquet
  build_sessions.py       → data/processed/sessions.parquet
  enrich_weather.py       → weather columns on sessions.parquet
  jupyter_serve.sh        JupyterLab on 127.0.0.1:8888
notebooks/              # jupytext-paired .ipynb + .py:percent
  01–03                   early EDA / hypothesis tests (03 superseded by 04)
  04_readiness            Plews/Altini rolling-z readiness, RHR/HRV/sleep
  05_performance_dash     PMC, MMP, clusters, zones, weather
  06_ml_exploration       CTL change-point, UMAP clusters, IsoForest
```

## Common operations

```bash
# Pull new Strava activities, then their streams
$PY scripts/strava_export.py
$PY scripts/strava_streams.py            # --since YYYY-MM-DD, --limit N

# Garmin: API pull (session cached in ~/.garminconnect), or GDPR archive
$PY scripts/garmin_fetch.py
$PY scripts/garmin_gdpr_ingest.py

# Fresh Apple Health export → SQLite (~35 s), then the daily CSV that
# notebooks 04–06 read (no script writes it)
$PY scripts/apple_health_ingest.py
$PY -c "from pathlib import Path; from src import apple_health as ah; ah.daily_metrics(Path('data/processed/apple_health.db')).to_csv('data/raw/apple_health.csv', index=False)"

# Rebuild sessions.parquet after any raw-data change, then re-enrich
# weather (idempotent, cache-aware)
$PY scripts/build_sessions.py
$PY scripts/enrich_weather.py

# Smoke-test all metrics (needs the private data/, so local only)
$PY tests/smoke_metrics.py

# Sync notebook pair (run whenever you edit either side)
cd notebooks && $JUPYTEXT --sync 05_performance_dashboard.py

# JupyterLab (dkbl1 is headless): start / url / stop
scripts/jupyter_serve.sh
```

`jupyter_serve.sh` binds 127.0.0.1:8888 and prints the token URL; the user
tunnels in from their laptop (see the script header). `stop` kills every
`jupyter-lab` on the host.

## Resumable jobs

dkbl1 can power off under load, so long pulls run detached (from the
repo root) and are built to resume:

```bash
nohup $PY scripts/strava_streams.py \
  > data/processed/streams/_strava_stdout.log 2>&1 &
```

It skips any activity whose `streams/strava_<id>.parquet` exists, so a
restart costs nothing. State is in
`data/processed/streams/_strava_progress.json` (last run, last pulled id)
and the tail of `_strava_stdout.log` (`✓ Done.` when complete); it
self-throttles under Strava's rate limit. `garmin_gdpr_ingest.py` is
idempotent the same way. After new streams land: `enrich_weather.py`,
then re-run notebooks 05 and 06.

## Known gotchas (load-bearing — don't forget)

- **RHR / HRV / walking-HR / SpO₂ / wrist temp / VO₂max must be
  segmented by `sourceName`** across multi-year spans: the primary wrist
  device changed partway through the archive, and optical sensors carry
  a several-bpm inter-device bias. Never plot a merged line, and never
  read a step change near a device switch as physiology before checking
  the `sourceName` mix on both sides. Use `ah.rhr_by_source(db)` instead
  of `ah.daily_metrics(db)["resting_hr"]`. Device-specific metrics (e.g.
  Apple Watch running power, VO₂max estimate, HRR1) have no comparator
  across a switch: don't fit a trend that spans it. Chest-strap-based
  load (TSS / hrTSS) is exempt.

- **Apple Watch HRV is SDNN, not RMSSD**, and only sampled during Breathe
  sessions (~1 / day, sparse). Ignore day-to-day changes < 10 ms.

- **`garth` is deprecated** (Cloudflare 429s since 2025). We use
  `python-garminconnect` (`curl_cffi` TLS impersonation, web-login
  fallback); if that breaks too, Garmin's GDPR archive is the fallback.

- **Stryd and Apple Watch running power are different constructs** —
  don't pool them for CP fits.

- **Per-year anchors, not archive-wide**. HRmax, LTHR, FTP change over
  10 years; `metrics.anchors_for_year()` returns them. HRmax is observed
  (`metrics.observed_hrmax`), never 220 − age.

- **CTL/ATL use canonical `α = 1 − exp(−1/τ)`**, not pandas' default
  `span = 2/(τ+1)` for `.ewm()`.

- **ACWR is computed but don't trust it** — Impellizzeri 2020 showed it's
  a statistical artefact. Report with skepticism.

## Docs

- [docs/ROADMAP.md](docs/ROADMAP.md) — phase status, settled decisions,
  next moves
- [docs/BIOMETRIC_INTEGRATION_GUIDE.md](docs/BIOMETRIC_INTEGRATION_GUIDE.md)
- [EXPERIMENT_TEMPLATE.md](EXPERIMENT_TEMPLATE.md) — hypothesis write-up
  template
- Early planning, partly stale (paths, env commands):
  [ANALYSIS_OPTIONS.md](ANALYSIS_OPTIONS.md),
  [BIOMETRIC_DATA_RESEARCH.md](BIOMETRIC_DATA_RESEARCH.md),
  [SETUP_SUMMARY.md](SETUP_SUMMARY.md). This file wins on conflict.
