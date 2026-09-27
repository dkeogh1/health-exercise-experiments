# Strava Data Science Analysis

Rigorous fitness data analysis with biometric integration.

## Project Structure

```
strava-analysis/
├── AGENTS.md                    # Setup, commands, layout, gotchas (start here)
├── config/
│   ├── .env.template           # Credentials template
│   ├── .env                     # Your actual credentials (DO NOT COMMIT)
│   └── strava_auth_setup.md    # OAuth flow guide
├── data/                        # gitignored: raw/ exports, processed/ SQLite + Parquet
├── src/                         # Pure modules: sessions, metrics, ml, weather, FIT
├── scripts/                     # One-command drivers (auth, export, ingest, build)
├── notebooks/                   # 01-06, jupytext-paired .ipynb + .py
├── tests/smoke_metrics.py
└── docs/ROADMAP.md
```

## Quick Start

Full setup and command list: [AGENTS.md](AGENTS.md).

1. **Set up credentials:**
   ```bash
   cp config/.env.template config/.env
   # Edit config/.env with your Strava Client ID & Secret
   ```

2. **Install dependencies** (Python 3.12 env; see AGENTS.md, *Setup*):
   ```bash
   $PY -m pip install -r scripts/requirements.txt \
       fitdecode scikit-learn umap-learn ruptures jupyterlab
   ```

3. **Authenticate with Strava** (headless flow):
   ```bash
   $PY scripts/strava_auth_manual.py
   ```

4. **Fetch your activities and build the data lake:**
   ```bash
   $PY scripts/strava_export.py
   $PY scripts/build_sessions.py
   ```

5. **Start analyzing:**
   `scripts/jupyter_serve.sh`, then open `notebooks/05_performance_dashboard.ipynb`

## Data Science Principles

- **Hypothesis-driven:** Define hypotheses before data exploration
- **Rigorous:** Effect sizes, 95% CIs, assumption checking
- **Reproducible:** All analysis in version control
- **Biometric-rich:** Apple Watch + Garmin + Strava integrated

## Next Steps

- [ ] Get fresh Strava access token
- [ ] Answer biometric setup questions
- [ ] Run initial data fetch
- [ ] Design experiments
