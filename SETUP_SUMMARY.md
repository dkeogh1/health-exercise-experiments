# Strava Analysis Project — Setup Complete ✅

## What's Built

A complete data science infrastructure for rigorous fitness analysis with:
- **1,651 Strava activities** downloaded
- **Hypothesis testing** framework (5 major tests)
- **Biometric integration** support (Apple Watch + Garmin)
- **Jupyter notebooks** for interactive analysis
- **Mamba environment** for reproducible Python

## Project Structure

```
~/strava-analysis/
├── README.md                          # Project overview
├── SETUP_SUMMARY.md                   # This file
├── config/
│   ├── .env                           # Your API credentials (DO NOT COMMIT)
│   ├── .env.template                  # Template for .env
│   └── strava_auth_setup.md           # OAuth guide
├── scripts/
│   ├── strava_auth_manual.py          # Headless OAuth flow ✅ (already done)
│   ├── strava_export.py               # Fetch activities ✅ (already done)
│   ├── garmin_fetch.py                # Garmin API integration (optional)
│   └── requirements.txt               # Python dependencies
├── data/
│   └── raw/
│       ├── activities_*.json          # Your 1,651 Strava activities ✅
│       ├── activities_*.csv           # CSV version ✅
│       ├── apple_health.csv           # (optional, add if you have Apple Watch)
│       └── garmin_activities.csv      # (optional, add if you have Garmin)
├── notebooks/
│   ├── 01_exploratory.ipynb           # Initial data review ✅
│   ├── 02_hypothesis_testing.ipynb    # Rigorous stats with 5 hypotheses ✅
│   └── 03_biometric_correlation.ipynb # Apple Watch / Garmin fusion ✅
└── docs/
    ├── BIOMETRIC_INTEGRATION_GUIDE.md # How to add Apple/Garmin data
    └── EXPERIMENT_TEMPLATE.md         # Reusable hypothesis framework
```

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Strava OAuth | ✅ Done | Tokens saved to `.env` |
| Data Export | ✅ Done | 1,651 activities in `data/raw/` |
| Python Environment | ✅ Done | Mamba env with pandas, scipy, jupyter |
| Jupyter Lab | 🟢 Running | Open in browser at `http://dkbl1:8888` |
| Exploratory Notebook | ✅ Ready | `01_exploratory.ipynb` |
| Hypothesis Testing | ✅ Ready | `02_hypothesis_testing.ipynb` |
| Biometric Integration | ⏳ Optional | `03_biometric_correlation.ipynb` (needs Apple/Garmin data) |

## Activate Your Environment

**Fresh terminal session:**

```bash
bash -l
micromamba activate strava-analysis
cd ~/strava-analysis
```

Or in a Jupyter cell:
```python
import subprocess
subprocess.run(['bash', '-c', 'source ~/.bashrc && micromamba activate strava-analysis'])
```

## Notebooks Quick Start

### 1. Exploratory Analysis (01_exploratory.ipynb)

**What it does:**
- Loads 1,651 activities
- Shows activity type distribution
- Monthly trends
- Distance, duration, pace breakdowns
- HR & power statistics
- Personal records
- Consistency patterns

**Key output:** Understand your data landscape

### 2. Hypothesis Testing (02_hypothesis_testing.ipynb)

**Tests run (pre-defined, no p-hacking):**

1. **H1: Pace Improvement Over Time**
   - Linear regression with rolling averages
   - Effect: {slope} km/h per activity
   - Result: p < 0.05? (Yes/No)

2. **H2: HR-Pace Correlation**
   - Pearson r with 95% CI
   - Does faster = higher HR?
   - Result: p < 0.05? (Yes/No)

3. **H3: Activity Type Differences**
   - ANOVA (Running vs Cycling vs Walking, etc.)
   - Different paces by activity?
   - Result: p < 0.05? (Yes/No)

4. **H4: Seasonal Effects**
   - Volume & intensity by season
   - Winter vs Summer training?
   - Result: p < 0.05? (Yes/No)

5. **H5: Recovery Impact**
   - Rest days vs next-day performance
   - More rest = better pace?
   - Result: p < 0.05? (Yes/No)

**Key output:** Actionable scientific findings with effect sizes

### 3. Biometric Correlation (03_biometric_correlation.ipynb)

**When you have Apple/Garmin data:**

- **HRV Analysis** — Heart rate variability predicts recovery
- **Training Stress Score** — Detect overtraining
- **Resting HR Trends** — Fitness progression
- **Aerobic Efficiency** — Do more with less HR effort
- **Cross-platform Correlations** — Device agreement

**How to enable:** See "Add Biometric Data" below

---

## Add Biometric Data (Optional)

### Apple Watch

1. iPhone Health app → Profile → Export Health Data → Save
2. Extract to get `.csv` with columns: `date, heart_rate_min, heart_rate_avg, heart_rate_max, hrv_rmssd_ms, steps, calories`
3. Save as `data/raw/apple_health.csv`
4. Rerun notebook 03 — new analyses unlock

**Full guide:** `docs/BIOMETRIC_INTEGRATION_GUIDE.md`

### Garmin Connect

1. https://connect.garmin.com → Settings → Download Your Data → zip
2. Extract CSV with columns: `date, activity_type, distance_km, duration_minutes, training_stress_score, avg_power_watts`
3. Save as `data/raw/garmin_activities.csv`
4. Rerun notebook 03 — TSS & power analysis activates

**Alternative (API):**
```bash
# Add to config/.env:
GARMIN_EMAIL=your_email@example.com
GARMIN_PASSWORD=your_password

# Run:
python scripts/garmin_fetch.py
```

---

## Next: Model Comparison

After running the hypothesis tests, the plan is:

**Compare AI models on same analysis:**
- **Haiku 4.5** — Fast baseline
- **Sonnet 4.6** — Mid-tier depth
- **Kimi 2.5** — Extended reasoning (if available)

Each model reviews the experimental design and suggests improvements.

---

## File Organization

### Never Commit
- `config/.env` (contains API tokens)
- `data/raw/` (keep locally only)

### Safe to Commit
- `README.md`
- `notebooks/*.ipynb`
- `scripts/*.py`
- `docs/`
- `.gitignore`

### Gitignore Already Set
```
config/.env
data/raw/*
__pycache__/
*.pyc
.ipynb_checkpoints/
```

---

## Troubleshooting

### Q: Jupyter shows "kernel not found"
**A:** Make sure kernel points to mamba env
```bash
micromamba activate strava-analysis
python -m ipykernel install --user --name strava-analysis
```

### Q: "No module named pandas" in Jupyter
**A:** Restart kernel (Kernel → Restart)

### Q: Want to run a notebook via CLI
```bash
micromamba activate strava-analysis
jupyter nbconvert --to script 02_hypothesis_testing.ipynb --execute
```

### Q: Tokens expired?
```bash
cd ~/strava-analysis
python scripts/strava_auth_manual.py  # Run again to refresh
```

---

## Key Stats

| Metric | Value |
|--------|-------|
| Total Activities | 1,651 |
| Date Range | {min_date} to {max_date} |
| Python Version | 3.12.13 |
| Environment | Mamba (micromamba) |
| Packages | 100+ (pandas, numpy, scipy, jupyter, etc.) |
| Notebook Cells | 80+ executable cells |
| Hypothesis Tests | 5 major tests with effect sizes |

---

## Next Steps

1. **Now:** Open Jupyter Lab, run 01_exploratory.ipynb
2. **Then:** Run 02_hypothesis_testing.ipynb (15 min)
3. **Optional:** Add Apple/Garmin data, run 03_biometric_correlation.ipynb
4. **Advanced:** Model comparison across Claude versions

---

## Commands You'll Use

```bash
# Activate environment
bash -l && micromamba activate strava-analysis

# Start Jupyter (already running at http://dkbl1:8888)
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# Fetch new data (if needed)
python scripts/strava_export.py

# Refresh tokens (if expired)
python scripts/strava_auth_manual.py
```

---

## Contact / Questions

- All code documented inline
- Notebooks have markdown explanations
- Check `docs/BIOMETRIC_INTEGRATION_GUIDE.md` for setup help
- See `EXPERIMENT_TEMPLATE.md` for hypothesis-testing framework

---

**Built:** March 26, 2026  
**Data points:** 1,651 activities  
**Analysis method:** Rigorous hypothesis-driven with effect sizes & confidence intervals  
**Ready for:** Exploratory analysis → hypothesis testing → biometric integration → model comparison

🚀 Let's analyze your fitness!
