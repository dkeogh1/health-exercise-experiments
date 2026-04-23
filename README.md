# Strava Data Science Analysis

Rigorous fitness data analysis with biometric integration.

## Project Structure

```
~/strava-analysis/
├── README.md                    # This file
├── config/
│   ├── .env.template           # Credentials template
│   ├── .env                     # Your actual credentials (DO NOT COMMIT)
│   └── strava_auth_setup.md    # OAuth flow guide
├── data/
│   ├── raw/                     # Raw exports from APIs
│   ├── processed/               # Cleaned data
│   └── exports/                 # Analysis outputs
├── scripts/
│   ├── strava_auth.py          # OAuth + token refresh
│   ├── strava_export.py        # Fetch activities
│   ├── biometric_fetch.py      # Apple HealthKit + Garmin
│   ├── merge_data.py           # Combine sources
│   └── requirements.txt
├── notebooks/
│   ├── 01_exploratory.ipynb    # Initial data review
│   ├── 02_hypothesis_testing.ipynb
│   └── 03_biometric_correlation.ipynb
└── docs/
    ├── BIOMETRIC_DATA_RESEARCH.md
    └── EXPERIMENT_TEMPLATE.md
```

## Quick Start

1. **Set up credentials:**
   ```bash
   cp config/.env.template config/.env
   # Edit config/.env with your Strava Client ID & Secret
   ```

2. **Install dependencies:**
   ```bash
   pip install -r scripts/requirements.txt
   ```

3. **Authenticate with Strava:**
   ```bash
   python scripts/strava_auth.py
   ```

4. **Fetch your activities:**
   ```bash
   python scripts/strava_export.py
   ```

5. **Start analyzing:**
   Open `notebooks/01_exploratory.ipynb` in Jupyter

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
