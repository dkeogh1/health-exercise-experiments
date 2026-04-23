# Biometric Data Integration Guide

Complete setup for Apple Watch, Garmin, and Strava data fusion.

## Overview

Three data sources:
1. **Strava** (already loaded) — Activities, HR during workouts, power
2. **Apple Watch** (optional) — Daily HR patterns, HRV, sleep, steps, activity rings
3. **Garmin** (optional) — Training Stress Score (TSS), Normalized Power, intensity metrics

## 1. Apple Watch / Health Kit

### Export from iPhone

1. **Health** app → **Profile** → **Export All Health Data**
2. Save the zip, extract, and place `export.xml` at
   `data/raw/apple_health_export/export.xml`.

### Ingest

The ingest converts the XML (typically 0.5–2 GB) into a SQLite database
at `data/processed/apple_health.db`, plus a daily-aggregate
`data/raw/apple_health.csv` for quick notebook consumption.

```bash
python scripts/apple_health_ingest.py
```

Implementation lives in `src/apple_health.py` and uses a streaming
parser with `INSERT OR IGNORE` to tolerate the duplicate workouts that
appear when a single session is recorded by multiple sources. Typical
ingest time: ~35 seconds for a 1.2 GB export.

### Record types stored

One SQLite table per HealthKit type (prefixed `r_`). Notable ones:

| Table | Typical signal |
|-------|----------------|
| `r_HeartRate` | Per-sample HR (sub-minute) |
| `r_HeartRateVariabilitySDNN` | HRV (SDNN, sampled during Breathe sessions) |
| `r_RestingHeartRate` | Apple-computed daily RHR |
| `r_VO2Max` | Apple's VO₂max estimate from outdoor workouts |
| `r_HeartRateRecoveryOneMinute` | HRR1 post-workout |
| `r_RunningPower` | Per-second running power (Apple Watch Series 10+) |
| `r_RunningSpeed` / `r_RunningStrideLength` | Running-dynamics streams |
| `r_RunningGroundContactTime` / `r_RunningVerticalOscillation` | Running-dynamics streams |
| `r_OxygenSaturation` | Per-reading SpO₂ |
| `r_RespiratoryRate` | Breaths/min during sleep |
| `r_SleepAnalysis` | Sleep-stage intervals (Asleep{Core,Deep,REM,Unspecified}, Awake, InBed) |
| `r_BodyMass` / `r_BodyFatPercentage` / `r_LeanBodyMass` | Withings body comp |
| `workouts` | Deduped workout summaries (natural key: type+start+end+source) |

Every record has an indexed `startDate_ts` column (Unix seconds, UTC)
for timezone-safe range scans.

### Query helpers (`src.apple_health`)

```python
from src import apple_health as ah
from pathlib import Path
db = Path("data/processed/apple_health.db")

daily  = ah.daily_metrics(db)          # one row per day, all aggregates
hr     = ah.records(db, "HeartRate")   # full-history samples of any record type
wk     = ah.workouts(db)               # deduped workout sessions

# Extract per-workout streams:
latest_run = wk[wk.workoutActivityType == "HKWorkoutActivityTypeRunning"].iloc[-1]
rp = ah.workout_records(db, latest_run.startDate, latest_run.endDate, "RunningPower")
```

### Daily-CSV columns

`data/raw/apple_health.csv` exposes these columns (one row per local day):

| Field | Description |
|-------|-------------|
| `heart_rate_min/avg/max` | From `r_HeartRate` per-sample data |
| `resting_hr` | Apple's daily RHR estimate |
| `walking_hr_avg` | WalkingHeartRateAverage |
| `hrv_sdnn_ms` | **SDNN** (not RMSSD — what Apple actually stores) |
| `vo2max` | Apple's daily VO₂max estimate |
| `oxygen_saturation` | Daily mean SpO₂ |
| `respiratory_rate` | Daily mean RR |
| `steps` / `flights_climbed` / `active_energy` / `basal_energy` | Daily totals |
| `exercise_min` / `stand_min` | Daily totals |
| `hrr1` | Best HRR-1-minute for the day |
| `wrist_temp_sleep` | Apple Sleeping Wrist Temperature |
| `sleep_minutes` | Sum of all `Asleep*` intervals ending that day |
| `stand_hours` | HKCategoryTypeIdentifierAppleStandHour | Stand hours (0-24) |

## 2. Garmin Connect

### Export via Web

**Steps:**
1. Go to https://connect.garmin.com
2. Log in with your Garmin credentials
3. Settings → Account → Download Your Data
4. Select date range → Download zip

**File structure:**
```
garmin_export/
├── DI_CONNECT/
│   ├── DI-Connect_Activity_*.csv
│   ├── DI-Connect_Biometrics.csv
│   └── ...
└── summaries/ (optional)
```

### Auto-fetch via API (requires garminconnect)

1. Install library:
   ```bash
   pip install garminconnect
   ```

2. Add credentials to `config/.env`:
   ```
   GARMIN_EMAIL=your_email@example.com
   GARMIN_PASSWORD=your_password
   ```

3. Run fetch script:
   ```bash
   python scripts/garmin_fetch.py
   ```

### Expected CSV Format

```csv
date,activity_type,distance_km,duration_minutes,avg_power_watts,normalized_power_watts,training_stress_score,intensity_factor,elevation_m
2024-01-01,Cycling,32.5,90,245,265,120,0.95,450
2024-01-01,Running,10.2,65,NULL,NULL,75,0.80,120
```

### Data Fields

| Field | Description | Use Case |
|-------|-------------|----------|
| `training_stress_score` | TSS = duration × intensity factor | Fatigue/load tracking |
| `normalized_power_watts` | Adjusted for effort variability | Power consistency |
| `intensity_factor` | NP / FTP (functional threshold power) | Training zone |
| `avg_power_watts` | Literal average power output | Cycling metrics |

## 3. File Placement

After export/conversion, place files in:

```
~/strava-analysis/data/raw/
├── activities_*.json        (Strava - already have)
├── apple_health.csv         (Apple Watch - optional)
└── garmin_activities.csv    (Garmin - optional)
```

## 4. Run Analysis

Once files are in place:

```bash
jupyter lab
# Open: notebooks/03_biometric_correlation.ipynb
```

The notebook automatically detects which data files exist and runs corresponding analyses.

## 5. Advanced: Data Quality Checks

### Apple Health Validation

```python
import pandas as pd

df = pd.read_csv('data/raw/apple_health.csv')

# Check for NaN
print(df.isnull().sum())

# Check date range
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Check HR ranges (should be 40-200 bpm)
print(f"HR min: {df['heart_rate_min'].min()}, max: {df['heart_rate_max'].max()}")

# Check HRV range (should be 20-200 ms)
print(f"HRV range: {df['hrv_rmssd_ms'].min()}-{df['hrv_rmssd_ms'].max()}")
```

### Garmin Validation

```python
df_garmin = pd.read_csv('data/raw/garmin_activities.csv')

# Check TSS range (0-500 typical)
print(f"TSS range: {df_garmin['training_stress_score'].min()}-{df_garmin['training_stress_score'].max()}")

# Check power range (watts should be 0-500 for most cyclists)
print(f"Power range: {df_garmin['avg_power_watts'].min()}-{df_garmin['avg_power_watts'].max()}")

# Check for duplicates
print(f"Duplicates: {df_garmin.duplicated().sum()}")
```

## 6. Privacy & Security

- **Never commit** `config/.env` with passwords
- Apple Health exports contain sensitive health data — store locally only
- Garmin API credentials in `.env` are not transmitted to analysis
- All data stays on your machine (no cloud upload)

## 7. Troubleshooting

### "No apple_health.csv found"
- Export from iPhone Health app (not just iCloud Health app)
- Verify file is in `data/raw/`
- Check filename spelling (lowercase, underscore)

### "Garmin login failed"
- Verify email & password in `config/.env`
- Check if 2FA is enabled (may block API access)
- Try generating Garmin app-specific password

### "Columns not found"
- Check column names match expected (case-sensitive)
- Run `df.columns.tolist()` in a Jupyter cell to see actual names
- Refer to "Data Fields" section above

## 8. Next Steps

Once biometric data is loaded:

1. **Recovery Tracking** — HRV trends predict performance
2. **Overtraining Detection** — Rising RHR + high TSS = warning
3. **Efficiency Metrics** — Same distance, lower HR = fitness gain
4. **Cross-device Validation** — Compare Strava HR vs Apple/Garmin
5. **Predictive Models** — Next-day performance from previous biometrics

---

**Questions?** See `docs/EXPERIMENT_TEMPLATE.md` for hypothesis-testing workflows.
