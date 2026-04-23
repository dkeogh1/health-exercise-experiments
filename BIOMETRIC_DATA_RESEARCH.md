# Biometric Data Integration — Research & Implementation Plan

**Goal:** Enrich Strava data with detailed biometric metrics from Apple Watch and Garmin for rigorous data science analysis.

---

## 📊 Data Sources Overview

### 1. Strava API (Primary)
**What's available:**
- Activity metadata (distance, duration, elevation, pace)
- Heart rate averages (if device synced)
- Power data (for Garmin devices on cycling)
- Streams endpoint: time-series data (lat/lng, altitude, HR, cadence, watts, temp)

**Limitations:**
- Streams are low-resolution and aggregated
- Not ideal for detailed HR zone analysis
- Power data sparse unless Garmin power meter

**Implementation:**
```python
# Strava API v3
GET /api/v3/activities/{id}/streams?keys=time,heartrate,cadence,watts,pace
# Returns time-series with 4-sample aggregate
```

---

### 2. Apple HealthKit (Running Data)
**What's available:**
- High-resolution heart rate data (per-second)
- Cadence (steps per minute)
- Distance, pace, elevation
- Recovery heart rate
- VO2 max estimates
- Blood oxygen (if available)
- Workout route & GPS precision

**Access methods:**
1. **Manual export:** Settings → Health → Profile → Export All Health Data (XML format, ~10GB max)
2. **Sideloaded app:** Read HealthKit directly on iPhone, bypass Strava
3. **Python libraries:**
   - `apple-health-parser` (PyPI) — Parse exported XML
   - Custom iOS app with HealthKit queries

**Best approach for you:**
- Export full HealthKit XML from iPhone
- Parse with Python `apple-health-parser`
- Extract time-series HR, cadence per workout
- Merge with Strava metadata

**Implementation:**
```python
from apple_health_parser import AppleHealthParser
parser = AppleHealthParser('export.xml')
workouts = parser.get_workouts()
hr_data = parser.get_heart_rate_data()  # Full resolution
```

---

### 3. Garmin Connect API (Cycling Data)
**What's available:**
- Power (watts) if bike has power meter
- Cadence (pedal RPM)
- Heart rate zones
- Training effect & aerobic/anaerobic benefit
- Body metrics (weight, HRV, resting HR)
- Activity summaries & detailed metrics

**Access methods:**
1. **Official Garmin API:** (Limited access, requires approval)
2. **Unofficial libraries:** (Reverse-engineered Garmin Connect API)
   - `garminconnect` (PyPI) — Actively maintained
   - `garmy` (PyPI) — AI-focused, newer
   - `garth` (GitHub) — Original reverse-engineered library

**Best approach for you:**
- Use `garminconnect` or `garmy` library
- Authenticate with Garmin Connect credentials
- Fetch detailed activity data (metrics, streams)
- Cross-reference with Strava

**Implementation:**
```python
from garminconnect import Garmin
client = Garmin("email@example.com", "password")
client.login()
activities = client.get_activities(0, 100)  # Last 100 activities
activity_details = client.get_activity_details(activity_id)
```

---

## 🔬 Data Science Implementation Plan

### Phase 1: Data Collection & Enrichment

**Tasks:**
1. **Export Apple HealthKit data**
   - Full XML export from iPhone
   - Parse running workouts with full HR resolution
   - Extract: HR (per-second), cadence, distance, pace, elevation, timestamps

2. **Fetch Garmin data**
   - Authenticate via `garminconnect` library
   - Fetch cycling activities with detailed metrics
   - Extract: power, cadence, HR zones, training effect, timestamps

3. **Enrich Strava data**
   - Merge Apple/Garmin data with Strava via timestamp & activity type
   - Create composite dataset: Strava metadata + biometric detail
   - CSV structure:
     ```
     activity_id, source, date, type, distance, duration, 
     avg_hr, max_hr, hr_zones, cadence, power, elevation, 
     gear_id, recovery_hr, vo2_max, temperature, ...
     ```

### Phase 2: Hypothesis Testing & Experiment Design

**Rigorous approach:**
- Define null hypotheses explicitly (not just fishing for patterns)
- Quantify effect sizes (not just p-values)
- Control for confounders (e.g., temperature, time of day, gear, fatigue)
- Use power analysis to ensure sample sizes

**Example experiments:**

| Hypothesis | Null H0 | Design | Metrics |
|-----------|---------|--------|---------|
| "Hard days + easy days improve fitness" | Training distribution has no effect on pace progression | Polarized vs. threshold: classify weeks, compare pace slopes | Effect size (Cohen's d), 95% CI |
| "Morning runs are faster" | Time of day has no effect on pace | Paired t-test: AM vs. PM pace for same runner | Cohen's d for paired samples |
| "HR recovery predicts fitness" | Recovery HR doesn't correlate with future performance | Lagged correlation + time series model | Spearman ρ, 95% CI, lag analysis |
| "Cross-training helps running" | Weeks with cycling don't improve running performance | Compare running pace improvement in mixed-sport vs. running-only weeks | ANCOVA (controlling for volume) |

### Phase 3: Analysis Workflow

**Reproducible pipeline:**
1. **Data validation:** Check completeness, outliers, data quality
2. **Descriptive stats:** Distributions, central tendencies, missingness
3. **Visual EDA:** Heatmaps, time-series plots, scatter matrices
4. **Hypothesis testing:** Each experiment as a separate notebook
5. **Effect size reporting:** Always include CI, not just p-values
6. **Sensitivity analysis:** How robust are findings to data assumptions?

**Tools:**
- `pandas` + `numpy` for wrangling
- `scipy.stats` for hypothesis testing
- `statsmodels` for time-series & regression
- `matplotlib` + `seaborn` + `plotly` for viz
- Jupyter notebooks for reproducibility

---

## 📋 Feasibility & Trade-offs

### Apple HealthKit
| Aspect | Status | Notes |
|--------|--------|-------|
| **Data richness** | ✅ High | Per-second resolution, many metrics |
| **Ease of access** | ⚠️ Medium | Manual export OR sideloaded app |
| **Frequency** | ❌ Manual | Can't auto-sync without app sideloading |
| **Privacy** | ✅ Good | You control all data, local parsing |

**Recommendation:** Export once, parse with `apple-health-parser`. Update periodically as needed.

### Garmin Connect API
| Aspect | Status | Notes |
|--------|--------|-------|
| **Data richness** | ✅ High | Power, cadence, training effect |
| **Ease of access** | ✅ Programmatic | `garminconnect` library handles auth |
| **Frequency** | ✅ Auto-sync | Can schedule daily/weekly refreshes |
| **Privacy** | ⚠️ Caution | Uses unofficial reverse-engineered API (ToS risk) |

**Recommendation:** Use `garminconnect` for automation, but be aware Garmin could block access. Have fallback export strategy.

---

## 🛠️ Implementation Roadmap

### Tomorrow (with fresh Strava token):
- [ ] Run basic Strava export
- [ ] Sketch data schema (what fields to collect)
- [ ] Prototype Apple HealthKit parser
- [ ] Test Garmin API access

### Next few days:
- [ ] Collect & validate all three data sources
- [ ] Build enriched dataset
- [ ] Design 3-5 core hypotheses with Sonnet 4.6

### Experiments phase:
- [ ] Run hypothesis tests (rigorous design)
- [ ] Report effect sizes + CIs
- [ ] Visualize findings
- [ ] Model recommendations from Kimi 2.5

---

## 📚 References

- **Apple HealthKit:** https://developer.apple.com/documentation/healthkit
- **apple-health-parser:** https://pypi.org/project/apple-health-parser/
- **Garmin API:** https://developer.garmin.com/gc-developer-program/activity-api/
- **garminconnect:** https://pypi.org/project/garminconnect/
- **Strava API v3:** https://developers.strava.com/docs/reference/

---

## Questions for Dan

1. Do you have a Garmin power meter on your bike, or cadence-only?
2. How far back do you want historical data? (All-time or last 6-12 months?)
3. Any known injuries/breaks in training we should account for in analysis?
4. Specific fitness goals? (Speed, endurance, cross-training optimization?)

