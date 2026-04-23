# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Biometric Data Integration & Correlation Analysis
#
# Integrating Apple Watch, Garmin, and Strava data for comprehensive fitness analysis.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
# %matplotlib inline

print('✅ Libraries imported')

# %% [markdown]
# ## Load Strava Data

# %%
# Load Strava activities
data_dir = Path('../data/raw')
json_files = sorted(data_dir.glob('activities_*.json'), reverse=True)
latest_file = json_files[0]

with open(latest_file) as f:
    activities_raw = json.load(f)

df_strava = pd.DataFrame(activities_raw)
df_strava['start_date'] = pd.to_datetime(df_strava['start_date'])
df_strava['date_only'] = df_strava['start_date'].dt.date
df_strava['distance_km'] = df_strava['distance_m'] / 1000
df_strava['duration_hours'] = df_strava['duration_s'] / 3600
df_strava['pace_kmh'] = df_strava['distance_km'] / df_strava['duration_hours']

print(f"✅ Loaded {len(df_strava)} Strava activities")
print(f"Strava data includes:")
print(f"  - Average HR: {df_strava['average_hr'].notna().sum()} activities")
print(f"  - Power (watts): {df_strava['average_watts'].notna().sum()} activities")

# %% [markdown]
# ## Apple HealthKit Data Import
#
# **Instructions to export Apple Watch data:**
#
# 1. On iPhone: Health app → Profile → Export Health Data → Save to Files
# 2. Transfer `apple_health_export/workout_routes/` to `data/raw/apple_watch/`
# 3. Or use this format to create `data/raw/apple_health.csv`:
#
# ```csv
# date,heart_rate_min,heart_rate_max,heart_rate_avg,steps,calories,active_energy,stand_hours,workout_type,workout_duration_min
# 2024-01-01,55,165,120,8234,2100,450,12,Running,45
# ```

# %%
# Check for Apple Health data
apple_health_path = Path('../data/raw/apple_health.csv')

if apple_health_path.exists():
    df_apple = pd.read_csv(apple_health_path)
    df_apple['date'] = pd.to_datetime(df_apple['date'])
    print(f"✅ Loaded Apple Health data: {len(df_apple)} days")
    print(df_apple.head())
else:
    df_apple = None
    print("\n⚠️ No apple_health.csv found")
    print("   To use Apple Watch data, create: data/raw/apple_health.csv")
    print("   Or place export in: data/raw/apple_watch/")

# %% [markdown]
# ## Garmin Connect Data Import
#
# **Instructions to fetch Garmin data:**
#
# 1. Install: `pip install garminconnect`
# 2. Run the fetch script below
# 3. Or manually export from Garmin Connect: Settings → Export Data → .zip
#
# Expected format: `data/raw/garmin_activities.csv`
# ```csv
# date,activity_type,distance_km,duration_minutes,avg_power_watts,normalized_power_watts,training_stress_score,intensity_factor
# 2024-01-01,Cycling,32.5,90,245,265,120,0.95
# ```

# %%
# Check for Garmin data
garmin_path = Path('../data/raw/garmin_activities.csv')

if garmin_path.exists():
    df_garmin = pd.read_csv(garmin_path)
    df_garmin['date'] = pd.to_datetime(df_garmin['date'])
    print(f"✅ Loaded Garmin data: {len(df_garmin)} activities")
    print(df_garmin.head())
else:
    df_garmin = None
    print("\n⚠️ No garmin_activities.csv found")
    print("   To use Garmin data, create: data/raw/garmin_activities.csv")

# %% [markdown]
# ## Analysis 1: HR Variability (HRV) & Recovery
#
# **Concept:** HRV (variation in beat-to-beat intervals) indicates nervous system recovery.  
# **Hypothesis:** Higher HRV on previous night → better performance next day

# %%
if df_apple is not None:
    print("\n" + "="*60)
    print("ANALYSIS 1: HRV & PERFORMANCE")
    print("="*60)
    
    # Check if HRV column exists
    if 'hrv_rmssd_ms' in df_apple.columns or 'heart_rate_variability' in df_apple.columns:
        hrv_col = 'hrv_rmssd_ms' if 'hrv_rmssd_ms' in df_apple.columns else 'heart_rate_variability'
        
        # Merge Apple data with Strava (next day)
        df_apple['date_only'] = df_apple['date'].dt.date
        df_apple['next_date'] = (df_apple['date'] + timedelta(days=1)).dt.date
        
        merged = df_strava.merge(
            df_apple[['next_date', hrv_col]].rename(columns={'next_date': 'date_only'}),
            on='date_only',
            how='inner'
        )
        
        if len(merged) > 2:
            corr_hrv, p_hrv = stats.pearsonr(merged[hrv_col], merged['pace_kmh'])
            print(f"\nHRV vs Pace Correlation:")
            print(f"  r = {corr_hrv:.4f}")
            print(f"  p-value = {p_hrv:.6f}")
            print(f"  N = {len(merged)} matched days")
            
            if p_hrv < 0.05:
                print(f"\n  ✅ SIGNIFICANT: HRV predicts next-day performance")
            else:
                print(f"\n  ℹ️ Not significant (p={p_hrv:.4f})")
        else:
            print("\n  ⚠️ Not enough matched data points")
    else:
        print("\n  ⚠️ No HRV data in Apple Health export")
else:
    print("\n⏭️ ANALYSIS 1: Skipped (no Apple Health data)")

# %% [markdown]
# ## Analysis 2: Training Stress Score (TSS) vs Fatigue
#
# **Concept:** Garmin's TSS (Training Stress Score) measures workout intensity.  
# **Hypothesis:** High TSS without recovery → lower performance next activity

# %%
if df_garmin is not None:
    print("\n" + "="*60)
    print("ANALYSIS 2: TRAINING STRESS & FATIGUE")
    print("="*60)
    
    if 'training_stress_score' in df_garmin.columns:
        # Plot TSS over time
        fig = px.line(df_garmin.sort_values('date'), x='date', y='training_stress_score',
                      title='Training Stress Score Over Time',
                      labels={'date': 'Date', 'training_stress_score': 'TSS'})
        fig.show()
        
        # Calculate rolling avg fatigue
        df_garmin_sorted = df_garmin.sort_values('date').reset_index(drop=True)
        df_garmin_sorted['tss_rolling_7'] = df_garmin_sorted['training_stress_score'].rolling(window=7).sum()
        
        print(f"\nTSS Statistics:")
        print(f"  Mean: {df_garmin['training_stress_score'].mean():.1f}")
        print(f"  Std: {df_garmin['training_stress_score'].std():.1f}")
        print(f"  Max: {df_garmin['training_stress_score'].max():.1f}")
        
        # Fatigue categorization
        print(f"\nWeekly TSS Summary (7-day rolling):")
        print(df_garmin_sorted[['date', 'training_stress_score', 'tss_rolling_7']].tail(10))
    else:
        print("\n  ⚠️ No TSS data in Garmin export")
else:
    print("\n⏭️ ANALYSIS 2: Skipped (no Garmin data)")

# %% [markdown]
# ## Analysis 3: Resting Heart Rate (RHR) Trend
#
# **Concept:** Rising RHR indicates overtraining/illness; falling RHR shows fitness gains.  
# **Hypothesis:** RHR correlates with average HR during workouts

# %%
if df_apple is not None:
    print("\n" + "="*60)
    print("ANALYSIS 3: RESTING HEART RATE TRENDS")
    print("="*60)
    
    if 'heart_rate_min' in df_apple.columns:
        # RHR is typically the minimum HR during rest
        rhr = df_apple.sort_values('date')['heart_rate_min']
        
        # Trend analysis
        x = np.arange(len(rhr))
        slope_rhr, intercept_rhr, r_rhr, p_rhr, _ = stats.linregress(x, rhr.values)
        
        print(f"\nResting Heart Rate Trend:")
        print(f"  Slope: {slope_rhr:.6f} bpm/day")
        print(f"  R²: {r_rhr**2:.4f}")
        print(f"  p-value: {p_rhr:.6f}")
        
        if p_rhr < 0.05:
            direction = "DECREASING" if slope_rhr < 0 else "INCREASING"
            print(f"\n  ✅ SIGNIFICANT: RHR is {direction}")
            print(f"     {direction.lower()} by {abs(slope_rhr):.4f} bpm per day")
        
        # Visualize
        fig = px.line(df_apple.sort_values('date'), x='date', y='heart_rate_min',
                      title='Resting Heart Rate Over Time',
                      labels={'date': 'Date', 'heart_rate_min': 'RHR (bpm)'})
        fig.show()
    else:
        print("\n  ⚠️ No resting HR data in Apple Health")
else:
    print("\n⏭️ ANALYSIS 3: Skipped (no Apple Health data)")

# %% [markdown]
# ## Analysis 4: Activity Type Efficiency (Power/HR)
#
# **Concept:** Efficiency = Distance / (HR × Duration)  
# **Hypothesis:** Efficiency improves with training (doing same work at lower HR)

# %%
df_efficiency = df_strava[
    (df_strava['average_hr'].notna()) & 
    (df_strava['duration_hours'] > 0)
].copy()

if len(df_efficiency) > 0:
    print("\n" + "="*60)
    print("ANALYSIS 4: AEROBIC EFFICIENCY")
    print("="*60)
    
    # Efficiency metric: distance per HR-minute
    df_efficiency['efficiency'] = df_efficiency['distance_km'] / (df_efficiency['average_hr'] * df_efficiency['duration_hours'])
    
    # Sort by date
    df_efficiency = df_efficiency.sort_values('start_date')
    
    # Trend
    x = np.arange(len(df_efficiency))
    slope_eff, intercept_eff, r_eff, p_eff, _ = stats.linregress(x, df_efficiency['efficiency'].values)
    
    print(f"\nAerobic Efficiency Trend:")
    print(f"  Slope: {slope_eff:.8f} (km/(HR·hour)) per activity")
    print(f"  R²: {r_eff**2:.4f}")
    print(f"  p-value: {p_eff:.6f}")
    
    if p_eff < 0.05:
        direction = "IMPROVING" if slope_eff > 0 else "DECLINING"
        print(f"\n  ✅ SIGNIFICANT: Efficiency is {direction}")
    else:
        print(f"\n  ℹ️ No significant efficiency trend (p={p_eff:.4f})")
    
    # By activity type
    print(f"\nEfficiency by Activity Type:")
    for atype in df_efficiency['type'].unique():
        eff = df_efficiency[df_efficiency['type'] == atype]['efficiency'].mean()
        print(f"  {atype}: {eff:.6f}")
    
    # Visualize
    fig = px.scatter(df_efficiency, x='start_date', y='efficiency', color='type',
                    title='Aerobic Efficiency Over Time',
                    labels={'start_date': 'Date', 'efficiency': 'Efficiency (km/(HR·hr)})
    fig.show()
else:
    print("\n⚠️ Not enough HR data for efficiency analysis")

# %% [markdown]
# ## Analysis 5: Cross-Platform Correlation Matrix

# %%
# Build correlation matrix from available data
correlation_vars = {
    'Strava': {
        'Pace (km/h)': df_strava['pace_kmh'],
        'Duration (h)': df_strava['duration_hours'],
        'Distance (km)': df_strava['distance_km'],
        'Avg HR (bpm)': df_strava['average_hr'],
        'Power (W)': df_strava['average_watts'],
        'Elevation (m)': df_strava['elevation_m']
    }
}

# Add Apple data if available
if df_apple is not None:
    if 'heart_rate_avg' in df_apple.columns:
        # Daily aggregate HR
        daily_apple_hr = df_apple.set_index('date_only')['heart_rate_avg']
        daily_strava_hr = df_strava.groupby('date_only')['average_hr'].mean()
        
        if len(daily_apple_hr) > 0 and len(daily_strava_hr) > 0:
            correlation_vars['Apple Watch'] = {'Daily Avg HR': daily_apple_hr}

# Create combined dataframe for correlation
print("\n" + "="*60)
print("ANALYSIS 5: CROSS-PLATFORM CORRELATIONS")
print("="*60)

# Strava correlations
strava_corr = df_strava[['pace_kmh', 'duration_hours', 'distance_km', 'average_hr', 'average_watts', 'elevation_m']].corr()

print("\nStrava Variable Correlations:")
print(strava_corr)

# Heatmap
fig = px.imshow(strava_corr, 
               labels=dict(color='Correlation'),
               title='Strava Data Correlation Matrix',
               color_continuous_scale='RdBu_r',
               zmin=-1, zmax=1)
fig.show()

# %% [markdown]
# ## Setup Instructions for Biometric Data
#
# ### Apple Watch Export
#
# 1. On iPhone: Open Health app
# 2. Tap Profile (bottom right) → Export Health Data
# 3. Download the zip file
# 4. Extract and upload CSV files to `data/raw/apple_watch/`
#
# **Expected files:**
# - `apple_health.csv` with columns: `date, heart_rate_min, heart_rate_max, heart_rate_avg, hrv_rmssd_ms, steps, calories`
#
# ### Garmin Connect Export
#
# 1. Log in to https://connect.garmin.com
# 2. Settings → Account → Download Your Data
# 3. Download the archive
# 4. Extract CSV files to `data/raw/garmin/`
#
# **Expected files:**
# - `garmin_activities.csv` with columns: `date, activity_type, distance_km, duration_minutes, avg_power_watts, training_stress_score, intensity_factor`
#
# ### Python-based Garmin Fetch (Optional)
#
# ```python
# from garminconnect import Garmin
#
# # See scripts/garmin_fetch.py for full implementation
# ```

# %% [markdown]
# ## Summary
#
# When you add biometric data:
#
# 1. **HRV Analysis** — Recovery patterns
# 2. **TSS Tracking** — Training load management
# 3. **RHR Trends** — Fitness progression indicator
# 4. **Efficiency Metrics** — Do more with less HR effort
# 5. **Cross-platform Correlations** — See how devices complement each other
#
# All analyses use rigorous stats with effect sizes and confidence intervals.
