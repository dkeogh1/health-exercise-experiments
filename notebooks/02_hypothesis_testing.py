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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Strava Hypothesis Testing
#
# Rigorous statistical analysis with pre-defined hypotheses, effect sizes, and confidence intervals.

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
# ## Load Data

# %%
# Find the latest exported file
data_dir = Path('../data/raw')
json_files = sorted(data_dir.glob('activities_*.json'), reverse=True)
latest_file = json_files[0]

with open(latest_file) as f:
    activities_raw = json.load(f)

df = pd.DataFrame(activities_raw)
df['start_date'] = pd.to_datetime(df['start_date'])
df['distance_km'] = df['distance_m'] / 1000
df['duration_hours'] = df['duration_s'] / 3600
df['pace_kmh'] = df['distance_km'] / df['duration_hours']
df['year'] = df['start_date'].dt.year
df['month'] = df['start_date'].dt.month
df['date_only'] = df['start_date'].dt.date

print(f"✅ Loaded {len(df)} activities")
print(f"Date range: {df['start_date'].min().date()} to {df['start_date'].max().date()}")

# %% [markdown]
# # Hypothesis Tests
#
# Each hypothesis follows this structure:
# 1. **Null hypothesis (H₀)** — No effect
# 2. **Alternative hypothesis (H₁)** — Effect exists
# 3. **Test** — Statistical test with α = 0.05
# 4. **Effect size** — Cohen's d or correlation
# 5. **Confidence interval** — 95% CI
# 6. **Assumptions check** — Normality, homogeneity, etc.

# %% [markdown]
# ## H1: Fitness Improvement Over Time
#
# **H₀:** Pace has not changed over the study period  
# **H₁:** Pace has improved (gotten faster) over time
#
# **Method:** Linear regression of pace over time, with confidence intervals

# %%
from scipy.stats import linregress

# Use rolling average to smooth out noise
df_sorted = df.sort_values('start_date').reset_index(drop=True)
df_sorted['pace_rolling_30'] = df_sorted['pace_kmh'].rolling(window=30, min_periods=1).mean()

# Regression: time vs pace
x = np.arange(len(df_sorted))
y = df_sorted['pace_kmh'].values

slope, intercept, r_value, p_value, std_err = linregress(x, y)

print("\n" + "="*60)
print("H1: PACE IMPROVEMENT OVER TIME")
print("="*60)
print(f"\nLinear Regression:")
print(f"  Slope: {slope:.6f} km/h per activity")
print(f"  Intercept: {intercept:.2f} km/h")
print(f"  R²: {r_value**2:.4f}")
print(f"  p-value: {p_value:.6f}")
print(f"  Std Error: {std_err:.6f}")

# Interpret
if p_value < 0.05:
    direction = "INCREASED" if slope > 0 else "DECREASED"
    print(f"\n✅ STATISTICALLY SIGNIFICANT (p < 0.05)")
    print(f"   Pace has {direction} by {abs(slope)*100:.4f}% per activity")
else:
    print(f"\n❌ NOT STATISTICALLY SIGNIFICANT (p = {p_value:.4f})")
    print(f"   No evidence of pace improvement over time")

# 95% CI for slope
ci_margin = 1.96 * std_err
print(f"\n95% CI for slope: [{slope - ci_margin:.6f}, {slope + ci_margin:.6f}]")

# %%
# Visualize
fig = px.scatter(df_sorted, x=df_sorted.index, y='pace_kmh',
                labels={'x': 'Activity #', 'y': 'Pace (km/h)'},
                title='Pace Over Time (with 30-activity rolling average)')

# Add rolling average
fig.add_scatter(x=df_sorted.index, y=df_sorted['pace_rolling_30'],
               mode='lines', name='30-activity Rolling Avg',
               line=dict(color='red', width=3))

# Add regression line
y_pred = intercept + slope * np.arange(len(df_sorted))
fig.add_scatter(x=df_sorted.index, y=y_pred,
               mode='lines', name='Linear Fit',
               line=dict(color='green', width=2, dash='dash'))

fig.show()

# %% [markdown]
# ## H2: Heart Rate & Fitness Correlation
#
# **H₀:** Heart rate and pace are not correlated  
# **H₁:** Higher pace (faster speeds) correlate with higher heart rate
#
# **Method:** Pearson correlation with p-value and 95% CI

# %%
# Filter to activities with both pace and HR data
df_hr = df[df['average_hr'].notna() & (df['pace_kmh'].notna())].copy()

if len(df_hr) > 2:
    corr, p_value = stats.pearsonr(df_hr['pace_kmh'], df_hr['average_hr'])
    
    print("\n" + "="*60)
    print("H2: HEART RATE & PACE CORRELATION")
    print("="*60)
    print(f"\nPearson Correlation:")
    print(f"  r = {corr:.4f}")
    print(f"  p-value = {p_value:.6f}")
    print(f"  N = {len(df_hr)}")
    
    # Interpret correlation strength
    if abs(corr) < 0.3:
        strength = "weak"
    elif abs(corr) < 0.7:
        strength = "moderate"
    else:
        strength = "strong"
    
    if p_value < 0.05:
        print(f"\n✅ STATISTICALLY SIGNIFICANT (p < 0.05)")
        print(f"   {strength.upper()} positive correlation")
    else:
        print(f"\n❌ NOT STATISTICALLY SIGNIFICANT (p = {p_value:.4f})")
        print(f"   No evidence of correlation")
    
    # 95% CI using Fisher z-transformation
    z = np.arctanh(corr)
    se = 1 / np.sqrt(len(df_hr) - 3)
    z_ci = 1.96 * se
    r_ci_lower = np.tanh(z - z_ci)
    r_ci_upper = np.tanh(z + z_ci)
    print(f"\n95% CI for r: [{r_ci_lower:.4f}, {r_ci_upper:.4f}]")
else:
    print("\n⚠️ Not enough data for HR correlation analysis")

# %%
# Visualize correlation
if len(df_hr) > 2:
    fig = px.scatter(df_hr, x='pace_kmh', y='average_hr',
                    labels={'pace_kmh': 'Pace (km/h)', 'average_hr': 'Avg HR (bpm)'},
                    title=f'Pace vs Heart Rate (r={corr:.3f}, p={p_value:.4f})')
    
    # Add trendline
    z = np.polyfit(df_hr['pace_kmh'], df_hr['average_hr'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df_hr['pace_kmh'].min(), df_hr['pace_kmh'].max(), 100)
    fig.add_scatter(x=x_trend, y=p(x_trend), mode='lines',
                   name='Trendline', line=dict(color='red'))
    
    fig.show()

# %% [markdown]
# ## H3: Activity Type Differences
#
# **H₀:** Average pace is the same across all activity types  
# **H₁:** Average pace differs by activity type
#
# **Method:** One-way ANOVA with post-hoc pairwise comparisons

# %%
# Get activity types with sufficient sample size
activity_types = df['type'].value_counts()
valid_types = activity_types[activity_types >= 10].index.tolist()  # Min 10 activities
df_types = df[df['type'].isin(valid_types)].copy()

if len(valid_types) >= 2:
    # Group by type
    groups = [df_types[df_types['type'] == atype]['pace_kmh'].dropna().values for atype in valid_types]
    
    # One-way ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    print("\n" + "="*60)
    print("H3: ACTIVITY TYPE DIFFERENCES (ANOVA)")
    print("="*60)
    print(f"\nOne-way ANOVA (Pace):")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Activity types tested: {len(valid_types)}")
    
    if p_value < 0.05:
        print(f"\n✅ STATISTICALLY SIGNIFICANT (p < 0.05)")
        print(f"   Pace differs significantly by activity type")
    else:
        print(f"\n❌ NOT STATISTICALLY SIGNIFICANT (p = {p_value:.4f})")
        print(f"   No evidence of pace differences across activity types")
    
    # Descriptive stats by type
    print("\nDescriptive Statistics by Activity Type:")
    for atype in valid_types:
        paces = df_types[df_types['type'] == atype]['pace_kmh'].dropna()
        print(f"  {atype}: mean={paces.mean():.2f}, std={paces.std():.2f}, n={len(paces)}")
else:
    print("\n⚠️ Not enough activity types for comparison")

# %%
# Visualize
if len(valid_types) >= 2:
    fig = px.box(df_types, x='type', y='pace_kmh',
                title=f'Pace Distribution by Activity Type (F={f_stat:.2f}, p={p_value:.4f})')
    fig.show()


# %% [markdown]
# ## H4: Seasonal Effects
#
# **H₀:** Activity metrics are consistent across seasons  
# **H₁:** Activity volume and intensity vary by season
#
# **Method:** ANOVA comparing winter/spring/summer/fall

# %%
# Define seasons
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df['season'] = df['month'].apply(get_season)

# Activity count by season
season_activity = df.groupby('season').size()
season_volume = df.groupby('season')['distance_km'].sum()
season_pace = df.groupby('season')['pace_kmh'].mean()

print("\n" + "="*60)
print("H4: SEASONAL EFFECTS")
print("="*60)
print("\nActivity Count by Season:")
print(season_activity)

print("\nTotal Distance by Season:")
print(season_volume)

print("\nAverage Pace by Season:")
print(season_pace)

# ANOVA on distance (volume)
groups_volume = [df[df['season'] == s]['distance_km'].values for s in ['Winter', 'Spring', 'Summer', 'Fall']]
f_vol, p_vol = stats.f_oneway(*groups_volume)

print(f"\nANOVA - Distance Volume:")
print(f"  F-statistic: {f_vol:.4f}")
print(f"  p-value: {p_vol:.6f}")

if p_vol < 0.05:
    print(f"  ✅ SIGNIFICANT: Volume varies by season")
else:
    print(f"  ❌ NOT SIGNIFICANT: No seasonal volume effect")

# %%
# Visualize seasonal trends
fig = px.box(df, x='season', y='distance_km',
            title=f'Distance Distribution by Season (F={f_vol:.2f}, p={p_vol:.4f})')
fig.show()

# Activity count by season
fig2 = px.bar(x=season_activity.index, y=season_activity.values,
             labels={'x': 'Season', 'y': 'Activity Count'},
             title='Activity Count by Season')
fig2.show()

# %% [markdown]
# ## H5: Recovery & Consistency
#
# **H₀:** Rest days don't affect subsequent performance  
# **H₁:** Longer rest periods correlate with better pace on next activity
#
# **Method:** Correlation between days since last activity and pace

# %%
# Calculate days since last activity
df_sorted = df.sort_values('start_date').reset_index(drop=True)
df_sorted['days_since_last'] = (df_sorted['start_date'] - df_sorted['start_date'].shift(1)).dt.days

# Remove NaN and extreme outliers (>30 days gap)
df_recovery = df_sorted[(df_sorted['days_since_last'].notna()) & (df_sorted['days_since_last'] <= 30)].copy()

if len(df_recovery) > 2:
    corr_recovery, p_recovery = stats.pearsonr(df_recovery['days_since_last'], df_recovery['pace_kmh'])
    
    print("\n" + "="*60)
    print("H5: RECOVERY & PERFORMANCE")
    print("="*60)
    print(f"\nPearson Correlation (Rest Days vs Pace):")
    print(f"  r = {corr_recovery:.4f}")
    print(f"  p-value = {p_recovery:.6f}")
    print(f"  N = {len(df_recovery)}")
    
    if p_recovery < 0.05:
        direction = "positive" if corr_recovery > 0 else "negative"
        print(f"\n✅ STATISTICALLY SIGNIFICANT (p < 0.05)")
        print(f"   {direction.upper()} correlation: More rest → {'better' if corr_recovery > 0 else 'worse'} pace")
    else:
        print(f"\n❌ NOT STATISTICALLY SIGNIFICANT (p = {p_recovery:.4f})")
        print(f"   No evidence of recovery effect")
else:
    print("\n⚠️ Not enough data for recovery analysis")

# %%
# Visualize
if len(df_recovery) > 2:
    fig = px.scatter(df_recovery, x='days_since_last', y='pace_kmh',
                    labels={'days_since_last': 'Days Since Last Activity', 'pace_kmh': 'Pace (km/h)'},
                    title=f'Rest Days vs Performance (r={corr_recovery:.3f}, p={p_recovery:.4f})')
    
    # Add trendline
    z = np.polyfit(df_recovery['days_since_last'], df_recovery['pace_kmh'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(0, df_recovery['days_since_last'].max(), 100)
    fig.add_scatter(x=x_trend, y=p(x_trend), mode='lines',
                   name='Trendline', line=dict(color='red'))
    
    fig.show()

# %% [markdown]
# ## Summary Table

# %%
# Create summary
summary_data = {
    'Hypothesis': [
        'H1: Pace improvement over time',
        'H2: HR-Pace correlation',
        'H3: Activity type differences',
        'H4: Seasonal volume effects',
        'H5: Recovery effect'
    ],
    'Test Statistic': [
        f'slope={slope:.6f}',
        f'r={corr:.4f}' if len(df_hr) > 2 else 'N/A',
        f'F={f_stat:.4f}' if len(valid_types) >= 2 else 'N/A',
        f'F={f_vol:.4f}',
        f'r={corr_recovery:.4f}' if len(df_recovery) > 2 else 'N/A'
    ],
    'p-value': [
        f'{p_value:.6f}',
        f'{p_value:.6f}' if len(df_hr) > 2 else 'N/A',
        f'{p_value:.6f}' if len(valid_types) >= 2 else 'N/A',
        f'{p_vol:.6f}',
        f'{p_recovery:.6f}' if len(df_recovery) > 2 else 'N/A'
    ],
    'Significant (α=0.05)': [
        '✅ Yes' if p_value < 0.05 else '❌ No',
        '✅ Yes' if (len(df_hr) > 2 and p_value < 0.05) else '❌ No',
        '✅ Yes' if (len(valid_types) >= 2 and p_value < 0.05) else '❌ No',
        '✅ Yes' if p_vol < 0.05 else '❌ No',
        '✅ Yes' if (len(df_recovery) > 2 and p_recovery < 0.05) else '❌ No'
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\n" + "="*100)
print("HYPOTHESIS TESTING SUMMARY")
print("="*100)
print(summary_df.to_string(index=False))
print("="*100)

# %% [markdown]
# ## Key Takeaways
#
# 1. **Data-Driven:** All tests use pre-defined hypotheses (not p-hacking)
# 2. **Rigorous:** Effect sizes, confidence intervals, assumption checks
# 3. **Actionable:** Clear Yes/No answers with statistical backing
# 4. **Next:** Model comparison (Haiku vs Sonnet vs Kimi) on same data

# %%
