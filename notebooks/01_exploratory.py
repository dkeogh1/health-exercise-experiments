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
# # Strava Data Exploratory Analysis
#
# Initial data review and descriptive statistics.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from pathlib import Path

# Set up plotting
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
# %matplotlib inline

print('✅ Libraries imported')

# %% [markdown]
# ## 1. Load Data

# %%
# Find the latest exported file
data_dir = Path('../data/raw')
json_files = sorted(data_dir.glob('activities_*.json'), reverse=True)

if not json_files:
    print("❌ No activity files found. Run strava_export.py first.")
else:
    latest_file = json_files[0]
    print(f"📂 Loading: {latest_file.name}")
    
    with open(latest_file) as f:
        activities_raw = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(activities_raw)
    
    # Convert date columns
    df['start_date'] = pd.to_datetime(df['start_date'])
    
    print(f"\n✅ Loaded {len(df)} activities")
    print(f"Date range: {df['start_date'].min().date()} to {df['start_date'].max().date()}")
    print(f"\nColumns: {df.shape[1]}")
    print(df.columns.tolist())

# %% [markdown]
# ## 2. Data Overview

# %%
print("\n📊 First few rows:")
print(df.head())

# %%
print("\n📈 Data types & missing values:")
print(df.info())

# %%
print("\n🔢 Descriptive statistics:")
print(df.describe())

# %% [markdown]
# ## 3. Activity Type Distribution

# %%
# Activity types
activity_counts = df['type'].value_counts()
print("\n🏃 Activity Types:")
print(activity_counts)

fig = px.pie(names=activity_counts.index, values=activity_counts.values, 
             title='Activity Distribution')
fig.show()

# %% [markdown]
# ## 4. Volume Trends Over Time

# %%
# Add time features
df['year'] = df['start_date'].dt.year
df['month'] = df['start_date'].dt.month
df['week'] = df['start_date'].dt.isocalendar().week
df['day_of_week'] = df['start_date'].dt.day_name()
df['date_only'] = df['start_date'].dt.date

# Monthly activity count
df['year_month'] = df['start_date'].dt.to_period('M')
monthly_activity = df.groupby('year_month').size()

fig = px.line(x=monthly_activity.index.astype(str), y=monthly_activity.values,
              labels={'x': 'Month', 'y': 'Activity Count'},
              title='Monthly Activity Count Over Time')
fig.show()

print(f"\nMonthly activity stats:")
print(f"Mean: {monthly_activity.mean():.1f}")
print(f"Median: {monthly_activity.median():.1f}")
print(f"Max: {monthly_activity.max()}")
print(f"Min: {monthly_activity.min()}")

# %% [markdown]
# ## 5. Distance & Duration Analysis

# %%
# Convert to more readable units
df['distance_km'] = df['distance_m'] / 1000
df['duration_hours'] = df['duration_s'] / 3600
df['pace_kmh'] = df['distance_km'] / df['duration_hours']

print("\n📏 Distance (km):")
print(df['distance_km'].describe())

print("\n⏱️ Duration (hours):")
print(df['duration_hours'].describe())

print("\n🏃 Pace (km/h):")
print(df['pace_kmh'].describe())

# %%
# Scatter: distance vs duration
fig = px.scatter(df, x='distance_km', y='duration_hours', color='type',
                 hover_data=['name', 'start_date'],
                 title='Distance vs Duration by Activity Type')
fig.show()

# %% [markdown]
# ## 6. Heart Rate Analysis

# %%
# Filter activities with HR data
df_with_hr = df[df['average_hr'].notna()]

print(f"\n❤️ Activities with HR data: {len(df_with_hr)} / {len(df)} ({len(df_with_hr)/len(df)*100:.1f}%)")
print("\nAverage Heart Rate:")
print(df_with_hr['average_hr'].describe())

print("\nMax Heart Rate:")
print(df_with_hr['max_hr'].describe())

# %%
# HR distribution by activity type
fig = px.box(df_with_hr, x='type', y='average_hr',
            title='Average Heart Rate by Activity Type')
fig.show()

# %% [markdown]
# ## 7. Elevation & Power Analysis

# %%
print("\n⛰️ Elevation Gain (m):")
print(df['elevation_m'].describe())

df_with_power = df[df['average_watts'].notna()]
print(f"\n⚡ Activities with power data: {len(df_with_power)} / {len(df)} ({len(df_with_power)/len(df)*100:.1f}%)")

if len(df_with_power) > 0:
    print("\nAverage Power (W):")
    print(df_with_power['average_watts'].describe())

# %% [markdown]
# ## 8. Consistency Analysis

# %%
# Activities per day of week
dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_counts = df['day_of_week'].value_counts().reindex(dow_order)

fig = px.bar(x=dow_counts.index, y=dow_counts.values,
            labels={'x': 'Day of Week', 'y': 'Activity Count'},
            title='Activities by Day of Week')
fig.show()

print("\n📅 Activity Distribution by Day:")
print(dow_counts)

# %%
# Weeks without activity (gaps)
df_sorted = df.sort_values('start_date')
df_sorted['date_only'] = df_sorted['start_date'].dt.date
unique_dates = df_sorted['date_only'].unique()

# Find gaps
gaps = []
for i in range(len(unique_dates)-1):
    gap = (unique_dates[i+1] - unique_dates[i]).days
    if gap > 7:  # Only gaps > 1 week
        gaps.append(gap)

if gaps:
    print(f"\n🔍 Long gaps (> 7 days):")
    print(f"Number of gaps: {len(gaps)}")
    print(f"Average gap: {np.mean(gaps):.0f} days")
    print(f"Max gap: {np.max(gaps)} days")
    print(f"Min gap: {np.min(gaps)} days")
else:
    print("\n✅ No long gaps detected")

# %% [markdown]
# ## 9. Personal Records (PRs)

# %%
print("\n🏆 Personal Records:")
print(f"Longest distance: {df['distance_km'].max():.2f} km")
print(f"Longest duration: {df['duration_hours'].max():.2f} hours")
print(f"Highest elevation: {df['elevation_m'].max():.0f} m")
if len(df_with_hr) > 0:
    print(f"Highest avg HR: {df_with_hr['average_hr'].max():.0f} bpm")
    print(f"Highest max HR: {df_with_hr['max_hr'].max():.0f} bpm")
if len(df_with_power) > 0:
    print(f"Highest avg power: {df_with_power['average_watts'].max():.0f} W")

# %% [markdown]
# ## 10. Heatmap: Activity Intensity by Month/Year

# %%
# Create a pivot table for heatmap
df['year_month_str'] = df['start_date'].dt.strftime('%Y-%m')
activity_heatmap = df.groupby(['year', 'month']).size().reset_index(name='count')

# Convert to pivot
pivot_data = activity_heatmap.pivot(index='year', columns='month', values='count')

fig = go.Figure(data=go.Heatmap(
    z=pivot_data.values,
    x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    y=pivot_data.index,
    colorscale='YlOrRd'
))
fig.update_layout(title='Activity Frequency Heatmap (Month/Year)', height=400)
fig.show()

# %% [markdown]
# ## 11. Summary

# %%
print("\n" + "="*60)
print("📊 DATA SUMMARY")
print("="*60)
print(f"Total activities: {len(df)}")
print(f"Date range: {df['start_date'].min().date()} to {df['start_date'].max().date()}")
print(f"Total distance: {df['distance_km'].sum():.0f} km")
print(f"Total duration: {df['duration_hours'].sum():.0f} hours")
print(f"Activity types: {', '.join(activity_counts.index.tolist())}")
print(f"\nHR data: {len(df_with_hr)} activities ({len(df_with_hr)/len(df)*100:.1f}%)")
print(f"Power data: {len(df_with_power)} activities ({len(df_with_power)/len(df)*100:.1f}%)")
print("="*60)

# %% [markdown]
# ## Next Steps
#
# 1. **02_hypothesis_testing.ipynb** — Test fitness improvement over time, correlations, etc.
# 2. **03_biometric_correlation.ipynb** — Merge with Apple Watch / Garmin data
# 3. **Model comparison** — Run same analyses across Haiku 4.5, Sonnet 4.6, Kimi 2.5

# %%
