# Data Science Experiment Template

Use this structure for rigorous hypothesis testing. Replace placeholders with your specific analysis.

---

## Experiment: [Catchy Title]

**Hypothesis:** [What are you testing?]

### 1. Research Question
**Null hypothesis (H₀):** [What would be true if the effect doesn't exist?]

*Example:* "Time of day has no effect on running pace."

**Alternative hypothesis (H₁):** [What would be true if the effect exists?]

*Example:* "Morning runs have statistically significantly different pace than evening runs."

---

### 2. Study Design

**Population:** [Who/what are you studying?]
- *Example:* All running activities from [Date] to [Date], n = [count]

**Independent variable(s):** [What you're comparing]
- *Example:* Time of day (categorical: morning/afternoon/evening)

**Dependent variable(s):** [What you're measuring]
- *Example:* Pace (numeric: min/km)

**Confounders to control:** [What else might affect the outcome?]
- *Example:* Distance, temperature, elevation, gear, fatigue (weekly volume)

**Sample size:** n = [count]. 
- *Justification:* Power analysis for [effect size, alpha, beta]

---

### 3. Methods

**Data preparation:**
```
1. Filter data: [criteria]
2. Handle missing values: [strategy]
3. Outlier detection: [method, thresholds]
4. Normalize/transform: [if needed]
```

**Statistical test:**
- **Type:** [t-test, ANOVA, correlation, regression, etc.]
- **Assumptions checked:**
  - Normality: [Shapiro-Wilk, visual inspection, etc.]
  - Homogeneity of variance: [Levene's test, etc.]
  - Independence: [sampling design ensures this]
- **Effect size metric:** [Cohen's d, η², r, etc.]
- **Significance level:** α = 0.05 (or justify differently)

---

### 4. Results

**Descriptive statistics:**

| Group | N | Mean | SD | Min | Max |
|-------|---|------|-----|-----|-----|
| Group A | | | | | |
| Group B | | | | | |

**Test results:**
- Test statistic: t = [value], df = [value]
- p-value: [value]
- Effect size: [Cohen's d = value, 95% CI: [lower, upper]]

**Interpretation:**
- ✅ **Reject H₀** if p < 0.05 and effect is meaningful
- ❌ **Fail to reject H₀** if p ≥ 0.05
- **Effect size:** [Small/medium/large] = [interpretation in your context]

---

### 5. Sensitivity Analysis

**What if we changed an assumption?**
- Robustness check 1: [method]
  - Result: [did conclusion hold?]
- Robustness check 2: [method]
  - Result: [did conclusion hold?]

---

### 6. Visualization

[Insert plots here]
- Distribution plots (for checking assumptions)
- Effect plots (group differences or trends)
- Residual plots (if applicable)

---

### 7. Conclusion

**Finding:** [Concise statement of result]

**Magnitude:** The effect size was [small/medium/large], meaning [practical significance].

**Limitations:** [What could affect this result?]
- Limited sample size during off-season?
- Confounding from seasonal temperature?
- Measurement error from GPS?

**Next steps:** [What should we investigate next?]

---

## Code Template (Python)

```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data/activities_enriched.csv')

# Filter & prepare
df_filtered = df[df['type'] == 'Run'].copy()
df_filtered = df_filtered.dropna(subset=['pace_avg', 'time_of_day'])

# Descriptive stats
print(df_filtered.groupby('time_of_day')['pace_avg'].describe())

# Test assumptions
groups = [df_filtered[df_filtered['time_of_day'] == t]['pace_avg'].values 
          for t in df_filtered['time_of_day'].unique()]
shapiro_results = [stats.shapiro(g) for g in groups]
print("Normality tests:", shapiro_results)

# Statistical test
t_stat, p_value = stats.ttest_ind(groups[0], groups[1])
cohens_d = (groups[0].mean() - groups[1].mean()) / np.sqrt(((len(groups[0])-1)*groups[0].std()**2 + (len(groups[1])-1)*groups[1].std()**2) / (len(groups[0]) + len(groups[1]) - 2))

print(f"t-test: t={t_stat:.3f}, p={p_value:.4f}")
print(f"Cohen's d: {cohens_d:.3f}")

# Visualization
sns.boxplot(data=df_filtered, x='time_of_day', y='pace_avg')
plt.title("Pace by Time of Day")
plt.show()
```

---

## Questions to Ask Before Running an Experiment

1. **Is this question actually interesting?** (Worth the analysis effort?)
2. **Do you have enough data?** (Sample size, temporal range, completeness?)
3. **Are confounders controlled?** (Or acknowledged as limitations?)
4. **What's the practical significance?** (Even if statistically significant, does it matter?)
5. **Would a reasonable person believe this result?** (Plausibility check)

---

*Use this template for every hypothesis you want to test. Reproducibility > cleverness.*

