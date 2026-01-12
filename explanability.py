# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 16:00:07 2025

@author: asgha
"""

# Install required packages if not already installed:
# pip install pm4py pygam lightgbm scikit-learn matplotlib seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pm4py
from pygam import LogisticGAM, s
from sklearn.inspection import PartialDependenceDisplay

# -----------------------------
# 1. Load Sepsis Event Log
# -----------------------------
log = pm4py.read_xes("Sepsis Cases - Event Log.xes")
df_log = pm4py.convert_to_dataframe(log)

print("Columns available:", df_log.columns.tolist())
print(df_log.head())

# -----------------------------
# 2. Feature Engineering
# -----------------------------
records = []

for case_id, group in df_log.groupby("case:concept:name"):
    group = group.sort_values("time:timestamp")
    
    # Antibiotic delay
    triage_time = group[group["concept:name"] == "ER Sepsis Triage"]["time:timestamp"].min()
    abx_time = group[group["concept:name"] == "IV Antibiotics"]["time:timestamp"].min()
    if pd.notnull(triage_time) and pd.notnull(abx_time):
        antibiotic_delay = (abx_time - triage_time).total_seconds() / 3600.0
    else:
        antibiotic_delay = None
    
    # Lactate level (numeric from LacticAcid column)
    lactate_event = group[group["concept:name"] == "LacticAcid"]
    lactate_level = None
    if not lactate_event.empty and "LacticAcid" in lactate_event.columns:
        lactate_level = pd.to_numeric(lactate_event["LacticAcid"], errors="coerce").max()
        # you can swap .max() for .mean() or .iloc[0] depending on study design
    
    # ICU admission outcome
    icu_flag = 0
    if "Admission NC" in group["concept:name"].values:
        icu_flag = 1
    
    records.append({
        "case_id": case_id,
        "antibiotic_delay": antibiotic_delay,
        "lactate_level": lactate_level,
        "icu_admission": icu_flag
    })

# -----------------------------
# 3. Build Final DataFrame
# -----------------------------
df = pd.DataFrame(records)
print(df.head())

# -----------------------------
# 4. Prepare data for modeling
# -----------------------------
X = df[["antibiotic_delay", "lactate_level"]].values
y = df["icu_admission"].values

# Drop rows with missing values
mask = ~pd.isnull(X).any(axis=1)
X = X[mask]
y = y[mask]

# -----------------------------
# 5. Fit GAM
# -----------------------------
gam = LogisticGAM(s(0) + s(1)).fit(X, y)

'''

# Plot GAM smooth curves (probability scale)
fig, axs = plt.subplots(1, figsize=(6, 5))
titles = ["Antibiotic Delay (hours)", "Lactate Level (mmol/L)"]

for i, ax in enumerate(axs):
    XX = gam.generate_X_grid(term=i)
    probs = gam.predict_mu(XX)  # predicted probability
    ax.plot(XX[:, i], probs)
    ax.plot(XX[:, i], gam.confidence_intervals(XX)[:,0], c='r', ls='--')
    ax.plot(XX[:, i], gam.confidence_intervals(XX)[:,1], c='r', ls='--')
    ax.set_title(f"GAM Effect: {titles[i]}")
    ax.set_xlabel(titles[i])
    ax.set_ylabel("Probability of ICU Admission")

plt.tight_layout()
plt.show()
'''

# Plot GAM smooth curve for Antibiotic Delay (probability scale)
fig, ax = plt.subplots(figsize=(6, 5))

# Generate grid for the antibiotic delay term (index 0)
XX = gam.generate_X_grid(term=0)
probs = gam.predict_mu(XX)  # predicted probability

# Plot the smooth curve and confidence intervals
ax.plot(XX[:, 0], probs)
ax.plot(XX[:, 0], gam.confidence_intervals(XX)[:, 0], c='r', ls='--')
ax.plot(XX[:, 0], gam.confidence_intervals(XX)[:, 1], c='r', ls='--')

# Labels and title
ax.set_title("GAM Effect: Antibiotic Delay (hours)")
ax.set_xlabel("Antibiotic Delay (hours)")
ax.set_ylabel("Probability of ICU Admission")

plt.tight_layout()
plt.show()
# -----------------------------
# 6. Monotonic Gradient Boosting (LightGBM)
# -----------------------------
import lightgbm as lgb

train_data = lgb.Dataset(X, label=y)
params = {
    "objective": "binary",
    "monotone_constraints": "(1,1)",  # enforce monotonic increase for both predictors
    "learning_rate": 0.05,
    "num_leaves": 31,
    "metric": "binary_logloss"
}
gbm = lgb.train(params, train_data, num_boost_round=200)

# -----------------------------
# 7. Partial Dependence Plots (using scikit-learn wrapper)
# -----------------------------
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# For PDP/ICE, use scikit-learn GBM (no monotonic constraints here)
gbm_sklearn = GradientBoostingClassifier(max_depth=3, random_state=42)
gbm_sklearn.fit(X, y)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
PartialDependenceDisplay.from_estimator(gbm_sklearn, X, [0], ax=axs[0])
PartialDependenceDisplay.from_estimator(gbm_sklearn, X, [1], ax=axs[1])
axs[0].set_title("PDP: Antibiotic Delay")
axs[1].set_title("PDP: Lactate Level")
plt.tight_layout()
plt.show()



# -----------------------------
# 8. ICE Plot for Antibiotic Delay
# -----------------------------
fig, ax = plt.subplots(figsize=(6,5))
PartialDependenceDisplay.from_estimator(
    gbm_sklearn, X, [0], kind="individual", ax=ax, subsample=50, random_state=42
)
ax.set_title("ICE Plot: Antibiotic Delay")
plt.show()