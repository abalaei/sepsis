# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 17:18:52 2025

@author: asgha
"""
import pm4py
from pygam import LogisticGAM, s
from sklearn.inspection import PartialDependenceDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Baseline predictions
baseline_probs = gam.predict_mu(X)

# Counterfactual: antibiotics capped at 1 hour
X_cf = X.copy()
X_cf[:,0] = np.minimum(X_cf[:,0], 1)  # antibiotic_delay capped at 1h

# Counterfactual predictions
cf_probs = gam.predict_mu(X_cf)

print("Baseline mean ICU risk:", baseline_probs.mean())
print("Counterfactual mean ICU risk:", cf_probs.mean())

# Plot comparison
plt.figure(figsize=(6,5))
sns.kdeplot(baseline_probs, label="Baseline", color="blue")
sns.kdeplot(cf_probs, label="Counterfactual (≤1h antibiotics)", color="green")
plt.title("Predicted ICU Admission Risk Distribution")
plt.xlabel("Probability of ICU Admission")
plt.legend()
plt.show()
"""----------------------
#For combined intervention
Create a new counterfactual dataset where:
- Antibiotic delay is capped at 1 hour.
- Lactate level is imputed for missing values (e.g., median or domain-informed value like 2.0 mmol/L)

"""

X_cf_combined = X.copy()
X_cf_combined[:,0] = np.minimum(X_cf_combined[:,0], 1)  # cap antibiotic delay
X_cf_combined[:,1] = np.where(np.isnan(X_cf_combined[:,1]), 2.0, X_cf_combined[:,1])  # impute lactate

# fitted GAM to predict probabilities

cf_combined_probs = gam.predict_mu(X_cf_combined)

plt.figure(figsize=(6,5))
sns.kdeplot(baseline_probs, label="Baseline", color="blue")
sns.kdeplot(cf_probs, label="Antibiotics ≤1h", color="green")
sns.kdeplot(cf_combined_probs, label="Combined Intervention", color="purple")
plt.title("Predicted ICU Admission Risk Distribution")
plt.xlabel("Probability of ICU Admission")
plt.legend()
plt.show()
# Total expected ICU admissions
baseline_total = baseline_probs.sum()
cf_combined_total = cf_combined_probs.sum()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Baseline predictions from your fitted GAM
baseline_probs = gam.predict_mu(X)


# Counterfactual: cap antibiotic delay at 1 hour
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

# Baseline predictions
baseline_probs = gam.predict_mu(X)


X_cf = X.copy()
X_cf[:,0] = np.minimum(X_cf[:,0], 1)  # antibiotic_delay capped at 1h
cf_probs = gam.predict_mu(X_cf)

# Calculate totals
baseline_total = baseline_probs.sum()
cf_total = cf_probs.sum()
icu_avoided = baseline_total - cf_total
percent_reduction = (icu_avoided / baseline_total) * 100

print(f"Baseline expected ICU admissions: {baseline_total:.2f}")
print(f"Counterfactual expected ICU admissions (≤1h antibiotics): {cf_total:.2f}")
print(f"Estimated ICU admissions avoided: {icu_avoided:.2f}")
print(f"Percent reduction: {percent_reduction:.2f}%")

# Visualization
plt.figure(figsize=(6,5))
sns.barplot(x=["Baseline", "≤1h Antibiotics"], 
            y=[baseline_total, cf_total], 
            palette=["blue", "green"])

plt.title("Expected ICU Admissions: Baseline vs ≤1h Antibiotics")
plt.ylabel("Expected ICU Admissions")

# Annotate avoided admissions
plt.text(0.5, max(baseline_total, cf_total), 
         f"Avoided: {icu_avoided:.2f} ({percent_reduction:.2f}%)", 
         ha="center", va="bottom", fontsize=12, color="black")

plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Example bin edges and labels
bins = [0, 2, 4, 6, 12, np.inf]
bin_labels = ["0-2h", "2-4h", "4-6h", "6-12h", ">12h"]

baseline_risk = []
counterfactual_risk = []

# Make sure antibiotic_delay is a NumPy array
antibiotic_delay = np.asarray(X[:, 0])  # first column is delay

for i in range(len(bins)-1):
    mask = (antibiotic_delay >= bins[i]) & (antibiotic_delay < bins[i+1])
    if np.any(mask):
        baseline_risk.append(np.mean(baseline_probs[mask]))
        counterfactual_risk.append(np.mean(cf_probs[mask]))
    else:
        baseline_risk.append(np.nan)
        counterfactual_risk.append(np.nan)

print("Bins:", bin_labels)
print("Baseline risk:", baseline_risk)
print("Counterfactual risk:", counterfactual_risk)

x = np.arange(len(bin_labels))  # length = 5
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))

bars1 = ax.bar(x - width/2, baseline_risk, width, label="Baseline", color="steelblue")
bars2 = ax.bar(x + width/2, counterfactual_risk, width, label="Counterfactual (≤1h antibiotics)", color="seagreen")

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

ax.set_ylabel("ICU Risk (Probability)")
ax.set_xlabel("Antibiotic Delay Bins")
ax.set_title("Per-bin ICU Risk: Baseline vs Counterfactual (≤1h antibiotics)")
ax.set_xticks(x)
ax.set_xticklabels(bin_labels)  # ✅ use labels, not bin edges
ax.legend()

plt.tight_layout()
plt.show()

import numpy as np

# Bin edges
bins = [0, 2, 4, 6, 12, np.inf]
bin_labels = ["0-2h", "2-4h", "4-6h", "6-12h", ">12h"]

# Antibiotic delays
antibiotic_delay = np.asarray(X[:, 0])

# Count patients per bin
patient_counts, _ = np.histogram(antibiotic_delay, bins=bins)

print("Bins:", bin_labels)
print("Patient counts:", patient_counts)

import numpy as np
import matplotlib.pyplot as plt

# Bin edges and labels
bins = [0, 2, 4, 6, 12, np.inf]
bin_labels = ["0-2h", "2-4h", "4-6h", "6-12h", ">12h"]

# Antibiotic delay and predictions
antibiotic_delay = np.asarray(X[:, 0])
baseline_probs = gam.predict_mu(X)

# Targeted counterfactual: only cap delays >6h
X_cf_targeted = X.copy()
X_cf_targeted[X_cf_targeted[:, 0] > 6, 0] = 1
cf_probs_targeted = gam.predict_mu(X_cf_targeted)

# Initialize arrays
baseline_risk = []
targeted_cf_risk = []
patient_counts = []

# Aggregate per bin
for i in range(len(bins)-1):
    mask = (antibiotic_delay >= bins[i]) & (antibiotic_delay < bins[i+1])
    patient_counts.append(np.sum(mask))
    if np.any(mask):
        baseline_risk.append(np.mean(baseline_probs[mask]))
        targeted_cf_risk.append(np.mean(cf_probs_targeted[mask]))
    else:
        baseline_risk.append(np.nan)
        targeted_cf_risk.append(np.nan)

# Convert to arrays
baseline_risk = np.array(baseline_risk)
targeted_cf_risk = np.array(targeted_cf_risk)
patient_counts = np.array(patient_counts)

# Compute admissions avoided and percent reduction
baseline_admissions = baseline_risk * patient_counts
cf_admissions = targeted_cf_risk * patient_counts
admissions_avoided = baseline_admissions - cf_admissions
percent_reduction = 100 * admissions_avoided / baseline_admissions

# Plot
x = np.arange(len(bin_labels))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))

bars1 = ax1.bar(x - width/2, admissions_avoided, width, label="Admissions Avoided", color="indianred")
ax1.set_ylabel("Admissions Avoided")
ax1.set_xlabel("Antibiotic Delay Bins")
ax1.set_title("ICU Admissions Avoided and Percent Reduction per Delay Bin (Targeted Intervention >6h)")
ax1.set_xticks(x)
ax1.set_xticklabels(bin_labels)

# Secondary axis for percent reduction
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, percent_reduction, width, label="Percent Reduction (%)", color="seagreen")
ax2.set_ylabel("Percent Reduction (%)")

# Add labels
for i, bar in enumerate(bars1):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
             f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

for i, bar in enumerate(bars2):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
             f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Bin edges and labels
bins = [0, 2, 4, 6, 12, np.inf]
bin_labels = ["0-2h", "2-4h", "4-6h", "6-12h", ">12h"]

# Antibiotic delay and predictions
antibiotic_delay = np.asarray(X[:, 0])
baseline_probs = gam.predict_mu(X)

# Blanket counterfactual: cap ALL delays to 1h
X_cf_blanket = X.copy()
X_cf_blanket[:, 0] = 1
cf_probs_blanket = gam.predict_mu(X_cf_blanket)

# Initialize arrays
baseline_admissions = []
cf_admissions = []
admissions_avoided = []
patient_counts = []

# Compute per-bin admissions
for i in range(len(bins)-1):
    mask = (antibiotic_delay >= bins[i]) & (antibiotic_delay < bins[i+1])
    patient_counts.append(np.sum(mask))
    if np.any(mask):
        baseline_adm = np.sum(baseline_probs[mask])
        cf_adm = np.sum(cf_probs_blanket[mask])
        baseline_admissions.append(baseline_adm)
        cf_admissions.append(cf_adm)
        admissions_avoided.append(baseline_adm - cf_adm)
    else:
        baseline_admissions.append(np.nan)
        cf_admissions.append(np.nan)
        admissions_avoided.append(np.nan)

# Plot admissions avoided per bin
x = np.arange(len(bin_labels))
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(x, admissions_avoided, color="indianred")

# Add labels above bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.2f}', ha='center', va='bottom', fontsize=9)

ax.set_ylabel("Admissions Avoided")
ax.set_xlabel("Antibiotic Delay Bins")
ax.set_title("ICU Admissions Avoided per Delay Bin (Blanket Intervention ≤1h)")
ax.set_xticks(x)
ax.set_xticklabels(bin_labels)

plt.tight_layout()
plt.show()

baseline_probs = gam.predict_mu(X)
# Blanket counterfactual: set ALL delays to 1h
X_cf_blanket = X.copy()
X_cf_blanket[:, 0] = 1
cf_probs_blanket = gam.predict_mu(X_cf_blanket)

import numpy as np

bins = [0, 2, 4, 6, 12, np.inf]
bin_labels = ["0-2h", "2-4h", "4-6h", "6-12h", ">12h"]

baseline_risk = []
blanket_cf_risk = []

antibiotic_delay = np.asarray(X[:, 0])

for i in range(len(bins)-1):
    mask = (antibiotic_delay >= bins[i]) & (antibiotic_delay < bins[i+1])
    if np.any(mask):
        baseline_risk.append(np.mean(baseline_probs[mask]))
        blanket_cf_risk.append(np.mean(cf_probs_blanket[mask]))
    else:
        baseline_risk.append(np.nan)
        blanket_cf_risk.append(np.nan)

baseline_risk = np.array(baseline_risk)
blanket_cf_risk = np.array(blanket_cf_risk)

print("Bins:", bin_labels)
print("Baseline risk:", baseline_risk)
print("Blanket counterfactual risk:", blanket_cf_risk)
# Compute admissions per bin
baseline_admissions = baseline_risk * patient_counts
blanket_cf_admissions = blanket_cf_risk * patient_counts
targeted_cf_admissions = targeted_cf_risk * patient_counts

# Admissions avoided
blanket_avoided = baseline_admissions - blanket_cf_admissions
targeted_avoided = baseline_admissions - targeted_cf_admissions

# Plot
x = np.arange(len(bin_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width/2, blanket_avoided, width,
               label="Blanket Intervention", color="indianred")
bars2 = ax.bar(x + width/2, targeted_avoided, width,
               label="Targeted Intervention (>6h)", color="seagreen")

# Labels and title
ax.set_ylabel("Admissions Avoided")
ax.set_xlabel("Antibiotic Delay Bins")
ax.set_title("ICU Admissions Avoided per Bin: Blanket vs Targeted Intervention")
ax.set_xticks(x)
ax.set_xticklabels(bin_labels)
ax.legend()

# Add labels above bars
for i in range(len(bin_labels)):
    ax.text(x[i] - width/2, blanket_avoided[i] + 0.5,
            f'{blanket_avoided[i]:.2f}', ha='center', va='bottom', fontsize=9)
    ax.text(x[i] + width/2, targeted_avoided[i] + 0.5,
            f'{targeted_avoided[i]:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

'''
simulate this refined intervention by 
creating a new counterfactual dataset 
where >12h patients are capped separately 
and not reassigned into 6–12h
'''
# Baseline predictions
baseline_probs = gam.predict_mu(X)

# Identify >12h patients
antibiotic_delay = np.asarray(X[:, 0])
mask_gt12 = antibiotic_delay > 12

# Create counterfactual dataset for >12h patients only

X_cf_gt12 = X[mask_gt12].copy()
X_cf_gt12[:, 0] = 1   # cap delay to 1h
cf_probs_gt12 = gam.predict_mu(X_cf_gt12)

# Aggregate admissions avoided per bin

bins = ["0-2h", "2-4h", "4-6h", "6-12h", ">12h"]
admissions_avoided = [0, 0, 0, 0,
                      np.sum(baseline_probs[mask_gt12]) - np.sum(cf_probs_gt12)]
total_avoided = np.sum(admissions_avoided)

print("Admissions avoided per bin:", dict(zip(bins, admissions_avoided)))
print("Total admissions avoided:", total_avoided)

import numpy as np
import matplotlib.pyplot as plt

# Strategies and their total avoided admissions
strategies = ["Baseline", "Blanket", "Targeted", "Refined"]
avoided = np.array([0.0, -8.03, -1.78, 0.66])

# Colors for each strategy
colors = ["gray", "indianred", "seagreen", "royalblue"]

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(strategies, avoided, color=colors)

# Labels and title
ax.set_ylabel("Total ICU Admissions Avoided")
ax.set_title("Comparison of Intervention Strategies")
ax.axhline(0, color="black", linewidth=0.8)  # horizontal line at 0

# Add value labels above bars
for bar, value in zip(bars, avoided):
    ax.text(bar.get_x() + bar.get_width()/2, value + (0.3 if value >= 0 else -0.6),
            f"{value:.2f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.show()