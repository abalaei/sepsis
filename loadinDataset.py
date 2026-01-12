# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 11:00:12 2025

@author: asgha
"""

import pm4py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pygam import LogisticGAM, s

# -----------------------------
# 1. Load Sepsis Event Log
# -----------------------------
log = pm4py.read_xes("Sepsis Cases - Event Log.xes")
df_log = pm4py.convert_to_dataframe(log)

# -----------------------------
# 2. Feature Engineering
# -----------------------------
records = []
for case_id, group in df_log.groupby("case:concept:name"):
    group = group.sort_values("time:timestamp")

    # Antibiotic delay (hours from triage to antibiotics)
    triage_time = group[group["concept:name"] == "ER Sepsis Triage"]["time:timestamp"].min()
    abx_time = group[group["concept:name"] == "IV Antibiotics"]["time:timestamp"].min()
    if pd.notnull(triage_time) and pd.notnull(abx_time):
        antibiotic_delay = (abx_time - triage_time).total_seconds() / 3600.0
    else:
        antibiotic_delay = None

    # Lactate measurement flag
    lactate_flag = 1 if "LacticAcid" in group["concept:name"].values else 0

    # ICU admission outcome
    icu_flag = 1 if "Admission NC" in group["concept:name"].values else 0

    records.append({
        "case_id": case_id,
        "antibiotic_delay": antibiotic_delay,
        "lactate_flag": lactate_flag,
        "icu_admission": icu_flag
    })

df = pd.DataFrame(records)

# -----------------------------
# 3. Prepare data for modeling
# -----------------------------
X = df[["antibiotic_delay", "lactate_flag"]].values
y = df["icu_admission"].values

mask = ~pd.isnull(X).any(axis=1)
X = X[mask]
y = y[mask]

# -----------------------------
# 4. Fit GAM
# -----------------------------
gam = LogisticGAM(s(0) + s(1)).fit(X, y)

df_model = df.loc[mask].copy()
df_model['icu_risk_baseline'] = gam.predict_mu(X)

# Counterfactual: antibiotics capped at 1 hour
X_cf = X.copy()
X_cf[:,0] = np.minimum(X_cf[:,0], 1)
df_model['icu_risk_counterfactual'] = gam.predict_mu(X_cf)

# -----------------------------
# 5. Add violation flags
# -----------------------------
# Normally these come from your alignment pipeline; here we simulate for demo
np.random.seed(42)
df_model['missing_antibiotics'] = np.random.randint(0,2,len(df_model))
df_model['missing_lactate'] = np.random.randint(0,2,len(df_model))
df_model['lab_order_swaps'] = np.random.randint(0,2,len(df_model))

# -----------------------------
# 6. Persist to CSV
# -----------------------------
df_model.to_csv("your_data.csv", index=False)

# -----------------------------
# 7. Aggregate ICU risk by violation type
# -----------------------------
summary = {}
for col in ['missing_antibiotics','missing_lactate','lab_order_swaps']:
    avg_risk = df_model.loc[df_model[col]==1,'icu_risk_baseline'].mean()
    summary[col] = avg_risk

summary_df = pd.DataFrame(list(summary.items()), columns=['Violation Type','Average ICU Risk'])

# -----------------------------
# 8. Visualization
# -----------------------------
sns.barplot(data=summary_df, x='Violation Type', y='Average ICU Risk')
plt.title("Average ICU Risk per Violation Type")
plt.ylabel("Mean ICU Risk (Baseline)")
plt.tight_layout()
plt.savefig("icu_risk_per_violation.png")
plt.show()