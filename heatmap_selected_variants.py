# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 12:51:21 2026

@author: asgha
"""

import pm4py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments_algorithm

# ============================================================
# 1. Load raw XES log
# ============================================================
raw_log_path = "Sepsis Cases - Event Log.xes"
event_log_raw = pm4py.read_xes(raw_log_path)
df_raw = pm4py.convert_to_dataframe(event_log_raw)

# ============================================================
# 2. Clean and keep only valid columns
# ============================================================
df_clean = df_raw[["case:concept:name", "concept:name", "time:timestamp"]].copy()
df_clean["case:concept:name"] = df_clean["case:concept:name"].astype(str)
df_clean["concept:name"] = df_clean["concept:name"].astype(str)
df_clean = df_clean.dropna(subset=["time:timestamp"])

df_clean["concept:name"] = df_clean["concept:name"].str.replace(
    r"[\x00-\x1F\x7F]", "", regex=True
)

# Remove consecutive duplicates
df_clean["prev"] = df_clean.groupby("case:concept:name")["concept:name"].shift()
df_clean = df_clean[df_clean["concept:name"] != df_clean["prev"]].copy()
df_clean = df_clean.drop(columns=["prev"])

# ============================================================
# 3. Convert to EventLog
# ============================================================
event_log_clean = pm4py.convert_to_event_log(df_clean)

# ============================================================
# 4. Build guideline Petri net
# ============================================================
net = PetriNet("Sepsis Guideline")

p_start = PetriNet.Place("start")
p_triage = PetriNet.Place("after_triage")
p_labs = PetriNet.Place("after_labs")
p_fluids = PetriNet.Place("after_fluids")
p_antibiotics = PetriNet.Place("after_antibiotics")
p_disposition = PetriNet.Place("end")

net.places.update([p_start, p_triage, p_labs, p_fluids, p_antibiotics, p_disposition])

t_triage = PetriNet.Transition("triage", "Triage")
t_crp = PetriNet.Transition("crp", "CRP")
t_lactate = PetriNet.Transition("lactate", "Lactate")
t_leucocytes = PetriNet.Transition("leucocytes", "Leucocytes")
t_fluids = PetriNet.Transition("fluids", "IV Fluids")
t_antibiotics = PetriNet.Transition("antibiotics", "IV Antibiotics")
t_disposition = PetriNet.Transition("disposition", "Disposition")

net.transitions.update([
    t_triage, t_crp, t_lactate, t_leucocytes,
    t_fluids, t_antibiotics, t_disposition
])

petri_utils.add_arc_from_to(p_start, t_triage, net)
petri_utils.add_arc_from_to(t_triage, p_triage, net)

for lab_t in [t_crp, t_lactate, t_leucocytes]:
    petri_utils.add_arc_from_to(p_triage, lab_t, net)
    petri_utils.add_arc_from_to(lab_t, p_labs, net)

petri_utils.add_arc_from_to(p_labs, t_fluids, net)
petri_utils.add_arc_from_to(t_fluids, p_fluids, net)

petri_utils.add_arc_from_to(p_fluids, t_antibiotics, net)
petri_utils.add_arc_from_to(t_antibiotics, p_antibiotics, net)

petri_utils.add_arc_from_to(p_antibiotics, t_disposition, net)
petri_utils.add_arc_from_to(t_disposition, p_disposition, net)

initial_marking = Marking({p_start: 1})
final_marking = Marking({p_disposition: 1})

# ============================================================
# 5. Align traces
# ============================================================
aligned_log = alignments_algorithm.apply_log(
    event_log_clean, net, initial_marking, final_marking
)

# ============================================================
# 6. Violation detection
# ============================================================
def detect_violations(alignment):
    moves = alignment["alignment"]
    v = {"missing_antibiotics": 0, "missing_lactate": 0, "lab_order_swaps": 0}

    if not any(m[1] == "IV Antibiotics" and m[0] != ">>" for m in moves):
        v["missing_antibiotics"] = 1

    if not any(m[1] == "Lactate" and m[0] != ">>" for m in moves):
        v["missing_lactate"] = 1

    labs = [m[1] for m in moves if m[1] in ["CRP", "Lactate", "Leucocytes"] and m[0] != ">>"]
    if "CRP" in labs and "Lactate" in labs and labs.index("CRP") < labs.index("Lactate"):
        v["lab_order_swaps"] = 1

    return v

df_viol = pd.DataFrame([
    {"case_id": trace.attributes["concept:name"], **detect_violations(aligned_log[i])}
    for i, trace in enumerate(event_log_clean)
])

# ============================================================
# 7. Extract and canonicalize variants (STRING VERSION)
# ============================================================
variants_raw = pm4py.statistics.variants.log.get.get_variants(event_log_clean)

variants_dict = {}
for variant, traces in variants_raw.items():
    canonical = " → ".join(str(a) for a in variant)
    if canonical not in variants_dict:
        variants_dict[canonical] = list(traces)
    else:
        variants_dict[canonical].extend(traces)

# ============================================================
# 8. Aggregate violations per variant
# ============================================================
variant_rows = []
for variant, traces in variants_dict.items():
    case_ids = [t.attributes["concept:name"] for t in traces]
    subset = df_viol[df_viol["case_id"].isin(case_ids)]
    variant_rows.append({
        "variant": variant,
        "missing_antibiotics": subset["missing_antibiotics"].sum(),
        "missing_lactate": subset["missing_lactate"].sum(),
        "lab_order_swaps": subset["lab_order_swaps"].sum()
    })

df_variant = pd.DataFrame(variant_rows)

# ============================================================
# 9. Select top-k variants
# ============================================================
k = 10
variant_counts = {variant: len(traces) for variant, traces in variants_dict.items()}
top_k_variants = sorted(variant_counts, key=variant_counts.get, reverse=True)[:k]

df_variant_topk = df_variant[df_variant["variant"].isin(top_k_variants)].copy()
df_variant_topk["case_count"] = df_variant_topk["variant"].map(variant_counts)
df_variant_topk = df_variant_topk.sort_values("case_count", ascending=False)

# ============================================================
# 10. Generate short labels (V1, V2, …)
# ============================================================
df_variant_topk["short_label"] = [f"V{i+1}" for i in range(len(df_variant_topk))]

# Legend table for appendix
legend_table = df_variant_topk[["short_label", "variant"]].copy()
print("\n=== Variant Legend Table ===")
print(legend_table.to_string(index=False))

# Replace index with short labels
df_variant_topk = df_variant_topk.set_index("short_label")

# ============================================================
# 11. Build heatmap with rotated labels
# ============================================================
heatmap_df = df_variant_topk[[
    "lab_order_swaps",
    "missing_antibiotics",
    "missing_lactate"
]]

plt.figure(figsize=(16, 8))
sns.heatmap(heatmap_df, annot=True, cmap="Reds", fmt="d")
plt.xticks(rotation=45, ha="right")
plt.title(f"Violation Types Across Top {k} Variants")
plt.xlabel("Violation Type")
plt.ylabel("Variant (Short Labels)")
plt.tight_layout()
plt.show()