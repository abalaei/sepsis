# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 08:22:09 2025

@author: asgha
"""

import pm4py
import pandas as pd
import numpy as np
from datetime import timedelta

# -----------------------------
# 1. Load and parse the XES log
# -----------------------------
log_path = "Sepsis Cases - Event Log.xes"
log = pm4py.read_xes(log_path)

# Convert to DataFrame for easier manipulation
df = pm4py.convert_to_dataframe(log)

# -----------------------------
# 2. Normalize timestamps
# -----------------------------
# Case-relative time: subtract start timestamp of each case
df['case_start'] = df.groupby('case:concept:name')['time:timestamp'].transform('min')
df['relative_time'] = (df['time:timestamp'] - df['case_start']).dt.total_seconds()

# -----------------------------
# 3. Map activity labels to categories
# -----------------------------
activity_map = {
    "ER Registration": "Triage",
    "ER Triage": "Triage",
    "Leucocytes": "Labs",
    "CRP": "Labs",
    "LacticAcid": "Labs",
    "IV Antibiotics": "Treatment",
    "IV Liquid": "Treatment",
    "Admission IC": "Disposition",
    "Admission NC": "Disposition",
    "Release A": "Disposition",
    "Release B": "Disposition"
}
df['clinical_category'] = df['concept:name'].map(activity_map).fillna("Other")

# -----------------------------
# 4. Clean traces
# -----------------------------
# Collapse duplicates (same activity repeated consecutively)
df['prev_activity'] = df.groupby('case:concept:name')['concept:name'].shift()
df['is_duplicate'] = df['concept:name'] == df['prev_activity']
df_clean = df[~df['is_duplicate']].copy()

# Mark rework loops (activity reappears later in same case)
df_clean['rework'] = df_clean.groupby('case:concept:name')['concept:name'].transform(
    lambda x: x.duplicated(keep=False)
)

# Handle missing values
df_clean.fillna({"clinical_category": "Unknown"}, inplace=True)

# -----------------------------
# 5. Case summary artifact
# -----------------------------
def summarize_case(case_df):
    case_id = case_df['case:concept:name'].iloc[0]
    start_time = case_df['case_start'].iloc[0]

    # Time-to-antibiotics
    antibiotics = case_df[case_df['concept:name'] == "IV Antibiotics"]
    tta = None
    if not antibiotics.empty:
        tta = (antibiotics['time:timestamp'].iloc[0] - start_time).total_seconds()

    # Labs performed
    labs = case_df[case_df['clinical_category'] == "Labs"]['concept:name'].unique().tolist()

    # Disposition
    disposition = case_df[case_df['clinical_category'] == "Disposition"]['concept:name'].tolist()

    return {
        "case_id": case_id,
        "time_to_antibiotics_sec": tta,
        "labs_performed": labs,
        "disposition": disposition
    }

case_summaries = df_clean.groupby('case:concept:name').apply(summarize_case)
case_summary_df = pd.DataFrame(case_summaries.tolist())

# -----------------------------
# 6. Persist artifacts
# -----------------------------
case_summary_df.to_csv("case_summaries.csv", index=False)
df_clean.to_csv("cleaned_traces.csv", index=False)

# Convert cleaned DataFrame back to EventLog
log_clean = pm4py.convert_to_event_log(df_clean)

# Export to XES
pm4py.write_xes(log_clean, "cleaned_traces.xes")


# Inductive Miner
inductive_net, im_initial_marking, im_final_marking = pm4py.discover_petri_net_inductive(log)

# Heuristic Miner
heu_net = pm4py.discover_heuristics_net(log)

# Persist models for reproducibility
pm4py.write_pnml(inductive_net, im_initial_marking, im_final_marking, "inductive_model_v1.pnml")

import pm4py
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer

# Discover heuristics net
heu_net = pm4py.discover_heuristics_net(log)

# Apply visualization
gviz = hn_visualizer.apply(heu_net)

# Save to file (PNG, SVG, DOT supported)
hn_visualizer.save(gviz, "heuristic_model_v1.png")
hn_visualizer.save(gviz, "heuristic_model_v1.dot")

from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils

# Build guideline Petri net
net = PetriNet("Sepsis Guideline")

# Places
p_start = PetriNet.Place("start")
p_triage = PetriNet.Place("after_triage")
p_labs = PetriNet.Place("after_labs")
p_fluids = PetriNet.Place("after_fluids")
p_antibiotics = PetriNet.Place("after_antibiotics")
p_disposition = PetriNet.Place("end")

net.places.update([p_start, p_triage, p_labs, p_fluids, p_antibiotics, p_disposition])

# Transitions
t_triage = PetriNet.Transition("triage", "Triage")
t_crp = PetriNet.Transition("crp", "CRP")
t_lactate = PetriNet.Transition("lactate", "Lactate")
t_leucocytes = PetriNet.Transition("leucocytes", "Leucocytes")
t_fluids = PetriNet.Transition("fluids", "IV Fluids")
t_antibiotics = PetriNet.Transition("antibiotics", "IV Antibiotics")
t_disposition = PetriNet.Transition("disposition", "Disposition")

net.transitions.update([t_triage, t_crp, t_lactate, t_leucocytes, t_fluids, t_antibiotics, t_disposition])

# Connect arcs
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

# Initial and final markings
initial_marking = Marking()
initial_marking[p_start] = 1
final_marking = Marking()
final_marking[p_disposition] = 1

# Persist guideline model
pm4py.write_pnml(net, initial_marking, final_marking, "guideline_model_v1.pnml")


# Load cleaned log and guideline model
log = pm4py.read_xes("cleaned_traces.xes")
net, initial_marking, final_marking = pm4py.read_pnml("guideline_model_v1.pnml")

from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments

aligned_traces = alignments.apply_log(log, net, initial_marking, final_marking)


# Fitness
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator

fitness = replay_fitness_evaluator.apply(log, net, initial_marking, final_marking)


# Precision (using token-based replay)
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator

# Compute precision
precision = precision_evaluator.apply(
    log, net, initial_marking, final_marking,
    variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN
)


# Custom violation counts
def violation_summary(alignment_result):
    violations = {"missing_antibiotics": 0, "missing_lactate": 0, "lab_order_swaps": 0}
    for a in alignment_result:
        moves = a['alignment']
        # Example: check if antibiotics transition never matched
        if not any(m[1] == "IV Antibiotics" and m[0] != ">>" for m in moves):
            violations["missing_antibiotics"] += 1
        # Example: check if lactate missing
        if not any(m[1] == "Lactate" and m[0] != ">>" for m in moves):
            violations["missing_lactate"] += 1
        # Example: lab order swap (CRP before lactate)
        labs = [m[1] for m in moves if m[1] in ["CRP", "Lactate", "Leucocytes"] and m[0] != ">>"]
        if "CRP" in labs and "Lactate" in labs and labs.index("CRP") < labs.index("Lactate"):
            violations["lab_order_swaps"] += 1
    return violations

violations = violation_summary(aligned_traces)

# Extract variants
variants = pm4py.statistics.variants.log.get.get_variants(log)

# Sort by frequency
variant_freq = sorted(variants.items(), key=lambda x: len(x[1]), reverse=True)

from pm4py.objects.log.importer.xes import importer as xes_importer

log = xes_importer.apply("cleaned_traces.xes")

from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments_algorithm

aligned_traces = alignments_algorithm.apply_log(log, net, initial_marking, final_marking)

# If you want case IDs attached:
for i, trace in enumerate(log):
    aligned_traces[i]["case_id"] = trace.attributes["concept:name"]    

import pandas as pd


from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments_algorithm

variant_summary = []
alignment_cache = {}

for variant, cases in variant_freq[:10]:  # top 10 variants
    # Check cache first
    if variant not in alignment_cache:
        # Align only one representative trace from this variant
        representative_trace = cases[0]
        case_id = representative_trace.attributes["concept:name"]
        alignment_result = alignments_algorithm.apply_log([representative_trace], net, initial_marking, final_marking)[0]
        alignment_result["case_id"] = case_id
        alignment_cache[variant] = alignment_result

    # Approximate violation check using cached alignment
    case_alignments = [alignment_cache[variant]]  # reuse alignment for all cases in variant
    v_violations = violation_summary(case_alignments)

    variant_summary.append({
        "variant": variant,
        "case_count": len(cases),
        "violations": v_violations
    })




# Flatten variant_summary into rows
rows = []
for v in variant_summary:
    for viol_type, count in v["violations"].items():
        rows.append({
            "variant": v["variant"],
            "case_count": v["case_count"],
            "violation_type": viol_type,
            "count": count
        })



df = pd.DataFrame(rows)

import seaborn as sns
import matplotlib.pyplot as plt

# Pivot for heatmap
heatmap_data = df.pivot_table(index="variant", columns="violation_type", values="count", fill_value=0)

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, cmap="Reds", fmt=".0f")  # changed fmt
plt.title("Violation Types Across Variants")
plt.ylabel("Variants")
plt.xlabel("Violation Types")
plt.tight_layout()
plt.show()
