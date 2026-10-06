"""
Dev45 — Final Experiment Packaging

FREEZES the current Dev39 method.

No parameter tuning.
No new optimization.
No modification of Dev10.

Purpose:
    Create a reproducible, paper-ready summary of the final experiment.

Sources:
    Dev40 — six development cases
    Dev43 — independent Case-7 validation
    Dev44 — cross-cohort evaluation
"""

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DEV40_FILE = (
    ROOT
    / "results"
    / "development_cases"
    / "dev40_robustness_audit"
    / "dev40_case_summary.csv"
)

DEV43_FILE = (
    ROOT
    / "results"
    / "development_cases"
    / "dev43_validation_case7"
    / "dev43_case_summary.csv"
)

DEV44_SUMMARY = (
    ROOT
    / "results"
    / "development_cases"
    / "dev44_cross_cohort_evaluation"
    / "dev44_cohort_summary.csv"
)

DEV44_IMPROVEMENT = (
    ROOT
    / "results"
    / "development_cases"
    / "dev44_cross_cohort_evaluation"
    / "dev44_improvement_summary.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "development_cases"
    / "dev45_final_package"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# CONFIGURATION — FROZEN DEV39
# =====================================================================

PARAMETERS = {
    "method": "Dev39",
    "candidate_strategy": "PET-only",
    "pet_threshold": 2.25,
    "hot_threshold": 3.0,
    "three_d_min_voxels": 75,
    "three_d_min_slices": 2,
    "ranking_method": "PET-quality",
    "hot_fraction_gate": 0.40,
    "top_k": 4,
    "core_policy": "P70",
}


# =====================================================================
# LOAD DATA
# =====================================================================

print("=" * 100)
print("DEV45 — FINAL EXPERIMENT PACKAGE")
print("=" * 100)

dev40 = pd.read_csv(DEV40_FILE)
dev43 = pd.read_csv(DEV43_FILE)
dev44_summary = pd.read_csv(DEV44_SUMMARY)
dev44_improvement = pd.read_csv(DEV44_IMPROVEMENT)


# =====================================================================
# DEVELOPMENT RESULTS
# =====================================================================

dev_rows = []

for _, row in dev40.iterrows():

    dev_rows.append({
        "cohort": "development",
        "case": row["case"],
        "method": "Dev10",
        "dice": row["dev10_dice"],
        "iou": row["dev10_iou"],
        "slice_recall": row["dev10_slice_recall"],
        "fpr": row["dev10_fpr"],
        "prediction_voxels":
            row["dev10_prediction_voxels"],
    })

    dev_rows.append({
        "cohort": "development",
        "case": row["case"],
        "method": "Dev39",
        "dice": row["dev39_dice"],
        "iou": row["dev39_iou"],
        "slice_recall": row["dev39_slice_recall"],
        "fpr": row["dev39_fpr"],
        "prediction_voxels":
            row["dev39_prediction_voxels"],
    })


development = pd.DataFrame(dev_rows)


# =====================================================================
# VALIDATION RESULTS
# =====================================================================

validation = dev43[
    [
        "case",
        "method",
        "dice",
        "iou",
        "slice_recall",
        "fpr",
        "prediction_voxels",
    ]
].copy()

validation.insert(
    0,
    "cohort",
    "validation"
)


# =====================================================================
# COMBINE
# =====================================================================

final_results = pd.concat(
    [
        development,
        validation
    ],
    ignore_index=True
)

final_results = final_results[
    [
        "cohort",
        "case",
        "method",
        "dice",
        "iou",
        "slice_recall",
        "fpr",
        "prediction_voxels",
    ]
]


# =====================================================================
# PER-CASE COMPARISON
# =====================================================================

pivot = final_results.pivot_table(
    index=["cohort", "case"],
    columns="method",
    values=[
        "dice",
        "iou",
        "slice_recall",
        "fpr",
        "prediction_voxels",
    ],
    aggfunc="first",
)

pivot.columns = [
    f"{metric}_{method.lower()}"
    for metric, method in pivot.columns
]

pivot = pivot.reset_index()


for metric in [
    "dice",
    "iou",
    "slice_recall",
    "fpr",
    "prediction_voxels",
]:

    d10 = f"{metric}_dev10"
    d39 = f"{metric}_dev39"

    if d10 in pivot.columns and d39 in pivot.columns:

        pivot[
            f"{metric}_delta_dev39_vs_dev10"
        ] = (
            pivot[d39] - pivot[d10]
        )


# =====================================================================
# FINAL METHOD COMPARISON
# =====================================================================

method_summary = (
    final_results
    .groupby(["cohort", "method"])
    .agg(
        n_cases=("case", "count"),
        mean_dice=("dice", "mean"),
        median_dice=("dice", "median"),
        mean_iou=("iou", "mean"),
        mean_slice_recall=("slice_recall", "mean"),
        mean_fpr=("fpr", "mean"),
        mean_prediction_voxels=(
            "prediction_voxels",
            "mean"
        ),
    )
    .reset_index()
)


# =====================================================================
# COMBINED COHORT SUMMARY
# =====================================================================

combined = (
    final_results
    .groupby("method")
    .agg(
        n_cases=("case", "count"),
        mean_dice=("dice", "mean"),
        median_dice=("dice", "median"),
        mean_iou=("iou", "mean"),
        mean_slice_recall=("slice_recall", "mean"),
        mean_fpr=("fpr", "mean"),
        mean_prediction_voxels=(
            "prediction_voxels",
            "mean"
        ),
    )
    .reset_index()
)

combined.insert(
    0,
    "cohort",
    "combined"
)

method_summary = pd.concat(
    [
        method_summary,
        combined
    ],
    ignore_index=True
)


# =====================================================================
# ABLATION / DEVELOPMENT IMPROVEMENT SUMMARY
# =====================================================================

ablation_rows = []

for cohort in [
    "development",
    "validation",
    "combined",
]:

    x = final_results[
        final_results["cohort"] == cohort
    ]

    if "Dev10" not in x["method"].values:
        continue

    if "Dev39" not in x["method"].values:
        continue

    d10 = x[
        x["method"] == "Dev10"
    ].set_index("case")

    d39 = x[
        x["method"] == "Dev39"
    ].set_index("case")

    common = d10.index.intersection(
        d39.index
    )

    if len(common) == 0:
        continue

    dice_delta = (
        d39.loc[common, "dice"]
        - d10.loc[common, "dice"]
    )

    iou_delta = (
        d39.loc[common, "iou"]
        - d10.loc[common, "iou"]
    )

    recall_delta = (
        d39.loc[common, "slice_recall"]
        - d10.loc[common, "slice_recall"]
    )

    fpr_delta = (
        d39.loc[common, "fpr"]
        - d10.loc[common, "fpr"]
    )

    voxel_delta = (
        d39.loc[common, "prediction_voxels"]
        - d10.loc[common, "prediction_voxels"]
    )

    ablation_rows.append({
        "cohort": cohort,
        "n_cases": len(common),
        "mean_dice_delta": dice_delta.mean(),
        "mean_iou_delta": iou_delta.mean(),
        "mean_slice_recall_delta":
            recall_delta.mean(),
        "mean_fpr_delta": fpr_delta.mean(),
        "mean_prediction_voxel_delta":
            voxel_delta.mean(),
        "dice_improved_cases":
            int((dice_delta > 0).sum()),
        "dice_equal_cases":
            int((dice_delta == 0).sum()),
        "dice_worse_cases":
            int((dice_delta < 0).sum()),
        "fpr_lower_cases":
            int((fpr_delta < 0).sum()),
        "fpr_equal_cases":
            int((fpr_delta == 0).sum()),
        "fpr_higher_cases":
            int((fpr_delta > 0).sum()),
    })


ablation = pd.DataFrame(
    ablation_rows
)


# =====================================================================
# PARAMETERS TABLE
# =====================================================================

parameters = pd.DataFrame(
    [
        {
            "parameter": key,
            "value": value,
        }
        for key, value in PARAMETERS.items()
    ]
)


# =====================================================================
# REPRODUCIBILITY REPORT
# =====================================================================

timestamp = datetime.now().isoformat(
    timespec="seconds"
)

# Extract final summary values first.
dev10_dev_dice = float(
    method_summary.loc[
        (method_summary["cohort"] == "development")
        & (method_summary["method"] == "Dev10"),
        "mean_dice"
    ].iloc[0]
)

dev39_dev_dice = float(
    method_summary.loc[
        (method_summary["cohort"] == "development")
        & (method_summary["method"] == "Dev39"),
        "mean_dice"
    ].iloc[0]
)

dev10_val_dice = float(
    method_summary.loc[
        (method_summary["cohort"] == "validation")
        & (method_summary["method"] == "Dev10"),
        "mean_dice"
    ].iloc[0]
)

dev39_val_dice = float(
    method_summary.loc[
        (method_summary["cohort"] == "validation")
        & (method_summary["method"] == "Dev39"),
        "mean_dice"
    ].iloc[0]
)

dev10_combined_dice = float(
    method_summary.loc[
        (method_summary["cohort"] == "combined")
        & (method_summary["method"] == "Dev10"),
        "mean_dice"
    ].iloc[0]
)

dev39_combined_dice = float(
    method_summary.loc[
        (method_summary["cohort"] == "combined")
        & (method_summary["method"] == "Dev39"),
        "mean_dice"
    ].iloc[0]
)


report = []

report.append(
    "DEV45 — FINAL REPRODUCIBILITY REPORT"
)

report.append("=" * 80)

report.append(
    f"Generated: {timestamp}"
)

report.append("")

report.append(
    "FINAL METHOD: Dev39"
)

report.append(
    "Dev39 was frozen before validation."
)

report.append("")

report.append(
    "PIPELINE"
)

report.append(
    "PET-only SUV threshold >= 2.25"
)

report.append(
    "-> exact Dev10 3D connected-component filtering"
)

report.append(
    "-> PET-quality component ranking"
)

report.append(
    "-> hot_fraction >= 0.40"
)

report.append(
    "-> Top-K = 4"
)

report.append(
    "-> P70 core"
)

report.append("")

report.append(
    "DEVELOPMENT COHORT"
)

report.append(
    "6 development cases"
)

report.append(
    f"Dev10 mean Dice = {dev10_dev_dice:.6f}"
)

report.append(
    f"Dev39 mean Dice = {dev39_dev_dice:.6f}"
)

report.append(
    f"Dev39 improvement = "
    f"{dev39_dev_dice - dev10_dev_dice:+.6f}"
)

report.append("")

report.append(
    "VALIDATION COHORT"
)

report.append(
    "1 independent Case-7 validation case"
)

report.append(
    f"Dev10 Dice = {dev10_val_dice:.6f}"
)

report.append(
    f"Dev39 Dice = {dev39_val_dice:.6f}"
)

report.append(
    f"Dev39 improvement = "
    f"{dev39_val_dice - dev10_val_dice:+.6f}"
)

report.append("")

report.append(
    "COMBINED COHORT"
)

report.append(
    "7 total evaluated cases"
)

report.append(
    f"Dev10 mean Dice = {dev10_combined_dice:.6f}"
)

report.append(
    f"Dev39 mean Dice = {dev39_combined_dice:.6f}"
)

report.append(
    f"Dev39 improvement = "
    f"{dev39_combined_dice - dev10_combined_dice:+.6f}"
)

report.append("")

report.append(
    "IMPORTANT LIMITATION"
)

report.append(
    "The validation cohort currently contains only one independent case."
)

report.append(
    "Therefore, the validation result provides preliminary evidence "
    "of generalization but is not sufficient for statistical "
    "generalization claims."
)

report.append("")

report.append(
    "DEV10 SOURCE FILE WAS NOT MODIFIED."
)

report.append(
    "Dev39 parameters were not changed during validation."
)


report_path = (
    OUTPUT_DIR
    / "dev45_reproducibility_report.txt"
)

report_path.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# =====================================================================
# SAVE TABLES
# =====================================================================

files = {}

files[
    "dev45_final_results.csv"
] = final_results

files[
    "dev45_per_case_results.csv"
] = pivot

files[
    "dev45_method_comparison.csv"
] = method_summary

files[
    "dev45_ablation_summary.csv"
] = ablation

files[
    "dev45_parameters.csv"
] = parameters

for filename, dataframe in files.items():

    path = OUTPUT_DIR / filename

    dataframe.to_csv(
        path,
        index=False
    )


# =====================================================================
# PRINT
# =====================================================================

print("\n")
print("=" * 100)
print("DEV45 — FINAL METHOD COMPARISON")
print("=" * 100)

print(
    method_summary.to_string(index=False)
)

print("\n")
print("=" * 100)
print("DEV45 — IMPROVEMENT / ABLATION SUMMARY")
print("=" * 100)

print(
    ablation.to_string(index=False)
)

print("\n")
print("=" * 100)
print("DEV45 — FROZEN PARAMETERS")
print("=" * 100)

print(
    parameters.to_string(index=False)
)

print("\n")
print("=" * 100)
print("OUTPUT FILES")
print("=" * 100)

for filename in files:
    print(OUTPUT_DIR / filename)

print(report_path)

print("\n===== DEV45 COMPLETE =====")