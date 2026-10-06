"""
Dev44 — Final cross-cohort evaluation

Development cohort:
    6 development cases from Dev40

Validation cohort:
    PETCT_0011f3deaf from Dev43

Frozen methods:
    Dev10
    Dev39

NO parameter tuning is performed.
"""

from pathlib import Path
import numpy as np
import pandas as pd


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

OUTPUT_DIR = (
    ROOT
    / "results"
    / "development_cases"
    / "dev44_cross_cohort_evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 100)
print("DEV44 — CROSS-COHORT EVALUATION")
print("=" * 100)


# =====================================================================
# LOAD DEVELOPMENT RESULTS
# =====================================================================

dev40 = pd.read_csv(DEV40_FILE)

print("\nDevelopment source:")
print(DEV40_FILE)

print("\nDevelopment cases:")
print(dev40.to_string(index=False))


# =====================================================================
# LOAD VALIDATION RESULTS
# =====================================================================

dev43 = pd.read_csv(DEV43_FILE)

print("\nValidation source:")
print(DEV43_FILE)

print("\nValidation result:")
print(dev43.to_string(index=False))


# =====================================================================
# CONVERT DEV40 WIDE FORMAT -> LONG FORMAT
# =====================================================================

dev10_cols = [
    "dev10_dice",
    "dev10_iou",
    "dev10_slice_recall",
    "dev10_fpr",
    "dev10_prediction_voxels",
]

dev39_cols = [
    "dev39_dice",
    "dev39_iou",
    "dev39_slice_recall",
    "dev39_fpr",
    "dev39_prediction_voxels",
]

for col in dev10_cols + dev39_cols:
    if col not in dev40.columns:
        raise ValueError(
            f"Dev40 is missing required column: {col}"
        )


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


dev = pd.DataFrame(dev_rows)


# =====================================================================
# CONVERT DEV43 -> LONG FORMAT
# =====================================================================

required_val_cols = [
    "case",
    "method",
    "dice",
    "iou",
    "slice_recall",
    "fpr",
    "prediction_voxels",
]

for col in required_val_cols:
    if col not in dev43.columns:
        raise ValueError(
            f"Dev43 is missing required column: {col}"
        )


val = dev43[
    required_val_cols
].copy()

val["cohort"] = "validation"

val = val[
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
# COMBINE
# =====================================================================

all_results = pd.concat(
    [dev, val],
    ignore_index=True
)


print("\n")
print("=" * 100)
print("ALL CASE RESULTS")
print("=" * 100)

print(
    all_results.to_string(index=False)
)


# =====================================================================
# PER-CASE DEV39 vs DEV10 COMPARISON
# =====================================================================

pivot = all_results.pivot_table(
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
        pivot[f"{metric}_delta_dev39_vs_dev10"] = (
            pivot[d39] - pivot[d10]
        )


# =====================================================================
# COHORT SUMMARY
# =====================================================================

def make_summary(df, cohort_name):

    rows = []

    for method in ["Dev10", "Dev39"]:

        x = df[
            (df["cohort"] == cohort_name)
            & (df["method"] == method)
        ]

        rows.append({
            "cohort": cohort_name,
            "method": method,
            "n_cases": len(x),
            "mean_dice": x["dice"].mean(),
            "median_dice": x["dice"].median(),
            "mean_iou": x["iou"].mean(),
            "mean_slice_recall":
                x["slice_recall"].mean(),
            "mean_fpr": x["fpr"].mean(),
            "mean_prediction_voxels":
                x["prediction_voxels"].mean(),
        })

    return pd.DataFrame(rows)


summary = pd.concat(
    [
        make_summary(all_results, "development"),
        make_summary(all_results, "validation"),
    ],
    ignore_index=True
)


# =====================================================================
# COMBINED COHORT SUMMARY
# =====================================================================

combined_rows = []

for method in ["Dev10", "Dev39"]:

    x = all_results[
        all_results["method"] == method
    ]

    combined_rows.append({
        "cohort": "combined",
        "method": method,
        "n_cases": len(x),
        "mean_dice": x["dice"].mean(),
        "median_dice": x["dice"].median(),
        "mean_iou": x["iou"].mean(),
        "mean_slice_recall":
            x["slice_recall"].mean(),
        "mean_fpr": x["fpr"].mean(),
        "mean_prediction_voxels":
            x["prediction_voxels"].mean(),
    })

summary = pd.concat(
    [
        summary,
        pd.DataFrame(combined_rows)
    ],
    ignore_index=True
)


# =====================================================================
# IMPROVEMENT ANALYSIS
# =====================================================================

def improvement_analysis(df, cohort_name):

    x = df[
        df["cohort"] == cohort_name
    ]

    d10 = x[
        x["method"] == "Dev10"
    ].set_index("case")

    d39 = x[
        x["method"] == "Dev39"
    ].set_index("case")

    common = d10.index.intersection(
        d39.index
    )

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

    return {
        "cohort": cohort_name,
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
    }


improvement = pd.DataFrame([
    improvement_analysis(
        all_results,
        "development"
    ),
    improvement_analysis(
        all_results,
        "validation"
    ),
    improvement_analysis(
        all_results,
        "combined"
    ),
])


# =====================================================================
# GENERALIZATION GAP
# =====================================================================

def get_mean_dice(cohort, method):

    row = summary[
        (summary["cohort"] == cohort)
        & (summary["method"] == method)
    ]

    return float(
        row["mean_dice"].iloc[0]
    )


dev10_dev = get_mean_dice(
    "development",
    "Dev10"
)

dev39_dev = get_mean_dice(
    "development",
    "Dev39"
)

dev10_val = get_mean_dice(
    "validation",
    "Dev10"
)

dev39_val = get_mean_dice(
    "validation",
    "Dev39"
)

generalization = pd.DataFrame([
    {
        "metric": "Dev10 mean Dice",
        "development": dev10_dev,
        "validation": dev10_val,
        "validation_minus_development":
            dev10_val - dev10_dev,
    },
    {
        "metric": "Dev39 mean Dice",
        "development": dev39_dev,
        "validation": dev39_val,
        "validation_minus_development":
            dev39_val - dev39_dev,
    },
    {
        "metric": "Dev39 improvement over Dev10",
        "development":
            dev39_dev - dev10_dev,
        "validation":
            dev39_val - dev10_val,
        "validation_minus_development":
            (
                dev39_val - dev10_val
            )
            -
            (
                dev39_dev - dev10_dev
            ),
    },
])


# =====================================================================
# SAVE
# =====================================================================

all_results_path = (
    OUTPUT_DIR
    / "dev44_all_case_results.csv"
)

per_case_path = (
    OUTPUT_DIR
    / "dev44_per_case_comparison.csv"
)

summary_path = (
    OUTPUT_DIR
    / "dev44_cohort_summary.csv"
)

improvement_path = (
    OUTPUT_DIR
    / "dev44_improvement_summary.csv"
)

generalization_path = (
    OUTPUT_DIR
    / "dev44_generalization.csv"
)

all_results.to_csv(
    all_results_path,
    index=False
)

pivot.to_csv(
    per_case_path,
    index=False
)

summary.to_csv(
    summary_path,
    index=False
)

improvement.to_csv(
    improvement_path,
    index=False
)

generalization.to_csv(
    generalization_path,
    index=False
)


# =====================================================================
# PRINT RESULTS
# =====================================================================

print("\n")
print("=" * 100)
print("DEV44 — COHORT SUMMARY")
print("=" * 100)

print(
    summary.to_string(index=False)
)

print("\n")
print("=" * 100)
print("DEV44 — IMPROVEMENT VS DEV10")
print("=" * 100)

print(
    improvement.to_string(index=False)
)

print("\n")
print("=" * 100)
print("DEV44 — GENERALIZATION")
print("=" * 100)

print(
    generalization.to_string(index=False)
)

print("\n")
print("=" * 100)
print("DEV44 — PER CASE COMPARISON")
print("=" * 100)

print(
    pivot.to_string(index=False)
)

print("\nSaved:")
print(all_results_path)
print(per_case_path)
print(summary_path)
print(improvement_path)
print(generalization_path)

print("\n===== DEV44 COMPLETE =====")