from pathlib import Path
import sys
from unittest import case
import numpy as np
import pandas as pd

# ============================================================
# DEV15 — CANDIDATE LOCALIZATION DIAGNOSTIC
# ============================================================
#
# Purpose:
#   Determine whether performance is limited by:
#   1. Candidate generation
#   2. 3D filtering
#   3. Ranking/selection
#
# Uses EXACT Dev10 candidate generation.
# No GT is used for candidate generation or selection.
#
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DEV10_PATH = ROOT / "src" / "dev10_lite_hot_core.py"

OUTPUT_DIR = (
    ROOT
    / "results"
    / "development_cases"
    / "dev15_candidate_localization_results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SLICE_CSV = OUTPUT_DIR / "dev15_slice_localization.csv"
CASE_CSV = OUTPUT_DIR / "dev15_case_summary.csv"
SUMMARY_CSV = OUTPUT_DIR / "dev15_summary.csv"


# ============================================================
# Load EXACT Dev10 definitions
# ============================================================

print("=" * 70)
print("DEV15 — Candidate Localization Diagnostic")
print("=" * 70)

if not DEV10_PATH.exists():
    raise FileNotFoundError(f"Dev10 file not found: {DEV10_PATH}")

source = DEV10_PATH.read_text(encoding="utf-8")

if "\n# MAIN" not in source:
    raise RuntimeError("Could not locate Dev10 '# MAIN' boundary.")

# IMPORTANT:
# Dev10 itself contains "# MAIN" in a validation check.
# Therefore use the LAST actual main boundary.
prefix = source.rsplit("\n# MAIN", 1)[0]

namespace = {
    "__file__": str(DEV10_PATH),
    "__name__": "dev10_imported_for_dev15",
}

exec(prefix, namespace)


# ============================================================
# Retrieve exact Dev10 functions
# ============================================================

load_case = namespace["load_case"]
generate_candidate_volume = namespace["generate_candidate_volume"]
build_3d_component_table = namespace["build_3d_component_table"]

# ============================================================
# Verified six development cases used throughout Dev9–Dev14
# ============================================================

DEV_CASES = [
    "PETCT_0168f65af8",
    "PETCT_04606080a0",
    "PETCT_04ab5c61c9",
    "PETCT_0b57b247b6",
    "PETCT_11afab3485",
    "PETCT_185da4c8b6",
]

print(f"Development cases loaded: {len(DEV_CASES)}")
print()


# ============================================================
# Helper: per-slice best component metrics
# ============================================================

def best_component_metrics(candidate_labels, gt_slice):
    """
    For one slice, calculate the best overlap and Dice
    among all non-zero 3D candidate components.

    Returns:
        best_dice
        best_gt_coverage
        best_overlap_pixels
        best_label
    """

    gt_count = int(gt_slice.sum())

    if gt_count == 0:
        return 0.0, 0.0, 0, 0

    labels = np.unique(candidate_labels)

    best_dice = 0.0
    best_coverage = 0.0
    best_overlap = 0
    best_label = 0

    for label in labels:
        if label == 0:
            continue

        comp = candidate_labels == label
        comp_count = int(comp.sum())

        if comp_count == 0:
            continue

        overlap = int(np.logical_and(comp, gt_slice).sum())

        if overlap == 0:
            continue

        coverage = overlap / gt_count

        union = comp_count + gt_count - overlap
        dice = (2.0 * overlap / union) if union > 0 else 0.0

        if dice > best_dice:
            best_dice = dice
            best_coverage = coverage
            best_overlap = overlap
            best_label = int(label)

    return best_dice, best_coverage, best_overlap, best_label


# ============================================================
# Storage
# ============================================================

slice_rows = []
case_rows = []


# ============================================================
# Process cases one at a time
# ============================================================

for case_idx, case_id in enumerate(DEV_CASES, start=1):

    print("-" * 70)
    print(f"CASE {case_idx}/{len(DEV_CASES)}: {case_id}")
    print("-" * 70)

    # --------------------------------------------------------
    # Load exact Dev10 case
    # --------------------------------------------------------

    case = load_case(case_id)

    ct = case["ct"]
    suv = case["suv"]
    gt = case["gt"]
    # --------------------------------------------------------
    # Exact Dev10 raw candidate generation
    # --------------------------------------------------------

    raw_volume = generate_candidate_volume(case)
    raw_volume = raw_volume.astype(np.uint8, copy=False)
    
    gt = gt.astype(bool)

    # --------------------------------------------------------
    # Exact Dev10 3D component filtering
    # --------------------------------------------------------

    component_table, cc = build_3d_component_table(
        case,
        raw_volume
    )

    cc = np.asarray(cc)

    # --------------------------------------------------------
    # Integrity checks
    # --------------------------------------------------------

    if raw_volume.shape != gt.shape:
        raise RuntimeError(
            f"{case_id}: raw candidate shape {raw_volume.shape} "
            f"!= GT shape {gt.shape}"
        )

    if cc.shape != gt.shape:
        raise RuntimeError(
            f"{case_id}: component shape {cc.shape} "
            f"!= GT shape {gt.shape}"
        )

    gt_positive_slices = np.where(gt.reshape(gt.shape[0], -1).any(axis=1))[0]

    print(f"Volume shape: {gt.shape}")
    print(f"GT-positive slices: {len(gt_positive_slices)}")
    print(f"Raw candidate voxels: {int(raw_volume.sum())}")
    print(f"3D eligible candidate voxels: {int((cc > 0).sum())}")
    print(f"3D components: {len(component_table)}")

    # --------------------------------------------------------
    # Case accumulators
    # --------------------------------------------------------

    raw_hits = 0
    cc_hits = 0

    raw_coverages = []
    cc_coverages = []

    raw_but_cc_miss = 0
    raw_miss = 0

    best_dices = []
    best_coverages = []

    # --------------------------------------------------------
    # Analyze every GT-positive slice
    # --------------------------------------------------------

    for z in gt_positive_slices:

        gt_slice = gt[z]
        raw_slice = raw_volume[z]
        cc_slice = cc[z]

        gt_pixels = int(gt_slice.sum())

        # ----------------------------------------------------
        # RAW candidate
        # ----------------------------------------------------

        raw_overlap = int(
            np.logical_and(raw_slice, gt_slice).sum()
        )

        raw_hit = raw_overlap > 0

        raw_coverage = (
            raw_overlap / gt_pixels
            if gt_pixels > 0
            else 0.0
        )

        if raw_hit:
            raw_hits += 1
        else:
            raw_miss += 1

        raw_coverages.append(raw_coverage)

        # ----------------------------------------------------
        # 3D eligible candidates
        # ----------------------------------------------------

        cc_binary = cc_slice > 0

        cc_overlap = int(
            np.logical_and(cc_binary, gt_slice).sum()
        )

        cc_hit = cc_overlap > 0

        cc_coverage = (
            cc_overlap / gt_pixels
            if gt_pixels > 0
            else 0.0
        )

        if cc_hit:
            cc_hits += 1

        cc_coverages.append(cc_coverage)

        # ----------------------------------------------------
        # Best individual 3D component on this slice
        # ----------------------------------------------------

        (
            best_dice,
            best_component_coverage,
            best_component_overlap,
            best_label,
        ) = best_component_metrics(
            cc_slice,
            gt_slice
        )

        best_dices.append(best_dice)
        best_coverages.append(best_component_coverage)

        # ----------------------------------------------------
        # Identify 3D filtering loss
        # ----------------------------------------------------

        raw_but_cc_miss_flag = raw_hit and not cc_hit

        if raw_but_cc_miss_flag:
            raw_but_cc_miss += 1

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        if not raw_hit:
            classification = "NO_RAW_CANDIDATE"

        elif raw_hit and not cc_hit:
            classification = "REMOVED_BY_3D_FILTER"

        elif best_dice >= 0.50:
            classification = "STRONG_COMPONENT"

        elif best_dice >= 0.20:
            classification = "MODERATE_COMPONENT"

        else:
            classification = "WEAK_COMPONENT"

        slice_rows.append(
            {
                "case_id": case_id,
                "slice_z": int(z),
                "gt_pixels": gt_pixels,

                "raw_candidate_pixels": int(raw_slice.sum()),
                "raw_overlap_pixels": raw_overlap,
                "raw_hit": raw_hit,
                "raw_gt_coverage": raw_coverage,

                "cc_candidate_pixels": int(cc_binary.sum()),
                "cc_overlap_pixels": cc_overlap,
                "cc_hit": cc_hit,
                "cc_gt_coverage": cc_coverage,

                "best_component_label": best_label,
                "best_component_overlap": best_component_overlap,
                "best_component_dice": best_dice,
                "best_component_gt_coverage": best_component_coverage,

                "classification": classification,
            }
        )

    # --------------------------------------------------------
    # Case summary
    # --------------------------------------------------------

    n_gt_slices = len(gt_positive_slices)

    raw_hit_rate = raw_hits / n_gt_slices if n_gt_slices else 0.0
    cc_hit_rate = cc_hits / n_gt_slices if n_gt_slices else 0.0

    mean_raw_coverage = (
        float(np.mean(raw_coverages))
        if raw_coverages
        else 0.0
    )

    mean_cc_coverage = (
        float(np.mean(cc_coverages))
        if cc_coverages
        else 0.0
    )

    median_best_dice = (
        float(np.median(best_dices))
        if best_dices
        else 0.0
    )

    max_best_dice = (
        float(np.max(best_dices))
        if best_dices
        else 0.0
    )

    mean_best_dice = (
        float(np.mean(best_dices))
        if best_dices
        else 0.0
    )

    case_rows.append(
        {
            "case_id": case_id,
            "gt_positive_slices": n_gt_slices,

            "raw_candidate_hit_slices": raw_hits,
            "raw_candidate_hit_rate": raw_hit_rate,

            "three_d_candidate_hit_slices": cc_hits,
            "three_d_candidate_hit_rate": cc_hit_rate,

            "raw_miss_slices": raw_miss,
            "raw_hit_but_3d_miss_slices": raw_but_cc_miss,

            "mean_raw_gt_coverage": mean_raw_coverage,
            "mean_3d_gt_coverage": mean_cc_coverage,

            "mean_best_component_dice": mean_best_dice,
            "median_best_component_dice": median_best_dice,
            "max_best_component_dice": max_best_dice,

            "num_3d_components": len(component_table),
        }
    )

    print(
        f"Raw candidate hit rate: "
        f"{raw_hits}/{n_gt_slices} "
        f"({100 * raw_hit_rate:.1f}%)"
    )

    print(
        f"3D candidate hit rate: "
        f"{cc_hits}/{n_gt_slices} "
        f"({100 * cc_hit_rate:.1f}%)"
    )

    print(
        f"Raw mean GT coverage: "
        f"{mean_raw_coverage:.4f}"
    )

    print(
        f"3D mean GT coverage: "
        f"{mean_cc_coverage:.4f}"
    )

    print(
        f"Best-component Dice — "
        f"mean: {mean_best_dice:.4f}, "
        f"median: {median_best_dice:.4f}, "
        f"max: {max_best_dice:.4f}"
    )

    print(
        f"Raw-hit but 3D-missed slices: "
        f"{raw_but_cc_miss}"
    )

    print(
        f"No-raw-candidate slices: "
        f"{raw_miss}"
    )

    print()


# ============================================================
# DataFrames
# ============================================================

slice_df = pd.DataFrame(slice_rows)
case_df = pd.DataFrame(case_rows)


# ============================================================
# Aggregate summary
# ============================================================

total_gt_slices = int(
    case_df["gt_positive_slices"].sum()
)

total_raw_hits = int(
    case_df["raw_candidate_hit_slices"].sum()
)

total_cc_hits = int(
    case_df["three_d_candidate_hit_slices"].sum()
)

total_raw_miss = int(
    case_df["raw_miss_slices"].sum()
)

total_raw_but_cc_miss = int(
    case_df["raw_hit_but_3d_miss_slices"].sum()
)

aggregate_raw_hit_rate = (
    total_raw_hits / total_gt_slices
    if total_gt_slices
    else 0.0
)

aggregate_cc_hit_rate = (
    total_cc_hits / total_gt_slices
    if total_gt_slices
    else 0.0
)

summary_rows = [
    {
        "metric": "GT-positive slices",
        "value": total_gt_slices,
    },
    {
        "metric": "Raw candidate hit slices",
        "value": total_raw_hits,
    },
    {
        "metric": "Raw candidate hit rate",
        "value": aggregate_raw_hit_rate,
    },
    {
        "metric": "3D candidate hit slices",
        "value": total_cc_hits,
    },
    {
        "metric": "3D candidate hit rate",
        "value": aggregate_cc_hit_rate,
    },
    {
        "metric": "No raw candidate slices",
        "value": total_raw_miss,
    },
    {
        "metric": "Raw hit but removed by 3D filter",
        "value": total_raw_but_cc_miss,
    },
    {
        "metric": "Mean raw GT coverage",
        "value": float(
            slice_df["raw_gt_coverage"].mean()
        ),
    },
    {
        "metric": "Mean 3D GT coverage",
        "value": float(
            slice_df["cc_gt_coverage"].mean()
        ),
    },
    {
        "metric": "Mean best component Dice",
        "value": float(
            slice_df["best_component_dice"].mean()
        ),
    },
    {
        "metric": "Median best component Dice",
        "value": float(
            slice_df["best_component_dice"].median()
        ),
    },
    {
        "metric": "Maximum best component Dice",
        "value": float(
            slice_df["best_component_dice"].max()
        ),
    },
]


summary_df = pd.DataFrame(summary_rows)


# ============================================================
# Save
# ============================================================

slice_df.to_csv(SLICE_CSV, index=False)
case_df.to_csv(CASE_CSV, index=False)
summary_df.to_csv(SUMMARY_CSV, index=False)


# ============================================================
# Print final summary
# ============================================================

print("=" * 70)
print("DEV15 COMPLETE")
print("=" * 70)

print()
print("Aggregate summary:")
print(summary_df.to_string(index=False))

print()
print("Case summary:")
print(case_df.to_string(index=False))

print()
print("Output files:")
print(f"Slice-level CSV : {SLICE_CSV}")
print(f"Case-level CSV  : {CASE_CSV}")
print(f"Summary CSV     : {SUMMARY_CSV}")

print()
print("=" * 70)
print("Interpretation guide")
print("=" * 70)

print(
    """
NO_RAW_CANDIDATE
    -> Candidate generation failed on that GT-positive slice.

REMOVED_BY_3D_FILTER
    -> A raw candidate existed, but Dev10's 3D filtering removed it.

WEAK_COMPONENT
    -> A surviving 3D candidate exists, but has weak slice overlap.

MODERATE_COMPONENT
MODERATE_COMPONENT
    -> A surviving candidate has meaningful overlap.

STRONG_COMPONENT
    -> A surviving candidate has Dice >= 0.50 on that slice.
"""
)