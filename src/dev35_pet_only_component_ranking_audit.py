"""
Dev35 - PET-only Component / Ranking Audit

Purpose
-------
Audit where PET-only candidates are lost between:

    PET-only candidate
        -> Dev10 3D component filtering
        -> ranking
        -> Top-K selection
        -> P70 core
        -> final metrics

IMPORTANT:
- src/dev10_lite_hot_core.py is NOT modified.
- GT is used ONLY for diagnostic auditing.
- GT is NEVER used for ranking or selection.
- Downstream component construction and final evaluation use
  the exact Dev10 implementation.

Outputs
-------
results/development_cases/dev35_pet_only_component_ranking_audit/
    dev35_component_audit.csv
    dev35_case_summary.csv
    dev35_macro_summary.csv
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import dev10_lite_hot_core as dev10


# ============================================================
# CONFIGURATION
# ============================================================

CASES = [
    "PETCT_0168f65af8",
    "PETCT_04606080a0",
    "PETCT_04ab5c61c9",
    "PETCT_0b57b247b6",
    "PETCT_11afab3485",
    "PETCT_185da4c8b6",
]

PET_THRESHOLD = 2.25

RANKING_FEATURE = "mean_suv_hot3_a0p5"
CORE_POLICY = "P70"
TOP_K = 4

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "development_cases"
    / "dev35_pet_only_component_ranking_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    """Convert numeric values safely to float."""
    try:
        return float(value)
    except Exception:
        return np.nan


def safe_int(value):
    """Convert numeric values safely to int."""
    try:
        return int(value)
    except Exception:
        return 0


def dice_from_counts(pred_count, gt_count, overlap):
    """Dice from voxel counts."""
    denom = pred_count + gt_count

    if denom == 0:
        return 0.0

    return (2.0 * overlap) / denom


def iou_from_counts(pred_count, gt_count, overlap):
    """IoU from voxel counts."""
    union = pred_count + gt_count - overlap

    if union == 0:
        return 0.0

    return overlap / union


# ============================================================
# PET-ONLY CANDIDATE
# ============================================================

def build_pet_only_candidate(case):
    """
    PET-only candidate:

        SUV >= 2.25

    No CT contour restriction.
    No additional morphology.
    """

    suv = case["suv"]

    candidate = (suv >= PET_THRESHOLD).astype(np.uint8)

    return candidate


# ============================================================
# COMPONENT AUDIT
# ============================================================

def audit_components(case, components, cc):
    """
    Audit every Dev10 3D component.

    GT is used only to understand whether a component is useful.
    """

    suv = case["suv"]
    gt = case["gt"]

    # Ensure boolean GT
    gt_bool = gt > 0

    # Exact Dev10 ordering
    ranked = components.sort_values(
        [RANKING_FEATURE, "mean_suv"],
        ascending=[False, False],
    ).reset_index(drop=True)

    rows = []

    for rank_idx, row in ranked.iterrows():

        label = safe_int(row["label"])

        component_mask = cc == label

        component_voxels = int(component_mask.sum())

        if component_voxels == 0:
            continue

        # ----------------------------------------------------
        # Component spatial information
        # ----------------------------------------------------

        coords = np.where(component_mask)

        z_values = coords[0]

        z_min = int(z_values.min())
        z_max = int(z_values.max())

        z_span = z_max - z_min + 1

        # ----------------------------------------------------
        # SUV information
        # ----------------------------------------------------

        component_suv = suv[component_mask]

        mean_suv = float(np.mean(component_suv))
        max_suv = float(np.max(component_suv))

        frac_ge_3 = float(np.mean(component_suv >= 3.0))

        # ----------------------------------------------------
        # Exact Dev10 ranking feature
        # ----------------------------------------------------

        ranking_value = safe_float(row[RANKING_FEATURE])

        # ----------------------------------------------------
        # GT overlap
        # ----------------------------------------------------

        gt_overlap = int(np.logical_and(component_mask, gt_bool).sum())

        gt_voxels = int(gt_bool.sum())

        component_dice = dice_from_counts(
            component_voxels,
            gt_voxels,
            gt_overlap,
        )

        component_iou = iou_from_counts(
            component_voxels,
            gt_voxels,
            gt_overlap,
        )

        # ----------------------------------------------------
        # Top-K status
        # ----------------------------------------------------

        rank = rank_idx + 1

        selected_top_k = rank <= TOP_K

        # ----------------------------------------------------
        # P70 core audit
        #
        # IMPORTANT:
        # Exact Dev10 build_core signature:
        #
        # build_core(cc, suv, label, policy)
        # ----------------------------------------------------

        try:
            core_mask = dev10.build_core(
                cc,
                suv,
                label,
                CORE_POLICY,
            )

            core_mask = core_mask.astype(bool)

        except Exception as exc:
            print(
                f"WARNING: build_core failed for "
                f"{case['name']} label {label}: {exc}"
            )

            core_mask = np.zeros_like(component_mask, dtype=bool)

        core_voxels = int(core_mask.sum())

        core_gt_overlap = int(
            np.logical_and(core_mask, gt_bool).sum()
        )

        core_dice = dice_from_counts(
            core_voxels,
            gt_voxels,
            core_gt_overlap,
        )

        core_iou = iou_from_counts(
            core_voxels,
            gt_voxels,
            core_gt_overlap,
        )

        core_reaches_gt = core_gt_overlap > 0

        # ----------------------------------------------------
        # Component row
        # ----------------------------------------------------

        result = {
            "case": case["name"],

            # Identity
            "label": label,

            # Ranking
            "rank": rank,
            "ranking_feature": RANKING_FEATURE,
            "ranking_value": ranking_value,

            # Component geometry
            "component_voxels": component_voxels,
            "z_min": z_min,
            "z_max": z_max,
            "z_span": z_span,

            # SUV
            "mean_suv": mean_suv,
            "max_suv": max_suv,
            "frac_ge_3": frac_ge_3,

            # Exact Dev10 ranking feature
            "mean_suv_hot3_a0p5": ranking_value,

            # GT diagnostic only
            "gt_overlap_voxels": gt_overlap,
            "component_gt_dice": component_dice,
            "component_gt_iou": component_iou,

            # Selection
            "selected_top4": selected_top_k,

            # P70
            "core_voxels": core_voxels,
            "core_gt_overlap_voxels": core_gt_overlap,
            "core_gt_dice": core_dice,
            "core_gt_iou": core_iou,
            "core_reaches_gt": core_reaches_gt,

            # Useful diagnostic flags
            "gt_overlapping_component": gt_overlap > 0,
        }

        rows.append(result)

    return pd.DataFrame(rows)


# ============================================================
# CASE EVALUATION
# ============================================================

def evaluate_case(case_name):
    print("\n" + "=" * 80)
    print(f"DEV35 | {case_name}")
    print("=" * 80)

    # --------------------------------------------------------
    # Load case using exact Dev10 loader
    # --------------------------------------------------------

    case = dev10.load_case(case_name)

    print("CT shape :", case["ct"].shape)
    print("SUV shape:", case["suv"].shape)
    print("GT shape :", case["gt"].shape)

    # --------------------------------------------------------
    # PET-only candidate
    # --------------------------------------------------------

    pet_only = build_pet_only_candidate(case)

    candidate_voxels = int(np.count_nonzero(pet_only))

    print(f"\nPET threshold : {PET_THRESHOLD}")
    print(f"PET candidate voxels: {candidate_voxels}")

    # --------------------------------------------------------
    # Exact Dev10 3D component construction
    # --------------------------------------------------------

    components, cc = dev10.build_3d_component_table(
        case,
        pet_only,
    )

    print(f"3D components after Dev10 filtering: {len(components)}")

    # --------------------------------------------------------
    # Audit every component
    # --------------------------------------------------------

    audit_df = audit_components(
        case,
        components,
        cc,
    )

    # --------------------------------------------------------
    # Exact Dev10 final evaluation
    # --------------------------------------------------------

    metrics = dev10.evaluate_selection(
        case,
        components,
        cc,
        RANKING_FEATURE,
        CORE_POLICY,
        TOP_K,
    )

    # --------------------------------------------------------
    # Diagnostic summaries
    # --------------------------------------------------------

    gt_components = audit_df[
        audit_df["gt_overlapping_component"]
    ]

    selected = audit_df[
        audit_df["selected_top4"]
    ]

    selected_gt = selected[
        selected["gt_overlapping_component"]
    ]

    gt_selected_count = len(selected_gt)

    total_gt_overlap_components = len(gt_components)

    # Best rank of a GT-overlapping component
    if len(gt_components) > 0:
        best_gt_rank = int(gt_components["rank"].min())
    else:
        best_gt_rank = np.nan

    # Best component Dice
    if len(gt_components) > 0:
        best_component_dice = float(
            gt_components["component_gt_dice"].max()
        )
    else:
        best_component_dice = 0.0

    # Best core Dice among GT-overlapping components
    if len(gt_components) > 0:
        best_core_dice = float(
            gt_components["core_gt_dice"].max()
        )
    else:
        best_core_dice = 0.0

    # --------------------------------------------------------
    # Print ranking audit
    # --------------------------------------------------------

    print("\nGT-overlapping components:")
    print("-" * 80)

    if len(gt_components) == 0:
        print("NONE")
    else:
        display_columns = [
            "label",
            "rank",
            "component_voxels",
            "z_span",
            "mean_suv",
            "max_suv",
            "frac_ge_3",
            "mean_suv_hot3_a0p5",
            "gt_overlap_voxels",
            "component_gt_dice",
            "selected_top4",
            "core_voxels",
            "core_gt_overlap_voxels",
            "core_gt_dice",
        ]

        print(
            gt_components[
                display_columns
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # Print Top-4
    # --------------------------------------------------------

    print("\nTop-4 components:")
    print("-" * 80)

    if len(selected) == 0:
        print("NONE")
    else:
        print(
            selected[
                [
                    "label",
                    "rank",
                    "component_voxels",
                    "z_span",
                    "mean_suv",
                    "max_suv",
                    "frac_ge_3",
                    "mean_suv_hot3_a0p5",
                    "gt_overlap_voxels",
                    "component_gt_dice",
                    "core_voxels",
                    "core_gt_overlap_voxels",
                    "core_gt_dice",
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # Print final metrics
    # --------------------------------------------------------

    print("\nPET-only final Dev10 metrics:")
    print("-" * 80)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    # --------------------------------------------------------
    # Case summary
    # --------------------------------------------------------

    case_summary = {
        "case": case_name,

        "pet_threshold": PET_THRESHOLD,

        "candidate_voxels": candidate_voxels,

        "num_3d_components": len(components),

        "num_gt_overlapping_components":
            total_gt_overlap_components,

        "num_gt_overlapping_top4":
            gt_selected_count,

        "best_gt_component_rank":
            best_gt_rank,

        "best_component_gt_dice":
            best_component_dice,

        "best_core_gt_dice":
            best_core_dice,

        "final_dice":
            safe_float(metrics.get("dice")),

        "final_iou":
            safe_float(metrics.get("iou")),

        "final_slice_recall":
            safe_float(metrics.get("slice_recall")),

        "final_fpr":
            safe_float(metrics.get("fpr")),

        "final_prediction_voxels":
            safe_int(metrics.get("prediction_voxels")),
    }

    return audit_df, case_summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("DEV35 - PET-ONLY COMPONENT / RANKING AUDIT")
    print("=" * 80)

    print("\nConfiguration:")
    print(f"PET threshold : {PET_THRESHOLD}")
    print(f"Ranking       : {RANKING_FEATURE}")
    print(f"Core policy   : {CORE_POLICY}")
    print(f"Top-K         : {TOP_K}")

    print("\nIMPORTANT:")
    print("- Dev10 is imported only.")
    print("- Dev10 source is NOT modified.")
    print("- GT is used only for diagnostic auditing.")
    print("- Ranking and selection do NOT use GT.")

    all_audits = []
    case_summaries = []

    # --------------------------------------------------------
    # Run all development cases
    # --------------------------------------------------------

    for case_name in CASES:

        try:
            audit_df, case_summary = evaluate_case(case_name)

            all_audits.append(audit_df)
            case_summaries.append(case_summary)

        except Exception as exc:

            print("\nERROR")
            print(f"Case: {case_name}")
            print(f"Reason: {exc}")

            import traceback
            traceback.print_exc()

    # --------------------------------------------------------
    # Combine component audits
    # --------------------------------------------------------

    if all_audits:

        component_audit_df = pd.concat(
            all_audits,
            ignore_index=True,
        )

    else:
        component_audit_df = pd.DataFrame()

    case_summary_df = pd.DataFrame(case_summaries)

    # --------------------------------------------------------
    # Macro summary
    # --------------------------------------------------------

    if len(case_summary_df) > 0:

        macro_summary = {
            "variant": "pet_only",

            "macro_final_dice":
                case_summary_df["final_dice"].mean(),

            "macro_final_iou":
                case_summary_df["final_iou"].mean(),

            "macro_final_slice_recall":
                case_summary_df["final_slice_recall"].mean(),

            "macro_final_fpr":
                case_summary_df["final_fpr"].mean(),

            "total_prediction_voxels":
                case_summary_df[
                    "final_prediction_voxels"
                ].sum(),

            "mean_candidate_voxels":
                case_summary_df[
                    "candidate_voxels"
                ].mean(),

            "mean_3d_components":
                case_summary_df[
                    "num_3d_components"
                ].mean(),

            "mean_gt_overlapping_components":
                case_summary_df[
                    "num_gt_overlapping_components"
                ].mean(),

            "mean_gt_overlapping_top4":
                case_summary_df[
                    "num_gt_overlapping_top4"
                ].mean(),
        }

        macro_summary_df = pd.DataFrame(
            [macro_summary]
        )

    else:
        macro_summary_df = pd.DataFrame()

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    component_path = (
        OUTPUT_DIR
        / "dev35_component_audit.csv"
    )

    case_path = (
        OUTPUT_DIR
        / "dev35_case_summary.csv"
    )

    macro_path = (
        OUTPUT_DIR
        / "dev35_macro_summary.csv"
    )

    component_audit_df.to_csv(
        component_path,
        index=False,
    )

    case_summary_df.to_csv(
        case_path,
        index=False,
    )

    macro_summary_df.to_csv(
        macro_path,
        index=False,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("DEV35 COMPLETE")
    print("=" * 80)

    print("\nSaved:")
    print(component_path)
    print(case_path)
    print(macro_path)

    print("\nCase summary:")

    if len(case_summary_df) > 0:

        print(
            case_summary_df[
                [
                    "case",
                    "num_3d_components",
                    "num_gt_overlapping_components",
                    "num_gt_overlapping_top4",
                    "best_gt_component_rank",
                    "best_component_gt_dice",
                    "best_core_gt_dice",
                    "final_dice",
                    "final_iou",
                    "final_slice_recall",
                    "final_fpr",
                ]
            ].to_string(index=False)
        )

    print("\nMacro summary:")

    if len(macro_summary_df) > 0:
        print(
            macro_summary_df.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()