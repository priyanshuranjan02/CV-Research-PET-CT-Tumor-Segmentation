"""
Dev40 - Per-Case Robustness Audit

Purpose
-------
Audit whether the Dev39 improvement is distributed across the six
development cases or dominated by one/few cases.

Comparison:
    1. Original Dev10
    2. Dev38: PET-only + PET Quality ranking
    3. Dev39: PET-only + PET Quality ranking + hot_fraction >= 0.40

Important
---------
- DO NOT modify dev10_lite_hot_core.py.
- Dev10 functions are imported and reused exactly.
- This script is an audit, not another parameter optimization.
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

PET_THRESHOLD = 2.25

CORE_POLICY = "P70"
TOP_K = 4

HOT_THRESHOLD = 3.0
HOT_FRACTION_GATE = 0.40

DEVELOPMENT_CASES = [
    "PETCT_0168f65af8",
    "PETCT_04606080a0",
    "PETCT_04ab5c61c9",
    "PETCT_0b57b247b6",
    "PETCT_11afab3485",
    "PETCT_185da4c8b6",
]

DEV10_RANKING = "mean_suv_hot3_a0p5"

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "development_cases"
    / "dev40_robustness_audit"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER: SAFE NUMERIC
# ============================================================

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


# ============================================================
# PET-ONLY CANDIDATE
# ============================================================

def generate_pet_only_volume(case):
    """
    PET-only candidate used in Dev38/Dev39.

    No CT contour gate.
    Candidate = SUV >= 2.25.
    """

    suv = case["suv"]

    return (suv >= PET_THRESHOLD).astype(np.uint8)


# ============================================================
# COMPONENT FEATURE EXTRACTION
# ============================================================

def add_pet_quality_features(case, components):
    """
    Recreate the Dev38 PET-quality features.

    The features are calculated from each 3D component.
    """

    suv = case["suv"]

    rows = []

    for _, row in components.iterrows():

        label = int(row["label"])

        # Component mask
        # The component label is interpreted using the exact
        # connected-component volume from Dev10.
        component_mask = row["_component_mask"]

        coords = np.argwhere(component_mask)

        if len(coords) == 0:
            continue

        z_values = coords[:, 0]

        z_min = int(z_values.min())
        z_max = int(z_values.max())

        z_span = z_max - z_min + 1

        unique_z = np.unique(z_values)
        slice_count = len(unique_z)

        # Longest consecutive z-run
        longest_run = 1
        current_run = 1

        for i in range(1, len(unique_z)):
            if unique_z[i] == unique_z[i - 1] + 1:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 1

        persistence = slice_count / max(z_span, 1)

        voxel_count = int(component_mask.sum())

        z0, z1 = z_min, z_max + 1
        y0, y1 = int(coords[:, 1].min()), int(coords[:, 1].max()) + 1
        x0, x1 = int(coords[:, 2].min()), int(coords[:, 2].max()) + 1

        bbox_volume = max(
            (z1 - z0) * (y1 - y0) * (x1 - x0),
            1
        )

        compactness = voxel_count / bbox_volume

        suv_values = suv[component_mask]

        mean_suv = float(np.mean(suv_values))
        max_suv = float(np.max(suv_values))

        hot_fraction = float(
            np.mean(suv_values >= HOT_THRESHOLD)
        )

        rows.append({
            "label": label,
            "voxels": voxel_count,
            "z_span": z_span,
            "slice_count": slice_count,
            "longest_run": longest_run,
            "persistence": persistence,
            "compactness": compactness,
            "mean_suv": mean_suv,
            "max_suv": max_suv,
            "hot_fraction": hot_fraction,
        })

    feature_df = pd.DataFrame(rows)

    if feature_df.empty:
        return feature_df

    return feature_df


# ============================================================
# NORMALIZATION
# ============================================================

def minmax_normalize(series):
    """
    Robust min-max normalization.
    """

    series = series.astype(float)

    min_value = series.min()
    max_value = series.max()

    if not np.isfinite(min_value) or not np.isfinite(max_value):
        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    if max_value - min_value < 1e-12:
        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    return (series - min_value) / (max_value - min_value)


# ============================================================
# DEV38 PET QUALITY RANKING
# ============================================================

def build_pet_quality_ranking(case, components):
    """
    Recreate the Dev38 PET-quality ranking.

    Dev38 score:

        0.25 * mean SUV
      + 0.15 * max SUV
      + 0.15 * hot fraction
      + 0.20 * persistence
      + 0.15 * longest run
      + 0.10 * compactness
    """

    if components.empty:
        return components.copy()

    feature_df = add_pet_quality_features(
        case,
        components
    )

    if feature_df.empty:
        return components.copy()

    for column in [
        "mean_suv",
        "max_suv",
        "hot_fraction",
        "persistence",
        "longest_run",
        "compactness",
    ]:
        feature_df[f"n_{column}"] = minmax_normalize(
            feature_df[column]
        )

    feature_df["pet_quality_score"] = (
        0.25 * feature_df["n_mean_suv"]
        + 0.15 * feature_df["n_max_suv"]
        + 0.15 * feature_df["n_hot_fraction"]
        + 0.20 * feature_df["n_persistence"]
        + 0.15 * feature_df["n_longest_run"]
        + 0.10 * feature_df["n_compactness"]
    )

    merged = components.copy()

    merged = merged.drop(
        columns=[
            "voxels",
            "mean_suv",
            "max_suv",
        ],
        errors="ignore"
    )

    merged = merged.merge(
        feature_df,
        on="label",
        how="left"
    )

    return merged


# ============================================================
# COMPONENT MASKS
# ============================================================

def attach_component_masks(components, cc):
    """
    Attach the actual 3D component mask to every component row.

    The mask is kept in a private column and is not written
    directly to CSV.
    """

    components = components.copy()

    masks = []

    for _, row in components.iterrows():
        label = int(row["label"])
        masks.append(cc == label)

    components["_component_mask"] = masks

    return components


# ============================================================
# GT OVERLAP AUDIT
# ============================================================

def calculate_component_gt_metrics(component_mask, gt):
    """
    Calculate component-level GT overlap.
    """

    component_mask = component_mask.astype(bool)
    gt = gt.astype(bool)

    intersection = int(
        np.logical_and(component_mask, gt).sum()
    )

    pred_voxels = int(component_mask.sum())
    gt_voxels = int(gt.sum())

    dice_den = pred_voxels + gt_voxels

    if dice_den == 0:
        dice = 1.0
    else:
        dice = (
            2.0 * intersection / dice_den
        )

    union = int(
        np.logical_or(component_mask, gt).sum()
    )

    if union == 0:
        iou = 1.0
    else:
        iou = intersection / union

    return {
        "gt_overlap_voxels": intersection,
        "component_dice": dice,
        "component_iou": iou,
    }


# ============================================================
# CORE EXTRACTION AUDIT
# ============================================================

def safe_build_core(case, cc, label, policy):
    """
    Handle the Dev10 build_core return type safely.

    Dev10 may return a tuple containing the core mask.
    """

    result = dev10.build_core(
        cc,
        case["suv"],
        int(label),
        policy
    )

    if isinstance(result, tuple):

        for item in result:
            if isinstance(item, np.ndarray):
                return item.astype(bool)

        raise RuntimeError(
            "build_core returned a tuple but no ndarray "
            "core mask was found."
        )

    if isinstance(result, np.ndarray):
        return result.astype(bool)

    raise RuntimeError(
        f"Unexpected build_core return type: "
        f"{type(result)}"
    )


# ============================================================
# EVALUATION HELPER
# ============================================================

def evaluate_exact(case, components, cc):
    """
    Use the exact Dev10 evaluation implementation.
    """

    if components.empty:
        return {
            "dice": 0.0,
            "iou": 0.0,
            "slice_recall": 0.0,
            "fpr": 0.0,
            "prediction_voxels": 0,
        }

    evaluation_components = components.copy()

    evaluation_components = evaluation_components.drop(
        columns=["_component_mask"],
        errors="ignore"
    )

    metrics = dev10.evaluate_selection(
        case,
        evaluation_components,
        cc,
        "pet_quality_score",
        CORE_POLICY,
        TOP_K
    )

    return {
        "dice": safe_float(metrics["dice"]),
        "iou": safe_float(metrics["iou"]),
        "slice_recall": safe_float(
            metrics["slice_recall"]
        ),
        "fpr": safe_float(metrics["fpr"]),
        "prediction_voxels": int(
            metrics["prediction_voxels"]
        ),
    }


# ============================================================
# SELECTED COMPONENT AUDIT
# ============================================================

def audit_selected_components(
    case,
    ranked_components,
    cc,
    gt,
    case_name,
    method_name,
):
    """
    Determine which components were selected by Top-K
    and whether they overlap GT.
    """

    rows = []

    ranked = ranked_components.sort_values(
        ["pet_quality_score", "mean_suv"],
        ascending=[False, False]
    ).reset_index(drop=True)

    selected = ranked.head(TOP_K)

    for rank, (_, row) in enumerate(
        selected.iterrows(),
        start=1
    ):

        label = int(row["label"])

        component_mask = cc == label

        gt_metrics = calculate_component_gt_metrics(
            component_mask,
            gt
        )

        core_mask = safe_build_core(
            case,
            cc,
            label,
            CORE_POLICY
        )

        core_metrics = calculate_component_gt_metrics(
            core_mask,
            gt
        )

        rows.append({
            "case": case_name,
            "method": method_name,
            "rank": rank,
            "label": label,
            "component_voxels": int(
                component_mask.sum()
            ),
            "mean_suv": safe_float(
                row["mean_suv"]
            ),
            "max_suv": safe_float(
                row["max_suv"]
            ),
            "hot_fraction": safe_float(
                row["hot_fraction"]
            ),
            "persistence": safe_float(
                row["persistence"]
            ),
            "longest_run": safe_float(
                row["longest_run"]
            ),
            "compactness": safe_float(
                row["compactness"]
            ),
            "pet_quality_score": safe_float(
                row["pet_quality_score"]
            ),
            "component_gt_overlap": gt_metrics[
                "gt_overlap_voxels"
            ],
            "component_dice": gt_metrics[
                "component_dice"
            ],
            "component_iou": gt_metrics[
                "component_iou"
            ],
            "core_voxels": int(
                core_mask.sum()
            ),
            "core_gt_overlap": core_metrics[
                "gt_overlap_voxels"
            ],
            "core_dice": core_metrics[
                "component_dice"
            ],
            "core_iou": core_metrics[
                "component_iou"
            ],
        })

    return rows


# ============================================================
# CASE PROCESSING
# ============================================================

def process_case(case_name):

    print("\n" + "=" * 80)
    print(f"DEV40 CASE: {case_name}")
    print("=" * 80)

    case = dev10.load_case(case_name)

    suv = case["suv"]
    gt = case["gt"]

    # --------------------------------------------------------
    # ORIGINAL DEV10
    # --------------------------------------------------------

    print("Running original Dev10...")

    dev10_volume = dev10.generate_candidate_volume(
        case
    )

    dev10_components, dev10_cc = (
        dev10.build_3d_component_table(
            case,
            dev10_volume
        )
    )

    dev10_metrics = dev10.evaluate_selection(
        case,
        dev10_components,
        dev10_cc,
        DEV10_RANKING,
        CORE_POLICY,
        TOP_K
    )

    # --------------------------------------------------------
    # PET-ONLY
    # --------------------------------------------------------

    print("Building PET-only candidate...")

    pet_volume = generate_pet_only_volume(case)

    pet_components, pet_cc = (
        dev10.build_3d_component_table(
            case,
            pet_volume
        )
    )

    pet_components = attach_component_masks(
        pet_components,
        pet_cc
    )

    # --------------------------------------------------------
    # DEV38
    # --------------------------------------------------------

    print("Running Dev38 PET Quality ranking...")

    ranked = build_pet_quality_ranking(
        case,
        pet_components
    )

    ranked = ranked.sort_values(
        ["pet_quality_score", "mean_suv"],
        ascending=[False, False]
    ).reset_index(drop=True)

    dev38_components = ranked.copy()

    dev38_metrics = evaluate_exact(
        case,
        dev38_components,
        pet_cc
    )

    # --------------------------------------------------------
    # DEV39
    # --------------------------------------------------------

    print("Applying Dev39 hot-fraction gate...")

    dev39_components = ranked[
        ranked["hot_fraction"] >= HOT_FRACTION_GATE
    ].copy()

    dev39_components = dev39_components.sort_values(
        ["pet_quality_score", "mean_suv"],
        ascending=[False, False]
    ).reset_index(drop=True)

    dev39_metrics = evaluate_exact(
        case,
        dev39_components,
        pet_cc
    )

    # --------------------------------------------------------
    # SELECTED COMPONENT AUDITS
    # --------------------------------------------------------

    component_rows = []

    component_rows.extend(
        audit_selected_components(
            case,
            dev38_components,
            pet_cc,
            gt,
            case_name,
            "Dev38"
        )
    )

    component_rows.extend(
        audit_selected_components(
            case,
            dev39_components,
            pet_cc,
            gt,
            case_name,
            "Dev39"
        )
    )

    # --------------------------------------------------------
    # CASE SUMMARY
    # --------------------------------------------------------

    summary = {
        "case": case_name,

        "dev10_dice": safe_float(
            dev10_metrics["dice"]
        ),
        "dev10_iou": safe_float(
            dev10_metrics["iou"]
        ),
        "dev10_slice_recall": safe_float(
            dev10_metrics["slice_recall"]
        ),
        "dev10_fpr": safe_float(
            dev10_metrics["fpr"]
        ),
        "dev10_prediction_voxels": int(
            dev10_metrics["prediction_voxels"]
        ),

        "dev38_dice": safe_float(
            dev38_metrics["dice"]
        ),
        "dev38_iou": safe_float(
            dev38_metrics["iou"]
        ),
        "dev38_slice_recall": safe_float(
            dev38_metrics["slice_recall"]
        ),
        "dev38_fpr": safe_float(
            dev38_metrics["fpr"]
        ),
        "dev38_prediction_voxels": int(
            dev38_metrics["prediction_voxels"]
        ),

        "dev39_dice": safe_float(
            dev39_metrics["dice"]
        ),
        "dev39_iou": safe_float(
            dev39_metrics["iou"]
        ),
        "dev39_slice_recall": safe_float(
            dev39_metrics["slice_recall"]
        ),
        "dev39_fpr": safe_float(
            dev39_metrics["fpr"]
        ),
        "dev39_prediction_voxels": int(
            dev39_metrics["prediction_voxels"]
        ),

        "dice_delta_dev39_vs_dev10":
            safe_float(dev39_metrics["dice"])
            - safe_float(dev10_metrics["dice"]),

        "dice_delta_dev39_vs_dev38":
            safe_float(dev39_metrics["dice"])
            - safe_float(dev38_metrics["dice"]),

        "fpr_delta_dev39_vs_dev10":
            safe_float(dev39_metrics["fpr"])
            - safe_float(dev10_metrics["fpr"]),

        "fpr_delta_dev39_vs_dev38":
            safe_float(dev39_metrics["fpr"])
            - safe_float(dev38_metrics["fpr"]),

        "prediction_voxel_delta_dev39_vs_dev10":
            int(dev39_metrics["prediction_voxels"])
            - int(dev10_metrics["prediction_voxels"]),

        "prediction_voxel_delta_dev39_vs_dev38":
            int(dev39_metrics["prediction_voxels"])
            - int(dev38_metrics["prediction_voxels"]),

        "dev38_eligible_components":
            int(len(dev38_components)),

        "dev39_eligible_components":
            int(len(dev39_components)),
    }

    print(
        f"Dev10 Dice = {summary['dev10_dice']:.6f}"
    )
    print(
        f"Dev38 Dice = {summary['dev38_dice']:.6f}"
    )
    print(
        f"Dev39 Dice = {summary['dev39_dice']:.6f}"
    )

    print(
        f"Dev39 Δ vs Dev10 = "
        f"{summary['dice_delta_dev39_vs_dev10']:+.6f}"
    )

    return summary, component_rows


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("DEV40 - PER-CASE ROBUSTNESS AUDIT")
    print("=" * 80)

    print(f"PET threshold       : {PET_THRESHOLD}")
    print(f"Hot threshold       : {HOT_THRESHOLD}")
    print(f"Hot fraction gate   : {HOT_FRACTION_GATE}")
    print(f"Core policy         : {CORE_POLICY}")
    print(f"Top-K               : {TOP_K}")
    print(f"Cases               : {len(DEVELOPMENT_CASES)}")
    print()
    print("Dev10 source remains untouched.")

    case_results = []
    component_results = []

    for case_name in DEVELOPMENT_CASES:

        try:

            summary, components = process_case(
                case_name
            )

            case_results.append(summary)
            component_results.extend(components)

        except Exception as exc:

            print(
                f"\nERROR in {case_name}: "
                f"{type(exc).__name__}: {exc}"
            )

    # --------------------------------------------------------
    # SAVE CASE RESULTS
    # --------------------------------------------------------

    case_df = pd.DataFrame(case_results)

    case_csv = RESULT_DIR / "dev40_case_summary.csv"

    case_df.to_csv(
        case_csv,
        index=False
    )

    # --------------------------------------------------------
    # SAVE COMPONENT AUDIT
    # --------------------------------------------------------

    component_df = pd.DataFrame(
        component_results
    )

    component_csv = (
        RESULT_DIR
        / "dev40_selected_component_audit.csv"
    )

    component_df.to_csv(
        component_csv,
        index=False
    )

    # --------------------------------------------------------
    # MACRO SUMMARY
    # --------------------------------------------------------

    if not case_df.empty:

        macro_rows = []

        for method in ["dev10", "dev38", "dev39"]:

            macro_rows.append({
                "method": method,

                "macro_dice": case_df[
                    f"{method}_dice"
                ].mean(),

                "macro_iou": case_df[
                    f"{method}_iou"
                ].mean(),

                "macro_slice_recall": case_df[
                    f"{method}_slice_recall"
                ].mean(),

                "macro_fpr": case_df[
                    f"{method}_fpr"
                ].mean(),

                "total_prediction_voxels": case_df[
                    f"{method}_prediction_voxels"
                ].sum(),

                "mean_prediction_voxels": case_df[
                    f"{method}_prediction_voxels"
                ].mean(),
            })

        macro_df = pd.DataFrame(
            macro_rows
        )

        macro_csv = (
            RESULT_DIR
            / "dev40_macro_summary.csv"
        )

        macro_df.to_csv(
            macro_csv,
            index=False
        )

        # ----------------------------------------------------
        # ROBUSTNESS STATISTICS
        # ----------------------------------------------------

        dev39_better_dice = int(
            (
                case_df["dev39_dice"]
                > case_df["dev10_dice"]
            ).sum()
        )

        dev39_better_or_equal_dice = int(
            (
                case_df["dev39_dice"]
                >= case_df["dev10_dice"]
            ).sum()
        )

        dev39_lower_fpr = int(
            (
                case_df["dev39_fpr"]
                < case_df["dev10_fpr"]
            ).sum()
        )

        dev39_better_vs_dev38 = int(
            (
                case_df["dev39_dice"]
                > case_df["dev38_dice"]
            ).sum()
        )

        print("\n" + "=" * 80)
        print("DEV40 MACRO RESULTS")
        print("=" * 80)

        print(
            macro_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}"
            )
        )

        print("\n" + "=" * 80)
        print("ROBUSTNESS SUMMARY")
        print("=" * 80)

        print(
            f"Dev39 better Dice than Dev10: "
            f"{dev39_better_dice}/{len(case_df)} cases"
        )

        print(
            f"Dev39 >= Dev10 Dice: "
            f"{dev39_better_or_equal_dice}/{len(case_df)} cases"
        )

        print(
            f"Dev39 lower FPR than Dev10: "
            f"{dev39_lower_fpr}/{len(case_df)} cases"
        )

        print(
            f"Dev39 better Dice than Dev38: "
            f"{dev39_better_vs_dev38}/{len(case_df)} cases"
        )

        # ----------------------------------------------------
        # BEST / WORST CASES
        # ----------------------------------------------------

        best_case = case_df.loc[
            case_df["dice_delta_dev39_vs_dev10"].idxmax()
        ]

        worst_case = case_df.loc[
            case_df["dice_delta_dev39_vs_dev10"].idxmin()
        ]

        print("\n" + "=" * 80)
        print("CASE CONTRIBUTION")
        print("=" * 80)

        print(
            f"Best Dev39 improvement: "
            f"{best_case['case']} "
            f"({best_case['dice_delta_dev39_vs_dev10']:+.6f})"
        )

        print(
            f"Worst Dev39 change: "
            f"{worst_case['case']} "
            f"({worst_case['dice_delta_dev39_vs_dev10']:+.6f})"
        )

        # ----------------------------------------------------
        # FINAL INTERPRETATION
        # ----------------------------------------------------

        print("\n" + "=" * 80)
        print("DEV40 INTERPRETATION")
        print("=" * 80)

        if dev39_better_dice >= 4:

            print(
                "RESULT: STRONG ROBUSTNESS"
            )
            print(
                "Dev39 improves Dice on most development cases."
            )
            print(
                "Proceed toward freezing the method and "
                "validation/generalization testing."
            )

        elif dev39_better_dice >= 3:

            print(
                "RESULT: MODERATE ROBUSTNESS"
            )
            print(
                "Dev39 improves at least half of the cases."
            )
            print(
                "Inspect the case-level component audit before "
                "freezing the method."
            )

        else:

            print(
                "RESULT: LIMITED ROBUSTNESS"
            )
            print(
                "The macro improvement may be concentrated "
                "in a small number of cases."
            )
            print(
                "Do not freeze the method yet."
            )

        print("\nOutput files:")
        print(f"  {case_csv}")
        print(f"  {component_csv}")
        print(f"  {macro_csv}")


if __name__ == "__main__":
    main()