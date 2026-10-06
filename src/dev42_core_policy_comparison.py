"""
Dev42 - Core Policy Comparison

Purpose
-------
Evaluate different SUV-percentile core policies while keeping
the entire Dev39 pipeline fixed.

Fixed Dev39 pipeline:
    PET-only candidate
    SUV >= 2.25
    Exact Dev10 3D connected components
    PET Quality ranking
    hot_fraction >= 0.40
    Top-K = 4

Variable:
    Core policy = P50, P60, P70, P80, P90

Important:
    dev10_lite_hot_core.py is imported only.
    It is NEVER modified.
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

HOT_THRESHOLD = 3.0
HOT_FRACTION_GATE = 0.40

TOP_K = 4

CORE_POLICIES = [
    "P50",
    "P60",
    "P70",
    "P80",
    "P90",
]

DEVELOPMENT_CASES = [
    "PETCT_0168f65af8",
    "PETCT_04606080a0",
    "PETCT_04ab5c61c9",
    "PETCT_0b57b247b6",
    "PETCT_11afab3485",
    "PETCT_185da4c8b6",
]

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "development_cases"
    / "dev42_core_policy_comparison"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def minmax(series):
    series = series.astype(float)

    lo = series.min()
    hi = series.max()

    if not np.isfinite(lo) or not np.isfinite(hi):
        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    if hi - lo < 1e-12:
        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    return (series - lo) / (hi - lo)


# ============================================================
# PET-ONLY CANDIDATE
# ============================================================

def generate_pet_only_volume(case):

    suv = case["suv"]

    return (
        suv >= PET_THRESHOLD
    ).astype(np.uint8)


# ============================================================
# COMPONENT FEATURES
# ============================================================

def extract_component_features(
    case,
    components,
    cc
):

    suv = case["suv"]

    rows = []

    for _, row in components.iterrows():

        label = int(row["label"])

        mask = cc == label

        coords = np.argwhere(mask)

        if len(coords) == 0:
            continue

        # ----------------------------------------------------
        # Z-axis structure
        # ----------------------------------------------------

        z_values = coords[:, 0]

        z_min = int(z_values.min())
        z_max = int(z_values.max())

        z_span = (
            z_max - z_min + 1
        )

        unique_z = np.unique(
            z_values
        )

        slice_count = len(
            unique_z
        )

        longest_run = 1
        current_run = 1

        for i in range(
            1,
            len(unique_z)
        ):

            if (
                unique_z[i]
                == unique_z[i - 1] + 1
            ):

                current_run += 1

                longest_run = max(
                    longest_run,
                    current_run
                )

            else:

                current_run = 1

        persistence = (
            slice_count
            / max(z_span, 1)
        )

        # ----------------------------------------------------
        # Component volume
        # ----------------------------------------------------

        voxel_count = int(
            mask.sum()
        )

        # ----------------------------------------------------
        # Bounding box / compactness
        # ----------------------------------------------------

        y_min = int(
            coords[:, 1].min()
        )

        y_max = int(
            coords[:, 1].max()
        )

        x_min = int(
            coords[:, 2].min()
        )

        x_max = int(
            coords[:, 2].max()
        )

        bbox_volume = max(
            (z_max - z_min + 1)
            * (y_max - y_min + 1)
            * (x_max - x_min + 1),
            1
        )

        compactness = (
            voxel_count
            / bbox_volume
        )

        # ----------------------------------------------------
        # PET statistics
        # ----------------------------------------------------

        suv_values = suv[mask]

        mean_suv = float(
            np.mean(suv_values)
        )

        max_suv = float(
            np.max(suv_values)
        )

        hot_fraction = float(
            np.mean(
                suv_values
                >= HOT_THRESHOLD
            )
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

    return pd.DataFrame(rows)


# ============================================================
# DEV39 PET QUALITY RANKING
# ============================================================

def build_dev39_ranking(
    case,
    components,
    cc
):

    features = extract_component_features(
        case,
        components,
        cc
    )

    if features.empty:
        return features

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    features["n_mean_suv"] = minmax(
        features["mean_suv"]
    )

    features["n_max_suv"] = minmax(
        features["max_suv"]
    )

    features["n_hot_fraction"] = minmax(
        features["hot_fraction"]
    )

    features["n_persistence"] = minmax(
        features["persistence"]
    )

    features["n_longest_run"] = minmax(
        features["longest_run"]
    )

    features["n_compactness"] = minmax(
        features["compactness"]
    )

    # --------------------------------------------------------
    # Exact Dev38/Dev39 PET quality score
    # --------------------------------------------------------

    features["pet_quality_score"] = (
        0.25
        * features["n_mean_suv"]

        + 0.15
        * features["n_max_suv"]

        + 0.15
        * features["n_hot_fraction"]

        + 0.20
        * features["n_persistence"]

        + 0.15
        * features["n_longest_run"]

        + 0.10
        * features["n_compactness"]
    )

    # --------------------------------------------------------
    # Dev39 hot-fraction gate
    # --------------------------------------------------------

    features = features[
        features["hot_fraction"]
        >= HOT_FRACTION_GATE
    ].copy()

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    features = features.sort_values(
        [
            "pet_quality_score",
            "mean_suv"
        ],
        ascending=[
            False,
            False
        ]
    ).reset_index(drop=True)

    return features


# ============================================================
# EXACT DEV10 EVALUATION
# ============================================================

def evaluate_policy(
    case,
    ranked_components,
    cc,
    core_policy
):

    if ranked_components.empty:

        return {
            "dice": 0.0,
            "iou": 0.0,
            "slice_recall": 0.0,
            "fpr": 0.0,
            "prediction_voxels": 0,
        }

    metrics = dev10.evaluate_selection(
        case,
        ranked_components,
        cc,
        "pet_quality_score",
        core_policy,
        TOP_K
    )

    return {
        "dice": safe_float(
            metrics["dice"]
        ),

        "iou": safe_float(
            metrics["iou"]
        ),

        "slice_recall": safe_float(
            metrics["slice_recall"]
        ),

        "fpr": safe_float(
            metrics["fpr"]
        ),

        "prediction_voxels": int(
            metrics["prediction_voxels"]
        ),
    }


# ============================================================
# SAFE CORE EXTRACTION
# ============================================================

def safe_build_core(
    case,
    cc,
    label,
    policy
):

    result = dev10.build_core(
        cc,
        case["suv"],
        int(label),
        policy
    )

    # Dev10 may return multiple values.
    if isinstance(result, tuple):

        for item in result:

            if isinstance(
                item,
                np.ndarray
            ):

                return item.astype(bool)

        raise RuntimeError(
            "build_core returned a tuple "
            "but no ndarray was found."
        )

    if isinstance(
        result,
        np.ndarray
    ):

        return result.astype(bool)

    raise RuntimeError(
        "Unexpected build_core return type: "
        f"{type(result)}"
    )


# ============================================================
# COMPONENT / CORE AUDIT
# ============================================================

def audit_components(
    case,
    ranked,
    cc,
    gt,
    case_name,
    policy
):

    rows = []

    ranked = ranked.sort_values(
        [
            "pet_quality_score",
            "mean_suv"
        ],
        ascending=[
            False,
            False
        ]
    ).reset_index(drop=True)

    selected = ranked.head(
        TOP_K
    )

    gt_bool = gt.astype(bool)

    for rank, (_, row) in enumerate(
        selected.iterrows(),
        start=1
    ):

        label = int(
            row["label"]
        )

        component_mask = (
            cc == label
        )

        component_voxels = int(
            component_mask.sum()
        )

        component_overlap = int(
            np.logical_and(
                component_mask,
                gt_bool
            ).sum()
        )

        gt_voxels = int(
            gt_bool.sum()
        )

        union = int(
            np.logical_or(
                component_mask,
                gt_bool
            ).sum()
        )

        if (
            component_voxels
            + gt_voxels
            == 0
        ):

            component_dice = 1.0

        else:

            component_dice = (
                2.0
                * component_overlap
                / (
                    component_voxels
                    + gt_voxels
                )
            )

        if union == 0:
            component_iou = 1.0
        else:
            component_iou = (
                component_overlap
                / union
            )

        # ----------------------------------------------------
        # Core for every policy will be added separately
        # by the caller.
        # ----------------------------------------------------

        rows.append({
            "case": case_name,
            "core_policy": policy,
            "rank": rank,
            "label": label,

            "component_voxels":
                component_voxels,

            "mean_suv":
                safe_float(
                    row["mean_suv"]
                ),

            "max_suv":
                safe_float(
                    row["max_suv"]
                ),

            "hot_fraction":
                safe_float(
                    row["hot_fraction"]
                ),

            "persistence":
                safe_float(
                    row["persistence"]
                ),

            "longest_run":
                safe_float(
                    row["longest_run"]
                ),

            "compactness":
                safe_float(
                    row["compactness"]
                ),

            "pet_quality_score":
                safe_float(
                    row["pet_quality_score"]
                ),

            "component_gt_overlap":
                component_overlap,

            "component_dice":
                component_dice,

            "component_iou":
                component_iou,
        })

    return rows


# ============================================================
# CORE AUDIT FOR ONE POLICY
# ============================================================

def add_core_metrics(
    case,
    cc,
    gt,
    audit_rows
):

    gt_bool = gt.astype(bool)

    for row in audit_rows:

        label = int(
            row["label"]
        )

        policy = row[
            "core_policy"
        ]

        core_mask = safe_build_core(
            case,
            cc,
            label,
            policy
        )

        core_voxels = int(
            core_mask.sum()
        )

        core_overlap = int(
            np.logical_and(
                core_mask,
                gt_bool
            ).sum()
        )

        gt_voxels = int(
            gt_bool.sum()
        )

        union = int(
            np.logical_or(
                core_mask,
                gt_bool
            ).sum()
        )

        if (
            core_voxels
            + gt_voxels
            == 0
        ):

            core_dice = 1.0

        else:

            core_dice = (
                2.0
                * core_overlap
                / (
                    core_voxels
                    + gt_voxels
                )
            )

        if union == 0:
            core_iou = 1.0
        else:
            core_iou = (
                core_overlap
                / union
            )

        row[
            "core_voxels"
        ] = core_voxels

        row[
            "core_gt_overlap"
        ] = core_overlap

        row[
            "core_dice"
        ] = core_dice

        row[
            "core_iou"
        ] = core_iou

    return audit_rows


# ============================================================
# PROCESS ONE CASE
# ============================================================

def process_case(case_name):

    print("\n" + "=" * 80)
    print(
        f"DEV42 CASE: {case_name}"
    )
    print("=" * 80)

    case = dev10.load_case(
        case_name
    )

    gt = case["gt"]

    # --------------------------------------------------------
    # PET-only candidate
    # --------------------------------------------------------

    pet_volume = (
        generate_pet_only_volume(
            case
        )
    )

    components, cc = (
        dev10.build_3d_component_table(
            case,
            pet_volume
        )
    )

    print(
        f"Initial 3D components: "
        f"{len(components)}"
    )

    # --------------------------------------------------------
    # Exact Dev39 ranking + gate
    # --------------------------------------------------------

    ranked = build_dev39_ranking(
        case,
        components,
        cc
    )

    print(
        f"Eligible components after "
        f"Dev39 gate: {len(ranked)}"
    )

    case_results = []
    audit_results = []

    # --------------------------------------------------------
    # Test every core policy
    # --------------------------------------------------------

    for core_policy in CORE_POLICIES:

        metrics = evaluate_policy(
            case,
            ranked,
            cc,
            core_policy
        )

        case_results.append({
            "case": case_name,
            "core_policy": core_policy,

            "dice":
                metrics["dice"],

            "iou":
                metrics["iou"],

            "slice_recall":
                metrics["slice_recall"],

            "fpr":
                metrics["fpr"],

            "prediction_voxels":
                metrics[
                    "prediction_voxels"
                ],

            "eligible_components":
                len(ranked),

            "top1_label":
                (
                    int(
                        ranked.iloc[0]["label"]
                    )
                    if len(ranked) >= 1
                    else -1
                ),

            "top2_label":
                (
                    int(
                        ranked.iloc[1]["label"]
                    )
                    if len(ranked) >= 2
                    else -1
                ),

            "top3_label":
                (
                    int(
                        ranked.iloc[2]["label"]
                    )
                    if len(ranked) >= 3
                    else -1
                ),

            "top4_label":
                (
                    int(
                        ranked.iloc[3]["label"]
                    )
                    if len(ranked) >= 4
                    else -1
                ),
        })

        rows = audit_components(
            case,
            ranked,
            cc,
            gt,
            case_name,
            core_policy
        )

        rows = add_core_metrics(
            case,
            cc,
            gt,
            rows
        )

        audit_results.extend(
            rows
        )

        print(
            f"{core_policy:>4s}  "
            f"Dice={metrics['dice']:.6f}  "
            f"IoU={metrics['iou']:.6f}  "
            f"Recall={metrics['slice_recall']:.6f}  "
            f"FPR={metrics['fpr']:.6f}  "
            f"Pred={metrics['prediction_voxels']}"
        )

    return (
        case_results,
        audit_results
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "DEV42 - CORE POLICY COMPARISON"
    )
    print("=" * 80)

    print()
    print(
        "Fixed Dev39 configuration:"
    )

    print(
        f"PET threshold     : "
        f"{PET_THRESHOLD}"
    )

    print(
        f"Hot threshold     : "
        f"{HOT_THRESHOLD}"
    )

    print(
        f"Hot fraction gate : "
        f"{HOT_FRACTION_GATE}"
    )

    print(
        f"Top-K             : "
        f"{TOP_K}"
    )

    print(
        f"Core policies     : "
        f"{CORE_POLICIES}"
    )

    print()
    print(
        "Only the core policy changes."
    )

    print(
        "dev10_lite_hot_core.py remains untouched."
    )

    all_case_results = []
    all_audit_results = []

    # --------------------------------------------------------
    # Run all cases
    # --------------------------------------------------------

    for case_name in DEVELOPMENT_CASES:

        try:

            case_results, audit_results = (
                process_case(
                    case_name
                )
            )

            all_case_results.extend(
                case_results
            )

            all_audit_results.extend(
                audit_results
            )

        except Exception as exc:

            print(
                f"\nERROR in {case_name}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    # --------------------------------------------------------
    # Save case results
    # --------------------------------------------------------

    case_df = pd.DataFrame(
        all_case_results
    )

    case_csv = (
        RESULT_DIR
        / "dev42_case_results.csv"
    )

    case_df.to_csv(
        case_csv,
        index=False
    )

    # --------------------------------------------------------
    # Save component/core audit
    # --------------------------------------------------------

    audit_df = pd.DataFrame(
        all_audit_results
    )

    audit_csv = (
        RESULT_DIR
        / "dev42_component_core_audit.csv"
    )

    audit_df.to_csv(
        audit_csv,
        index=False
    )

    # --------------------------------------------------------
    # Macro summary
    # --------------------------------------------------------

    if case_df.empty:

        print(
            "\nNo results generated."
        )

        return

    macro_rows = []

    for policy in CORE_POLICIES:

        subset = case_df[
            case_df[
                "core_policy"
            ] == policy
        ]

        macro_rows.append({
            "core_policy": policy,

            "macro_dice":
                subset["dice"].mean(),

            "macro_iou":
                subset["iou"].mean(),

            "macro_slice_recall":
                subset[
                    "slice_recall"
                ].mean(),

            "macro_fpr":
                subset["fpr"].mean(),

            "total_prediction_voxels":
                subset[
                    "prediction_voxels"
                ].sum(),

            "mean_prediction_voxels":
                subset[
                    "prediction_voxels"
                ].mean(),

            "mean_eligible_components":
                subset[
                    "eligible_components"
                ].mean(),
        })

    macro_df = pd.DataFrame(
        macro_rows
    )

    macro_csv = (
        RESULT_DIR
        / "dev42_macro_summary.csv"
    )

    macro_df.to_csv(
        macro_csv,
        index=False
    )

    # ========================================================
    # PRINT MACRO
    # ========================================================

    print("\n" + "=" * 80)
    print(
        "DEV42 MACRO RESULTS"
    )
    print("=" * 80)

    print(
        macro_df.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.6f}"
        )
    )

    # ========================================================
    # DELTAS VS P70
    # ========================================================

    p70 = macro_df[
        macro_df[
            "core_policy"
        ] == "P70"
    ]

    if not p70.empty:

        p70_dice = float(
            p70.iloc[0]["macro_dice"]
        )

        p70_fpr = float(
            p70.iloc[0]["macro_fpr"]
        )

        comparison = macro_df.copy()

        comparison[
            "dice_delta_vs_P70"
        ] = (
            comparison["macro_dice"]
            - p70_dice
        )

        comparison[
            "fpr_delta_vs_P70"
        ] = (
            comparison["macro_fpr"]
            - p70_fpr
        )

        print(
            "\n" + "=" * 80
        )

        print(
            "DEV42 DELTAS VS P70"
        )

        print(
            "=" * 80
        )

        print(
            comparison[
                [
                    "core_policy",
                    "dice_delta_vs_P70",
                    "fpr_delta_vs_P70",
                    "total_prediction_voxels",
                ]
            ].to_string(
                index=False,
                float_format=lambda x:
                f"{x:+.6f}"
            )
        )

    # ========================================================
    # BEST DICE
    # ========================================================

    best_idx = macro_df[
        "macro_dice"
    ].idxmax()

    best = macro_df.loc[
        best_idx
    ]

    print("\n" + "=" * 80)
    print(
        "BEST CORE POLICY BY DICE"
    )
    print("=" * 80)

    print(
        f"Policy : "
        f"{best['core_policy']}"
    )

    print(
        f"Dice   : "
        f"{best['macro_dice']:.6f}"
    )

    print(
        f"IoU    : "
        f"{best['macro_iou']:.6f}"
    )

    print(
        f"Recall : "
        f"{best['macro_slice_recall']:.6f}"
    )

    print(
        f"FPR    : "
        f"{best['macro_fpr']:.6f}"
    )

    print(
        f"Pred   : "
        f"{int(best['total_prediction_voxels'])}"
    )

    # ========================================================
    # FPR-CONSTRAINED BEST
    # ========================================================

    constrained = macro_df[
        (
            macro_df["macro_fpr"]
            < 0.10
        )
    ]

    if not constrained.empty:

        constrained_idx = (
            constrained[
                "macro_dice"
            ].idxmax()
        )

        constrained_best = (
            constrained.loc[
                constrained_idx
            ]
        )

        print(
            "\n" + "=" * 80
        )

        print(
            "BEST CORE POLICY WITH FPR < 0.10"
        )

        print(
            "=" * 80
        )

        print(
            f"Policy : "
            f"{constrained_best['core_policy']}"
        )

        print(
            f"Dice   : "
            f"{constrained_best['macro_dice']:.6f}"
        )

        print(
            f"IoU    : "
            f"{constrained_best['macro_iou']:.6f}"
        )

        print(
            f"Recall : "
            f"{constrained_best['macro_slice_recall']:.6f}"
        )

        print(
            f"FPR    : "
            f"{constrained_best['macro_fpr']:.6f}"
        )

    # ========================================================
    # OUTPUTS
    # ========================================================

    print("\n" + "=" * 80)
    print(
        "OUTPUT FILES"
    )
    print("=" * 80)

    print(case_csv)
    print(audit_csv)
    print(macro_csv)


if __name__ == "__main__":
    main()