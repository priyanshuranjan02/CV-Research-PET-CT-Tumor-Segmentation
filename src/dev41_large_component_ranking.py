"""
Dev41 - Large Component Suppression / Ranking Robustness

Goal
----
Test whether very large PET components are dominating the ranking
and suppressing useful tumour components.

Fixed from Dev39:
    PET-only candidate
    SUV >= 2.25
    Exact Dev10 3D connected components
    Hot fraction >= 0.40
    Top-K = 4
    P70 core

Changed:
    ONLY the component ranking.

Policies:
    1. dev39_baseline
    2. volume_penalty
    3. sqrt_volume_penalty
    4. compactness_volume
    5. quality_volume
    6. strong_volume_penalty

Important:
    dev10_lite_hot_core.py is imported but NEVER modified.
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

CORE_POLICY = "P70"
TOP_K = 4

DEV10_RANKING = "mean_suv_hot3_a0p5"

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
    / "dev41_large_component_ranking"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)


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

    return (suv >= PET_THRESHOLD).astype(np.uint8)


# ============================================================
# COMPONENT FEATURES
# ============================================================

def extract_component_features(case, components, cc):

    suv = case["suv"]

    rows = []

    for _, row in components.iterrows():

        label = int(row["label"])

        mask = cc == label

        coords = np.argwhere(mask)

        if len(coords) == 0:
            continue

        z_values = coords[:, 0]

        z_min = int(z_values.min())
        z_max = int(z_values.max())

        z_span = z_max - z_min + 1

        unique_z = np.unique(z_values)

        slice_count = len(unique_z)

        # ----------------------------------------------------
        # Longest consecutive z-run
        # ----------------------------------------------------

        longest_run = 1
        current_run = 1

        for i in range(1, len(unique_z)):

            if unique_z[i] == unique_z[i - 1] + 1:

                current_run += 1
                longest_run = max(
                    longest_run,
                    current_run
                )

            else:

                current_run = 1

        persistence = (
            slice_count / max(z_span, 1)
        )

        # ----------------------------------------------------
        # Volume
        # ----------------------------------------------------

        voxel_count = int(mask.sum())

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        y_min = int(coords[:, 1].min())
        y_max = int(coords[:, 1].max())

        x_min = int(coords[:, 2].min())
        x_max = int(coords[:, 2].max())

        bbox_volume = max(
            (z_max - z_min + 1)
            * (y_max - y_min + 1)
            * (x_max - x_min + 1),
            1
        )

        compactness = (
            voxel_count / bbox_volume
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
                suv_values >= HOT_THRESHOLD
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
# BUILD DEV39 FEATURES
# ============================================================

def build_features(case, components, cc):

    features = extract_component_features(
        case,
        components,
        cc
    )

    if features.empty:
        return features

    # --------------------------------------------------------
    # Normalized features
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

    # Log volume is used to reduce the dominance of
    # extremely large components.
    features["log_voxels"] = np.log1p(
        features["voxels"]
    )

    features["n_log_voxels"] = minmax(
        features["log_voxels"]
    )

    # --------------------------------------------------------
    # Dev38 / Dev39 PET quality score
    # --------------------------------------------------------

    features["pet_quality_score"] = (
        0.25 * features["n_mean_suv"]
        + 0.15 * features["n_max_suv"]
        + 0.15 * features["n_hot_fraction"]
        + 0.20 * features["n_persistence"]
        + 0.15 * features["n_longest_run"]
        + 0.10 * features["n_compactness"]
    )

    return features


# ============================================================
# DEV41 RANKING POLICIES
# ============================================================

def apply_ranking_policy(features, policy):

    df = features.copy()

    # --------------------------------------------------------
    # Dev39 baseline
    # --------------------------------------------------------

    if policy == "dev39_baseline":

        df["ranking_score"] = (
            df["pet_quality_score"]
        )

    # --------------------------------------------------------
    # Moderate volume penalty
    #
    # Penalizes large components while retaining most
    # of the PET-quality information.
    # --------------------------------------------------------

    elif policy == "volume_penalty":

        df["ranking_score"] = (
            df["pet_quality_score"]
            - 0.10 * df["n_log_voxels"]
        )

    # --------------------------------------------------------
    # Square-root style volume penalty
    #
    # Less aggressive than direct volume suppression.
    # --------------------------------------------------------

    elif policy == "sqrt_volume_penalty":

        volume_factor = np.sqrt(
            df["n_log_voxels"]
        )

        df["ranking_score"] = (
            df["pet_quality_score"]
            - 0.10 * volume_factor
        )

    # --------------------------------------------------------
    # Compactness + volume
    #
    # Rewards compact structures while penalizing large
    # components.
    # --------------------------------------------------------

    elif policy == "compactness_volume":

        df["ranking_score"] = (
            df["pet_quality_score"]
            + 0.15 * df["n_compactness"]
            - 0.10 * df["n_log_voxels"]
        )

    # --------------------------------------------------------
    # PET quality + volume
    #
    # Slightly stronger volume suppression.
    # --------------------------------------------------------

    elif policy == "quality_volume":

        df["ranking_score"] = (
            0.90 * df["pet_quality_score"]
            + 0.10 * df["n_compactness"]
            - 0.15 * df["n_log_voxels"]
        )

    # --------------------------------------------------------
    # Strong volume penalty
    #
    # Stress test. This is deliberately aggressive.
    # --------------------------------------------------------

    elif policy == "strong_volume_penalty":

        df["ranking_score"] = (
            df["pet_quality_score"]
            + 0.10 * df["n_compactness"]
            - 0.25 * df["n_log_voxels"]
        )

    else:

        raise ValueError(
            f"Unknown ranking policy: {policy}"
        )

    return df


# ============================================================
# EXACT DEV10 EVALUATION
# ============================================================

def evaluate_selection(case, ranked, cc):

    if ranked.empty:

        return {
            "dice": 0.0,
            "iou": 0.0,
            "slice_recall": 0.0,
            "fpr": 0.0,
            "prediction_voxels": 0,
        }

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Dev10 evaluate_selection expects a ranking column.
    # We provide ranking_score but preserve the exact
    # downstream selection/core implementation.
    # --------------------------------------------------------

    evaluation_df = ranked.copy()

    metrics = dev10.evaluate_selection(
        case,
        evaluation_df,
        cc,
        "ranking_score",
        CORE_POLICY,
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
# SELECTED COMPONENT AUDIT
# ============================================================

def audit_selected_components(
    case,
    ranked,
    cc,
    gt,
    case_name,
    policy
):

    rows = []

    ranked = ranked.sort_values(
        ["ranking_score", "mean_suv"],
        ascending=[False, False]
    ).reset_index(drop=True)

    selected = ranked.head(TOP_K)

    for rank, (_, row) in enumerate(
        selected.iterrows(),
        start=1
    ):

        label = int(row["label"])

        component_mask = cc == label

        gt_bool = gt.astype(bool)

        intersection = int(
            np.logical_and(
                component_mask,
                gt_bool
            ).sum()
        )

        component_voxels = int(
            component_mask.sum()
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

        if component_voxels + gt_voxels == 0:
            component_dice = 1.0
        else:
            component_dice = (
                2.0 * intersection
                / (component_voxels + gt_voxels)
            )

        if union == 0:
            component_iou = 1.0
        else:
            component_iou = (
                intersection / union
            )

        rows.append({
            "case": case_name,
            "policy": policy,
            "rank": rank,
            "label": label,
            "voxels": component_voxels,
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
            "n_log_voxels": safe_float(
                row["n_log_voxels"]
            ),
            "ranking_score": safe_float(
                row["ranking_score"]
            ),
            "gt_overlap_voxels": intersection,
            "component_dice": component_dice,
            "component_iou": component_iou,
        })

    return rows


# ============================================================
# PROCESS ONE CASE
# ============================================================

def process_case(case_name):

    print("\n" + "=" * 80)
    print(f"DEV41 CASE: {case_name}")
    print("=" * 80)

    case = dev10.load_case(
        case_name
    )

    gt = case["gt"]

    # --------------------------------------------------------
    # PET-only candidate
    # --------------------------------------------------------

    pet_volume = generate_pet_only_volume(
        case
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
    # Feature extraction
    # --------------------------------------------------------

    features = build_features(
        case,
        components,
        cc
    )

    # --------------------------------------------------------
    # Dev39 hot-fraction gate
    #
    # This is FIXED for every Dev41 policy.
    # --------------------------------------------------------

    eligible = features[
        features["hot_fraction"]
        >= HOT_FRACTION_GATE
    ].copy()

    print(
        f"Eligible after hot-fraction gate: "
        f"{len(eligible)}"
    )

    case_results = []
    component_results = []

    policies = [
        "dev39_baseline",
        "volume_penalty",
        "sqrt_volume_penalty",
        "compactness_volume",
        "quality_volume",
        "strong_volume_penalty",
    ]

    for policy in policies:

        ranked = apply_ranking_policy(
            eligible,
            policy
        )

        ranked = ranked.sort_values(
            ["ranking_score", "mean_suv"],
            ascending=[False, False]
        ).reset_index(drop=True)

        metrics = evaluate_selection(
            case,
            ranked,
            cc
        )

        case_results.append({
            "case": case_name,
            "policy": policy,

            "dice": metrics["dice"],
            "iou": metrics["iou"],
            "slice_recall": metrics[
                "slice_recall"
            ],
            "fpr": metrics["fpr"],
            "prediction_voxels": metrics[
                "prediction_voxels"
            ],

            "eligible_components": len(
                ranked
            ),

            "top1_label": (
                int(ranked.iloc[0]["label"])
                if len(ranked) >= 1
                else -1
            ),

            "top2_label": (
                int(ranked.iloc[1]["label"])
                if len(ranked) >= 2
                else -1
            ),

            "top3_label": (
                int(ranked.iloc[2]["label"])
                if len(ranked) >= 3
                else -1
            ),

            "top4_label": (
                int(ranked.iloc[3]["label"])
                if len(ranked) >= 4
                else -1
            ),
        })

        component_results.extend(
            audit_selected_components(
                case,
                ranked,
                cc,
                gt,
                case_name,
                policy
            )
        )

        print(
            f"{policy:25s} "
            f"Dice={metrics['dice']:.6f} "
            f"IoU={metrics['iou']:.6f} "
            f"Recall={metrics['slice_recall']:.6f} "
            f"FPR={metrics['fpr']:.6f}"
        )

    return case_results, component_results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("DEV41 - LARGE COMPONENT RANKING ROBUSTNESS")
    print("=" * 80)

    print()
    print("Fixed configuration:")
    print(f"PET threshold       : {PET_THRESHOLD}")
    print(f"Hot threshold       : {HOT_THRESHOLD}")
    print(f"Hot fraction gate   : {HOT_FRACTION_GATE}")
    print(f"Core policy         : {CORE_POLICY}")
    print(f"Top-K               : {TOP_K}")
    print()

    print(
        "Dev10 source remains untouched."
    )

    all_case_results = []
    all_component_results = []

    for case_name in DEVELOPMENT_CASES:

        try:

            case_results, component_results = (
                process_case(case_name)
            )

            all_case_results.extend(
                case_results
            )

            all_component_results.extend(
                component_results
            )

        except Exception as exc:

            print(
                f"\nERROR in {case_name}: "
                f"{type(exc).__name__}: {exc}"
            )

    # ========================================================
    # SAVE CASE RESULTS
    # ========================================================

    case_df = pd.DataFrame(
        all_case_results
    )

    case_csv = (
        RESULT_DIR
        / "dev41_case_results.csv"
    )

    case_df.to_csv(
        case_csv,
        index=False
    )

    # ========================================================
    # SAVE COMPONENT AUDIT
    # ========================================================

    component_df = pd.DataFrame(
        all_component_results
    )

    component_csv = (
        RESULT_DIR
        / "dev41_component_audit.csv"
    )

    component_df.to_csv(
        component_csv,
        index=False
    )

    # ========================================================
    # MACRO SUMMARY
    # ========================================================

    if case_df.empty:
        print(
            "\nNo case results were generated."
        )
        return

    macro_rows = []

    for policy in case_df["policy"].unique():

        subset = case_df[
            case_df["policy"] == policy
        ]

        macro_rows.append({
            "policy": policy,

            "macro_dice": subset[
                "dice"
            ].mean(),

            "macro_iou": subset[
                "iou"
            ].mean(),

            "macro_slice_recall": subset[
                "slice_recall"
            ].mean(),

            "macro_fpr": subset[
                "fpr"
            ].mean(),

            "total_prediction_voxels": subset[
                "prediction_voxels"
            ].sum(),

            "mean_prediction_voxels": subset[
                "prediction_voxels"
            ].mean(),

            "mean_eligible_components": subset[
                "eligible_components"
            ].mean(),
        })

    macro_df = pd.DataFrame(
        macro_rows
    )

    macro_csv = (
        RESULT_DIR
        / "dev41_macro_summary.csv"
    )

    macro_df.to_csv(
        macro_csv,
        index=False
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 80)
    print("DEV41 MACRO RESULTS")
    print("=" * 80)

    print(
        macro_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # ========================================================
    # COMPARE AGAINST DEV39
    # ========================================================

    baseline = macro_df[
        macro_df["policy"]
        == "dev39_baseline"
    ]

    if not baseline.empty:

        baseline_dice = float(
            baseline.iloc[0]["macro_dice"]
        )

        baseline_fpr = float(
            baseline.iloc[0]["macro_fpr"]
        )

        print("\n" + "=" * 80)
        print("DEV41 DELTAS VS DEV39")
        print("=" * 80)

        comparison = macro_df.copy()

        comparison[
            "dice_delta_vs_dev39"
        ] = (
            comparison["macro_dice"]
            - baseline_dice
        )

        comparison[
            "fpr_delta_vs_dev39"
        ] = (
            comparison["macro_fpr"]
            - baseline_fpr
        )

        print(
            comparison[
                [
                    "policy",
                    "dice_delta_vs_dev39",
                    "fpr_delta_vs_dev39",
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

    best_dice_idx = macro_df[
        "macro_dice"
    ].idxmax()

    best_dice = macro_df.loc[
        best_dice_idx
    ]

    print("\n" + "=" * 80)
    print("BEST DICE POLICY")
    print("=" * 80)

    print(
        f"Policy : {best_dice['policy']}"
    )

    print(
        f"Dice   : "
        f"{best_dice['macro_dice']:.6f}"
    )

    print(
        f"IoU    : "
        f"{best_dice['macro_iou']:.6f}"
    )

    print(
        f"Recall : "
        f"{best_dice['macro_slice_recall']:.6f}"
    )

    print(
        f"FPR    : "
        f"{best_dice['macro_fpr']:.6f}"
    )

    # ========================================================
    # BEST FPR-CONSTRAINED
    # ========================================================

    constrained = macro_df[
        (macro_df["macro_dice"] >= 0.114357)
        & (macro_df["macro_fpr"] < 0.10)
    ]

    if not constrained.empty:

        best_idx = constrained[
            "macro_dice"
        ].idxmax()

        best = constrained.loc[
            best_idx
        ]

        print("\n" + "=" * 80)
        print("BEST POLICY UNDER DEV39 TARGET")
        print("=" * 80)

        print(
            f"Policy : {best['policy']}"
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

    # ========================================================
    # OUTPUTS
    # ========================================================

    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    print(case_csv)
    print(component_csv)
    print(macro_csv)

if __name__ == "__main__":
    main()