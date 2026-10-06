# src/dev37_adaptive_topk.py

import os
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import dev10_lite_hot_core as dev10


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PET_THRESHOLD = 2.25
CORE_POLICY = "P70"

MAX_K = 4

ADAPTIVE_ALPHAS = [
    0.90,
    0.80,
    0.70,
    0.60,
]

RANKING_POLICIES = [
    "dev10",
    "mean_suv",
    "mean_suv_logvol",
    "combined",
]


# ---------------------------------------------------------------------
# Development cases
# ---------------------------------------------------------------------

DEVELOPMENT_CASES = [
    "PETCT_0168f65af8",
    "PETCT_04606080a0",
    "PETCT_04ab5c61c9",
    "PETCT_0b57b247b6",
    "PETCT_11afab3485",
    "PETCT_185da4c8b6",
]


# ---------------------------------------------------------------------
# PET-only candidate generation
# ---------------------------------------------------------------------

def generate_pet_only_candidate(case):

    suv = case["suv"]

    candidate = (
        suv >= PET_THRESHOLD
    ).astype(np.uint8)

    return candidate


# ---------------------------------------------------------------------
# Component volume column
# ---------------------------------------------------------------------

def get_volume_column(df):

    for col in [
        "voxels",
        "voxel_count",
        "component_voxels",
    ]:
        if col in df.columns:
            return col

    raise KeyError(
        "Could not find component volume column. "
        f"Available columns: {list(df.columns)}"
    )


# ---------------------------------------------------------------------
# Ranking features
# ---------------------------------------------------------------------

def add_ranking_features(components):

    df = components.copy()

    volume_col = get_volume_column(df)

    # -------------------------------------------------------------
    # Basic features
    # -------------------------------------------------------------

    df["dev37_volume"] = (
        df[volume_col].astype(float)
    )

    if "mean_suv" in df.columns:
        df["dev37_mean_suv"] = (
            df["mean_suv"].astype(float)
        )
    else:
        df["dev37_mean_suv"] = 0.0

    if "max_suv" in df.columns:
        df["dev37_max_suv"] = (
            df["max_suv"].astype(float)
        )
    else:
        df["dev37_max_suv"] = 0.0

    if "frac_ge_3" in df.columns:
        df["dev37_frac_ge_3"] = (
            df["frac_ge_3"].astype(float)
        )
    else:
        df["dev37_frac_ge_3"] = 0.0

    # -------------------------------------------------------------
    # Original Dev10 ranking
    # -------------------------------------------------------------

    if "mean_suv_hot3_a0p5" in df.columns:
        df["dev37_dev10"] = (
            df["mean_suv_hot3_a0p5"].astype(float)
        )
    else:
        df["dev37_dev10"] = 0.0

    # -------------------------------------------------------------
    # Mean SUV × log(volume)
    # -------------------------------------------------------------

    df["dev37_mean_suv_logvol"] = (
        df["dev37_mean_suv"]
        * np.log1p(df["dev37_volume"])
    )

    # -------------------------------------------------------------
    # Combined ranking
    # -------------------------------------------------------------

    def normalize(series):

        minimum = series.min()
        maximum = series.max()

        if maximum <= minimum:
            return pd.Series(
                np.ones(len(series)),
                index=series.index,
            )

        return (
            (series - minimum)
            / (maximum - minimum)
        )

    normalized_mean = normalize(
        df["dev37_mean_suv"]
    )

    normalized_volume = normalize(
        np.log1p(df["dev37_volume"])
    )

    normalized_hot = normalize(
        df["dev37_dev10"]
    )

    df["dev37_combined"] = (
        0.40 * normalized_mean
        + 0.30 * normalized_volume
        + 0.30 * normalized_hot
    )

    return df


# ---------------------------------------------------------------------
# Ranking column mapping
# ---------------------------------------------------------------------

def ranking_column(policy):

    mapping = {
        "dev10": "dev37_dev10",
        "mean_suv": "dev37_mean_suv",
        "mean_suv_logvol": "dev37_mean_suv_logvol",
        "combined": "dev37_combined",
    }

    if policy not in mapping:
        raise ValueError(
            f"Unknown ranking policy: {policy}"
        )

    return mapping[policy]


# ---------------------------------------------------------------------
# Adaptive component selection
# ---------------------------------------------------------------------

def adaptive_select_components(
    components,
    ranking_col,
    alpha,
    max_k=MAX_K,
):

    df = components.copy()

    if len(df) == 0:
        return df

    # Match Dev10 ordering:
    # primary = ranking feature
    # secondary = mean SUV
    df = df.sort_values(
        by=[
            ranking_col,
            "mean_suv",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)

    best_score = float(
        df.iloc[0][ranking_col]
    )

    # If all scores are zero/non-positive,
    # keep the best-ranked component.
    if best_score <= 0:

        return df.head(1).copy()

    threshold = alpha * best_score

    selected = df[
        df[ranking_col] >= threshold
    ].head(max_k).copy()

    return selected.reset_index(drop=True)


# ---------------------------------------------------------------------
# CORRECTED adaptive evaluation
# ---------------------------------------------------------------------

def evaluate_adaptive_selection(
    case,
    components,
    cc,
    ranking_col,
    alpha,
):

    # -------------------------------------------------------------
    # Select components adaptively
    # -------------------------------------------------------------

    ranked = adaptive_select_components(
        components=components,
        ranking_col=ranking_col,
        alpha=alpha,
        max_k=MAX_K,
    )

    if len(ranked) == 0:

        return {
            "dice": 0.0,
            "iou": 0.0,
            "slice_recall": 0.0,
            "fpr": 0.0,
            "prediction_voxels": 0,
            "selected_components": 0,
        }

    # -------------------------------------------------------------
    # IMPORTANT FIX
    #
    # Pass ONLY the selected components to Dev10.
    #
    # Previously we passed the entire component table and used
    # a binary temporary score. That allowed evaluate_selection()
    # to select additional components and invalidated the adaptive
    # K experiment.
    # -------------------------------------------------------------

    metrics = dev10.evaluate_selection(
        case,
        ranked,
        cc,
        ranking_col,
        CORE_POLICY,
        MAX_K,
    )

    return {
        "dice": float(
            metrics["dice"]
        ),
        "iou": float(
            metrics["iou"]
        ),
        "slice_recall": float(
            metrics["slice_recall"]
        ),
        "fpr": float(
            metrics["fpr"]
        ),
        "prediction_voxels": int(
            metrics["prediction_voxels"]
        ),
        "selected_components": int(
            len(ranked)
        ),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 100)
    print("DEV37 - CORRECTED ADAPTIVE TOP-K COMPONENT SELECTION")
    print("=" * 100)

    print()
    print("Configuration")
    print("-" * 80)

    print(
        f"PET threshold : {PET_THRESHOLD}"
    )

    print(
        f"Core policy   : {CORE_POLICY}"
    )

    print(
        f"Maximum K     : {MAX_K}"
    )

    print(
        f"Adaptive α    : {ADAPTIVE_ALPHAS}"
    )

    print(
        f"Ranking       : {RANKING_POLICIES}"
    )

    all_results = []
    audit_rows = []

    # -----------------------------------------------------------------
    # Cases
    # -----------------------------------------------------------------

    for case_name in DEVELOPMENT_CASES:

        print()
        print("=" * 100)
        print(f"CASE: {case_name}")
        print("=" * 100)

        case = dev10.load_case(
            case_name
        )

        print(
            f"CT: {case['ct'].shape}"
        )

        print(
            f"PET: {case['suv'].shape}"
        )

        print(
            f"SUV max: {case['suv'].max():.4f}"
        )

        print(
            f"GT voxels: "
            f"{int(np.count_nonzero(case['gt']))}"
        )

        # -------------------------------------------------------------
        # PET-only candidate
        # -------------------------------------------------------------

        candidate = generate_pet_only_candidate(
            case
        )

        print(
            f"PET candidate voxels: "
            f"{int(np.count_nonzero(candidate))}"
        )

        # -------------------------------------------------------------
        # Exact Dev10 3D component construction
        # -------------------------------------------------------------

        components, cc = (
            dev10.build_3d_component_table(
                case,
                candidate,
            )
        )

        print(
            f"Dev10 3D components: "
            f"{len(components)}"
        )

        components = add_ranking_features(
            components
        )

        # -------------------------------------------------------------
        # Ranking policies
        # -------------------------------------------------------------

        for policy in RANKING_POLICIES:

            ranking_col = ranking_column(
                policy
            )

            print()
            print(
                f"  Ranking policy: {policy}"
            )

            for alpha in ADAPTIVE_ALPHAS:

                metrics = (
                    evaluate_adaptive_selection(
                        case,
                        components,
                        cc,
                        ranking_col,
                        alpha,
                    )
                )

                print(
                    f"    α={alpha:.2f} | "
                    f"K={metrics['selected_components']} | "
                    f"Dice={metrics['dice']:.6f} | "
                    f"IoU={metrics['iou']:.6f} | "
                    f"Recall={metrics['slice_recall']:.6f} | "
                    f"FPR={metrics['fpr']:.6f} | "
                    f"Pred={metrics['prediction_voxels']}"
                )

                all_results.append({
                    "case": case_name,
                    "policy": policy,
                    "alpha": alpha,
                    "dice": metrics["dice"],
                    "iou": metrics["iou"],
                    "slice_recall": metrics[
                        "slice_recall"
                    ],
                    "fpr": metrics["fpr"],
                    "prediction_voxels": metrics[
                        "prediction_voxels"
                    ],
                    "selected_components": metrics[
                        "selected_components"
                    ],
                })

        # -------------------------------------------------------------
        # Component audit
        # -------------------------------------------------------------

        gt = case["gt"].astype(bool)

        ranking_cols = {
            policy: ranking_column(policy)
            for policy in RANKING_POLICIES
        }

        for _, row in components.iterrows():

            label = int(
                row["label"]
            )

            component_mask = (
                cc == label
            )

            overlap = int(
                np.count_nonzero(
                    component_mask & gt
                )
            )

            component_voxels = int(
                np.count_nonzero(
                    component_mask
                )
            )

            if component_voxels > 0:

                dice = (
                    2.0 * overlap
                    /
                    (
                        component_voxels
                        + int(
                            np.count_nonzero(gt)
                        )
                    )
                )

            else:

                dice = 0.0

            ranks = {}

            for policy, col in ranking_cols.items():

                ordered = (
                    components
                    .sort_values(
                        by=[
                            col,
                            "mean_suv",
                        ],
                        ascending=[
                            False,
                            False,
                        ],
                    )
                    .reset_index(drop=True)
                )

                matches = np.where(
                    ordered["label"]
                    .astype(int)
                    .values
                    == label
                )[0]

                ranks[policy] = (
                    int(matches[0]) + 1
                    if len(matches) > 0
                    else -1
                )

            audit_rows.append({
                "case": case_name,
                "label": label,
                "voxels": component_voxels,
                "overlap_gt": overlap,
                "component_dice": dice,
                "rank_dev10": ranks[
                    "dev10"
                ],
                "rank_mean_suv": ranks[
                    "mean_suv"
                ],
                "rank_mean_suv_logvol": ranks[
                    "mean_suv_logvol"
                ],
                "rank_combined": ranks[
                    "combined"
                ],
            })

    # -----------------------------------------------------------------
    # Macro results
    # -----------------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    macro = (
        results_df
        .groupby(
            [
                "policy",
                "alpha",
            ],
            as_index=False,
        )
        .agg(
            macro_dice=(
                "dice",
                "mean",
            ),
            macro_iou=(
                "iou",
                "mean",
            ),
            macro_slice_recall=(
                "slice_recall",
                "mean",
            ),
            macro_fpr=(
                "fpr",
                "mean",
            ),
            total_prediction_voxels=(
                "prediction_voxels",
                "sum",
            ),
            mean_prediction_voxels=(
                "prediction_voxels",
                "mean",
            ),
            mean_selected_components=(
                "selected_components",
                "mean",
            ),
        )
        .sort_values(
            "macro_dice",
            ascending=False,
        )
    )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    output_dir = os.path.join(
        PROJECT_ROOT,
        "results",
        "development_cases",
        "dev37_adaptive_topk",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    results_path = os.path.join(
        output_dir,
        "dev37_case_results.csv",
    )

    macro_path = os.path.join(
        output_dir,
        "dev37_macro_summary.csv",
    )

    audit_path = os.path.join(
        output_dir,
        "dev37_component_audit.csv",
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    macro.to_csv(
        macro_path,
        index=False,
    )

    pd.DataFrame(
        audit_rows
    ).to_csv(
        audit_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # Print macro results
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print("DEV37 MACRO RESULTS")
    print("=" * 100)

    print(
        macro.to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------
    # Best configuration
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print("BEST CONFIGURATION")
    print("=" * 100)

    if len(macro) > 0:

        best = macro.iloc[0]

        print(
            f"Policy       : "
            f"{best['policy']}"
        )

        print(
            f"Alpha        : "
            f"{best['alpha']:.2f}"
        )

        print(
            f"Macro Dice   : "
            f"{best['macro_dice']:.6f}"
        )

        print(
            f"Macro IoU    : "
            f"{best['macro_iou']:.6f}"
        )

        print(
            f"Slice Recall : "
            f"{best['macro_slice_recall']:.6f}"
        )

        print(
            f"FPR          : "
            f"{best['macro_fpr']:.6f}"
        )

        print(
            f"Total Pred   : "
            f"{int(best['total_prediction_voxels'])}"
        )

        print(
            f"Mean K       : "
            f"{best['mean_selected_components']:.3f}"
        )

    # -----------------------------------------------------------------
    # Complete
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print("DEV37 COMPLETE")
    print("=" * 100)

    print()
    print("Saved:")
    print(results_path)
    print(macro_path)
    print(audit_path)


if __name__ == "__main__":
    main()