# src/dev38_pet_component_quality_ranking.py

import os
import sys
import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import dev10_lite_hot_core as dev10


# ============================================================================
# CONFIGURATION
# ============================================================================

PET_THRESHOLD = 2.25

CORE_POLICY = "P70"
TOP_K = 4

HOT_THRESHOLD = 3.0

DEVELOPMENT_CASES = [
    "PETCT_0168f65af8",
    "PETCT_04606080a0",
    "PETCT_04ab5c61c9",
    "PETCT_0b57b247b6",
    "PETCT_11afab3485",
    "PETCT_185da4c8b6",
]


# ============================================================================
# OUTPUT
# ============================================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "development_cases",
    "dev38_pet_component_quality_ranking",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================================
# PET-ONLY CANDIDATE
# ============================================================================

def generate_pet_only_candidate(case):
    """
    PET-only candidate.

    This intentionally removes the Dev10 CT contour gate so that
    PET localization can be studied independently.
    """

    suv = case["suv"]

    candidate = (
        suv >= PET_THRESHOLD
    ).astype(np.uint8)

    return candidate


# ============================================================================
# COMPONENT FEATURE EXTRACTION
# ============================================================================

def add_component_features(
    components,
    cc,
    suv,
):
    """
    Add interpretable structural and PET features to each 3D component.

    No GT information is used here.
    """

    df = components.copy()

    labels = df["label"].astype(int).values

    feature_rows = []

    for label in labels:

        component_mask = (
            cc == label
        )

        voxel_count = int(
            np.count_nonzero(
                component_mask
            )
        )

        if voxel_count == 0:
            feature_rows.append({
                "dev38_voxels": 0,
                "dev38_z_span": 0,
                "dev38_slice_count": 0,
                "dev38_longest_run": 0,
                "dev38_persistence": 0.0,
                "dev38_compactness": 0.0,
                "dev38_mean_suv": 0.0,
                "dev38_max_suv": 0.0,
                "dev38_hot_fraction": 0.0,
                "dev38_intensity_persistence": 0.0,
            })
            continue

        coords = np.where(
            component_mask
        )

        z_values = coords[0]

        z_start = int(
            z_values.min()
        )

        z_end = int(
            z_values.max()
        )

        z_span = (
            z_end - z_start + 1
        )

        unique_z = np.unique(
            z_values
        )

        slice_count = len(
            unique_z
        )

        # --------------------------------------------------------------
        # Longest consecutive z-slice run
        # --------------------------------------------------------------

        longest_run = 1
        current_run = 1

        for i in range(
            1,
            len(unique_z),
        ):

            if (
                unique_z[i]
                == unique_z[i - 1] + 1
            ):
                current_run += 1
            else:
                current_run = 1

            longest_run = max(
                longest_run,
                current_run,
            )

        # --------------------------------------------------------------
        # Persistence
        # --------------------------------------------------------------

        persistence = (
            longest_run
            / max(
                z_span,
                1,
            )
        )

        # --------------------------------------------------------------
        # SUV statistics
        # --------------------------------------------------------------

        component_suv = suv[
            component_mask
        ]

        mean_suv = float(
            np.mean(
                component_suv
            )
        )

        max_suv = float(
            np.max(
                component_suv
            )
        )

        hot_fraction = float(
            np.mean(
                component_suv
                >= HOT_THRESHOLD
            )
        )

        # --------------------------------------------------------------
        # Compactness
        #
        # volume / bounding-box volume
        #
        # Higher value = more spatially compact.
        # --------------------------------------------------------------

        x_values = coords[1]
        y_values = coords[2]

        x_span = (
            int(x_values.max())
            - int(x_values.min())
            + 1
        )

        y_span = (
            int(y_values.max())
            - int(y_values.min())
            + 1
        )

        bbox_volume = (
            z_span
            * x_span
            * y_span
        )

        compactness = (
            voxel_count
            / max(
                bbox_volume,
                1,
            )
        )

        # --------------------------------------------------------------
        # Intensity × persistence
        # --------------------------------------------------------------

        intensity_persistence = (
            mean_suv
            * persistence
        )

        feature_rows.append({
            "dev38_voxels": voxel_count,
            "dev38_z_span": z_span,
            "dev38_slice_count": slice_count,
            "dev38_longest_run": longest_run,
            "dev38_persistence": persistence,
            "dev38_compactness": compactness,
            "dev38_mean_suv": mean_suv,
            "dev38_max_suv": max_suv,
            "dev38_hot_fraction": hot_fraction,
            "dev38_intensity_persistence":
                intensity_persistence,
        })

    feature_df = pd.DataFrame(
        feature_rows,
        index=df.index,
    )

    df = pd.concat(
        [
            df.reset_index(drop=True),
            feature_df.reset_index(drop=True),
        ],
        axis=1,
    )

    return df


# ============================================================================
# NORMALIZATION
# ============================================================================

def minmax_normalize(series):

    series = series.astype(float)

    minimum = series.min()
    maximum = series.max()

    if (
        not np.isfinite(minimum)
        or not np.isfinite(maximum)
        or maximum <= minimum
    ):
        return pd.Series(
            np.zeros(
                len(series)
            ),
            index=series.index,
        )

    return (
        (series - minimum)
        / (maximum - minimum)
    )


# ============================================================================
# RANKING FEATURES
# ============================================================================

def build_ranking_scores(df):

    result = df.copy()

    # ------------------------------------------------------------------
    # Normalized primitive features
    # ------------------------------------------------------------------

    result["n_mean_suv"] = (
        minmax_normalize(
            result["dev38_mean_suv"]
        )
    )

    result["n_max_suv"] = (
        minmax_normalize(
            result["dev38_max_suv"]
        )
    )

    result["n_hot_fraction"] = (
        minmax_normalize(
            result["dev38_hot_fraction"]
        )
    )

    result["n_volume"] = (
        minmax_normalize(
            np.log1p(
                result["dev38_voxels"]
            )
        )
    )

    result["n_persistence"] = (
        minmax_normalize(
            result["dev38_persistence"]
        )
    )

    result["n_longest_run"] = (
        minmax_normalize(
            np.log1p(
                result["dev38_longest_run"]
            )
        )
    )

    result["n_compactness"] = (
        minmax_normalize(
            result["dev38_compactness"]
        )
    )

    result["n_intensity_persistence"] = (
        minmax_normalize(
            result[
                "dev38_intensity_persistence"
            ]
        )
    )

    # ------------------------------------------------------------------
    # Dev10 reference ranking
    # ------------------------------------------------------------------

    if "mean_suv_hot3_a0p5" in result.columns:

        result["dev38_dev10"] = (
            result[
                "mean_suv_hot3_a0p5"
            ].astype(float)
        )

    else:

        result["dev38_dev10"] = (
            result["dev38_mean_suv"]
        )

    # ------------------------------------------------------------------
    # Ranking 1: intensity
    # ------------------------------------------------------------------

    result["dev38_intensity"] = (
        0.50 * result["n_mean_suv"]
        + 0.30 * result["n_hot_fraction"]
        + 0.20 * result["n_max_suv"]
    )

    # ------------------------------------------------------------------
    # Ranking 2: intensity + persistence
    # ------------------------------------------------------------------

    result["dev38_intensity_persistence"] = (
        0.35 * result["n_mean_suv"]
        + 0.20 * result["n_hot_fraction"]
        + 0.20 * result["n_max_suv"]
        + 0.25 * result["n_persistence"]
    )

    # ------------------------------------------------------------------
    # Ranking 3: volume + persistence
    # ------------------------------------------------------------------

    result["dev38_volume_persistence"] = (
        0.25 * result["n_mean_suv"]
        + 0.20 * result["n_hot_fraction"]
        + 0.20 * result["n_volume"]
        + 0.20 * result["n_persistence"]
        + 0.15 * result["n_longest_run"]
    )

    # ------------------------------------------------------------------
    # Ranking 4: structural
    # ------------------------------------------------------------------

    result["dev38_structural"] = (
        0.25 * result["n_mean_suv"]
        + 0.15 * result["n_hot_fraction"]
        + 0.15 * result["n_volume"]
        + 0.20 * result["n_persistence"]
        + 0.15 * result["n_longest_run"]
        + 0.10 * result["n_compactness"]
    )

    # ------------------------------------------------------------------
    # Ranking 5: PET quality
    #
    # Strong intensity + persistence + compactness.
    # ------------------------------------------------------------------

    result["dev38_pet_quality"] = (
        0.25 * result["n_mean_suv"]
        + 0.15 * result["n_max_suv"]
        + 0.15 * result["n_hot_fraction"]
        + 0.20 * result["n_persistence"]
        + 0.15 * result["n_longest_run"]
        + 0.10 * result["n_compactness"]
    )

    # ------------------------------------------------------------------
    # Ranking 6: combined structural
    #
    # This is deliberately balanced rather than dominated by volume.
    # ------------------------------------------------------------------

    result["dev38_combined"] = (
        0.25 * result["n_mean_suv"]
        + 0.15 * result["n_max_suv"]
        + 0.15 * result["n_hot_fraction"]
        + 0.15 * result["n_volume"]
        + 0.15 * result["n_persistence"]
        + 0.10 * result["n_longest_run"]
        + 0.05 * result["n_compactness"]
    )

    return result


# ============================================================================
# RANKING POLICY MAP
# ============================================================================

RANKING_POLICIES = {
    "dev10": "dev38_dev10",
    "intensity": "dev38_intensity",
    "intensity_persistence":
        "dev38_intensity_persistence",
    "volume_persistence":
        "dev38_volume_persistence",
    "structural":
        "dev38_structural",
    "pet_quality":
        "dev38_pet_quality",
    "combined":
        "dev38_combined",
}


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_policy(
    case,
    components,
    cc,
    ranking_col,
):

    # ------------------------------------------------------------------
    # Exact Dev10 ordering convention
    # ------------------------------------------------------------------

    ranked = (
        components
        .sort_values(
            by=[
                ranking_col,
                "mean_suv",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    # ------------------------------------------------------------------
    # IMPORTANT:
    #
    # Pass the complete ranked component table to Dev10.
    # Dev10 itself performs the exact Top-K selection.
    # ------------------------------------------------------------------

    metrics = dev10.evaluate_selection(
        case,
        ranked,
        cc,
        ranking_col,
        CORE_POLICY,
        TOP_K,
    )

    return metrics, ranked


# ============================================================================
# COMPONENT GT AUDIT
# ============================================================================

def component_gt_audit(
    case,
    components,
    cc,
):

    gt = case["gt"].astype(bool)

    rows = []

    for _, row in components.iterrows():

        label = int(
            row["label"]
        )

        component_mask = (
            cc == label
        )

        component_voxels = int(
            np.count_nonzero(
                component_mask
            )
        )

        overlap = int(
            np.count_nonzero(
                component_mask
                & gt
            )
        )

        gt_voxels = int(
            np.count_nonzero(gt)
        )

        if (
            component_voxels
            + gt_voxels
            > 0
        ):

            dice = (
                2.0 * overlap
                /
                (
                    component_voxels
                    + gt_voxels
                )
            )

            union = (
                component_voxels
                + gt_voxels
                - overlap
            )

            iou = (
                overlap / union
                if union > 0
                else 0.0
            )

        else:

            dice = 0.0
            iou = 0.0

        rows.append({
            "label": label,
            "component_voxels":
                component_voxels,
            "gt_overlap":
                overlap,
            "component_dice":
                dice,
            "component_iou":
                iou,
        })

    return pd.DataFrame(rows)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 100)
    print("DEV38 - PET COMPONENT QUALITY RANKING")
    print("=" * 100)

    print()
    print("Configuration")
    print("-" * 80)

    print(
        f"PET threshold : {PET_THRESHOLD}"
    )

    print(
        f"Hot threshold : {HOT_THRESHOLD}"
    )

    print(
        f"Core policy   : {CORE_POLICY}"
    )

    print(
        f"Top-K         : {TOP_K}"
    )

    print(
        "Ranking policies:"
    )

    for name in RANKING_POLICIES:
        print(
            f"  - {name}"
        )

    all_results = []
    all_audits = []

    # ==================================================================
    # CASE LOOP
    # ==================================================================

    for case_name in DEVELOPMENT_CASES:

        print()
        print("=" * 100)
        print(
            f"CASE: {case_name}"
        )
        print("=" * 100)

        case = dev10.load_case(
            case_name
        )

        print(
            f"CT shape : "
            f"{case['ct'].shape}"
        )

        print(
            f"PET shape: "
            f"{case['suv'].shape}"
        )

        print(
            f"SUV max  : "
            f"{case['suv'].max():.4f}"
        )

        print(
            f"GT voxels: "
            f"{int(np.count_nonzero(case['gt']))}"
        )

        # --------------------------------------------------------------
        # PET-only candidate
        # --------------------------------------------------------------

        candidate = (
            generate_pet_only_candidate(
                case
            )
        )

        print(
            f"PET candidate voxels: "
            f"{int(np.count_nonzero(candidate))}"
        )

        # --------------------------------------------------------------
        # Exact Dev10 3D component extraction
        # --------------------------------------------------------------

        components, cc = (
            dev10.build_3d_component_table(
                case,
                candidate,
            )
        )

        print(
            f"3D components: "
            f"{len(components)}"
        )

        # --------------------------------------------------------------
        # Add features
        # --------------------------------------------------------------

        components = add_component_features(
            components,
            cc,
            case["suv"],
        )

        components = build_ranking_scores(
            components
        )

        # --------------------------------------------------------------
        # GT audit
        # --------------------------------------------------------------

        gt_audit = component_gt_audit(
            case,
            components,
            cc,
        )

        gt_audit["case"] = (
            case_name
        )

        all_audits.append(
            gt_audit
        )

        # --------------------------------------------------------------
        # Evaluate every ranking
        # --------------------------------------------------------------

        for policy_name, ranking_col in (
            RANKING_POLICIES.items()
        ):

            print()
            print(
                f"  Ranking: "
                f"{policy_name}"
            )

            metrics, ranked = (
                evaluate_policy(
                    case,
                    components,
                    cc,
                    ranking_col,
                )
            )

            # ----------------------------------------------------------
            # Determine Top-K labels for audit
            # ----------------------------------------------------------

            top_labels = (
                ranked
                .head(TOP_K)["label"]
                .astype(int)
                .tolist()
            )

            top_gt_overlap = []

            gt = case["gt"].astype(bool)

            for label in top_labels:

                mask = (
                    cc == label
                )

                overlap = int(
                    np.count_nonzero(
                        mask & gt
                    )
                )

                top_gt_overlap.append(
                    overlap
                )

            print(
                f"    Top-K labels: "
                f"{top_labels}"
            )

            print(
                f"    GT overlap: "
                f"{top_gt_overlap}"
            )

            print(
                f"    Dice: "
                f"{metrics['dice']:.6f}"
            )

            print(
                f"    IoU: "
                f"{metrics['iou']:.6f}"
            )

            print(
                f"    Slice recall: "
                f"{metrics['slice_recall']:.6f}"
            )

            print(
                f"    FPR: "
                f"{metrics['fpr']:.6f}"
            )

            print(
                f"    Prediction voxels: "
                f"{metrics['prediction_voxels']}"
            )

            all_results.append({
                "case":
                    case_name,

                "policy":
                    policy_name,

                "ranking_column":
                    ranking_col,

                "core_policy":
                    CORE_POLICY,

                "top_k":
                    TOP_K,

                "dice":
                    float(
                        metrics["dice"]
                    ),

                "iou":
                    float(
                        metrics["iou"]
                    ),

                "slice_recall":
                    float(
                        metrics[
                            "slice_recall"
                        ]
                    ),

                "fpr":
                    float(
                        metrics["fpr"]
                    ),

                "prediction_voxels":
                    int(
                        metrics[
                            "prediction_voxels"
                        ]
                    ),

                "top1_label":
                    top_labels[0]
                    if len(top_labels) > 0
                    else -1,

                "top2_label":
                    top_labels[1]
                    if len(top_labels) > 1
                    else -1,

                "top3_label":
                    top_labels[2]
                    if len(top_labels) > 2
                    else -1,

                "top4_label":
                    top_labels[3]
                    if len(top_labels) > 3
                    else -1,

                "top1_gt_overlap":
                    top_gt_overlap[0]
                    if len(top_gt_overlap) > 0
                    else 0,

                "top2_gt_overlap":
                    top_gt_overlap[1]
                    if len(top_gt_overlap) > 1
                    else 0,

                "top3_gt_overlap":
                    top_gt_overlap[2]
                    if len(top_gt_overlap) > 2
                    else 0,

                "top4_gt_overlap":
                    top_gt_overlap[3]
                    if len(top_gt_overlap) > 3
                    else 0,
            })

    # ==================================================================
    # DATAFRAMES
    # ==================================================================

    results_df = pd.DataFrame(
        all_results
    )

    audit_df = pd.concat(
        all_audits,
        ignore_index=True,
    )

    # ==================================================================
    # MACRO SUMMARY
    # ==================================================================

    macro_df = (
        results_df
        .groupby(
            [
                "policy",
                "core_policy",
                "top_k",
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
        )
        .sort_values(
            "macro_dice",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # ==================================================================
    # RANKING POSITION AUDIT
    # ==================================================================

    ranking_audit_rows = []

    for policy_name, ranking_col in (
        RANKING_POLICIES.items()
    ):

        for case_name in DEVELOPMENT_CASES:

            case_components = (
                components
                if False
                else None
            )

            # Get rows belonging to this case
            # from the global feature/audit information
            # by rebuilding the case ordering.
            case = dev10.load_case(
                case_name
            )

            candidate = (
                generate_pet_only_candidate(
                    case
                )
            )

            case_components, case_cc = (
                dev10.build_3d_component_table(
                    case,
                    candidate,
                )
            )

            case_components = (
                add_component_features(
                    case_components,
                    case_cc,
                    case["suv"],
                )
            )

            case_components = (
                build_ranking_scores(
                    case_components
                )
            )

            ordered = (
                case_components
                .sort_values(
                    by=[
                        ranking_col,
                        "mean_suv",
                    ],
                    ascending=[
                        False,
                        False,
                    ],
                )
                .reset_index(
                    drop=True
                )
            )

            gt = case["gt"].astype(bool)

            for rank, row in (
                ordered.iterrows()
            ):

                label = int(
                    row["label"]
                )

                mask = (
                    case_cc == label
                )

                overlap = int(
                    np.count_nonzero(
                        mask & gt
                    )
                )

                ranking_audit_rows.append({
                    "case":
                        case_name,

                    "policy":
                        policy_name,

                    "rank":
                        rank + 1,

                    "label":
                        label,

                    "voxels":
                        int(
                            row[
                                "dev38_voxels"
                            ]
                        ),

                    "mean_suv":
                        float(
                            row[
                                "dev38_mean_suv"
                            ]
                        ),

                    "max_suv":
                        float(
                            row[
                                "dev38_max_suv"
                            ]
                        ),

                    "hot_fraction":
                        float(
                            row[
                                "dev38_hot_fraction"
                            ]
                        ),

                    "z_span":
                        int(
                            row[
                                "dev38_z_span"
                            ]
                        ),

                    "slice_count":
                        int(
                            row[
                                "dev38_slice_count"
                            ]
                        ),

                    "longest_run":
                        int(
                            row[
                                "dev38_longest_run"
                            ]
                        ),

                    "persistence":
                        float(
                            row[
                                "dev38_persistence"
                            ]
                        ),

                    "compactness":
                        float(
                            row[
                                "dev38_compactness"
                            ]
                        ),

                    "gt_overlap":
                        overlap,
                })

    ranking_audit_df = pd.DataFrame(
        ranking_audit_rows
    )

    # ==================================================================
    # SAVE
    # ==================================================================

    results_path = os.path.join(
        OUTPUT_DIR,
        "dev38_case_results.csv",
    )

    macro_path = os.path.join(
        OUTPUT_DIR,
        "dev38_macro_summary.csv",
    )

    component_path = os.path.join(
        OUTPUT_DIR,
        "dev38_component_gt_audit.csv",
    )

    ranking_path = os.path.join(
        OUTPUT_DIR,
        "dev38_ranking_audit.csv",
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    macro_df.to_csv(
        macro_path,
        index=False,
    )

    audit_df.to_csv(
        component_path,
        index=False,
    )

    ranking_audit_df.to_csv(
        ranking_path,
        index=False,
    )

    # ==================================================================
    # PRINT RESULTS
    # ==================================================================

    print()
    print("=" * 100)
    print("DEV38 MACRO RESULTS")
    print("=" * 100)

    print(
        macro_df.to_string(
            index=False
        )
    )

    # ==================================================================
    # BEST CONFIGURATION
    # ==================================================================

    print()
    print("=" * 100)
    print("DEV38 BEST CONFIGURATION")
    print("=" * 100)

    if len(macro_df) > 0:

        best = macro_df.iloc[0]

        print(
            f"Policy       : "
            f"{best['policy']}"
        )

        print(
            f"Core         : "
            f"{best['core_policy']}"
        )

        print(
            f"Top-K        : "
            f"{int(best['top_k'])}"
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

    # ==================================================================
    # BENCHMARK
    # ==================================================================

    DEV10_BENCHMARK = 0.100912

    print()
    print("=" * 100)
    print("DEV10 BENCHMARK")
    print("=" * 100)

    print(
        f"Dev10 P70 K4 Dice : "
        f"{DEV10_BENCHMARK:.6f}"
    )

    if len(macro_df) > 0:

        delta = (
            best["macro_dice"]
            - DEV10_BENCHMARK
        )

        print(
            f"Dev38 best delta  : "
            f"{delta:+.6f}"
        )

    # ==================================================================
    # OUTPUT PATHS
    # ==================================================================

    print()
    print("=" * 100)
    print("DEV38 COMPLETE")
    print("=" * 100)

    print()
    print("Saved:")
    print(results_path)
    print(macro_path)
    print(component_path)
    print(ranking_path)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()