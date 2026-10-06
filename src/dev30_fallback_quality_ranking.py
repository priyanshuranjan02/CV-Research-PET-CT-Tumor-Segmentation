"""
Dev30 — Fallback Quality-Aware Ranking

Purpose
-------
Dev29 showed that fallback-aware ranking can improve the final
Dev10 result.

Dev30 tests whether the QUALITY of fallback information is more
useful than simply giving every fallback component priority.

IMPORTANT
---------
DO NOT MODIFY:

    src/dev10_lite_hot_core.py

The exact Dev10:
    - 3D component construction
    - P70 core construction
    - Top-K selection
    - Dice / IoU / slice recall / FPR

are retained.

Only the ranking feature is changed.
"""

from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import dev10_lite_hot_core as dev10


# ============================================================
# CASES
# ============================================================

CASES = [
    "PETCT_0168f65af8",
    "PETCT_04606080a0",
    "PETCT_04ab5c61c9",
    "PETCT_0b57b247b6",
    "PETCT_11afab3485",
    "PETCT_185da4c8b6",
]


# ============================================================
# FALLBACK CONFIGURATION
# ============================================================

PET_THRESHOLD = 2.25
MIN_COMPONENT_AREA = 100
MIN_MAX_SUV = 4.0


# ============================================================
# DEV10 FINAL CONFIGURATION
# ============================================================

CORE_POLICY = "P70"
TOP_K = 4

ORIGINAL_RANKING = (
    "mean_suv_hot3_a0p5"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "results"
    / "development_cases"
    / "dev30_fallback_quality_ranking"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PET FALLBACK
# ============================================================

def build_pet_fallback(
    suv_slice
):
    """
    Exact Dev26 / Dev27 fallback.

    PET >= 2.25
    area >= 100
    max SUV >= 4.0
    """

    pet_hot = (
        suv_slice >= PET_THRESHOLD
    )

    (
        num_labels,
        labels,
        stats,
        _
    ) = cv2.connectedComponentsWithStats(
        pet_hot.astype(np.uint8),
        connectivity=8
    )

    fallback = np.zeros(
        suv_slice.shape,
        dtype=np.uint8
    )

    accepted = []

    for label in range(
        1,
        num_labels
    ):

        component = (
            labels == label
        )

        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA
            ]
        )

        if area < MIN_COMPONENT_AREA:
            continue

        values = suv_slice[
            component
        ]

        if values.size == 0:
            continue

        max_suv = float(
            np.max(values)
        )

        if max_suv < MIN_MAX_SUV:
            continue

        fallback[
            component
        ] = 1

        accepted.append({
            "label": label,
            "area": area,
            "max_suv": max_suv,
        })

    return (
        fallback,
        accepted
    )


# ============================================================
# EXACT DEV27 EXPERIMENTAL VOLUME
# ============================================================

def build_experimental_volume(
    case
):
    """
    Exact Dev27 architecture.

    Fallback is used ONLY when the exact Dev10 candidate
    is empty on a slice.
    """

    baseline_volume = (
        dev10.generate_candidate_volume(
            case
        )
    )

    experimental_volume = (
        baseline_volume.copy()
    )

    fallback_slices = 0
    fallback_components = 0

    for z in range(
        case["ct"].shape[0]
    ):

        if np.count_nonzero(
            baseline_volume[z]
        ) != 0:
            continue

        (
            fallback,
            accepted
        ) = build_pet_fallback(
            case["suv"][z]
        )

        if accepted:

            experimental_volume[z] = (
                np.maximum(
                    experimental_volume[z],
                    fallback
                )
            )

            fallback_slices += 1
            fallback_components += len(
                accepted
            )

    return (
        baseline_volume,
        experimental_volume,
        fallback_slices,
        fallback_components
    )


# ============================================================
# FALLBACK COMPONENT INFORMATION
# ============================================================

def get_fallback_component_info(
    components,
    cc,
    baseline_volume,
    experimental_volume,
    suv
):
    """
    Connect newly-added fallback voxels to the final
    experimental 3D components.
    """

    fallback_added = (
        (experimental_volume > 0)
        &
        (baseline_volume == 0)
    )

    info = {}

    for label in (
        components["label"]
        .astype(int)
        .tolist()
    ):

        component_mask = (
            cc == label
        )

        fallback_mask = (
            component_mask
            &
            fallback_added
        )

        fallback_voxels = int(
            fallback_mask.sum()
        )

        if fallback_voxels == 0:
            continue

        component_voxels = int(
            component_mask.sum()
        )

        fraction = (
            fallback_voxels
            / component_voxels
            if component_voxels > 0
            else 0.0
        )

        fallback_values = suv[
            fallback_mask
        ]

        fallback_mean_suv = (
            float(
                np.mean(
                    fallback_values
                )
            )
            if fallback_values.size > 0
            else 0.0
        )

        fallback_max_suv = (
            float(
                np.max(
                    fallback_values
                )
            )
            if fallback_values.size > 0
            else 0.0
        )

        fallback_hot_fraction = (
            float(
                np.mean(
                    fallback_values >= 3.0
                )
            )
            if fallback_values.size > 0
            else 0.0
        )

        info[label] = {
            "fallback_voxels":
                fallback_voxels,

            "component_voxels":
                component_voxels,

            "fallback_fraction":
                fraction,

            "fallback_mean_suv":
                fallback_mean_suv,

            "fallback_max_suv":
                fallback_max_suv,

            "fallback_hot_fraction":
                fallback_hot_fraction,
        }

    return info


# ============================================================
# BUILD RANKING TABLE
# ============================================================

def build_ranking_table(
    components,
    fallback_info
):
    """
    Add fallback-quality features to a COPY of the
    Dev10 component table.
    """

    df = components.copy()

    df["is_fallback"] = (
        df["label"]
        .astype(int)
        .isin(
            fallback_info.keys()
        )
    )

    def get_value(
        label,
        key,
        default=0.0
    ):

        return fallback_info.get(
            int(label),
            {}
        ).get(
            key,
            default
        )

    df["fallback_voxels"] = (
        df["label"]
        .map(
            lambda x:
                get_value(
                    x,
                    "fallback_voxels",
                    0
                )
        )
    )

    df["fallback_fraction"] = (
        df["label"]
        .map(
            lambda x:
                get_value(
                    x,
                    "fallback_fraction",
                    0.0
                )
        )
    )

    df["fallback_mean_suv"] = (
        df["label"]
        .map(
            lambda x:
                get_value(
                    x,
                    "fallback_mean_suv",
                    0.0
                )
        )
    )

    df["fallback_max_suv"] = (
        df["label"]
        .map(
            lambda x:
                get_value(
                    x,
                    "fallback_max_suv",
                    0.0
                )
        )
    )

    df["fallback_hot_fraction"] = (
        df["label"]
        .map(
            lambda x:
                get_value(
                    x,
                    "fallback_hot_fraction",
                    0.0
                )
        )
    )

    return df


# ============================================================
# NORMALIZATION
# ============================================================

def minmax_normalize(
    values
):
    """
    Normalize an array to [0, 1].
    """

    values = np.asarray(
        values,
        dtype=float
    )

    if values.size == 0:
        return values

    vmin = np.min(
        values
    )

    vmax = np.max(
        values
    )

    if vmax <= vmin:
        return np.zeros_like(
            values,
            dtype=float
        )

    return (
        (values - vmin)
        /
        (vmax - vmin)
    )


# ============================================================
# ADD QUALITY RANKINGS
# ============================================================

def add_ranking_features(
    df
):
    """
    Create several candidate ranking functions.

    All are derived from the same experimental
    component table.
    """

    base = (
        df[
            ORIGINAL_RANKING
        ]
        .astype(float)
        .to_numpy()
    )

    fallback_fraction = (
        df[
            "fallback_fraction"
        ]
        .astype(float)
        .to_numpy()
    )

    fallback_mean_suv = (
        df[
            "fallback_mean_suv"
        ]
        .astype(float)
        .to_numpy()
    )

    fallback_max_suv = (
        df[
            "fallback_max_suv"
        ]
        .astype(float)
        .to_numpy()
    )

    fallback_hot_fraction = (
        df[
            "fallback_hot_fraction"
        ]
        .astype(float)
        .to_numpy()
    )

    fallback_fraction_n = (
        minmax_normalize(
            fallback_fraction
        )
    )

    fallback_mean_n = (
        minmax_normalize(
            fallback_mean_suv
        )
    )

    fallback_max_n = (
        minmax_normalize(
            fallback_max_suv
        )
    )

    fallback_hot_n = (
        minmax_normalize(
            fallback_hot_fraction
        )
    )

    # --------------------------------------------------------
    # 1. Original Dev10
    # --------------------------------------------------------

    df[
        "rank_original"
    ] = base

    # --------------------------------------------------------
    # 2. Fallback fraction
    #
    # Moderate influence.
    # --------------------------------------------------------

    df[
        "rank_fraction"
    ] = (
        base
        +
        1.0
        *
        fallback_fraction_n
    )

    # --------------------------------------------------------
    # 3. Fallback PET intensity
    # --------------------------------------------------------

    df[
        "rank_intensity"
    ] = (
        base
        +
        1.0
        *
        fallback_mean_n
    )

    # --------------------------------------------------------
    # 4. Fallback quality
    #
    # Combines:
    #     fraction
    #     mean SUV
    #     max SUV
    #     hot fraction
    # --------------------------------------------------------

    quality = (
        0.35 * fallback_fraction_n
        +
        0.30 * fallback_mean_n
        +
        0.20 * fallback_max_n
        +
        0.15 * fallback_hot_n
    )

    df[
        "fallback_quality"
    ] = quality

    df[
        "rank_quality"
    ] = (
        base
        +
        quality
    )

    # --------------------------------------------------------
    # 5. Moderate quality-aware priority
    #
    # Give fallback quality an additional but controlled
    # influence.
    # --------------------------------------------------------

    df[
        "rank_quality_bonus"
    ] = (
        base
        +
        1.5
        *
        quality
    )

    return df


# ============================================================
# EVALUATION
# ============================================================

def evaluate_policy(
    case,
    ranked_df,
    cc,
    feature
):
    """
    Exact Dev10 evaluator.

    Only the ranking feature changes.
    """

    return (
        dev10.evaluate_selection(
            case,
            ranked_df,
            cc,
            feature,
            CORE_POLICY,
            TOP_K
        )
    )


# ============================================================
# PRINT TOP-K
# ============================================================

def print_top_k(
    df,
    feature,
    name
):
    """
    Print exactly what will be passed to Dev10.
    """

    selected = (
        df
        .sort_values(
            [
                feature,
                "mean_suv",
            ],
            ascending=[
                False,
                False,
            ]
        )
        .head(
            TOP_K
        )
    )

    print()
    print(
        f"{name} Top-{TOP_K}"
    )

    for i, (_, row) in enumerate(
        selected.iterrows(),
        start=1
    ):

        print(
            f"  #{i}: "
            f"label={int(row['label']):<4} "
            f"score="
            f"{float(row[feature]):.6f} "
            f"fallback="
            f"{bool(row['is_fallback'])} "
            f"fraction="
            f"{float(row['fallback_fraction']):.4f} "
            f"meanSUV="
            f"{float(row['fallback_mean_suv']):.4f} "
            f"GT_overlap="
            f"{float(row.get('overlap_gt', 0)):.0f}"
        )

    return selected


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print(
        "DEV30 — FALLBACK QUALITY-AWARE RANKING"
    )
    print("=" * 100)

    print()
    print("Configuration")
    print("-------------")
    print(
        f"PET threshold   : {PET_THRESHOLD}"
    )
    print(
        f"Minimum area    : {MIN_COMPONENT_AREA}"
    )
    print(
        f"Minimum max SUV : {MIN_MAX_SUV}"
    )
    print(
        f"Core policy     : {CORE_POLICY}"
    )
    print(
        f"Top-K           : {TOP_K}"
    )

    policies = [
        (
            "baseline",
            "rank_original"
        ),
        (
            "fraction",
            "rank_fraction"
        ),
        (
            "intensity",
            "rank_intensity"
        ),
        (
            "quality",
            "rank_quality"
        ),
        (
            "quality_bonus",
            "rank_quality_bonus"
        ),
    ]

    summary_rows = []
    ranking_rows = []

    for case_name in CASES:

        print()
        print("=" * 100)
        print(
            f"CASE: {case_name}"
        )
        print("=" * 100)

        case = dev10.load_case(
            case_name
        )

        (
            baseline_volume,
            experimental_volume,
            fallback_slices,
            fallback_components,
        ) = build_experimental_volume(
            case
        )

        print(
            f"Fallback slices      : "
            f"{fallback_slices}"
        )

        print(
            f"Fallback 2D comps    : "
            f"{fallback_components}"
        )

        # ----------------------------------------------------
        # Exact Dev10 3D component construction
        # ----------------------------------------------------

        baseline_components, baseline_cc = (
            dev10.build_3d_component_table(
                case,
                baseline_volume
            )
        )

        experimental_components, experimental_cc = (
            dev10.build_3d_component_table(
                case,
                experimental_volume
            )
        )

        print(
            f"Baseline 3D comps    : "
            f"{len(baseline_components)}"
        )

        print(
            f"Experimental 3D comps: "
            f"{len(experimental_components)}"
        )

        # ----------------------------------------------------
        # Link fallback to components
        # ----------------------------------------------------

        fallback_info = (
            get_fallback_component_info(
                experimental_components,
                experimental_cc,
                baseline_volume,
                experimental_volume,
                case["suv"]
            )
        )

        print(
            f"Fallback-linked comps: "
            f"{len(fallback_info)}"
        )

        # ----------------------------------------------------
        # Build ranking features
        # ----------------------------------------------------

        ranked = build_ranking_table(
            experimental_components,
            fallback_info
        )

        ranked = add_ranking_features(
            ranked
        )

        # ----------------------------------------------------
        # Evaluate each policy
        # ----------------------------------------------------

        for policy_name, feature in policies:

            print()
            print("-" * 100)
            print(
                f"POLICY: {policy_name}"
            )
            print("-" * 100)

            selected = print_top_k(
                ranked,
                feature,
                policy_name
            )

            metrics = evaluate_policy(
                case,
                ranked,
                experimental_cc,
                feature
            )

            print()
            print(
                f"Dice        : "
                f"{metrics['dice']:.6f}"
            )

            print(
                f"IoU         : "
                f"{metrics['iou']:.6f}"
            )

            print(
                f"Slice recall: "
                f"{metrics['slice_recall']:.6f}"
            )

            print(
                f"FPR         : "
                f"{metrics['fpr']:.6f}"
            )

            print(
                f"Pred voxels : "
                f"{metrics['prediction_voxels']}"
            )

            summary_rows.append({
                "case":
                    case_name,

                "policy":
                    policy_name,

                "fallback_slices":
                    fallback_slices,

                "fallback_2d_components":
                    fallback_components,

                "fallback_3d_components":
                    len(fallback_info),

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

                "top_k_labels":
                    ",".join(
                        str(
                            int(x)
                        )
                        for x in selected[
                            "label"
                        ]
                    ),
            })

            # ------------------------------------------------
            # Save ranking information
            # ------------------------------------------------

            ordered = (
                ranked
                .sort_values(
                    [
                        feature,
                        "mean_suv",
                    ],
                    ascending=[
                        False,
                        False,
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            selected_labels = set(
                selected[
                    "label"
                ]
                .astype(int)
                .tolist()
            )

            for rank, (_, row) in enumerate(
                ordered.iterrows(),
                start=1
            ):

                ranking_rows.append({
                    "case":
                        case_name,

                    "policy":
                        policy_name,

                    "rank":
                        rank,

                    "label":
                        int(row["label"]),

                    "ranking_score":
                        float(
                            row[feature]
                        ),

                    "original_score":
                        float(
                            row[
                                ORIGINAL_RANKING
                            ]
                        ),

                    "is_fallback":
                        bool(
                            row[
                                "is_fallback"
                            ]
                        ),

                    "fallback_voxels":
                        int(
                            row[
                                "fallback_voxels"
                            ]
                        ),

                    "fallback_fraction":
                        float(
                            row[
                                "fallback_fraction"
                            ]
                        ),

                    "fallback_mean_suv":
                        float(
                            row[
                                "fallback_mean_suv"
                            ]
                        ),

                    "fallback_max_suv":
                        float(
                            row[
                                "fallback_max_suv"
                            ]
                        ),

                    "fallback_hot_fraction":
                        float(
                            row[
                                "fallback_hot_fraction"
                            ]
                        ),

                    "gt_overlap":
                        float(
                            row.get(
                                "overlap_gt",
                                0
                            )
                        ),

                    "selected_topk":
                        int(
                            row["label"]
                        )
                        in selected_labels,
                })

    # ========================================================
    # SAVE
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    ranking_df = pd.DataFrame(
        ranking_rows
    )

    summary_path = (
        OUTPUT_DIR
        / "dev30_policy_summary.csv"
    )

    ranking_path = (
        OUTPUT_DIR
        / "dev30_component_rankings.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    ranking_df.to_csv(
        ranking_path,
        index=False
    )

    # ========================================================
    # MACRO
    # ========================================================

    macro = (
        summary_df
        .groupby("policy")
        .agg(
            macro_dice=(
                "dice",
                "mean"
            ),
            macro_iou=(
                "iou",
                "mean"
            ),
            macro_slice_recall=(
                "slice_recall",
                "mean"
            ),
            macro_fpr=(
                "fpr",
                "mean"
            ),
            total_prediction_voxels=(
                "prediction_voxels",
                "sum"
            ),
        )
        .reset_index()
    )

    print()
    print("=" * 100)
    print(
        "DEV30 MACRO SUMMARY"
    )
    print("=" * 100)

    print(
        macro.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    # ========================================================
    # DELTAS
    # ========================================================

    baseline_rows = macro[
        macro["policy"]
        == "baseline"
    ]

    if len(baseline_rows) == 1:

        baseline = (
            baseline_rows.iloc[0]
        )

        print()
        print("=" * 100)
        print(
            "DEV30 DELTA FROM BASELINE"
        )
        print("=" * 100)

        for _, row in macro.iterrows():

            if (
                row["policy"]
                == "baseline"
            ):
                continue

            print()
            print(
                row["policy"]
            )

            print(
                f"  Dice        : "
                f"{row['macro_dice'] - baseline['macro_dice']:+.6f}"
            )

            print(
                f"  IoU         : "
                f"{row['macro_iou'] - baseline['macro_iou']:+.6f}"
            )

            print(
                f"  Slice recall: "
                f"{row['macro_slice_recall'] - baseline['macro_slice_recall']:+.6f}"
            )

            print(
                f"  FPR         : "
                f"{row['macro_fpr'] - baseline['macro_fpr']:+.6f}"
            )

            print(
                f"  Pred voxels : "
                f"{int(row['total_prediction_voxels'] - baseline['total_prediction_voxels']):+d}"
            )

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    print()
    print("=" * 100)
    print("OUTPUT FILES")
    print("=" * 100)

    print(
        summary_path
    )

    print(
        ranking_path
    )

    print()
    print(
        "DEV30 COMPLETE"
    )


if __name__ == "__main__":
    main()