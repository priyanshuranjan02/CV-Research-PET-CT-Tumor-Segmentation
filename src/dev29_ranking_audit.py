"""
Dev29 — Fallback-Aware Ranking Audit

Purpose
-------
Test whether useful PET-fallback components are suppressed by
the original Dev10 ranking / Top-K mechanism.

IMPORTANT
---------
DO NOT MODIFY:
    src/dev10_lite_hot_core.py

Dev29 uses:
    Exact Dev27 selective PET fallback
    Exact Dev10 3D component construction
    Exact Dev10 evaluate_selection()

Only the ranking feature is changed on a COPY of the
component table.

Policies
--------
1. baseline
2. bonus_0.25
3. bonus_0.50
4. bonus_0.75
5. bonus_1.00
6. fallback_priority

Core:
    P70

Top-K:
    4

Original ranking:
    mean_suv_hot3_a0p5
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


# ============================================================
# IMPORT DEV10
# ============================================================

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
MIN_COMPONENT_AREA = 100
MIN_MAX_SUV = 4.0

CORE_POLICY = "P70"
TOP_K = 4

ORIGINAL_RANKING = "mean_suv_hot3_a0p5"

BONUSES = [
    0.25,
    0.50,
    0.75,
    1.00,
]


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "results"
    / "development_cases"
    / "dev29_ranking_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEV27 EXACT PET FALLBACK
# ============================================================

def build_pet_fallback(
    suv_slice
):
    """
    Exact Dev27/Dev26 fallback:

        PET >= 2.25
        component area >= 100
        component max SUV >= 4.0
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
# DEV27 EXACT SELECTIVE FALLBACK
# ============================================================

def build_dev27_experimental_volume(
    case
):
    """
    Exact Dev27 architecture:

        baseline = exact Dev10 candidate volume

        fallback is added ONLY on slices where
        the baseline CT-gated candidate is empty.
    """

    ct = case["ct"]
    suv = case["suv"]

    baseline_volume = (
        dev10.generate_candidate_volume(
            case
        )
    )

    experimental_volume = (
        baseline_volume.copy()
    )

    fallback_slice_count = 0
    fallback_component_count = 0

    for z in range(
        ct.shape[0]
    ):

        # Only use fallback when exact Dev10
        # candidate is completely empty.
        if np.count_nonzero(
            baseline_volume[z]
        ) != 0:
            continue

        (
            fallback,
            accepted
        ) = build_pet_fallback(
            suv[z]
        )

        if accepted:

            experimental_volume[z] = (
                np.maximum(
                    experimental_volume[z],
                    fallback
                )
            )

            fallback_slice_count += 1

            fallback_component_count += (
                len(accepted)
            )

    return (
        baseline_volume,
        experimental_volume,
        fallback_slice_count,
        fallback_component_count,
    )


# ============================================================
# FALLBACK → 3D COMPONENT LINK
# ============================================================

def get_fallback_component_info(
    components,
    cc,
    experimental_volume,
    baseline_volume
):
    """
    Identify which experimental 3D components contain
    newly-added fallback voxels.

    IMPORTANT:
    We compare experimental volume against the exact
    baseline volume. Therefore only genuinely added
    fallback voxels are counted.
    """

    fallback_added = (
        (
            experimental_volume > 0
        )
        &
        (
            baseline_volume == 0
        )
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

        overlap = (
            component_mask
            &
            fallback_added
        )

        fallback_voxels = int(
            overlap.sum()
        )

        if fallback_voxels == 0:
            continue

        component_voxels = int(
            component_mask.sum()
        )

        fallback_slices = int(
            np.count_nonzero(
                np.any(
                    overlap,
                    axis=(1, 2)
                )
            )
        )

        fraction = (
            fallback_voxels
            / component_voxels
            if component_voxels > 0
            else 0.0
        )

        info[label] = {
            "fallback_voxels": fallback_voxels,
            "fallback_slices": fallback_slices,
            "fallback_fraction": fraction,
        }

    return info


# ============================================================
# CREATE RANKING TABLE
# ============================================================

def make_ranking_table(
    components,
    fallback_info,
    policy,
    bonus=None
):
    """
    Create a COPY of the component table and add
    a temporary ranking feature.

    Dev10 itself is never changed.
    """

    ranked = components.copy()

    ranked["is_fallback"] = (
        ranked["label"]
        .astype(int)
        .isin(
            fallback_info.keys()
        )
    )

    ranked["fallback_voxels"] = (
        ranked["label"]
        .astype(int)
        .map(
            lambda x:
                fallback_info.get(
                    x,
                    {}
                ).get(
                    "fallback_voxels",
                    0
                )
        )
    )

    ranked["fallback_fraction"] = (
        ranked["label"]
        .astype(int)
        .map(
            lambda x:
                fallback_info.get(
                    x,
                    {}
                ).get(
                    "fallback_fraction",
                    0.0
                )
        )
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    if policy == "baseline":

        ranked[
            "dev29_rank_score"
        ] = ranked[
            ORIGINAL_RANKING
        ].astype(float)

    # --------------------------------------------------------
    # Controlled fallback bonus
    # --------------------------------------------------------

    elif policy.startswith(
        "bonus_"
    ):

        if bonus is None:
            raise ValueError(
                "Bonus value is required."
            )

        ranked[
            "dev29_rank_score"
        ] = (
            ranked[
                ORIGINAL_RANKING
            ].astype(float)
            +
            np.where(
                ranked["is_fallback"],
                bonus,
                0.0
            )
        )

    # --------------------------------------------------------
    # Fallback priority
    # --------------------------------------------------------

    elif policy == "fallback_priority":

        # Large enough to put fallback components
        # before normal components, while retaining
        # original score within each group.
        max_score = float(
            ranked[
                ORIGINAL_RANKING
            ].max()
        )

        ranked[
            "dev29_rank_score"
        ] = (
            np.where(
                ranked["is_fallback"],
                max_score + 1.0,
                0.0
            )
            +
            ranked[
                ORIGINAL_RANKING
            ].astype(float)
            / (
                max_score + 1.0
            )
        )

    else:

        raise ValueError(
            f"Unknown policy: {policy}"
        )

    return ranked


# ============================================================
# PRINT RANKING AUDIT
# ============================================================

def print_fallback_ranking(
    case_name,
    ranked,
    fallback_info
):
    """
    Print all fallback-containing components in
    their current ranking order.
    """

    fallback_rows = (
        ranked[
            ranked["is_fallback"]
        ]
        .sort_values(
            "dev29_rank_score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print()
    print(
        "Fallback-containing components:"
    )

    if fallback_rows.empty:

        print(
            "  None"
        )

        return

    for rank, (_, row) in enumerate(
        fallback_rows.iterrows(),
        start=1
    ):

        label = int(
            row["label"]
        )

        original_score = float(
            row[
                ORIGINAL_RANKING
            ]
        )

        rank_score = float(
            row[
                "dev29_rank_score"
            ]
        )

        overlap_gt = float(
            row.get(
                "overlap_gt",
                0.0
            )
        )

        print(
            f"  fallback_order={rank:<3} "
            f"label={label:<4} "
            f"original={original_score:>9.6f} "
            f"rank_score={rank_score:>9.6f} "
            f"fallback_voxels="
            f"{int(row['fallback_voxels']):<7} "
            f"GT_overlap="
            f"{overlap_gt:.0f}"
        )


# ============================================================
# PRINT TOP-K
# ============================================================

def print_top_k(
    ranked,
    policy
):
    """
    Print exact Top-K that Dev10 will receive.
    """

    selected = (
        ranked
        .sort_values(
            [
                "dev29_rank_score",
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
        f"{policy} Top-{TOP_K}:"
    )

    for position, (_, row) in enumerate(
        selected.iterrows(),
        start=1
    ):

        label = int(
            row["label"]
        )

        print(
            f"  #{position}: "
            f"label={label:<4} "
            f"original="
            f"{float(row[ORIGINAL_RANKING]):.6f} "
            f"rank_score="
            f"{float(row['dev29_rank_score']):.6f} "
            f"fallback="
            f"{bool(row['is_fallback'])} "
            f"GT_overlap="
            f"{float(row.get('overlap_gt', 0.0)):.0f}"
        )

    return selected


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print(
        "DEV29 — FALLBACK-AWARE RANKING AUDIT"
    )
    print("=" * 90)

    print()
    print("Configuration")
    print("-------------")
    print(
        f"PET threshold       : {PET_THRESHOLD}"
    )
    print(
        f"Minimum area        : {MIN_COMPONENT_AREA}"
    )
    print(
        f"Minimum max SUV     : {MIN_MAX_SUV}"
    )
    print(
        f"Core policy         : {CORE_POLICY}"
    )
    print(
        f"Top-K               : {TOP_K}"
    )
    print(
        f"Original ranking    : {ORIGINAL_RANKING}"
    )

    summary_rows = []
    ranking_rows = []

    for case_name in CASES:

        print()
        print("=" * 90)
        print(case_name)
        print("=" * 90)

        # ----------------------------------------------------
        # Load case
        # ----------------------------------------------------

        case = dev10.load_case(
            case_name
        )

        # ----------------------------------------------------
        # Exact Dev27 volumes
        # ----------------------------------------------------

        (
            baseline_volume,
            experimental_volume,
            fallback_slices,
            fallback_components,
        ) = build_dev27_experimental_volume(
            case
        )

        print(
            f"Fallback slices added     : "
            f"{fallback_slices}"
        )

        print(
            f"Fallback 2D components    : "
            f"{fallback_components}"
        )

        # ----------------------------------------------------
        # Exact Dev10 3D filtering
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
            f"Baseline 3D components    : "
            f"{len(baseline_components)}"
        )

        print(
            f"Experimental 3D components: "
            f"{len(experimental_components)}"
        )

        # ----------------------------------------------------
        # Link newly added fallback voxels
        # ----------------------------------------------------

        fallback_info = (
            get_fallback_component_info(
                experimental_components,
                experimental_cc,
                experimental_volume,
                baseline_volume
            )
        )

        print(
            f"Fallback-linked 3D comps  : "
            f"{len(fallback_info)}"
        )

        # ----------------------------------------------------
        # POLICIES
        # ----------------------------------------------------

        policies = [
            ("baseline", None),
            ("bonus_0.25", 0.25),
            ("bonus_0.50", 0.50),
            ("bonus_0.75", 0.75),
            ("bonus_1.00", 1.00),
            ("fallback_priority", None),
        ]

        # ====================================================
        # EACH RANKING POLICY
        # ====================================================

        for policy, bonus in policies:

            print()
            print("-" * 90)
            print(
                f"POLICY: {policy}"
            )
            print("-" * 90)

            ranked = make_ranking_table(
                experimental_components,
                fallback_info,
                policy,
                bonus
            )

            print_fallback_ranking(
                case_name,
                ranked,
                fallback_info
            )

            selected = print_top_k(
                ranked,
                policy
            )

            # ------------------------------------------------
            # Use EXACT Dev10 evaluation.
            #
            # Only ranking feature is changed.
            # ------------------------------------------------

            metrics = (
                dev10.evaluate_selection(
                    case,
                    ranked,
                    experimental_cc,
                    "dev29_rank_score",
                    CORE_POLICY,
                    TOP_K
                )
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

            # ------------------------------------------------
            # Summary
            # ------------------------------------------------

            summary_rows.append({
                "case": case_name,
                "policy": policy,
                "bonus": (
                    bonus
                    if bonus is not None
                    else 0.0
                ),
                "fallback_slices": (
                    fallback_slices
                ),
                "fallback_2d_components": (
                    fallback_components
                ),
                "fallback_3d_components": (
                    len(fallback_info)
                ),
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "slice_recall": (
                    metrics[
                        "slice_recall"
                    ]
                ),
                "fpr": metrics["fpr"],
                "prediction_voxels": (
                    metrics[
                        "prediction_voxels"
                    ]
                ),
                "top_k_labels": ",".join(
                    str(
                        int(x)
                    )
                    for x in selected[
                        "label"
                    ]
                ),
            })

            # ------------------------------------------------
            # Complete ranking table
            # ------------------------------------------------

            for _, row in (
                ranked
                .sort_values(
                    "dev29_rank_score",
                    ascending=False
                )
                .reset_index(
                    drop=True
                )
                .iterrows()
            ):

                ranking_rows.append({
                    "case": case_name,
                    "policy": policy,
                    "rank": (
                        len(
                            [
                                r
                                for r in ranking_rows
                                if (
                                    r["case"]
                                    == case_name
                                    and
                                    r["policy"]
                                    == policy
                                )
                            ]
                        )
                        + 1
                    ),
                    "label": int(
                        row["label"]
                    ),
                    "original_score": float(
                        row[
                            ORIGINAL_RANKING
                        ]
                    ),
                    "dev29_rank_score": float(
                        row[
                            "dev29_rank_score"
                        ]
                    ),
                    "is_fallback": bool(
                        row[
                            "is_fallback"
                        ]
                    ),
                    "fallback_voxels": int(
                        row[
                            "fallback_voxels"
                        ]
                    ),
                    "fallback_fraction": float(
                        row[
                            "fallback_fraction"
                        ]
                    ),
                    "gt_overlap": float(
                        row.get(
                            "overlap_gt",
                            0.0
                        )
                    ),
                    "dice_gt": float(
                        row.get(
                            "dice_gt",
                            0.0
                        )
                    ),
                    "mean_suv": float(
                        row[
                            "mean_suv"
                        ]
                    ),
                    "selected_topk": (
                        int(
                            row["label"]
                        )
                        in selected[
                            "label"
                        ].astype(int).tolist()
                    ),
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
        / "dev29_policy_summary.csv"
    )

    ranking_path = (
        OUTPUT_DIR
        / "dev29_component_rankings.csv"
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
    # MACRO SUMMARY
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
    print("=" * 90)
    print("DEV29 MACRO SUMMARY")
    print("=" * 90)

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

    baseline_row = macro[
        macro["policy"]
        == "baseline"
    ]

    if len(baseline_row) == 1:

        baseline = (
            baseline_row.iloc[0]
        )

        print()
        print("=" * 90)
        print("DELTA FROM BASELINE")
        print("=" * 90)

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
    # FILES
    # ========================================================

    print()
    print("=" * 90)
    print("FILES")
    print("=" * 90)

    print(summary_path)
    print(ranking_path)

    print()
    print(
        "DEV29 COMPLETE"
    )


if __name__ == "__main__":
    main()