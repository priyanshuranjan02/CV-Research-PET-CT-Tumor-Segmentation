"""
Dev31 — Fallback Spatial / 3D Structural Ranking

Goal
----
Determine whether structural properties of fallback-linked
3D components can distinguish useful fallback candidates
from false-positive fallback candidates.

IMPORTANT
---------
src/dev10_lite_hot_core.py is NOT modified.

The exact Dev10:
    - candidate generation
    - 3D component filtering
    - core construction
    - Top-K selection
    - evaluation

are reused.

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
# DEV10 CONFIGURATION
# ============================================================

CORE_POLICY = "P70"
TOP_K = 4

ORIGINAL_RANKING = "mean_suv_hot3_a0p5"


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "results"
    / "development_cases"
    / "dev31_fallback_structural_ranking"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PET FALLBACK
# ============================================================

def build_pet_fallback(suv_slice):

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

    return fallback, accepted


# ============================================================
# EXACT DEV27 EXPERIMENTAL VOLUME
# ============================================================

def build_experimental_volume(case):

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

        fallback, accepted = (
            build_pet_fallback(
                case["suv"][z]
            )
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
# CONNECTED COMPONENT INFORMATION
# ============================================================

def component_bbox(mask):

    coords = np.argwhere(mask)

    if coords.size == 0:
        return None

    z0, y0, x0 = np.min(
        coords,
        axis=0
    )

    z1, y1, x1 = np.max(
        coords,
        axis=0
    )

    return (
        int(z0),
        int(z1),
        int(y0),
        int(y1),
        int(x0),
        int(x1)
    )


# ============================================================
# FALLBACK STRUCTURAL FEATURES
# ============================================================

def get_fallback_structural_info(
    components,
    cc,
    baseline_volume,
    experimental_volume,
    suv
):

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

        # ----------------------------------------------------
        # Total component geometry
        # ----------------------------------------------------

        component_voxels = int(
            component_mask.sum()
        )

        bbox = component_bbox(
            component_mask
        )

        if bbox is None:
            continue

        (
            z0,
            z1,
            y0,
            y1,
            x0,
            x1
        ) = bbox

        z_span = (
            z1 - z0 + 1
        )

        y_span = (
            y1 - y0 + 1
        )

        x_span = (
            x1 - x0 + 1
        )

        # ----------------------------------------------------
        # Fallback slice persistence
        # ----------------------------------------------------

        fallback_z = np.any(
            fallback_mask,
            axis=(1, 2)
        )

        fallback_slice_indices = (
            np.where(
                fallback_z
            )[0]
        )

        fallback_slice_count = int(
            len(
                fallback_slice_indices
            )
        )

        # ----------------------------------------------------
        # Consecutive fallback slice runs
        # ----------------------------------------------------

        longest_run = 0

        if fallback_slice_count > 0:

            current_run = 1

            for i in range(
                1,
                fallback_slice_count
            ):

                if (
                    fallback_slice_indices[i]
                    ==
                    fallback_slice_indices[i - 1] + 1
                ):
                    current_run += 1
                else:
                    longest_run = max(
                        longest_run,
                        current_run
                    )
                    current_run = 1

            longest_run = max(
                longest_run,
                current_run
            )

        # ----------------------------------------------------
        # Spatial compactness
        #
        # Number of fallback voxels relative to bounding-box
        # volume.
        # ----------------------------------------------------

        bbox_volume = (
            z_span
            *
            y_span
            *
            x_span
        )

        compactness = (
            fallback_voxels
            /
            bbox_volume
            if bbox_volume > 0
            else 0.0
        )

        # ----------------------------------------------------
        # Fallback fraction
        # ----------------------------------------------------

        fallback_fraction = (
            fallback_voxels
            /
            component_voxels
            if component_voxels > 0
            else 0.0
        )

        # ----------------------------------------------------
        # PET statistics
        # ----------------------------------------------------

        fallback_values = suv[
            fallback_mask
        ]

        fallback_mean_suv = (
            float(
                np.mean(
                    fallback_values
                )
            )
            if fallback_values.size
            else 0.0
        )

        fallback_max_suv = (
            float(
                np.max(
                    fallback_values
                )
            )
            if fallback_values.size
            else 0.0
        )

        info[label] = {

            "fallback_voxels":
                fallback_voxels,

            "component_voxels":
                component_voxels,

            "fallback_fraction":
                fallback_fraction,

            "fallback_slice_count":
                fallback_slice_count,

            "fallback_z_span":
                z_span,

            "longest_fallback_run":
                longest_run,

            "bbox_volume":
                bbox_volume,

            "compactness":
                compactness,

            "fallback_mean_suv":
                fallback_mean_suv,

            "fallback_max_suv":
                fallback_max_suv,

            "z_start":
                z0,

            "z_end":
                z1,

            "y_span":
                y_span,

            "x_span":
                x_span,
        }

    return info


# ============================================================
# PROXIMITY TO BASELINE COMPONENTS
# ============================================================

def get_proximity_to_baseline(
    experimental_cc,
    fallback_mask,
    baseline_volume
):
    """
    Approximate spatial proximity of fallback voxels to
    existing baseline candidate voxels.

    Returns minimum Euclidean distance in pixels.

    No GT information is used.
    """

    baseline_coords = np.argwhere(
        baseline_volume > 0
    )

    fallback_coords = np.argwhere(
        fallback_mask
    )

    if (
        baseline_coords.size == 0
        or fallback_coords.size == 0
    ):
        return np.inf

    # Downsample if unusually large to keep diagnostic
    # computation manageable.

    if len(baseline_coords) > 5000:

        idx = np.linspace(
            0,
            len(baseline_coords) - 1,
            5000
        ).astype(int)

        baseline_coords = (
            baseline_coords[idx]
        )

    if len(fallback_coords) > 5000:

        idx = np.linspace(
            0,
            len(fallback_coords) - 1,
            5000
        ).astype(int)

        fallback_coords = (
            fallback_coords[idx]
        )

    minimum_distance = np.inf

    # Process fallback coordinates in batches.
    batch_size = 500

    for start in range(
        0,
        len(fallback_coords),
        batch_size
    ):

        batch = fallback_coords[
            start:
            start + batch_size
        ]

        diff = (
            batch[:, None, :]
            -
            baseline_coords[None, :, :]
        )

        dist_sq = np.sum(
            diff * diff,
            axis=2
        )

        batch_min = np.sqrt(
            np.min(
                dist_sq
            )
        )

        minimum_distance = min(
            minimum_distance,
            float(batch_min)
        )

    return minimum_distance


# ============================================================
# BUILD RANKING TABLE
# ============================================================

def build_ranking_table(
    components,
    structural_info,
    cc,
    baseline_volume,
    experimental_volume
):

    df = components.copy()

    labels = (
        df["label"]
        .astype(int)
        .tolist()
    )

    df["is_fallback"] = [
        label in structural_info
        for label in labels
    ]

    def get_value(
        label,
        key,
        default=0.0
    ):

        return structural_info.get(
            int(label),
            {}
        ).get(
            key,
            default
        )

    feature_names = [
        "fallback_voxels",
        "component_voxels",
        "fallback_fraction",
        "fallback_slice_count",
        "fallback_z_span",
        "longest_fallback_run",
        "bbox_volume",
        "compactness",
        "fallback_mean_suv",
        "fallback_max_suv",
        "z_start",
        "z_end",
        "y_span",
        "x_span",
    ]

    for feature in feature_names:

        df[feature] = (
            df["label"]
            .map(
                lambda x:
                    get_value(
                        x,
                        feature,
                        0.0
                    )
            )
        )

    # --------------------------------------------------------
    # Proximity
    # --------------------------------------------------------

    proximity = {}

    fallback_added = (
        (experimental_volume > 0)
        &
        (baseline_volume == 0)
    )

    for label in labels:

        component_mask = (
            cc == label
        )

        fallback_mask = (
            component_mask
            &
            fallback_added
        )

        if not np.any(
            fallback_mask
        ):
            continue

        proximity[label] = (
            get_proximity_to_baseline(
                cc,
                fallback_mask,
                baseline_volume
            )
        )

    df["distance_to_baseline"] = (
        df["label"]
        .map(
            lambda x:
                proximity.get(
                    int(x),
                    np.inf
                )
        )
    )

    return df


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_active(
    values,
    mask
):

    values = np.asarray(
        values,
        dtype=float
    )

    output = np.zeros_like(
        values,
        dtype=float
    )

    active = np.asarray(
        mask,
        dtype=bool
    )

    if not np.any(active):
        return output

    active_values = (
        values[active]
    )

    vmin = np.min(
        active_values
    )

    vmax = np.max(
        active_values
    )

    if vmax <= vmin:
        output[active] = 1.0
        return output

    output[active] = (
        (
            active_values
            - vmin
        )
        /
        (
            vmax
            - vmin
        )
    )

    return output


# ============================================================
# ADD STRUCTURAL RANKING FEATURES
# ============================================================

def add_ranking_features(df):

    base = (
        df[
            ORIGINAL_RANKING
        ]
        .astype(float)
        .to_numpy()
    )

    fallback = (
        df[
            "is_fallback"
        ]
        .astype(bool)
        .to_numpy()
    )

    volume = (
        np.log1p(
            df[
                "fallback_voxels"
            ]
            .astype(float)
            .to_numpy()
        )
    )

    persistence = (
        df[
            "fallback_slice_count"
        ]
        .astype(float)
        .to_numpy()
    )

    longest_run = (
        df[
            "longest_fallback_run"
        ]
        .astype(float)
        .to_numpy()
    )

    compactness = (
        df[
            "compactness"
        ]
        .astype(float)
        .to_numpy()
    )

    distance = (
        df[
            "distance_to_baseline"
        ]
        .astype(float)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Normalize only fallback components.
    # Non-fallback components receive zero structural bonus.
    # --------------------------------------------------------

    volume_n = normalize_active(
        volume,
        fallback
    )

    persistence_n = normalize_active(
        persistence,
        fallback
    )

    longest_run_n = normalize_active(
        longest_run,
        fallback
    )

    compactness_n = normalize_active(
        compactness,
        fallback
    )

    # Smaller distance is better.
    proximity_score = np.zeros_like(
        base,
        dtype=float
    )

    finite_fallback = (
        fallback
        &
        np.isfinite(distance)
    )

    if np.any(
        finite_fallback
    ):

        d = distance[
            finite_fallback
        ]

        dmax = np.max(d)

        if dmax > 0:

            proximity_score[
                finite_fallback
            ] = (
                1.0
                -
                (
                    d / dmax
                )
            )

        else:

            proximity_score[
                finite_fallback
            ] = 1.0

    # ========================================================
    # POLICY 1 — 3D VOLUME
    # ========================================================

    df[
        "rank_volume"
    ] = (
        base
        +
        volume_n
    )

    # ========================================================
    # POLICY 2 — SLICE PERSISTENCE
    # ========================================================

    df[
        "rank_persistence"
    ] = (
        base
        +
        persistence_n
    )

    # ========================================================
    # POLICY 3 — LONGEST CONTINUOUS RUN
    # ========================================================

    df[
        "rank_continuity"
    ] = (
        base
        +
        longest_run_n
    )

    # ========================================================
    # POLICY 4 — COMPACTNESS
    # ========================================================

    df[
        "rank_compactness"
    ] = (
        base
        +
        compactness_n
    )

    # ========================================================
    # POLICY 5 — COMBINED STRUCTURE
    # ========================================================

    structural = (
        0.30 * volume_n
        +
        0.30 * persistence_n
        +
        0.25 * longest_run_n
        +
        0.15 * compactness_n
    )

    df[
        "structural_score"
    ] = structural

    df[
        "rank_structural"
    ] = (
        base
        +
        structural
    )

    # ========================================================
    # POLICY 6 — STRUCTURE + PROXIMITY
    # ========================================================

    structural_proximity = (
        0.25 * volume_n
        +
        0.25 * persistence_n
        +
        0.20 * longest_run_n
        +
        0.15 * compactness_n
        +
        0.15 * proximity_score
    )

    df[
        "structural_proximity_score"
    ] = (
        structural_proximity
    )

    df[
        "rank_structural_proximity"
    ] = (
        base
        +
        structural_proximity
    )

    return df


# ============================================================
# TOP-K DISPLAY
# ============================================================

def print_top_k(
    df,
    feature,
    name
):

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
            f"score={float(row[feature]):.6f} "
            f"fallback={bool(row['is_fallback'])} "
            f"vox={int(row['fallback_voxels'])} "
            f"slices={int(row['fallback_slice_count'])} "
            f"run={int(row['longest_fallback_run'])} "
            f"compact={float(row['compactness']):.6f} "
            f"dist={float(row['distance_to_baseline']):.2f} "
            f"GT_overlap={float(row.get('overlap_gt', 0)):.0f}"
        )

    return selected


# ============================================================
# EVALUATE
# ============================================================

def evaluate_policy(
    case,
    ranked,
    cc,
    feature
):

    return (
        dev10.evaluate_selection(
            case,
            ranked,
            cc,
            feature,
            CORE_POLICY,
            TOP_K
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print(
        "DEV31 — FALLBACK SPATIAL / 3D STRUCTURAL RANKING"
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
            ORIGINAL_RANKING
        ),
        (
            "volume",
            "rank_volume"
        ),
        (
            "persistence",
            "rank_persistence"
        ),
        (
            "continuity",
            "rank_continuity"
        ),
        (
            "compactness",
            "rank_compactness"
        ),
        (
            "structural",
            "rank_structural"
        ),
        (
            "structural_proximity",
            "rank_structural_proximity"
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
            fallback_components
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
        # Exact Dev10 3D construction
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
        # Structural information
        # ----------------------------------------------------

        structural_info = (
            get_fallback_structural_info(
                experimental_components,
                experimental_cc,
                baseline_volume,
                experimental_volume,
                case["suv"]
            )
        )

        print(
            f"Fallback-linked comps: "
            f"{len(structural_info)}"
        )

        # ----------------------------------------------------
        # Ranking table
        # ----------------------------------------------------

        ranked = build_ranking_table(
            experimental_components,
            structural_info,
            experimental_cc,
            baseline_volume,
            experimental_volume
        )

        ranked = add_ranking_features(
            ranked
        )

        # ----------------------------------------------------
        # Policies
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
                    len(structural_info),

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
            # Save ranking audit
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
                            row["is_fallback"]
                        ),

                    "fallback_voxels":
                        int(
                            row[
                                "fallback_voxels"
                            ]
                        ),

                    "component_voxels":
                        int(
                            row[
                                "component_voxels"
                            ]
                        ),

                    "fallback_slice_count":
                        int(
                            row[
                                "fallback_slice_count"
                            ]
                        ),

                    "longest_fallback_run":
                        int(
                            row[
                                "longest_fallback_run"
                            ]
                        ),

                    "fallback_fraction":
                        float(
                            row[
                                "fallback_fraction"
                            ]
                        ),

                    "compactness":
                        float(
                            row[
                                "compactness"
                            ]
                        ),

                    "distance_to_baseline":
                        float(
                            row[
                                "distance_to_baseline"
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
        / "dev31_policy_summary.csv"
    )

    ranking_path = (
        OUTPUT_DIR
        / "dev31_component_rankings.csv"
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
        "DEV31 MACRO SUMMARY"
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
            "DEV31 DELTA FROM BASELINE"
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
    # OUTPUT
    # ========================================================

    print()
    print("=" * 100)
    print(
        "OUTPUT FILES"
    )
    print("=" * 100)

    print(
        summary_path
    )

    print(
        ranking_path
    )

    print()
    print(
        "DEV31 COMPLETE"
    )


if __name__ == "__main__":
    main()