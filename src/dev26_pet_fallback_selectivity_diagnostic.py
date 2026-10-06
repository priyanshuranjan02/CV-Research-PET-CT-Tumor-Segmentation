"""
Dev26 — PET Fallback Selectivity Diagnostic

Purpose:
    Analyze how selective a PET-only fallback should be when
    the Dev10 CT-gated candidate is empty.

Dev10 is NOT modified.

We test:

    PET threshold:
        2.25, 3.0, 4.0

    Minimum PET component area:
        20, 50, 100 pixels

    Minimum component MAX SUV:
        2.25, 3.0, 4.0

The fallback is only considered when the exact Dev10
CT-gated candidate is empty.

Evaluation is performed against GT for diagnosis only.
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

RESULT_DIR = (
    ROOT
    / "results"
    / "development_cases"
    / "dev26_pet_fallback_selectivity_results"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

sys.path.insert(0, str(SRC))

import dev10_lite_hot_core as dev10


# ============================================================
# DEVELOPMENT CASES
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
# EXPERIMENT PARAMETERS
# ============================================================

PET_THRESHOLDS = [
    2.25,
    3.0,
    4.0,
]

MIN_COMPONENT_AREAS = [
    20,
    50,
    100,
]

MIN_MAX_SUV = [
    2.25,
    3.0,
    4.0,
]


# ============================================================
# DEV10 CT CONTOUR
# ============================================================

def get_dev10_ct_contours(
    ct_slice,
    body_slice
):
    """
    Reproduce the Dev10 CT contour stage exactly.
    """

    ct_display = dev10.normalize_ct(
        ct_slice
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    opened = cv2.morphologyEx(
        ct_display,
        cv2.MORPH_OPEN,
        kernel
    )

    binary = cv2.adaptiveThreshold(
        opened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        3
    )

    binary[
        body_slice == 0
    ] = 0

    binary[:5, :] = 0
    binary[-5:, :] = 0
    binary[:, :5] = 0
    binary[:, -5:] = 0

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return [
        c
        for c in contours
        if cv2.contourArea(c) >= 100
    ]


# ============================================================
# DEV10 CT-GATED PET CANDIDATE
# ============================================================

def build_dev10_candidate(
    ct_slice,
    suv_slice,
    body_slice,
    pet_threshold
):
    """
    Exact 2D equivalent of the Dev10 CT-gated candidate stage.
    """

    contours = get_dev10_ct_contours(
        ct_slice,
        body_slice
    )

    contour_mask = np.zeros(
        ct_slice.shape,
        dtype=np.uint8
    )

    if contours:

        cv2.drawContours(
            contour_mask,
            contours,
            -1,
            1,
            -1
        )

    pet_hot = (
        suv_slice >= pet_threshold
    )

    gated = np.logical_and(
        contour_mask > 0,
        pet_hot
    )

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            gated.astype(np.uint8),
            connectivity=8
        )
    )

    candidate = np.zeros(
        ct_slice.shape,
        dtype=np.uint8
    )

    components = []

    for label in range(
        1,
        num_labels
    ):

        area = int(
            stats[label, cv2.CC_STAT_AREA]
        )

        if area < 20:
            continue

        component = (
            labels == label
        )

        candidate[component] = 1

        components.append(
            {
                "label": label,
                "area": area,
            }
        )

    return (
        candidate,
        components,
        contours
    )


# ============================================================
# PET-ONLY COMPONENT EXTRACTION
# ============================================================

def get_pet_components(
    suv_slice,
    pet_threshold
):
    """
    Extract all PET-only connected components above
    the selected PET threshold.

    No area or SUV filtering is applied here.
    """

    pet_hot = (
        suv_slice >= pet_threshold
    )

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            pet_hot.astype(np.uint8),
            connectivity=8
        )
    )

    components = []

    for label in range(
        1,
        num_labels
    ):

        component = (
            labels == label
        )

        area = int(
            stats[label, cv2.CC_STAT_AREA]
        )

        values = suv_slice[
            component
        ]

        if values.size == 0:
            continue

        max_suv = float(
            np.max(values)
        )

        mean_suv = float(
            np.mean(values)
        )

        min_suv = float(
            np.min(values)
        )

        # Bounding box
        x = int(
            stats[
                label,
                cv2.CC_STAT_LEFT
            ]
        )

        y = int(
            stats[
                label,
                cv2.CC_STAT_TOP
            ]
        )

        w = int(
            stats[
                label,
                cv2.CC_STAT_WIDTH
            ]
        )

        h = int(
            stats[
                label,
                cv2.CC_STAT_HEIGHT
            ]
        )

        bbox_area = max(
            w * h,
            1
        )

        fill_ratio = (
            area / bbox_area
        )

        components.append(
            {
                "label": label,
                "area": area,
                "max_suv": max_suv,
                "mean_suv": mean_suv,
                "min_suv": min_suv,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "bbox_area": bbox_area,
                "fill_ratio": fill_ratio,
                "mask": component,
            }
        )

    return components


# ============================================================
# BUILD SELECTIVE FALLBACK
# ============================================================

def build_selective_fallback(
    suv_slice,
    pet_threshold,
    min_area,
    min_max_suv
):
    """
    Build PET-only fallback using:

        PET threshold
        +
        minimum component area
        +
        minimum maximum SUV
    """

    components = get_pet_components(
        suv_slice,
        pet_threshold
    )

    candidate = np.zeros(
        suv_slice.shape,
        dtype=np.uint8
    )

    accepted = []

    for component in components:

        if component["area"] < min_area:
            continue

        if component["max_suv"] < min_max_suv:
            continue

        candidate[
            component["mask"]
        ] = 1

        accepted.append(
            component
        )

    return (
        candidate,
        accepted,
        components
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction,
    gt
):
    """
    Binary segmentation metrics.
    """

    prediction = (
        prediction.astype(bool)
    )

    gt = (
        gt.astype(bool)
    )

    gt_area = int(
        gt.sum()
    )

    pred_area = int(
        prediction.sum()
    )

    overlap = int(
        np.logical_and(
            prediction,
            gt
        ).sum()
    )

    union = int(
        np.logical_or(
            prediction,
            gt
        ).sum()
    )

    dice = (
        2.0 * overlap /
        (pred_area + gt_area)
        if pred_area + gt_area > 0
        else 0.0
    )

    iou = (
        overlap / union
        if union > 0
        else 0.0
    )

    coverage = (
        overlap / gt_area
        if gt_area > 0
        else 0.0
    )

    false_positive = int(
        np.logical_and(
            prediction,
            ~gt
        ).sum()
    )

    fpr = (
        false_positive /
        prediction.size
    )

    return {
        "gt_area": gt_area,
        "prediction_area": pred_area,
        "overlap": overlap,
        "dice": float(dice),
        "iou": float(iou),
        "coverage": float(coverage),
        "fpr": float(fpr),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 110)
    print(
        "DEV26 — PET FALLBACK SELECTIVITY DIAGNOSTIC"
    )
    print("=" * 110)

    all_rows = []

    for case_name in CASES:

        print("\n" + "=" * 110)
        print(
            f"===== CASE: {case_name} ====="
        )
        print("=" * 110)

        case = dev10.load_case(
            case_name
        )

        ct = case["ct"]
        suv = case["suv"]
        gt = case["gt"]

        body = dev10.build_union_body_mask(
            ct
        )

        gt_bool = (
            gt.astype(bool)
        )

        positive_slices = np.where(
            np.any(
                gt_bool,
                axis=(1, 2)
            )
        )[0]

        print(
            f"GT-positive slices: "
            f"{len(positive_slices)}"
        )

        for z in positive_slices:

            gt_slice = gt_bool[z]

            for pet_threshold in PET_THRESHOLDS:

                # ------------------------------------------------
                # Exact Dev10 candidate
                # ------------------------------------------------

                (
                    baseline,
                    baseline_components,
                    contours
                ) = build_dev10_candidate(
                    ct[z],
                    suv[z],
                    body[z],
                    pet_threshold
                )

                baseline_metrics = (
                    calculate_metrics(
                        baseline,
                        gt_slice
                    )
                )

                ct_gate_empty = (
                    len(
                        baseline_components
                    ) == 0
                )

                # ------------------------------------------------
                # Only investigate fallback when
                # Dev10 candidate is empty.
                # ------------------------------------------------

                if not ct_gate_empty:
                    continue

                # ------------------------------------------------
                # Test all selectivity combinations
                # ------------------------------------------------

                for min_area in MIN_COMPONENT_AREAS:

                    for min_max_suv in MIN_MAX_SUV:

                        (
                            fallback,
                            accepted_components,
                            all_components
                        ) = build_selective_fallback(
                            suv[z],
                            pet_threshold,
                            min_area,
                            min_max_suv
                        )

                        fallback_metrics = (
                            calculate_metrics(
                                fallback,
                                gt_slice
                            )
                        )

                        baseline_overlap = (
                            baseline_metrics[
                                "overlap"
                            ] > 0
                        )

                        fallback_overlap = (
                            fallback_metrics[
                                "overlap"
                            ] > 0
                        )

                        recovered = (
                            (not baseline_overlap)
                            and
                            fallback_overlap
                        )

                        # Component statistics
                        if accepted_components:

                            max_component_suv = max(
                                c["max_suv"]
                                for c in accepted_components
                            )

                            largest_component = max(
                                c["area"]
                                for c in accepted_components
                            )

                            mean_component_suv = np.mean(
                                [
                                    c["mean_suv"]
                                    for c in accepted_components
                                ]
                            )

                        else:

                            max_component_suv = 0.0
                            largest_component = 0
                            mean_component_suv = 0.0

                        all_rows.append(
                            {
                                "case_id": case_name,
                                "slice": int(z),

                                "pet_threshold":
                                    pet_threshold,

                                "min_area":
                                    min_area,

                                "min_max_suv":
                                    min_max_suv,

                                "ct_contours":
                                    len(contours),

                                "baseline_components":
                                    len(baseline_components),

                                "total_pet_components":
                                    len(all_components),

                                "accepted_components":
                                    len(accepted_components),

                                "largest_component_area":
                                    largest_component,

                                "max_accepted_component_suv":
                                    max_component_suv,

                                "mean_accepted_component_suv":
                                    float(
                                        mean_component_suv
                                    ),

                                "baseline_dice":
                                    baseline_metrics[
                                        "dice"
                                    ],

                                "baseline_coverage":
                                    baseline_metrics[
                                        "coverage"
                                    ],

                                "baseline_iou":
                                    baseline_metrics[
                                        "iou"
                                    ],

                                "baseline_fpr":
                                    baseline_metrics[
                                        "fpr"
                                    ],

                                "fallback_dice":
                                    fallback_metrics[
                                        "dice"
                                    ],

                                "fallback_coverage":
                                    fallback_metrics[
                                        "coverage"
                                    ],

                                "fallback_iou":
                                    fallback_metrics[
                                        "iou"
                                    ],

                                "fallback_fpr":
                                    fallback_metrics[
                                        "fpr"
                                    ],

                                "recovered":
                                    recovered,
                            }
                        )

    df = pd.DataFrame(
        all_rows
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_rows = []

    for pet_threshold in PET_THRESHOLDS:

        for min_area in MIN_COMPONENT_AREAS:

            for min_max_suv in MIN_MAX_SUV:

                sub = df[
                    (df["pet_threshold"] == pet_threshold)
                    &
                    (df["min_area"] == min_area)
                    &
                    (df["min_max_suv"] == min_max_suv)
                ]

                recovered = sub[
                    sub["recovered"]
                ]

                summary_rows.append(
                    {
                        "pet_threshold":
                            pet_threshold,

                        "min_area":
                            min_area,

                        "min_max_suv":
                            min_max_suv,

                        "ct_failure_slices":
                            len(sub),

                        "recovered_slices":
                            len(recovered),

                        "recovery_rate":
                            (
                                len(recovered) /
                                len(sub)
                                if len(sub) > 0
                                else 0.0
                            ),

                        "mean_fallback_dice":
                            sub[
                                "fallback_dice"
                            ].mean(),

                        "mean_fallback_coverage":
                            sub[
                                "fallback_coverage"
                            ].mean(),

                        "mean_fallback_iou":
                            sub[
                                "fallback_iou"
                            ].mean(),

                        "mean_fallback_fpr":
                            sub[
                                "fallback_fpr"
                            ].mean(),

                        "mean_accepted_components":
                            sub[
                                "accepted_components"
                            ].mean(),

                        "mean_largest_component_area":
                            sub[
                                "largest_component_area"
                            ].mean(),
                    }
                )

    summary = pd.DataFrame(
        summary_rows
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n" + "=" * 110)
    print(
        "DEV26 SELECTIVITY SUMMARY"
    )
    print("=" * 110)

    print(
        summary.to_string(
            index=False
        )
    )

    # ========================================================
    # RECOVERY-FOCUSED SUMMARY
    # ========================================================

    recovery_summary = (
        summary
        .sort_values(
            [
                "recovered_slices",
                "mean_fallback_dice",
                "mean_fallback_coverage",
            ],
            ascending=False
        )
        .head(15)
    )

    print("\n" + "=" * 110)
    print(
        "TOP 15 CONFIGURATIONS BY RECOVERY / QUALITY"
    )
    print("=" * 110)

    print(
        recovery_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # CASE-SPECIFIC 0b57 SUMMARY
    # ========================================================

    case_0b57 = df[
        df["case_id"]
        ==
        "PETCT_0b57b247b6"
    ]

    case_0b57_summary = (
        case_0b57
        .groupby(
            [
                "pet_threshold",
                "min_area",
                "min_max_suv",
            ],
            as_index=False
        )
        .agg(
            ct_failure_slices=(
                "slice",
                "count"
            ),
            recovered_slices=(
                "recovered",
                "sum"
            ),
            mean_dice=(
                "fallback_dice",
                "mean"
            ),
            mean_coverage=(
                "fallback_coverage",
                "mean"
            ),
            mean_fpr=(
                "fallback_fpr",
                "mean"
            ),
            mean_components=(
                "accepted_components",
                "mean"
            ),
        )
    )

    print("\n" + "=" * 110)
    print(
        "PETCT_0b57 SELECTIVITY RESULTS"
    )
    print("=" * 110)

    print(
        case_0b57_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    slice_path = (
        RESULT_DIR /
        "dev26_slice_results.csv"
    )

    summary_path = (
        RESULT_DIR /
        "dev26_selectivity_summary.csv"
    )

    top_path = (
        RESULT_DIR /
        "dev26_top15_configurations.csv"
    )

    case_path = (
        RESULT_DIR /
        "dev26_0b57_summary.csv"
    )

    df.to_csv(
        slice_path,
        index=False
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    recovery_summary.to_csv(
        top_path,
        index=False
    )

    case_0b57_summary.to_csv(
        case_path,
        index=False
    )

    print("\n" + "=" * 110)
    print(
        "DEV26 COMPLETE"
    )
    print("=" * 110)

    print(
        f"Slice results:\n{slice_path}"
    )

    print(
        f"Selectivity summary:\n{summary_path}"
    )

    print(
        f"Top configurations:\n{top_path}"
    )

    print(
        f"0b57 summary:\n{case_path}"
    )

    print(
        "\nDev10 was NOT modified."
    )


if __name__ == "__main__":
    main()