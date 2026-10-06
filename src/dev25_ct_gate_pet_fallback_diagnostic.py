"""
Dev25 — CT-Gate PET Fallback Refinement Diagnostic

Purpose:
    Test a conservative PET fallback for slices where
    the Dev10 CT contour gate produces no PET candidate.

    Baseline:
        Exact Dev10 candidate generation.

    Experimental:
        If Dev10 produces no candidate on a slice,
        test PET-only connected components.

    PET fallback is NOT added to Dev10 itself.

    Dev10 remains untouched.
"""

from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

RESULT_DIR = (
    ROOT
    / "results"
    / "development_cases"
    / "dev25_ct_gate_pet_fallback_results"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

sys.path.insert(0, str(SRC))

import dev10_lite_hot_core as dev10


CASES = [
    "PETCT_0168f65af8",
    "PETCT_04606080a0",
    "PETCT_04ab5c61c9",
    "PETCT_0b57b247b6",
    "PETCT_11afab3485",
    "PETCT_185da4c8b6",
]

PET_THRESHOLDS = [
    2.25,
    3.0,
    4.0,
]

MIN_PET_COMPONENT_AREA = 20


def get_dev10_ct_contours(
    ct_slice,
    body_slice
):
    """
    Exact Dev10 CT contour generation.
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


def build_dev10_slice_candidate(
    ct_slice,
    suv_slice,
    body_slice,
    pet_threshold
):
    """
    Reproduce the essential Dev10 2D candidate logic.
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

        if area < MIN_PET_COMPONENT_AREA:
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

    return candidate, components, contours


def build_pet_only_candidate(
    suv_slice,
    pet_threshold
):
    """
    PET-only fallback.

    No CT contour restriction.

    Only PET connected components >= 20 pixels
    are retained.
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

    candidate = np.zeros(
        suv_slice.shape,
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

        if area < MIN_PET_COMPONENT_AREA:
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

    return candidate, components


def calculate_metrics(
    prediction,
    gt
):
    """
    Calculate binary segmentation metrics.
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

    false_positive = (
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
        "coverage": float(coverage),
        "dice": float(dice),
        "iou": float(iou),
        "fpr": float(fpr),
    }


def main():

    print("=" * 100)
    print(
        "DEV25 — CT-GATE PET FALLBACK "
        "REFINEMENT DIAGNOSTIC"
    )
    print("=" * 100)

    all_rows = []

    for case_name in CASES:

        print("\n" + "=" * 100)
        print(
            f"===== CASE: {case_name} ====="
        )
        print("=" * 100)

        case = dev10.load_case(
            case_name
        )

        ct = case["ct"]
        suv = case["suv"]
        gt = case["gt"]

        body = dev10.build_union_body_mask(
            ct
        )

        gt_bool = gt.astype(bool)

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

            for threshold in PET_THRESHOLDS:

                baseline, baseline_components, contours = (
                    build_dev10_slice_candidate(
                        ct[z],
                        suv[z],
                        body[z],
                        threshold
                    )
                )

                baseline_metrics = (
                    calculate_metrics(
                        baseline,
                        gt_slice
                    )
                )

                ct_gate_empty = (
                    len(baseline_components) == 0
                )

                # PET fallback only when
                # baseline CT-gated candidate is empty.
                if ct_gate_empty:

                    fallback, fallback_components = (
                        build_pet_only_candidate(
                            suv[z],
                            threshold
                        )
                    )

                else:

                    fallback = baseline.copy()

                    fallback_components = []

                fallback_metrics = (
                    calculate_metrics(
                        fallback,
                        gt_slice
                    )
                )

                baseline_overlap = (
                    baseline_metrics["overlap"] > 0
                )

                fallback_overlap = (
                    fallback_metrics["overlap"] > 0
                )

                recovered = (
                    (not baseline_overlap)
                    and fallback_overlap
                )

                row = {
                    "case_id": case_name,
                    "slice": int(z),
                    "threshold": threshold,

                    "ct_contours": len(contours),

                    "baseline_components":
                        len(baseline_components),

                    "fallback_components":
                        len(fallback_components),

                    "ct_gate_empty":
                        ct_gate_empty,

                    "baseline_dice":
                        baseline_metrics["dice"],

                    "baseline_iou":
                        baseline_metrics["iou"],

                    "baseline_coverage":
                        baseline_metrics["coverage"],

                    "baseline_fpr":
                        baseline_metrics["fpr"],

                    "fallback_dice":
                        fallback_metrics["dice"],

                    "fallback_iou":
                        fallback_metrics["iou"],

                    "fallback_coverage":
                        fallback_metrics["coverage"],

                    "fallback_fpr":
                        fallback_metrics["fpr"],

                    "recovered_by_fallback":
                        recovered,
                }

                all_rows.append(
                    row
                )

    df = pd.DataFrame(
        all_rows
    )

    # ------------------------------------------------
    # Slice-level output
    # ------------------------------------------------

    print("\n" + "=" * 100)
    print(
        "DEV25 SLICE-LEVEL RESULTS"
    )
    print("=" * 100)

    print(
        df.to_string(
            index=False
        )
    )

    # ------------------------------------------------
    # Threshold summary
    # ------------------------------------------------

    threshold_rows = []

    for threshold in PET_THRESHOLDS:

        sub = df[
            df["threshold"] == threshold
        ]

        empty = sub[
            sub["ct_gate_empty"]
        ]

        recovered = sub[
            sub["recovered_by_fallback"]
        ]

        threshold_rows.append(
            {
                "threshold": threshold,

                "gt_positive_slices":
                    len(sub),

                "ct_gate_empty_slices":
                    len(empty),

                "recovered_slices":
                    len(recovered),

                "recovery_rate_among_ct_failures":
                    (
                        len(recovered) /
                        len(empty)
                        if len(empty) > 0
                        else 0.0
                    ),

                "mean_baseline_dice":
                    sub["baseline_dice"].mean(),

                "mean_fallback_dice":
                    sub["fallback_dice"].mean(),

                "mean_baseline_coverage":
                    sub["baseline_coverage"].mean(),

                "mean_fallback_coverage":
                    sub["fallback_coverage"].mean(),

                "mean_baseline_fpr":
                    sub["baseline_fpr"].mean(),

                "mean_fallback_fpr":
                    sub["fallback_fpr"].mean(),
            }
        )

    threshold_summary = pd.DataFrame(
        threshold_rows
    )

    print("\n" + "=" * 100)
    print(
        "DEV25 THRESHOLD SUMMARY"
    )
    print("=" * 100)

    print(
        threshold_summary.to_string(
            index=False
        )
    )

    # ------------------------------------------------
    # Case summary
    # ------------------------------------------------

    case_rows = []

    for case_name in CASES:

        for threshold in PET_THRESHOLDS:

            sub = df[
                (df["case_id"] == case_name)
                &
                (df["threshold"] == threshold)
            ]

            empty = sub[
                sub["ct_gate_empty"]
            ]

            recovered = sub[
                sub["recovered_by_fallback"]
            ]

            case_rows.append(
                {
                    "case_id": case_name,
                    "threshold": threshold,

                    "gt_slices":
                        len(sub),

                    "ct_gate_empty":
                        len(empty),

                    "recovered":
                        len(recovered),

                    "recovery_rate":
                        (
                            len(recovered) /
                            len(empty)
                            if len(empty) > 0
                            else 0.0
                        ),

                    "mean_baseline_dice":
                        sub["baseline_dice"].mean(),

                    "mean_fallback_dice":
                        sub["fallback_dice"].mean(),

                    "mean_baseline_coverage":
                        sub["baseline_coverage"].mean(),

                    "mean_fallback_coverage":
                        sub["fallback_coverage"].mean(),
                }
            )

    case_summary = pd.DataFrame(
        case_rows
    )

    print("\n" + "=" * 100)
    print(
        "DEV25 CASE SUMMARY"
    )
    print("=" * 100)

    print(
        case_summary.to_string(
            index=False
        )
    )

    # ------------------------------------------------
    # Save
    # ------------------------------------------------

    slice_path = (
        RESULT_DIR /
        "dev25_slice_results.csv"
    )

    threshold_path = (
        RESULT_DIR /
        "dev25_threshold_summary.csv"
    )

    case_path = (
        RESULT_DIR /
        "dev25_case_summary.csv"
    )

    df.to_csv(
        slice_path,
        index=False
    )

    threshold_summary.to_csv(
        threshold_path,
        index=False
    )

    case_summary.to_csv(
        case_path,
        index=False
    )

    print("\n" + "=" * 100)
    print(
        "DEV25 COMPLETE"
    )
    print("=" * 100)

    print(
        f"Slice results: {slice_path}"
    )

    print(
        f"Threshold summary: {threshold_path}"
    )

    print(
        f"Case summary: {case_path}"
    )

    print(
        "\nDev10 was NOT modified."
    )


if __name__ == "__main__":
    main()