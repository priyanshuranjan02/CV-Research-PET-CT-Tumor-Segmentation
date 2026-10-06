"""
Dev20 — PET Localization vs CT Gate Diagnostic

Purpose:
    Analyze the exact Dev10 baseline CT-contour failures.

    Question:
        Can PET alone localize the GT on these slices?

    PET thresholds:
        2.25
        3.0
        4.0
        5.0

    For each threshold:
        - PET mask overlap with GT
        - GT coverage
        - Dice
        - IoU
        - overlapping connected components
        - largest overlapping component
        - mean SUV in GT
        - max SUV in GT

    Dev10 is NOT modified.
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
    / "dev20_pet_localization_gate_results"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)

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
# PET THRESHOLDS
# ============================================================

PET_THRESHOLDS = [
    2.25,
    3.0,
    4.0,
    5.0,
]


# ============================================================
# CT CONTOUR REPRODUCTION
# ============================================================

def get_dev10_ct_contours(ct_slice, body_slice):

    ct_display = dev10.normalize_ct(ct_slice)

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

    binary[body_slice == 0] = 0

    binary[:5, :] = 0
    binary[-5:, :] = 0
    binary[:, :5] = 0
    binary[:, -5:] = 0

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 100:
            continue

        valid.append(contour)

    return valid


# ============================================================
# FIND TRUE DEV10 FAILURES
# ============================================================

def find_baseline_failures(case):

    ct = case["ct"]
    gt = case["gt"]

    body = dev10.build_union_body_mask(ct)

    gt_bool = gt.astype(bool)

    positive_slices = np.where(
        np.any(gt_bool, axis=(1, 2))
    )[0]

    failures = []

    for z in positive_slices:

        gt_slice = gt_bool[z]

        contours = get_dev10_ct_contours(
            ct[z],
            body[z]
        )

        contour_mask = np.zeros(
            gt_slice.shape,
            dtype=np.uint8
        )

        if contours:

            cv2.drawContours(
                contour_mask,
                contours,
                -1,
                1,
                thickness=-1
            )

        overlap = np.logical_and(
            contour_mask > 0,
            gt_slice
        ).sum()

        if overlap == 0:
            failures.append(z)

    return failures


# ============================================================
# PET SLICE METRICS
# ============================================================

def analyze_pet_slice(
    suv_slice,
    gt_slice,
    threshold
):

    gt = gt_slice.astype(bool)

    pet_mask = suv_slice >= threshold

    gt_area = int(gt.sum())
    pet_area = int(pet_mask.sum())

    overlap = int(
        np.logical_and(
            pet_mask,
            gt
        ).sum()
    )

    union = int(
        np.logical_or(
            pet_mask,
            gt
        ).sum()
    )

    if gt_area + pet_area > 0:

        dice = (
            2.0 * overlap /
            (gt_area + pet_area)
        )

    else:

        dice = 0.0

    if union > 0:

        iou = overlap / union

    else:

        iou = 0.0

    coverage = (
        overlap / gt_area
        if gt_area > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            pet_mask.astype(np.uint8),
            connectivity=8
        )
    )

    overlapping_components = 0
    largest_overlap = 0
    largest_overlap_area = 0

    for label in range(1, num_labels):

        component = labels == label

        component_overlap = int(
            np.logical_and(
                component,
                gt
            ).sum()
        )

        if component_overlap > 0:

            overlapping_components += 1

            if component_overlap > largest_overlap:

                largest_overlap = component_overlap

                largest_overlap_area = int(
                    stats[label, cv2.CC_STAT_AREA]
                )

    # --------------------------------------------------------
    # SUV statistics inside GT
    # --------------------------------------------------------

    gt_suv = suv_slice[gt]

    if len(gt_suv) > 0:

        mean_gt_suv = float(
            np.mean(gt_suv)
        )

        max_gt_suv = float(
            np.max(gt_suv)
        )

        median_gt_suv = float(
            np.median(gt_suv)
        )

    else:

        mean_gt_suv = np.nan
        max_gt_suv = np.nan
        median_gt_suv = np.nan

    return {
        "pet_area": pet_area,
        "gt_area": gt_area,
        "overlap": overlap,
        "gt_coverage": coverage,
        "dice": dice,
        "iou": iou,
        "overlapping_components":
            overlapping_components,
        "largest_component_overlap":
            largest_overlap,
        "largest_component_area":
            largest_overlap_area,
        "mean_gt_suv":
            mean_gt_suv,
        "median_gt_suv":
            median_gt_suv,
        "max_gt_suv":
            max_gt_suv,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("DEV20 — PET LOCALIZATION VS CT GATE DIAGNOSTIC")
    print("=" * 100)

    all_rows = []

    for case_name in CASES:

        print("\n" + "=" * 100)
        print(f"===== CASE: {case_name} =====")
        print("=" * 100)

        case = dev10.load_case(case_name)

        ct = case["ct"]
        suv = case["suv"]
        gt = case["gt"]

        print(f"CT:  {ct.shape}")
        print(f"PET: {suv.shape}")

        failures = find_baseline_failures(case)

        print(
            f"Exact Dev10 baseline CT failures: "
            f"{len(failures)}"
        )

        for z in failures:

            gt_slice = gt[z].astype(bool)
            suv_slice = suv[z]

            for threshold in PET_THRESHOLDS:

                metrics = analyze_pet_slice(
                    suv_slice,
                    gt_slice,
                    threshold
                )

                row = {
                    "case_id": case_name,
                    "slice": int(z),
                    "pet_threshold": threshold,
                }

                row.update(metrics)

                all_rows.append(row)

    df = pd.DataFrame(all_rows)

    print("\n" + "=" * 100)
    print("DEV20 COMPLETE")
    print("=" * 100)

    print(
        f"\nTotal failed slices analyzed: "
        f"{df['slice'].nunique()}"
    )

    # ========================================================
    # THRESHOLD SUMMARY
    # ========================================================

    summary_rows = []

    for threshold in PET_THRESHOLDS:

        sub = df[
            df["pet_threshold"] == threshold
        ]

        summary_rows.append({
            "pet_threshold": threshold,

            "slices_analyzed":
                len(sub),

            "slices_with_pet_overlap":
                int(
                    (sub["overlap"] > 0).sum()
                ),

            "recovery_rate":
                (
                    sub["overlap"] > 0
                ).mean(),

            "mean_gt_coverage":
                sub["gt_coverage"].mean(),

            "mean_dice":
                sub["dice"].mean(),

            "max_dice":
                sub["dice"].max(),

            "mean_iou":
                sub["iou"].mean(),

            "max_iou":
                sub["iou"].max(),

            "mean_overlapping_components":
                sub[
                    "overlapping_components"
                ].mean(),

            "mean_gt_suv":
                sub["mean_gt_suv"].mean(),

            "mean_max_gt_suv":
                sub["max_gt_suv"].mean(),
        })

    summary = pd.DataFrame(
        summary_rows
    )

    print("\nPET threshold summary:")
    print(
        summary.to_string(index=False)
    )

    # ========================================================
    # CASE SUMMARY
    # ========================================================

    case_rows = []

    for case_name in CASES:

        for threshold in PET_THRESHOLDS:

            sub = df[
                (df["case_id"] == case_name) &
                (df["pet_threshold"] == threshold)
            ]

            if len(sub) == 0:
                continue

            case_rows.append({
                "case_id": case_name,
                "pet_threshold": threshold,
                "slices_analyzed": len(sub),
                "slices_with_overlap": int(
                    (sub["overlap"] > 0).sum()
                ),
                "recovery_rate": (
                    sub["overlap"] > 0
                ).mean(),
                "mean_dice":
                    sub["dice"].mean(),
                "max_dice":
                    sub["dice"].max(),
                "mean_gt_coverage":
                    sub["gt_coverage"].mean(),
                "mean_gt_suv":
                    sub["mean_gt_suv"].mean(),
            })

    case_summary = pd.DataFrame(
        case_rows
    )

    print("\nCase × threshold summary:")
    print(
        case_summary.to_string(index=False)
    )

    # ========================================================
    # SAVE
    # ========================================================

    slice_path = (
        RESULT_DIR /
        "dev20_slice_results.csv"
    )

    summary_path = (
        RESULT_DIR /
        "dev20_pet_threshold_summary.csv"
    )

    case_path = (
        RESULT_DIR /
        "dev20_case_threshold_summary.csv"
    )

    df.to_csv(
        slice_path,
        index=False
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    case_summary.to_csv(
        case_path,
        index=False
    )

    print("\nOutput files:")
    print(f"Slice-level : {slice_path}")
    print(f"Threshold   : {summary_path}")
    print(f"Case×threshold: {case_path}")


if __name__ == "__main__":
    main()