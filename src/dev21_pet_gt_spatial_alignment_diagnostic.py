"""
Dev21 — PET/GT Spatial Alignment Diagnostic

Purpose:
    Investigate the 21 exact Dev10 CT-contour failure slices.

    Main question:
        Is PET spatially aligned with the GT on these difficult slices?

    Measures:
        - GT centroid
        - PET maximum-SUV location
        - distance from PET max to GT centroid
        - distance from PET max to nearest GT pixel
        - mean / median / max SUV inside GT
        - PET threshold overlap
        - PET threshold centroid
        - PET threshold centroid distance to GT centroid

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
    / "dev21_pet_gt_alignment_results"
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

THRESHOLDS = [2.25, 3.0, 4.0]


# ============================================================
# FIND EXACT DEV10 CT FAILURES
# ============================================================

def get_dev10_ct_contours(ct_slice, body_slice):

    ct_display = dev10.normalize_ct(ct_slice)

    kernel = np.ones((5, 5), np.uint8)

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

    return [
        c for c in contours
        if cv2.contourArea(c) >= 100
    ]


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

        mask = np.zeros(
            gt_slice.shape,
            dtype=np.uint8
        )

        if contours:
            cv2.drawContours(
                mask,
                contours,
                -1,
                1,
                -1
            )

        overlap = np.logical_and(
            mask > 0,
            gt_slice
        ).sum()

        if overlap == 0:
            failures.append(z)

    return failures


# ============================================================
# CENTROID
# ============================================================

def centroid(mask):

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return np.nan, np.nan

    return float(xs.mean()), float(ys.mean())


# ============================================================
# DISTANCE TO GT
# ============================================================

def distance_to_gt(point_x, point_y, gt):

    ys, xs = np.where(gt)

    if len(xs) == 0:
        return np.nan

    distances = np.sqrt(
        (xs - point_x) ** 2 +
        (ys - point_y) ** 2
    )

    return float(np.min(distances))


# ============================================================
# ANALYZE SLICE
# ============================================================

def analyze_slice(suv, gt, threshold):

    gt_y, gt_x = np.where(gt)

    gt_cx = float(gt_x.mean())
    gt_cy = float(gt_y.mean())

    gt_suv = suv[gt]

    # --------------------------------------------------------
    # Maximum SUV
    # --------------------------------------------------------

    max_index = np.unravel_index(
        np.argmax(suv),
        suv.shape
    )

    max_y, max_x = max_index

    max_suv = float(suv[max_y, max_x])

    max_to_gt_centroid = float(
        np.sqrt(
            (max_x - gt_cx) ** 2 +
            (max_y - gt_cy) ** 2
        )
    )

    max_to_gt_nearest = distance_to_gt(
        max_x,
        max_y,
        gt
    )

    # --------------------------------------------------------
    # PET threshold mask
    # --------------------------------------------------------

    pet_mask = suv >= threshold

    pet_overlap = np.logical_and(
        pet_mask,
        gt
    ).sum()

    gt_area = gt.sum()
    pet_area = pet_mask.sum()

    union = np.logical_or(
        pet_mask,
        gt
    ).sum()

    dice = (
        2.0 * pet_overlap /
        (gt_area + pet_area)
        if gt_area + pet_area > 0
        else 0.0
    )

    iou = (
        pet_overlap / union
        if union > 0
        else 0.0
    )

    coverage = (
        pet_overlap / gt_area
        if gt_area > 0
        else 0.0
    )

    pet_cx, pet_cy = centroid(
        pet_mask
    )

    if np.isnan(pet_cx):

        pet_centroid_distance = np.nan

    else:

        pet_centroid_distance = float(
            np.sqrt(
                (pet_cx - gt_cx) ** 2 +
                (pet_cy - gt_cy) ** 2
            )
        )

    return {
        "gt_cx": gt_cx,
        "gt_cy": gt_cy,

        "max_suv": max_suv,
        "max_suv_x": int(max_x),
        "max_suv_y": int(max_y),

        "max_to_gt_centroid_px":
            max_to_gt_centroid,

        "max_to_gt_nearest_px":
            max_to_gt_nearest,

        "mean_gt_suv":
            float(np.mean(gt_suv)),

        "median_gt_suv":
            float(np.median(gt_suv)),

        "max_gt_suv":
            float(np.max(gt_suv)),

        "pet_area":
            int(pet_area),

        "pet_overlap":
            int(pet_overlap),

        "pet_gt_coverage":
            float(coverage),

        "pet_dice":
            float(dice),

        "pet_iou":
            float(iou),

        "pet_centroid_x":
            pet_cx,

        "pet_centroid_y":
            pet_cy,

        "pet_centroid_distance_px":
            pet_centroid_distance,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("DEV21 — PET/GT SPATIAL ALIGNMENT DIAGNOSTIC")
    print("=" * 100)

    rows = []

    for case_name in CASES:

        print("\n" + "=" * 100)
        print(f"===== CASE: {case_name} =====")
        print("=" * 100)

        case = dev10.load_case(case_name)

        ct = case["ct"]
        suv = case["suv"]
        gt = case["gt"]

        failures = find_baseline_failures(case)

        print(
            f"Exact Dev10 CT failures: {len(failures)}"
        )

        for z in failures:

            gt_slice = gt[z].astype(bool)
            suv_slice = suv[z]

            for threshold in THRESHOLDS:

                metrics = analyze_slice(
                    suv_slice,
                    gt_slice,
                    threshold
                )

                row = {
                    "case_id": case_name,
                    "slice": int(z),
                    "threshold": threshold,
                }

                row.update(metrics)

                rows.append(row)

    df = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print("DEV21 COMPLETE")
    print("=" * 100)

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_rows = []

    for threshold in THRESHOLDS:

        sub = df[
            df["threshold"] == threshold
        ]

        summary_rows.append({
            "threshold": threshold,
            "slices": len(sub),

            "mean_gt_suv":
                sub["mean_gt_suv"].mean(),

            "mean_max_gt_suv":
                sub["max_gt_suv"].mean(),

            "mean_max_to_gt_nearest":
                sub["max_to_gt_nearest_px"].mean(),

            "median_max_to_gt_nearest":
                sub["max_to_gt_nearest_px"].median(),

            "mean_pet_coverage":
                sub["pet_gt_coverage"].mean(),

            "mean_pet_dice":
                sub["pet_dice"].mean(),

            "max_pet_dice":
                sub["pet_dice"].max(),

            "mean_pet_centroid_distance":
                sub["pet_centroid_distance_px"].mean(),
        })

    summary = pd.DataFrame(
        summary_rows
    )

    print("\nThreshold summary:")
    print(
        summary.to_string(index=False)
    )

    # ========================================================
    # CASE SUMMARY
    # ========================================================

    case_summary = (
        df.groupby(
            ["case_id", "threshold"]
        )
        .agg(
            slices=("slice", "count"),
            mean_gt_suv=(
                "mean_gt_suv",
                "mean"
            ),
            mean_max_to_gt_nearest=(
                "max_to_gt_nearest_px",
                "mean"
            ),
            mean_pet_coverage=(
                "pet_gt_coverage",
                "mean"
            ),
            mean_pet_dice=(
                "pet_dice",
                "mean"
            ),
            max_pet_dice=(
                "pet_dice",
                "max"
            ),
        )
        .reset_index()
    )

    print("\nCase summary:")
    print(
        case_summary.to_string(index=False)
    )

    # ========================================================
    # SAVE
    # ========================================================

    slice_path = (
        RESULT_DIR /
        "dev21_slice_results.csv"
    )

    summary_path = (
        RESULT_DIR /
        "dev21_threshold_summary.csv"
    )

    case_path = (
        RESULT_DIR /
        "dev21_case_summary.csv"
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
    print(f"Case summary: {case_path}")


if __name__ == "__main__":
    main()