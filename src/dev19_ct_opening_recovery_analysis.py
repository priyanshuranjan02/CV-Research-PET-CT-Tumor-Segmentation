"""
Dev19 — CT Opening Recovery Analysis

Purpose:
    Analyze the exact 21 slices where the Dev10 baseline CT contour
    has ZERO overlap with GT.

    Compare:
        A. Dev10 baseline: 5x5 opening, area >=100
        B. No opening:      area >=100
        C. 3x3 opening:     area >=100
        D. 5x5 opening:     area >=50
        E. 5x5 opening:     area >=25

    For each variant measure:
        - whether a contour overlaps GT
        - number of overlapping contours
        - best contour GT coverage
        - best Dice
        - best IoU
        - best contour area
        - total contour count

    Dev10 itself is NOT modified.
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
    / "dev19_ct_opening_recovery_results"
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
# VARIANTS
# ============================================================

VARIANTS = [
    ("DEV10_BASELINE", 5, 100),
    ("NO_OPENING", 0, 100),
    ("OPEN_3X3", 3, 100),
    ("AREA_50", 5, 50),
    ("AREA_25", 5, 25),
]


# ============================================================
# CT CONTOURS
# ============================================================

def get_contours(
    ct_slice,
    body_slice,
    opening_size,
    min_area
):

    ct_display = dev10.normalize_ct(ct_slice)

    # --------------------------------------------------------
    # Opening
    # --------------------------------------------------------

    if opening_size > 0:

        kernel = np.ones(
            (opening_size, opening_size),
            np.uint8
        )

        processed = cv2.morphologyEx(
            ct_display,
            cv2.MORPH_OPEN,
            kernel
        )

    else:

        processed = ct_display.copy()

    # --------------------------------------------------------
    # Adaptive threshold
    # --------------------------------------------------------

    binary = cv2.adaptiveThreshold(
        processed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        3
    )

    # --------------------------------------------------------
    # Body restriction
    # --------------------------------------------------------

    binary[body_slice == 0] = 0

    # --------------------------------------------------------
    # Border removal
    # --------------------------------------------------------

    binary[:5, :] = 0
    binary[-5:, :] = 0
    binary[:, :5] = 0
    binary[:, -5:] = 0

    # --------------------------------------------------------
    # External contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        valid.append(contour)

    return valid


# ============================================================
# METRICS
# ============================================================

def contour_metrics(
    contours,
    gt_slice
):

    gt = gt_slice.astype(bool)

    gt_area = int(gt.sum())

    if not contours:

        return {
            "num_contours": 0,
            "overlapping_contours": 0,
            "best_overlap": 0,
            "best_dice": 0.0,
            "best_iou": 0.0,
            "best_gt_coverage": 0.0,
            "best_contour_area": np.nan,
        }

    best_overlap = 0
    best_dice = 0.0
    best_iou = 0.0
    best_gt_coverage = 0.0
    best_area = np.nan

    overlapping = 0

    for contour in contours:

        mask = np.zeros(
            gt.shape,
            dtype=np.uint8
        )

        cv2.drawContours(
            mask,
            [contour],
            -1,
            1,
            thickness=-1
        )

        pred = mask.astype(bool)

        overlap = int(
            np.logical_and(
                pred,
                gt
            ).sum()
        )

        if overlap > 0:
            overlapping += 1

        pred_area = int(pred.sum())

        union = int(
            np.logical_or(
                pred,
                gt
            ).sum()
        )

        if pred_area + gt_area > 0:

            dice = (
                2.0 * overlap /
                (pred_area + gt_area)
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

        # Rank by Dice
        if dice > best_dice:

            best_dice = dice
            best_iou = iou
            best_overlap = overlap
            best_gt_coverage = coverage
            best_area = cv2.contourArea(contour)

    return {
        "num_contours": len(contours),
        "overlapping_contours": overlapping,
        "best_overlap": best_overlap,
        "best_dice": best_dice,
        "best_iou": best_iou,
        "best_gt_coverage": best_gt_coverage,
        "best_contour_area": best_area,
    }


# ============================================================
# FIND EXACT DEV10 FAILURES
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

        contours = get_contours(
            ct[z],
            body[z],
            opening_size=5,
            min_area=100
        )

        metrics = contour_metrics(
            contours,
            gt_slice
        )

        if metrics["best_overlap"] == 0:

            failures.append(z)

    return failures


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("DEV19 — CT OPENING RECOVERY ANALYSIS")
    print("=" * 100)

    all_rows = []

    for case_name in CASES:

        print("\n" + "=" * 100)
        print(f"===== CASE: {case_name} =====")
        print("=" * 100)

        case = dev10.load_case(case_name)

        ct = case["ct"]
        gt = case["gt"]

        body = dev10.build_union_body_mask(ct)

        failures = find_baseline_failures(case)

        print(
            f"Exact Dev10 baseline failures: {len(failures)}"
        )

        if not failures:
            continue

        for z in failures:

            gt_slice = gt[z].astype(bool)

            for (
                variant,
                opening_size,
                min_area
            ) in VARIANTS:

                contours = get_contours(
                    ct[z],
                    body[z],
                    opening_size,
                    min_area
                )

                metrics = contour_metrics(
                    contours,
                    gt_slice
                )

                row = {
                    "case_id": case_name,
                    "slice": int(z),
                    "variant": variant,
                    "opening_size": opening_size,
                    "min_area": min_area,
                }

                row.update(metrics)

                all_rows.append(row)

    df = pd.DataFrame(all_rows)

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 100)
    print("DEV19 COMPLETE")
    print("=" * 100)

    # ========================================================
    # VARIANT SUMMARY
    # ========================================================

    summary_rows = []

    for variant in [
        v[0]
        for v in VARIANTS
    ]:

        sub = df[
            df["variant"] == variant
        ]

        if len(sub) == 0:
            continue

        summary_rows.append({
            "variant": variant,
            "slices_analyzed": len(sub),
            "slices_with_overlap": int(
                (
                    sub["best_overlap"] > 0
                ).sum()
            ),
            "recovery_rate": (
                (
                    sub["best_overlap"] > 0
                ).mean()
            ),
            "mean_best_dice":
                sub["best_dice"].mean(),
            "max_best_dice":
                sub["best_dice"].max(),
            "mean_gt_coverage":
                sub["best_gt_coverage"].mean(),
            "mean_overlapping_contours":
                sub["overlapping_contours"].mean(),
        })

    summary = pd.DataFrame(
        summary_rows
    )

    print("\nVariant summary:")
    print(
        summary.to_string(index=False)
    )

    # ========================================================
    # CASE × VARIANT SUMMARY
    # ========================================================

    case_summary = (
        df
        .groupby(
            ["case_id", "variant"]
        )
        .agg(
            slices_analyzed=(
                "slice",
                "count"
            ),
            slices_with_overlap=(
                "best_overlap",
                lambda x:
                int((x > 0).sum())
            ),
            mean_best_dice=(
                "best_dice",
                "mean"
            ),
            max_best_dice=(
                "best_dice",
                "max"
            ),
            mean_gt_coverage=(
                "best_gt_coverage",
                "mean"
            ),
        )
        .reset_index()
    )

    print("\nCase × variant summary:")
    print(
        case_summary.to_string(index=False)
    )

    # ========================================================
    # SAVE
    # ========================================================

    slice_path = (
        RESULT_DIR /
        "dev19_slice_results.csv"
    )

    summary_path = (
        RESULT_DIR /
        "dev19_variant_summary.csv"
    )

    case_path = (
        RESULT_DIR /
        "dev19_case_variant_summary.csv"
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
    print(f"Variant     : {summary_path}")
    print(f"Case×variant: {case_path}")


if __name__ == "__main__":
    main()