"""
Dev18.1 — Corrected CT Contour Spatial-Distance Diagnostic

Purpose:
    Analyze ONLY GT-positive slices where the exact Dev10 baseline
    CT contour union has ZERO overlap with the GT.

    This fixes Dev18's failure-selection bug, which incorrectly
    treated all GT-positive slices as baseline failures.

Dev10 baseline CT contour generation:
    - normalize CT
    - 5x5 morphological opening
    - adaptive Gaussian threshold
    - restrict to body
    - zero 5-pixel border
    - external contours
    - contour area >= 100

Diagnostic:
    - nearest contour-to-GT distance
    - contour centroid distance
    - contour area
    - number of contours within 2/5/10/20/30/50 px
    - hypothetical dilation overlap
    - distance classification

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
    / "dev18_1_ct_contour_spatial_results"
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
# PARAMETERS — EXACT DEV10 BASELINE
# ============================================================

MIN_CONTOUR_AREA = 100

DISTANCE_THRESHOLDS = [2, 5, 10, 20, 30, 50]

DILATION_RADII = [1, 3, 5, 10, 20]


# ============================================================
# DEV10 BASELINE CT CONTOURS
# ============================================================

def get_dev10_baseline_contours(ct_slice, body_slice):
    """
    Reproduce the CT contour stage from Dev10 exactly.
    """

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

    # Restrict to body
    binary[body_slice == 0] = 0

    # Zero 5-pixel border
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

        if area < MIN_CONTOUR_AREA:
            continue

        valid.append(contour)

    return valid


# ============================================================
# CONTOUR MASK
# ============================================================

def contours_to_mask(contours, shape):

    mask = np.zeros(shape, dtype=np.uint8)

    if contours:
        cv2.drawContours(
            mask,
            contours,
            -1,
            1,
            thickness=-1
        )

    return mask


# ============================================================
# NEAREST DISTANCE
# ============================================================

def nearest_distance(contour_mask, gt_mask):

    if not np.any(contour_mask):
        return np.nan

    if not np.any(gt_mask):
        return np.nan

    # Distance from every background pixel to nearest contour-mask pixel
    inverse = (contour_mask == 0).astype(np.uint8)

    distance_map = cv2.distanceTransform(
        inverse,
        cv2.DIST_L2,
        5
    )

    gt_distances = distance_map[gt_mask > 0]

    if len(gt_distances) == 0:
        return np.nan

    return float(np.min(gt_distances))


# ============================================================
# CENTROID DISTANCE
# ============================================================

def centroid_distance(contour, gt_mask):

    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        return np.nan

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    ys, xs = np.where(gt_mask > 0)

    if len(xs) == 0:
        return np.nan

    gt_cx = float(xs.mean())
    gt_cy = float(ys.mean())

    return float(
        np.sqrt(
            (cx - gt_cx) ** 2 +
            (cy - gt_cy) ** 2
        )
    )


# ============================================================
# DISTANCE CLASSIFICATION
# ============================================================

def classify_distance(distance):

    if np.isnan(distance):
        return "NO_VALID_CONTOURS"

    if distance <= 5:
        return "VERY_NEAR_CONTOUR"

    if distance <= 10:
        return "NEAR_CONTOUR"

    if distance <= 20:
        return "MODERATE_DISTANCE"

    if distance <= 50:
        return "FAR_CONTOUR"

    return "VERY_FAR"


# ============================================================
# HYPOTHETICAL DILATION
# ============================================================

def dilation_metrics(contour_mask, gt_mask):

    results = {}

    for radius in DILATION_RADII:

        kernel_size = 2 * radius + 1

        kernel = np.ones(
            (kernel_size, kernel_size),
            np.uint8
        )

        dilated = cv2.dilate(
            contour_mask.astype(np.uint8),
            kernel
        )

        pred = dilated > 0
        gt = gt_mask > 0

        intersection = np.logical_and(pred, gt).sum()

        pred_area = pred.sum()
        gt_area = gt.sum()

        union = np.logical_or(pred, gt).sum()

        if pred_area + gt_area > 0:
            dice = (
                2.0 * intersection /
                (pred_area + gt_area)
            )
        else:
            dice = 0.0

        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0

        results[f"dilation_{radius}px_dice"] = float(dice)
        results[f"dilation_{radius}px_iou"] = float(iou)

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("DEV18.1 — CORRECTED CT CONTOUR SPATIAL-DISTANCE DIAGNOSTIC")
    print("=" * 100)

    all_rows = []

    for case_name in CASES:

        print("\n" + "=" * 100)
        print(f"===== CASE: {case_name} =====")
        print("=" * 100)

        case = dev10.load_case(case_name)

        ct = case["ct"]
        gt = case["gt"]

        print(f"CT: {ct.shape}")
        print(f"GT voxels: {int(np.sum(gt))}")

        # Body mask exactly as Dev10
        body = dev10.build_union_body_mask(ct)

        gt_bool = gt.astype(bool)

        gt_positive_slices = np.where(
            np.any(gt_bool, axis=(1, 2))
        )[0]

        case_failed = 0

        for z in gt_positive_slices:

            gt_slice = gt_bool[z]

            body_slice = body[z]

            contours = get_dev10_baseline_contours(
                ct[z],
                body_slice
            )

            contour_mask = contours_to_mask(
                contours,
                gt_slice.shape
            )

            # ====================================================
            # CRITICAL CORRECTED FAILURE TEST
            # ====================================================
            #
            # A slice is a baseline failure ONLY when the exact
            # Dev10 contour union has ZERO overlap with GT.
            #
            overlap = np.logical_and(
                contour_mask > 0,
                gt_slice
            ).sum()

            if overlap > 0:
                continue

            # This is a true baseline-failed slice.
            case_failed += 1

            distance = nearest_distance(
                contour_mask,
                gt_slice
            )

            # Per-contour measurements
            contour_distances = []
            contour_centroid_distances = []
            contour_areas = []

            for contour in contours:

                contour_area = cv2.contourArea(contour)

                contour_dist = nearest_distance(
                    contours_to_mask(
                        [contour],
                        gt_slice.shape
                    ),
                    gt_slice
                )

                centroid_dist = centroid_distance(
                    contour,
                    gt_slice
                )

                contour_distances.append(contour_dist)
                contour_centroid_distances.append(
                    centroid_dist
                )
                contour_areas.append(
                    contour_area
                )

            valid_distances = [
                d for d in contour_distances
                if not np.isnan(d)
            ]

            valid_centroid_distances = [
                d
                for d in contour_centroid_distances
                if not np.isnan(d)
            ]

            if valid_distances:

                nearest_contour_distance = min(
                    valid_distances
                )

                nearest_index = contour_distances.index(
                    nearest_contour_distance
                )

                nearest_contour_area = (
                    contour_areas[nearest_index]
                )

            else:

                nearest_contour_distance = np.nan
                nearest_contour_area = np.nan

            if valid_centroid_distances:
                nearest_centroid_distance = min(
                    valid_centroid_distances
                )
            else:
                nearest_centroid_distance = np.nan

            row = {
                "case_id": case_name,
                "slice": int(z),

                "num_valid_contours": len(contours),

                "gt_area": int(gt_slice.sum()),
                "baseline_overlap": int(overlap),

                "nearest_distance_px":
                    nearest_contour_distance,

                "nearest_centroid_distance_px":
                    nearest_centroid_distance,

                "nearest_contour_area":
                    nearest_contour_area,

                "distance_classification":
                    classify_distance(
                        nearest_contour_distance
                    ),
            }

            # Count contours within distance thresholds
            for threshold in DISTANCE_THRESHOLDS:

                count = sum(
                    d <= threshold
                    for d in valid_distances
                )

                row[
                    f"contours_within_{threshold}px"
                ] = count

            # Hypothetical dilation
            dilation = dilation_metrics(
                contour_mask,
                gt_slice
            )

            row.update(dilation)

            all_rows.append(row)

        print(
            f"Baseline-failed slices: {case_failed}"
        )

    # ============================================================
    # DATAFRAME
    # ============================================================

    df = pd.DataFrame(all_rows)

    print("\n" + "=" * 100)
    print("DEV18.1 COMPLETE")
    print("=" * 100)

    print(
        f"\nCorrected baseline-failed slices analyzed: {len(df)}"
    )

    # ============================================================
    # DISTANCE CLASSIFICATION
    # ============================================================

    if len(df) > 0:

        classification = (
            df["distance_classification"]
            .value_counts()
            .rename_axis("classification")
            .reset_index(name="slice_count")
        )

    else:

        classification = pd.DataFrame(
            columns=[
                "classification",
                "slice_count"
            ]
        )

    print("\nDistance classification:")
    print(classification.to_string(index=False))

    # ============================================================
    # AGGREGATE SUMMARY
    # ============================================================

    if len(df) > 0:

        aggregate = {
            "baseline_failed_slices": len(df),

            "mean_nearest_distance":
                df["nearest_distance_px"].mean(),

            "median_nearest_distance":
                df["nearest_distance_px"].median(),

        }

        for threshold in DISTANCE_THRESHOLDS:

            aggregate[
                f"within_{threshold}px"
            ] = int(
                (
                    df["nearest_distance_px"]
                    <= threshold
                ).sum()
            )

        dice_columns = [
            f"dilation_{r}px_dice"
            for r in DILATION_RADII
        ]

        aggregate[
            "best_dilation_dice_mean"
        ] = df[dice_columns].max(axis=1).mean()

        aggregate[
            "best_dilation_dice_max"
        ] = df[dice_columns].max(axis=1).max()

        aggregate_df = pd.DataFrame(
            [aggregate]
        )

    else:

        aggregate_df = pd.DataFrame(
            [{
                "baseline_failed_slices": 0
            }]
        )

    print("\nAggregate spatial summary:")
    print(
        aggregate_df.to_string(index=False)
    )

    # ============================================================
    # CASE SUMMARY
    # ============================================================

    case_rows = []

    for case_name in CASES:

        sub = df[
            df["case_id"] == case_name
        ]

        row = {
            "case_id": case_name,
            "baseline_failed_slices": len(sub),
        }

        for threshold in [5, 10, 20, 50]:

            row[
                f"within_{threshold}px"
            ] = int(
                (
                    sub["nearest_distance_px"]
                    <= threshold
                ).sum()
            )

        row["mean_nearest_distance"] = (
            sub["nearest_distance_px"].mean()
        )

        row["median_nearest_distance"] = (
            sub["nearest_distance_px"].median()
        )

        dice_columns = [
            f"dilation_{r}px_dice"
            for r in DILATION_RADII
        ]

        if len(sub) > 0:

            best_dice = sub[
                dice_columns
            ].max(axis=1)

            row["mean_best_dilation_dice"] = (
                best_dice.mean()
            )

            row["max_best_dilation_dice"] = (
                best_dice.max()
            )

        else:

            row["mean_best_dilation_dice"] = np.nan
            row["max_best_dilation_dice"] = np.nan

        case_rows.append(row)

    case_summary = pd.DataFrame(case_rows)

    print("\nCase summary:")
    print(
        case_summary.to_string(index=False)
    )

    # ============================================================
    # SAVE
    # ============================================================

    slice_path = (
        RESULT_DIR /
        "dev18_1_slice_spatial_distance.csv"
    )

    classification_path = (
        RESULT_DIR /
        "dev18_1_distance_classification.csv"
    )

    aggregate_path = (
        RESULT_DIR /
        "dev18_1_aggregate_summary.csv"
    )

    case_path = (
        RESULT_DIR /
        "dev18_1_case_summary.csv"
    )

    df.to_csv(
        slice_path,
        index=False
    )

    classification.to_csv(
        classification_path,
        index=False
    )

    aggregate_df.to_csv(
        aggregate_path,
        index=False
    )

    case_summary.to_csv(
        case_path,
        index=False
    )

    print("\nOutput files:")
    print(f"Slice-level : {slice_path}")
    print(f"Classification: {classification_path}")
    print(f"Aggregate   : {aggregate_path}")
    print(f"Case summary: {case_path}")


if __name__ == "__main__":
    main()