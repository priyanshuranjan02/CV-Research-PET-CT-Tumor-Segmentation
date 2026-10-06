"""
DEV18 — CT Contour Spatial-Distance Diagnostic

Purpose
-------
For GT-positive slices where the Dev10 baseline CT-contour stage
does not overlap the GT, measure how close the GT tumor is to the
available valid CT contours.

This helps distinguish:

    A. Near-miss:
       GT is just outside a CT contour.

    B. Moderate spatial mismatch:
       GT is separated from contours by a meaningful distance.

    C. Complete contour failure:
       No useful contour exists near the GT.

For each baseline-failed slice we measure:

    - nearest contour distance to GT
    - nearest contour centroid distance
    - nearest contour area
    - number of valid contours
    - number of contours within several distance thresholds
    - best overlap after hypothetical dilation

IMPORTANT:
    Dev10 is NOT modified.
    This is a diagnostic only.
"""

from pathlib import Path
import sys
import io
from contextlib import redirect_stdout

import cv2
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

DEV10_PATH = SRC / "dev10_lite_hot_core.py"


# ============================================================
# Import exact Dev10 definitions
# ============================================================

sys.path.insert(
    0,
    str(SRC)
)

with redirect_stdout(io.StringIO()):
    import dev10_lite_hot_core as dev10


load_case = dev10.load_case
build_union_body_mask = dev10.build_union_body_mask
normalize_ct = dev10.normalize_ct

MIN_CONTOUR_AREA = (
    dev10.MIN_CONTOUR_AREA
)


# ============================================================
# Development cases
# ============================================================

DEV_CASES = [
    "PETCT_0168f65af8",
    "PETCT_04606080a0",
    "PETCT_04ab5c61c9",
    "PETCT_0b57b247b6",
    "PETCT_11afab3485",
    "PETCT_185da4c8b6",
]


# ============================================================
# Distance thresholds
# ============================================================

DISTANCE_THRESHOLDS = [
    2,
    5,
    10,
    20,
    30,
    50,
]


# ============================================================
# Hypothetical dilation radii
#
# Diagnostic only — NOT used to modify Dev10.
# ============================================================

DILATION_RADII = [
    1,
    3,
    5,
    10,
    20,
]


# ============================================================
# Output
# ============================================================

OUT_DIR = (
    ROOT
    / "results"
    / "development_cases"
    / "dev18_ct_contour_spatial_results"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Build exact Dev10 baseline CT contours
# ============================================================

def get_valid_contours(
    ct_display_slice,
    body_slice
):

    opening_kernel = np.ones(
        (5, 5),
        np.uint8
    )

    # --------------------------------------------------------
    # Exact Dev10 adaptive threshold
    # --------------------------------------------------------

    adaptive = cv2.adaptiveThreshold(
        ct_display_slice,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        3
    )

    # --------------------------------------------------------
    # Exact Dev10 5x5 opening
    # --------------------------------------------------------

    opened = cv2.morphologyEx(
        adaptive,
        cv2.MORPH_OPEN,
        opening_kernel
    )

    # --------------------------------------------------------
    # Exact Dev10 body restriction
    # --------------------------------------------------------

    opened = np.where(
        body_slice,
        opened,
        0
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Exact Dev10 border suppression
    # --------------------------------------------------------

    opened[:5, :] = 0
    opened[-5:, :] = 0
    opened[:, :5] = 0
    opened[:, -5:] = 0

    # --------------------------------------------------------
    # Exact Dev10 external contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        opened,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid = []

    for contour in contours:

        area = float(
            cv2.contourArea(
                contour
            )
        )

        if area < MIN_CONTOUR_AREA:
            continue

        mask = np.zeros(
            opened.shape,
            dtype=np.uint8
        )

        cv2.drawContours(
            mask,
            [contour],
            -1,
            1,
            -1
        )

        valid.append(
            {
                "contour": contour,
                "area": area,
                "mask": mask > 0,
            }
        )

    return valid


# ============================================================
# Compute spatial distance
# ============================================================

def compute_distance(
    gt,
    contour_mask
):

    gt_u8 = (
        gt.astype(
            np.uint8
        )
    )

    contour_u8 = (
        contour_mask.astype(
            np.uint8
        )
    )

    # --------------------------------------------------------
    # If overlap already exists
    # --------------------------------------------------------

    if np.any(
        gt & contour_mask
    ):

        return 0.0

    # --------------------------------------------------------
    # Distance transform:
    #
    # distanceTransform calculates distance to zero pixels.
    # Therefore invert contour mask.
    # --------------------------------------------------------

    inverse = (
        1
        - contour_u8
    ).astype(
        np.uint8
    )

    distance_map = cv2.distanceTransform(
        inverse,
        cv2.DIST_L2,
        3
    )

    gt_distances = (
        distance_map[gt]
    )

    if len(gt_distances) == 0:
        return float("inf")

    return float(
        np.min(
            gt_distances
        )
    )


# ============================================================
# Compute centroid distance
# ============================================================

def compute_centroid_distance(
    gt,
    contour_mask
):

    gt_yx = np.argwhere(
        gt
    )

    contour_yx = np.argwhere(
        contour_mask
    )

    if (
        len(gt_yx) == 0
        or len(contour_yx) == 0
    ):
        return float("inf")

    gt_centroid = (
        gt_yx.mean(
            axis=0
        )
    )

    contour_centroid = (
        contour_yx.mean(
            axis=0
        )
    )

    return float(
        np.linalg.norm(
            gt_centroid
            - contour_centroid
        )
    )


# ============================================================
# Dice after hypothetical contour dilation
# ============================================================

def dilated_overlap_metrics(
    gt,
    contour_mask,
    radius
):

    kernel_size = (
        2 * radius
        + 1
    )

    kernel = np.ones(
        (
            kernel_size,
            kernel_size
        ),
        np.uint8
    )

    dilated = cv2.dilate(
        contour_mask.astype(
            np.uint8
        ),
        kernel
    ) > 0

    overlap = int(
        np.sum(
            gt & dilated
        )
    )

    gt_area = int(
        gt.sum()
    )

    pred_area = int(
        dilated.sum()
    )

    dice_denominator = (
        gt_area
        + pred_area
    )

    if dice_denominator > 0:

        dice = (
            2.0
            * overlap
            / dice_denominator
        )

    else:

        dice = 0.0

    union = (
        gt_area
        + pred_area
        - overlap
    )

    if union > 0:

        iou = (
            overlap
            / union
        )

    else:

        iou = 0.0

    return (
        overlap,
        dice,
        iou
    )


# ============================================================
# Diagnose one case
# ============================================================

def diagnose_case(
    case_id
):

    print()
    print("=" * 100)
    print(
        f"===== CASE: {case_id} ====="
    )
    print("=" * 100)

    case = load_case(
        case_id
    )

    ct = case["ct"]
    gt_volume = case["gt"]

    body = build_union_body_mask(
        ct
    )

    ct_display = normalize_ct(
        ct
    )

    gt_positive_slices = np.where(
        gt_volume.reshape(
            gt_volume.shape[0],
            -1
        ).any(axis=1)
    )[0]

    rows = []

    for z in gt_positive_slices:

        gt = (
            gt_volume[z] > 0
        )

        valid_contours = get_valid_contours(
            ct_display[z],
            body[z]
        )

        # ----------------------------------------------------
        # Determine baseline overlap
        # ----------------------------------------------------

        baseline_overlap = False

        for item in valid_contours:

            if np.any(
                gt
                &
                item["mask"]
            ):
                baseline_overlap = True
                break

        # ----------------------------------------------------
        # We are primarily interested in baseline failures.
        # ----------------------------------------------------

        if baseline_overlap:
            continue

        # ----------------------------------------------------
        # No contours at all
        # ----------------------------------------------------

        if len(valid_contours) == 0:

            rows.append(
                {
                    "case_id": case_id,
                    "slice": int(z),
                    "gt_area": int(gt.sum()),
                    "valid_contours": 0,
                    "nearest_contour_distance": float(
                        "inf"
                    ),
                    "nearest_centroid_distance": float(
                        "inf"
                    ),
                    "nearest_contour_area": 0.0,
                    "contours_within_2px": 0,
                    "contours_within_5px": 0,
                    "contours_within_10px": 0,
                    "contours_within_20px": 0,
                    "contours_within_30px": 0,
                    "contours_within_50px": 0,
                    "best_dilation_radius": 0,
                    "best_dilation_dice": 0.0,
                    "best_dilation_iou": 0.0,
                    "classification": "NO_VALID_CONTOURS",
                }
            )

            continue

        # ----------------------------------------------------
        # Measure every contour
        # ----------------------------------------------------

        contour_measurements = []

        for item in valid_contours:

            mask = item[
                "mask"
            ]

            distance = compute_distance(
                gt,
                mask
            )

            centroid_distance = (
                compute_centroid_distance(
                    gt,
                    mask
                )
            )

            contour_measurements.append(
                {
                    "area": item["area"],
                    "distance": distance,
                    "centroid_distance": (
                        centroid_distance
                    ),
                    "mask": mask,
                }
            )

        # ----------------------------------------------------
        # Nearest contour
        # ----------------------------------------------------

        nearest = min(
            contour_measurements,
            key=lambda x: x[
                "distance"
            ]
        )

        nearest_distance = nearest[
            "distance"
        ]

        nearest_centroid_distance = (
            nearest[
                "centroid_distance"
            ]
        )

        nearest_area = nearest[
            "area"
        ]

        # ----------------------------------------------------
        # Count contours within thresholds
        # ----------------------------------------------------

        within_counts = {}

        for threshold in (
            DISTANCE_THRESHOLDS
        ):

            within_counts[
                threshold
            ] = sum(
                1
                for item
                in contour_measurements
                if item[
                    "distance"
                ] <= threshold
            )

        # ----------------------------------------------------
        # Hypothetical dilation
        # ----------------------------------------------------

        best_radius = 0
        best_dice = 0.0
        best_iou = 0.0

        for radius in (
            DILATION_RADII
        ):

            radius_best_dice = 0.0
            radius_best_iou = 0.0

            for item in (
                contour_measurements
            ):

                (
                    overlap,
                    dice,
                    iou
                ) = dilated_overlap_metrics(
                    gt,
                    item["mask"],
                    radius
                )

                radius_best_dice = max(
                    radius_best_dice,
                    dice
                )

                radius_best_iou = max(
                    radius_best_iou,
                    iou
                )

            if (
                radius_best_dice
                > best_dice
            ):

                best_dice = (
                    radius_best_dice
                )

                best_iou = (
                    radius_best_iou
                )

                best_radius = (
                    radius
                )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        if nearest_distance <= 5:

            classification = (
                "VERY_NEAR_CONTOUR"
            )

        elif nearest_distance <= 10:

            classification = (
                "NEAR_CONTOUR"
            )

        elif nearest_distance <= 20:

            classification = (
                "MODERATE_DISTANCE"
            )

        elif nearest_distance <= 50:

            classification = (
                "FAR_CONTOUR"
            )

        else:

            classification = (
                "VERY_FAR_OR_MISSING"
            )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        rows.append(
            {
                "case_id": case_id,
                "slice": int(z),
                "gt_area": int(gt.sum()),
                "valid_contours": len(
                    valid_contours
                ),

                "nearest_contour_distance": (
                    nearest_distance
                ),

                "nearest_centroid_distance": (
                    nearest_centroid_distance
                ),

                "nearest_contour_area": (
                    nearest_area
                ),

                "contours_within_2px": (
                    within_counts[2]
                ),

                "contours_within_5px": (
                    within_counts[5]
                ),

                "contours_within_10px": (
                    within_counts[10]
                ),

                "contours_within_20px": (
                    within_counts[20]
                ),

                "contours_within_30px": (
                    within_counts[30]
                ),

                "contours_within_50px": (
                    within_counts[50]
                ),

                "best_dilation_radius": (
                    best_radius
                ),

                "best_dilation_dice": (
                    best_dice
                ),

                "best_dilation_iou": (
                    best_iou
                ),

                "classification": (
                    classification
                ),
            }
        )

    return rows


# ============================================================
# MAIN
# ============================================================

print("=" * 100)
print("DEV18 — CT Contour Spatial-Distance Diagnostic")
print("=" * 100)
print()

all_rows = []

for case_id in DEV_CASES:

    rows = diagnose_case(
        case_id
    )

    all_rows.extend(
        rows
    )


# ============================================================
# DataFrame
# ============================================================

df = pd.DataFrame(
    all_rows
)


# ============================================================
# Save slice-level results
# ============================================================

slice_path = (
    OUT_DIR
    / "dev18_slice_spatial_distance.csv"
)

df.to_csv(
    slice_path,
    index=False
)


# ============================================================
# Classification summary
# ============================================================

if len(df) > 0:

    classification_summary = (
        df[
            "classification"
        ]
        .value_counts()
        .rename_axis(
            "classification"
        )
        .reset_index(
            name="slice_count"
        )
    )

else:

    classification_summary = (
        pd.DataFrame(
            columns=[
                "classification",
                "slice_count",
            ]
        )
    )


classification_path = (
    OUT_DIR
    / "dev18_distance_classification.csv"
)

classification_summary.to_csv(
    classification_path,
    index=False
)


# ============================================================
# Aggregate summary
# ============================================================

if len(df) > 0:

    aggregate = {
        "baseline_failed_slices": len(df),

        "mean_nearest_distance": (
            df[
                "nearest_contour_distance"
            ].replace(
                np.inf,
                np.nan
            ).mean()
        ),

        "median_nearest_distance": (
            df[
                "nearest_contour_distance"
            ].replace(
                np.inf,
                np.nan
            ).median()
        ),

        "within_5px": int(
            (
                df[
                    "nearest_contour_distance"
                ] <= 5
            ).sum()
        ),

        "within_10px": int(
            (
                df[
                    "nearest_contour_distance"
                ] <= 10
            ).sum()
        ),

        "within_20px": int(
            (
                df[
                    "nearest_contour_distance"
                ] <= 20
            ).sum()
        ),

        "within_30px": int(
            (
                df[
                    "nearest_contour_distance"
                ] <= 30
            ).sum()
        ),

        "within_50px": int(
            (
                df[
                    "nearest_contour_distance"
                ] <= 50
            ).sum()
        ),

        "best_dilation_dice_mean": (
            df[
                "best_dilation_dice"
            ].mean()
        ),

        "best_dilation_dice_max": (
            df[
                "best_dilation_dice"
            ].max()
        ),
    }

else:

    aggregate = {
        "baseline_failed_slices": 0,
        "mean_nearest_distance": 0.0,
        "median_nearest_distance": 0.0,
        "within_5px": 0,
        "within_10px": 0,
        "within_20px": 0,
        "within_30px": 0,
        "within_50px": 0,
        "best_dilation_dice_mean": 0.0,
        "best_dilation_dice_max": 0.0,
    }


aggregate_df = pd.DataFrame(
    [aggregate]
)

aggregate_path = (
    OUT_DIR
    / "dev18_aggregate_summary.csv"
)

aggregate_df.to_csv(
    aggregate_path,
    index=False
)


# ============================================================
# Case summary
# ============================================================

case_rows = []

for case_id in DEV_CASES:

    group = df[
        df["case_id"] == case_id
    ]

    if len(group) == 0:

        case_rows.append(
            {
                "case_id": case_id,
                "baseline_failed_slices": 0,
                "within_5px": 0,
                "within_10px": 0,
                "within_20px": 0,
                "within_50px": 0,
                "mean_nearest_distance": np.nan,
                "median_nearest_distance": np.nan,
                "mean_best_dilation_dice": np.nan,
                "max_best_dilation_dice": np.nan,
            }
        )

        continue

    case_rows.append(
        {
            "case_id": case_id,
            "baseline_failed_slices": len(
                group
            ),

            "within_5px": int(
                (
                    group[
                        "nearest_contour_distance"
                    ] <= 5
                ).sum()
            ),

            "within_10px": int(
                (
                    group[
                        "nearest_contour_distance"
                    ] <= 10
                ).sum()
            ),

            "within_20px": int(
                (
                    group[
                        "nearest_contour_distance"
                    ] <= 20
                ).sum()
            ),

            "within_50px": int(
                (
                    group[
                        "nearest_contour_distance"
                    ] <= 50
                ).sum()
            ),

            "mean_nearest_distance": (
                group[
                    "nearest_contour_distance"
                ].replace(
                    np.inf,
                    np.nan
                ).mean()
            ),

            "median_nearest_distance": (
                group[
                    "nearest_contour_distance"
                ].replace(
                    np.inf,
                    np.nan
                ).median()
            ),

            "mean_best_dilation_dice": (
                group[
                    "best_dilation_dice"
                ].mean()
            ),

            "max_best_dilation_dice": (
                group[
                    "best_dilation_dice"
                ].max()
            ),
        }
    )


case_df = pd.DataFrame(
    case_rows
)

case_path = (
    OUT_DIR
    / "dev18_case_summary.csv"
)

case_df.to_csv(
    case_path,
    index=False
)


# ============================================================
# Print results
# ============================================================

print()
print("=" * 100)
print("DEV18 COMPLETE")
print("=" * 100)
print()

print(
    f"Baseline-failed slices analyzed: "
    f"{len(df)}"
)

print()

print(
    "Distance classification:"
)

print(
    classification_summary.to_string(
        index=False
    )
)

print()

print(
    "Aggregate spatial summary:"
)

print(
    aggregate_df.to_string(
        index=False
    )
)

print()

print(
    "Case summary:"
)

print(
    case_df.to_string(
        index=False
    )
)

print()

print(
    "Output files:"
)

print(
    f"Slice-level : {slice_path}"
)

print(
    f"Classification: {classification_path}"
)

print(
    f"Aggregate   : {aggregate_path}"
)

print(
    f"Case summary: {case_path}"
)

print()