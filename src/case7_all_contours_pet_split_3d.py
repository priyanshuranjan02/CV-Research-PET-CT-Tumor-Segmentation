from pathlib import Path
import sys
import io
from contextlib import redirect_stdout

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ============================================================
# IMPORT EXISTING CASE-7 DEFINITIONS
# ============================================================

with redirect_stdout(io.StringIO()):
    from v4_validate_case7 import (
        prepare_case7 as prepare_case7_original,
        normalize_ct,
    )


# ============================================================
# CONFIGURATION
# ============================================================

CASE_NAME = "PETCT_0011f3deaf"

PET_THRESHOLDS = [
    1.25,
    1.50,
    1.75,
    2.00,
    2.25,
]

# Original V4 lower contour-area boundary.
MIN_CONTOUR_AREA = 100

# IMPORTANT:
# No upper contour-area limit.
#
# This includes:
#   normal contours <= 10,000 px
#   oversized contours > 10,000 px
#
MAX_CONTOUR_AREA = None

MIN_PET_COMPONENT_AREA = 20

MIN_3D_VOXELS = 75
MIN_3D_SLICES = 2


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "validation_results"
    / CASE_NAME
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "case7_all_contours_pet_split_3d_results.csv"
)

COMPONENT_CSV = (
    OUTPUT_DIR
    / "case7_all_contours_pet_split_3d_components.csv"
)


# ============================================================
# UNION BODY MASK
# ============================================================

def build_union_body_mask(ct):

    body = np.zeros_like(
        ct,
        dtype=np.uint8
    )

    kernel = np.ones(
        (11, 11),
        np.uint8
    )

    for z in range(
        ct.shape[0]
    ):

        binary = (
            ct[z] > -900
        ).astype(
            np.uint8
        ) * 255

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            if cv2.contourArea(
                contour
            ) <= 0:
                continue

            cv2.drawContours(
                body[z],
                [contour],
                -1,
                1,
                -1
            )

    return body.astype(bool)


# ============================================================
# PATCH BODY MASK USED BY prepare_case7()
# ============================================================

prepare_case7_original.__globals__[
    "build_body_mask"
] = build_union_body_mask


# ============================================================
# LOAD CASE
# ============================================================

print()
print("=" * 100)
print("===== CASE 7 — ALL-CONTOUR PET SPLIT 3D AGGREGATION =====")
print("=" * 100)

print()
print("Case:", CASE_NAME)
print(
    "PET thresholds:",
    PET_THRESHOLDS
)
print(
    "Minimum CT contour area:",
    MIN_CONTOUR_AREA
)
print(
    "Maximum CT contour area:",
    "NONE"
)
print(
    "Minimum PET component area:",
    MIN_PET_COMPONENT_AREA
)
print(
    "Minimum 3D voxels:",
    MIN_3D_VOXELS
)
print(
    "Minimum 3D slices:",
    MIN_3D_SLICES
)


with redirect_stdout(io.StringIO()):
    prepared, ct_img, ct, body = (
        prepare_case7_original()
    )

suv = prepared[
    "suv"
]


# ============================================================
# GT
# ============================================================

GT_PATH = (
    ROOT
    / "verified_case"
    / "tumor_mask_ct_visible.nii.gz"
)

gt_img = sitk.ReadImage(
    str(GT_PATH)
)

gt = (
    sitk.GetArrayFromImage(
        gt_img
    ) > 0
)

gt_positive = np.any(
    gt,
    axis=(1, 2)
)


print()
print("CT shape:", ct.shape)
print("SUV shape:", suv.shape)
print(
    "Union body voxels:",
    int(body.sum())
)
print(
    "GT voxels:",
    int(gt.sum())
)
print(
    "GT-positive slices:",
    int(gt_positive.sum())
)


# ============================================================
# CT NORMALIZATION
# ============================================================

ct_display = normalize_ct(
    ct
)

opening_kernel = np.ones(
    (5, 5),
    np.uint8
)


# ============================================================
# BUILD RAW PET COMPONENT VOLUME
#
# Every eligible CT contour is processed.
#
# No GT is used to select components.
# ============================================================

def build_raw_volume(
    threshold
):

    raw_volume = np.zeros(
        suv.shape,
        dtype=np.uint8
    )

    component_rows = []

    total_components = 0
    contour_count = 0
    oversized_contours = 0
    normal_contours = 0

    slices_with_components = set()

    for z in range(
        ct.shape[0]
    ):

        body_slice = body[z]

        if not np.any(
            body_slice
        ):
            continue

        adaptive = cv2.adaptiveThreshold(
            ct_display[z],
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15,
            3
        )

        opened = cv2.morphologyEx(
            adaptive,
            cv2.MORPH_OPEN,
            opening_kernel
        )

        opened = np.where(
            body_slice,
            opened,
            0
        ).astype(
            np.uint8
        )

        opened[:5, :] = 0
        opened[-5:, :] = 0
        opened[:, :5] = 0
        opened[:, -5:] = 0

        contours, _ = cv2.findContours(
            opened,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour_index, contour in enumerate(
            contours
        ):

            contour_area = float(
                cv2.contourArea(
                    contour
                )
            )

            if (
                contour_area
                < MIN_CONTOUR_AREA
            ):
                continue

            if (
                MAX_CONTOUR_AREA is not None
                and contour_area
                > MAX_CONTOUR_AREA
            ):
                continue

            contour_count += 1

            if contour_area > 10000:
                oversized_contours += 1
                contour_type = "oversized"
            else:
                normal_contours += 1
                contour_type = "normal"

            contour_mask = np.zeros(
                opened.shape,
                dtype=np.uint8
            )

            cv2.drawContours(
                contour_mask,
                [contour],
                -1,
                1,
                -1
            )

            contour_pixels = (
                contour_mask > 0
            )

            hot = (
                contour_pixels
                & np.isfinite(
                    suv[z]
                )
                & (
                    suv[z]
                    >= threshold
                )
            ).astype(
                np.uint8
            )

            if not np.any(
                hot
            ):
                continue

            (
                n_components,
                labels,
                stats,
                centroids
            ) = cv2.connectedComponentsWithStats(
                hot,
                connectivity=8
            )

            for component_id in range(
                1,
                n_components
            ):

                area = int(
                    stats[
                        component_id,
                        cv2.CC_STAT_AREA
                    ]
                )

                if (
                    area
                    < MIN_PET_COMPONENT_AREA
                ):
                    continue

                component = (
                    labels
                    == component_id
                )

                values = suv[z][
                    component
                ]

                if values.size == 0:
                    continue

                # Add EVERY eligible component into the 3D volume.
                raw_volume[z][
                    component
                ] = 1

                total_components += 1

                slices_with_components.add(
                    int(z)
                )

                component_rows.append(
                    {
                        "slice": int(z),
                        "contour_index": int(
                            contour_index
                        ),
                        "contour_area": int(
                            contour_pixels.sum()
                        ),
                        "contour_type":
                            contour_type,
                        "threshold": float(
                            threshold
                        ),
                        "component_id": int(
                            component_id
                        ),
                        "component_area":
                            area,
                        "mean_suv": float(
                            np.mean(values)
                        ),
                        "median_suv": float(
                            np.median(values)
                        ),
                        "max_suv": float(
                            np.max(values)
                        ),
                        "p95_suv": float(
                            np.percentile(
                                values,
                                95
                            )
                        ),
                        "overlap_gt": int(
                            np.logical_and(
                                component,
                                gt[z]
                            ).sum()
                        ),
                    }
                )

    return (
        raw_volume,
        component_rows,
        total_components,
        contour_count,
        oversized_contours,
        normal_contours,
        slices_with_components
    )


# ============================================================
# 3D CONNECTED COMPONENT FILTER
# ============================================================

def build_3d_components(
    raw_volume
):

    raw_img = sitk.GetImageFromArray(
        raw_volume
    )

    cc_img = sitk.ConnectedComponent(
        raw_img
    )

    cc = sitk.GetArrayFromImage(
        cc_img
    )

    stats = (
        sitk.LabelShapeStatisticsImageFilter()
    )

    stats.Execute(
        cc_img
    )

    final_volume = np.zeros_like(
        raw_volume,
        dtype=np.uint8
    )

    rows = []

    for label in stats.GetLabels():

        component = (
            cc == label
        )

        voxels = int(
            component.sum()
        )

        z_indices = np.where(
            component
        )[0]

        if len(
            z_indices
        ) == 0:
            continue

        z_start = int(
            z_indices.min()
        )

        z_end = int(
            z_indices.max()
        )

        z_span = (
            z_end
            - z_start
            + 1
        )

        kept = (
            voxels >= MIN_3D_VOXELS
            and
            z_span >= MIN_3D_SLICES
        )

        overlap = int(
            np.logical_and(
                component,
                gt
            ).sum()
        )

        union = int(
            np.logical_or(
                component,
                gt
            ).sum()
        )

        dice = (
            2.0
            * overlap
            / (
                voxels
                + int(gt.sum())
            )
            if (
                voxels
                + int(gt.sum())
            ) > 0
            else 0.0
        )

        iou = (
            overlap
            / union
            if union > 0
            else 0.0
        )

        values = suv[
            component
        ]

        mean_suv = float(
            np.mean(values)
        ) if values.size else 0.0

        median_suv = float(
            np.median(values)
        ) if values.size else 0.0

        max_suv = float(
            np.max(values)
        ) if values.size else 0.0

        rows.append(
            {
                "label": int(label),
                "voxels": voxels,
                "z_start": z_start,
                "z_end": z_end,
                "z_span": z_span,
                "mean_suv": mean_suv,
                "median_suv": median_suv,
                "max_suv": max_suv,
                "overlap_gt": overlap,
                "dice_gt": float(dice),
                "iou_gt": float(iou),
                "kept": int(kept),
            }
        )

        if kept:
            final_volume[
                component
            ] = 1

    return (
        final_volume,
        rows
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction
):

    pred = (
        prediction > 0
    )

    intersection = np.logical_and(
        pred,
        gt
    ).sum()

    union = np.logical_or(
        pred,
        gt
    ).sum()

    pred_voxels = int(
        pred.sum()
    )

    gt_voxels = int(
        gt.sum()
    )

    dice = (
        2.0
        * intersection
        / (
            pred_voxels
            + gt_voxels
        )
        if (
            pred_voxels
            + gt_voxels
        ) > 0
        else 0.0
    )

    iou = (
        intersection
        / union
        if union > 0
        else 0.0
    )

    pred_positive = np.any(
        pred,
        axis=(1, 2)
    )

    tp = np.logical_and(
        gt_positive,
        pred_positive
    ).sum()

    fp = np.logical_and(
        ~gt_positive,
        pred_positive
    ).sum()

    fn = np.logical_and(
        gt_positive,
        ~pred_positive
    ).sum()

    return {
        "dice": float(dice),
        "iou": float(iou),
        "slice_recall": float(
            tp
            / max(
                gt_positive.sum(),
                1
            )
        ),
        "fpr": float(
            fp
            / max(
                (~gt_positive).sum(),
                1
            )
        ),
        "voxels": pred_voxels,
        "predicted_slices": int(
            pred_positive.sum()
        ),
        "tp_slices": int(tp),
        "fp_slices": int(fp),
        "fn_slices": int(fn),
    }


# ============================================================
# RUN ALL THRESHOLDS
# ============================================================

result_rows = []
component_rows_all = []


for threshold in PET_THRESHOLDS:

    print()
    print("=" * 100)
    print(
        f"===== PET THRESHOLD {threshold:.2f} ====="
    )
    print("=" * 100)

    (
        raw_volume,
        component_rows,
        total_components,
        contour_count,
        oversized_contours,
        normal_contours,
        slices_with_components
    ) = build_raw_volume(
        threshold
    )

    raw_metrics = calculate_metrics(
        raw_volume
    )

    (
        final_volume,
        three_d_rows
    ) = build_3d_components(
        raw_volume
    )

    final_metrics = calculate_metrics(
        final_volume
    )

    kept_3d = sum(
        row["kept"] == 1
        for row in three_d_rows
    )

    overlapping_kept = sum(
        (
            row["kept"] == 1
            and
            row["overlap_gt"] > 0
        )
        for row in three_d_rows
    )

    # --------------------------------------------------------
    # Add threshold to component audit
    # --------------------------------------------------------

    for row in three_d_rows:

        row_copy = row.copy()

        row_copy[
            "threshold"
        ] = threshold

        component_rows_all.append(
            row_copy
        )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print()
    print("CT contours processed:", contour_count)
    print(
        "Normal contours (100-10000):",
        normal_contours
    )
    print(
        "Oversized contours (>10000):",
        oversized_contours
    )
    print(
        "2D PET components:",
        total_components
    )
    print(
        "Slices with PET components:",
        len(slices_with_components)
    )

    print()
    print("RAW VOLUME")
    print(
        "Voxels:",
        raw_metrics["voxels"]
    )
    print(
        "Predicted slices:",
        raw_metrics[
            "predicted_slices"
        ]
    )
    print(
        "Dice:",
        f"{raw_metrics['dice']:.4f}"
    )
    print(
        "IoU:",
        f"{raw_metrics['iou']:.4f}"
    )
    print(
        "Slice Recall:",
        f"{raw_metrics['slice_recall']:.4f}"
    )
    print(
        "FPR:",
        f"{raw_metrics['fpr']:.4f}"
    )

    print()
    print("AFTER 3D FILTER")
    print(
        "Total 3D components:",
        len(three_d_rows)
    )
    print(
        "Kept 3D components:",
        kept_3d
    )
    print(
        "Kept components overlapping GT:",
        overlapping_kept
    )
    print(
        "Voxels:",
        final_metrics["voxels"]
    )
    print(
        "Predicted slices:",
        final_metrics[
            "predicted_slices"
        ]
    )
    print(
        "Dice:",
        f"{final_metrics['dice']:.4f}"
    )
    print(
        "IoU:",
        f"{final_metrics['iou']:.4f}"
    )
    print(
        "Slice Recall:",
        f"{final_metrics['slice_recall']:.4f}"
    )
    print(
        "FPR:",
        f"{final_metrics['fpr']:.4f}"
    )

    # --------------------------------------------------------
    # Save NIfTI
    # --------------------------------------------------------

    prediction_img = (
        sitk.GetImageFromArray(
            final_volume
        )
    )

    prediction_img.CopyInformation(
        ct_img
    )

    prediction_path = (
        OUTPUT_DIR
        / (
            "case7_all_contours_pet_split_3d_"
            f"{threshold:.2f}.nii.gz"
        )
    )

    sitk.WriteImage(
        prediction_img,
        str(prediction_path)
    )

    # --------------------------------------------------------
    # Summary row
    # --------------------------------------------------------

    result_rows.append(
        {
            "threshold": threshold,
            "normal_contours":
                normal_contours,
            "oversized_contours":
                oversized_contours,
            "2d_pet_components":
                total_components,
            "component_slices":
                len(slices_with_components),

            "raw_voxels":
                raw_metrics["voxels"],
            "raw_dice":
                raw_metrics["dice"],
            "raw_iou":
                raw_metrics["iou"],
            "raw_slice_recall":
                raw_metrics[
                    "slice_recall"
                ],
            "raw_fpr":
                raw_metrics["fpr"],

            "3d_components":
                len(three_d_rows),
            "kept_3d_components":
                kept_3d,
            "kept_gt_overlapping":
                overlapping_kept,

            "final_voxels":
                final_metrics["voxels"],
            "final_dice":
                final_metrics["dice"],
            "final_iou":
                final_metrics["iou"],
            "final_slice_recall":
                final_metrics[
                    "slice_recall"
                ],
            "final_fpr":
                final_metrics["fpr"],
            "predicted_slices":
                final_metrics[
                    "predicted_slices"
                ],
            "tp_slices":
                final_metrics[
                    "tp_slices"
                ],
            "fp_slices":
                final_metrics[
                    "fp_slices"
                ],
            "fn_slices":
                final_metrics[
                    "fn_slices"
                ],
        }
    )


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(
    result_rows
)

components_df = pd.DataFrame(
    component_rows_all
)

results_df.to_csv(
    SUMMARY_CSV,
    index=False
)

components_df.to_csv(
    COMPONENT_CSV,
    index=False
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print()
print("=" * 110)
print("===== ALL-CONTOUR PET SPLIT 3D AGGREGATION COMPARISON =====")
print("=" * 110)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# INSPECT COMPONENTS THAT OVERLAP GT
#
# GT is used only here for audit/evaluation.
# ============================================================

print()
print("=" * 110)
print("===== GT-OVERLAPPING KEPT 3D COMPONENTS =====")
print("=" * 110)

if not components_df.empty:

    overlap_df = components_df[
        (
            components_df["kept"] == 1
        )
        &
        (
            components_df["overlap_gt"] > 0
        )
    ].sort_values(
        [
            "threshold",
            "dice_gt"
        ],
        ascending=[
            True,
            False
        ]
    )

    if overlap_df.empty:

        print(
            "No kept 3D component overlaps GT."
        )

    else:

        print(
            overlap_df[
                [
                    "threshold",
                    "label",
                    "voxels",
                    "z_start",
                    "z_end",
                    "z_span",
                    "mean_suv",
                    "median_suv",
                    "max_suv",
                    "overlap_gt",
                    "dice_gt",
                    "iou_gt",
                ]
            ].to_string(
                index=False
            )
        )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 110)
print("===== ALL-CONTOUR EXPERIMENT COMPLETE =====")
print("=" * 110)

print()
print("Summary:")
print(SUMMARY_CSV)

print()
print("Component audit:")
print(COMPONENT_CSV)

print()
print("NIfTI predictions:")
print(OUTPUT_DIR)
