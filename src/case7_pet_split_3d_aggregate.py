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
#
# Suppress the validation output printed during module import.
# The frozen V4 code itself is NOT modified.
# ============================================================

with redirect_stdout(io.StringIO()):
    from v4_validate_case7 import (
        prepare_case7 as prepare_case7_original,
        normalize_ct,
    )


# ============================================================
# CASE CONFIGURATION
# ============================================================

CASE_NAME = "PETCT_0011f3deaf"

PET_THRESHOLDS = [
    1.25,
    1.50,
    1.75,
    2.00,
    2.25,
]

# Same oversized-contour definition
MAX_CANDIDATE_AREA = 10000

# Same minimum 2D PET component size
MIN_COMPONENT_AREA = 20

# Same V4 3D coherence requirements
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
    / "case7_pet_split_3d_aggregate_results.csv"
)

COMPONENT_CSV = (
    OUTPUT_DIR
    / "case7_pet_split_3d_components.csv"
)


# ============================================================
# UNION BODY MASK
#
# Case 7 showed that selecting only the largest CT component
# excluded substantial GT.
#
# This experiment therefore uses the union of all CT body
# components as a diagnostic.
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
# PATCH THE ACTUAL GLOBAL USED BY prepare_case7()
# ============================================================

prepare_case7_original.__globals__[
    "build_body_mask"
] = build_union_body_mask


# ============================================================
# LOAD CASE 7
# ============================================================

print()
print("=" * 90)
print("===== CASE 7 — PET SPLIT 3D AGGREGATION EXPERIMENT =====")
print("=" * 90)

print()
print("Case:", CASE_NAME)
print(
    "PET thresholds:",
    PET_THRESHOLDS
)
print(
    "Oversized contour area >",
    MAX_CANDIDATE_AREA
)
print(
    "Minimum 2D PET component area:",
    MIN_COMPONENT_AREA
)
print(
    "Minimum 3D component voxels:",
    MIN_3D_VOXELS
)
print(
    "Minimum 3D component slices:",
    MIN_3D_SLICES
)


# Suppress the repeated diagnostic output from prepare_case7().
with redirect_stdout(io.StringIO()):
    prepared, ct_img, ct, body = (
        prepare_case7_original()
    )


suv = prepared[
    "suv"
]


# ============================================================
# GT
#
# Used ONLY for final evaluation.
# Never used for candidate/component selection.
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
    "Union-body voxels:",
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
# CT PREPROCESSING
# ============================================================

ct_display = normalize_ct(
    ct
)

opening_kernel = np.ones(
    (5, 5),
    np.uint8
)


# ============================================================
# GENERATE 2D PET-SPLIT COMPONENTS
#
# IMPORTANT:
# All eligible PET components are retained.
# We DO NOT select the hottest component per slice.
#
# They are aggregated into a 3D volume afterwards.
# ============================================================

def build_raw_pet_split_volume(
    threshold
):

    raw_volume = np.zeros(
        suv.shape,
        dtype=np.uint8
    )

    rows = []

    total_2d_components = 0

    slices_with_components = set()

    for z in range(
        ct.shape[0]
    ):

        body_slice = body[z]

        if not np.any(
            body_slice
        ):
            continue

        # ----------------------------------------------------
        # Same V4 CT morphology
        # ----------------------------------------------------

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

        # Remove border artifacts
        opened[:5, :] = 0
        opened[-5:, :] = 0
        opened[:, :5] = 0
        opened[:, -5:] = 0

        contours, _ = cv2.findContours(
            opened,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # ----------------------------------------------------
        # Inspect oversized CT contours
        # ----------------------------------------------------

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
                <= MAX_CANDIDATE_AREA
            ):
                continue

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

            # ------------------------------------------------
            # PET threshold inside contour
            # ------------------------------------------------

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
                    < MIN_COMPONENT_AREA
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

                # Add this PET component to the 3D volume.
                raw_volume[z][
                    component
                ] = 1

                total_2d_components += 1

                slices_with_components.add(
                    int(z)
                )

                rows.append(
                    {
                        "slice": int(z),
                        "contour_index": int(
                            contour_index
                        ),
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
                        "cx": float(
                            centroids[
                                component_id
                            ][0]
                        ),
                        "cy": float(
                            centroids[
                                component_id
                            ][1]
                        ),
                    }
                )

    return (
        raw_volume,
        total_2d_components,
        slices_with_components,
        rows
    )


# ============================================================
# 3D CONNECTED COMPONENT FILTER
#
# This is the important part of the experiment:
# instead of choosing one PET component independently on each
# slice, all PET components are aggregated and then evaluated
# as 3D structures.
# ============================================================

def apply_3d_coherence_filter(
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

    component_rows = []

    kept_count = 0

    for label in stats.GetLabels():

        component = (
            cc == label
        )

        voxel_count = int(
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

        # ----------------------------------------------------
        # V4-style 3D filtering
        # ----------------------------------------------------

        kept = True

        if (
            voxel_count
            < MIN_3D_VOXELS
        ):
            kept = False

        if (
            z_span
            < MIN_3D_SLICES
        ):
            kept = False

        overlap_gt = int(
            np.logical_and(
                component,
                gt
            ).sum()
        )

        component_union = int(
            np.logical_or(
                component,
                gt
            ).sum()
        )

        component_dice = (
            2.0
            * overlap_gt
            / (
                voxel_count
                + int(gt.sum())
            )
            if (
                voxel_count
                + int(gt.sum())
            ) > 0
            else 0.0
        )

        component_iou = (
            overlap_gt
            / component_union
            if component_union > 0
            else 0.0
        )

        component_values = suv[
            component
        ]

        component_rows.append(
            {
                "label": int(label),
                "voxels": voxel_count,
                "z_start": z_start,
                "z_end": z_end,
                "z_span": z_span,
                "overlap_gt": overlap_gt,
                "dice_gt": float(
                    component_dice
                ),
                "iou_gt": float(
                    component_iou
                ),
                "mean_suv": float(
                    np.mean(
                        component_values
                    )
                ) if component_values.size
                else 0.0,
                "median_suv": float(
                    np.median(
                        component_values
                    )
                ) if component_values.size
                else 0.0,
                "max_suv": float(
                    np.max(
                        component_values
                    )
                ) if component_values.size
                else 0.0,
                "kept": int(
                    kept
                ),
            }
        )

        if kept:

            final_volume[
                component
            ] = 1

            kept_count += 1

    return (
        final_volume,
        kept_count,
        component_rows
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

    slice_recall = (
        tp
        / max(
            gt_positive.sum(),
            1
        )
    )

    fpr = (
        fp
        / max(
            (~gt_positive).sum(),
            1
        )
    )

    return {
        "dice": float(
            dice
        ),
        "iou": float(
            iou
        ),
        "slice_recall": float(
            slice_recall
        ),
        "fpr": float(
            fpr
        ),
        "prediction_voxels":
            pred_voxels,
        "predicted_slices":
            int(
                pred_positive.sum()
            ),
        "tp_slices":
            int(tp),
        "fp_slices":
            int(fp),
        "fn_slices":
            int(fn),
    }


# ============================================================
# RUN EXPERIMENT
# ============================================================

results = []
all_component_rows = []


for threshold in PET_THRESHOLDS:

    print()
    print("=" * 90)
    print(
        f"===== PET THRESHOLD = {threshold:.2f} ====="
    )
    print("=" * 90)

    # --------------------------------------------------------
    # Build all PET-split 2D components
    # --------------------------------------------------------

    (
        raw_volume,
        total_2d_components,
        component_slices,
        component_rows
    ) = build_raw_pet_split_volume(
        threshold
    )

    raw_metrics = calculate_metrics(
        raw_volume
    )

    print()
    print("2D PET COMPONENT AGGREGATION")
    print(
        "2D PET components:",
        total_2d_components
    )
    print(
        "Slices with PET components:",
        len(component_slices)
    )
    print(
        "Raw voxels:",
        raw_metrics[
            "prediction_voxels"
        ]
    )
    print(
        "Raw predicted slices:",
        raw_metrics[
            "predicted_slices"
        ]
    )
    print(
        "Raw Dice:",
        f"{raw_metrics['dice']:.4f}"
    )
    print(
        "Raw IoU:",
        f"{raw_metrics['iou']:.4f}"
    )
    print(
        "Raw Slice Recall:",
        f"{raw_metrics['slice_recall']:.4f}"
    )
    print(
        "Raw FPR:",
        f"{raw_metrics['fpr']:.4f}"
    )

    # --------------------------------------------------------
    # 3D coherence filtering
    # --------------------------------------------------------

    (
        final_volume,
        kept_count,
        three_d_rows
    ) = apply_3d_coherence_filter(
        raw_volume
    )

    final_metrics = calculate_metrics(
        final_volume
    )

    print()
    print("3D COHERENCE FILTER")
    print(
        "Kept 3D components:",
        kept_count
    )
    print(
        "Final voxels:",
        final_metrics[
            "prediction_voxels"
        ]
    )
    print(
        "Final predicted slices:",
        final_metrics[
            "predicted_slices"
        ]
    )
    print(
        "Final Dice:",
        f"{final_metrics['dice']:.4f}"
    )
    print(
        "Final IoU:",
        f"{final_metrics['iou']:.4f}"
    )
    print(
        "Final Slice Recall:",
        f"{final_metrics['slice_recall']:.4f}"
    )
    print(
        "Final FPR:",
        f"{final_metrics['fpr']:.4f}"
    )

    # --------------------------------------------------------
    # Save final prediction
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
            "case7_pet_split_3d_"
            f"{threshold:.2f}.nii.gz"
        )
    )

    sitk.WriteImage(
        prediction_img,
        str(prediction_path)
    )

    # --------------------------------------------------------
    # Save component information
    # --------------------------------------------------------

    for row in three_d_rows:

        row_copy = row.copy()

        row_copy[
            "threshold"
        ] = float(threshold)

        all_component_rows.append(
            row_copy
        )

    # --------------------------------------------------------
    # Save result row
    # --------------------------------------------------------

    results.append(
        {
            "threshold": float(
                threshold
            ),

            "raw_2d_components":
                total_2d_components,

            "raw_component_slices":
                len(component_slices),

            "raw_voxels":
                raw_metrics[
                    "prediction_voxels"
                ],

            "raw_dice":
                raw_metrics[
                    "dice"
                ],

            "raw_iou":
                raw_metrics[
                    "iou"
                ],

            "raw_slice_recall":
                raw_metrics[
                    "slice_recall"
                ],

            "raw_fpr":
                raw_metrics[
                    "fpr"
                ],

            "kept_3d_components":
                kept_count,

            "final_voxels":
                final_metrics[
                    "prediction_voxels"
                ],

            "final_dice":
                final_metrics[
                    "dice"
                ],

            "final_iou":
                final_metrics[
                    "iou"
                ],

            "final_slice_recall":
                final_metrics[
                    "slice_recall"
                ],

            "final_fpr":
                final_metrics[
                    "fpr"
                ],

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

            "prediction_path":
                str(
                    prediction_path
                ),
        }
    )


# ============================================================
# SAVE CSVs
# ============================================================

results_df = pd.DataFrame(
    results
)

components_df = pd.DataFrame(
    all_component_rows
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
# FINAL TABLE
# ============================================================

print()
print("=" * 90)
print("===== PET SPLIT 3D AGGREGATION COMPARISON =====")
print("=" * 90)

print(
    results_df[
        [
            "threshold",
            "raw_2d_components",
            "raw_component_slices",
            "raw_voxels",
            "raw_dice",
            "raw_iou",
            "raw_slice_recall",
            "raw_fpr",
            "kept_3d_components",
            "final_voxels",
            "final_dice",
            "final_iou",
            "final_slice_recall",
            "final_fpr",
            "predicted_slices",
            "tp_slices",
            "fp_slices",
            "fn_slices",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# COMPONENT AUDIT SUMMARY
# ============================================================

print()
print("=" * 90)
print("===== 3D COMPONENT AUDIT SUMMARY =====")
print("=" * 90)

if not components_df.empty:

    print()

    for threshold in PET_THRESHOLDS:

        t = components_df[
            components_df[
                "threshold"
            ]
            == threshold
        ]

        if t.empty:
            continue

        kept = t[
            t["kept"] == 1
        ]

        overlapping = kept[
            kept["overlap_gt"] > 0
        ]

        print(
            f"Threshold {threshold:.2f}: "
            f"3D components={len(t)}, "
            f"kept={len(kept)}, "
            f"kept components overlapping GT="
            f"{len(overlapping)}"
        )


# ============================================================
# FILES
# ============================================================

print()
print("=" * 90)
print("===== EXPERIMENT COMPLETE =====")
print("=" * 90)

print()
print("Saved summary:")
print(SUMMARY_CSV)

print()
print("Saved 3D component audit:")
print(COMPONENT_CSV)

print()
print(
    "Saved NIfTI predictions under:"
)
print(OUTPUT_DIR)
