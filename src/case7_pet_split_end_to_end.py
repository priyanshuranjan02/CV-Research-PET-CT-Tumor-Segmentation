from pathlib import Path
import sys

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
# IMPORT EXISTING CASE-7 PIPELINE DEFINITIONS
# ============================================================

from v4_validate_case7 import (  # noqa: E402
    prepare_case7 as prepare_case7_original,
    normalize_ct,
)


# ============================================================
# CASE CONFIGURATION
# ============================================================

CASE_NAME = "PETCT_0011f3deaf"

# PET thresholds to compare
PET_THRESHOLDS = [
    1.25,
    1.50,
    1.75,
    2.00,
    2.25,
]

# Same oversized contour definition used in previous audits
MAX_CANDIDATE_AREA = 10000

# Ignore extremely small PET components
MIN_COMPONENT_AREA = 20

# Same 3D coherence filter used by the frozen V4 pipeline
MIN_3D_VOXELS = 75
MIN_3D_SLICES = 2


# ============================================================
# OUTPUT DIRECTORY
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


# ============================================================
# UNION BODY MASK
#
# Original V4 keeps only the largest CT body component.
# Case 7 showed that this excluded part of the GT.
#
# This experiment intentionally uses the union of all
# connected body components.
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
        ).astype(np.uint8) * 255

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

            area = cv2.contourArea(
                contour
            )

            if area <= 0:
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
# IMPORTANT:
# prepare_case7() uses its own global namespace.
# Therefore patch its actual __globals__ dictionary.
# ============================================================

prepare_case7_original.__globals__[
    "build_body_mask"
] = build_union_body_mask


# ============================================================
# LOAD CASE 7
# ============================================================

print()
print("=" * 90)
print("===== CASE 7 — PET SPLIT END-TO-END EXPERIMENT =====")
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
    "Minimum PET component area:",
    MIN_COMPONENT_AREA
)
print(
    "3D minimum voxels:",
    MIN_3D_VOXELS
)
print(
    "3D minimum slices:",
    MIN_3D_SLICES
)


prepared, ct_img, ct, body = (
    prepare_case7_original()
)

suv = prepared[
    "suv"
]


# ============================================================
# LOAD GT
#
# GT is used ONLY for evaluation.
# It is never used to select PET components.
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

gt_positive_slices = (
    np.where(
        np.any(
            gt,
            axis=(1, 2)
        )
    )[0]
)


print()
print("CT shape:", ct.shape)
print("SUV shape:", suv.shape)
print(
    "Union-body voxels:",
    int(body.sum())
)
print(
    "GT-positive slices:",
    len(gt_positive_slices)
)
print(
    "GT slices:",
    gt_positive_slices.tolist()
)


# ============================================================
# PREPARE CT DISPLAY
# ============================================================

ct_display = normalize_ct(
    ct
)

opening_kernel = np.ones(
    (5, 5),
    np.uint8
)


# ============================================================
# FUNCTION:
# BUILD ONE PET-SPLIT PREDICTION
#
# Selection rule:
#   - inspect oversized CT contours
#   - threshold PET inside each contour
#   - find connected PET components
#   - remove components < 20 px
#   - select the component with highest mean SUV
#
# NO GT IS USED FOR SELECTION.
# ============================================================

def build_pet_split_prediction(
    threshold
):

    prediction = np.zeros(
        suv.shape,
        dtype=np.uint8
    )

    selected_rows = []

    selected_components = 0

    candidate_slices = set()

    # --------------------------------------------------------
    # Examine every CT slice
    # --------------------------------------------------------

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
        ).astype(np.uint8)

        # Border removal
        opened[:5, :] = 0
        opened[-5:, :] = 0
        opened[:, :5] = 0
        opened[:, -5:] = 0

        contours, _ = cv2.findContours(
            opened,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        slice_options = []

        # ----------------------------------------------------
        # Inspect oversized CT contours only
        # ----------------------------------------------------

        for contour_index, contour in enumerate(
            contours
        ):

            contour_area = float(
                cv2.contourArea(
                    contour
                )
            )

            if contour_area <= MAX_CANDIDATE_AREA:
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
            # PET threshold
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

            if not np.any(hot):
                continue

            (
                num_components,
                labels,
                stats,
                centroids
            ) = cv2.connectedComponentsWithStats(
                hot,
                connectivity=8
            )

            # ------------------------------------------------
            # Evaluate PET components
            # ------------------------------------------------

            for component_id in range(
                1,
                num_components
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

                mean_suv = float(
                    np.mean(
                        values
                    )
                )

                median_suv = float(
                    np.median(
                        values
                    )
                )

                max_suv = float(
                    np.max(
                        values
                    )
                )

                p95_suv = float(
                    np.percentile(
                        values,
                        95
                    )
                )

                fraction_ge_1_5 = float(
                    np.mean(
                        values >= 1.5
                    )
                )

                fraction_ge_2_25 = float(
                    np.mean(
                        values >= 2.25
                    )
                )

                cx = float(
                    centroids[
                        component_id
                    ][0]
                )

                cy = float(
                    centroids[
                        component_id
                    ][1]
                )

                slice_options.append(
                    {
                        "slice": int(z),
                        "contour_index": int(
                            contour_index
                        ),
                        "contour_area": int(
                            contour_pixels.sum()
                        ),
                        "threshold": float(
                            threshold
                        ),
                        "component_id": int(
                            component_id
                        ),
                        "component_area": area,
                        "mean_suv": mean_suv,
                        "median_suv": median_suv,
                        "max_suv": max_suv,
                        "p95_suv": p95_suv,
                        "frac_ge_1.5":
                            fraction_ge_1_5,
                        "frac_ge_2.25":
                            fraction_ge_2_25,
                        "cx": cx,
                        "cy": cy,
                        "mask": component.copy()
                    }
                )

        # ----------------------------------------------------
        # Select highest-mean-SUV component on this slice
        #
        # This is the actual inference rule.
        # GT is NOT used here.
        # ----------------------------------------------------

        if not slice_options:
            continue

        selected = max(
            slice_options,
            key=lambda row: (
                row["mean_suv"],
                row["median_suv"],
                -row["component_area"]
            )
        )

        prediction[z][
            selected["mask"]
        ] = 1

        candidate_slices.add(
            int(z)
        )

        selected_components += 1

        selected_rows.append(
            selected
        )

    return (
        prediction,
        selected_components,
        selected_rows
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

    gt_positive = np.any(
        gt,
        axis=(1, 2)
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

    false_positive_rate = (
        fp
        / max(
            (~gt_positive).sum(),
            1
        )
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "slice_recall": float(
            slice_recall
        ),
        "fpr": float(
            false_positive_rate
        ),
        "prediction_voxels": pred_voxels,
        "predicted_slices": int(
            pred_positive.sum()
        ),
        "gt_slices": int(
            gt_positive.sum()
        ),
        "tp_slices": int(tp),
        "fp_slices": int(fp),
        "fn_slices": int(fn),
    }


# ============================================================
# 3D FILTER
#
# Same connected-component filtering used by V4.
# ============================================================

def apply_3d_filter(
    prediction
):

    candidate_img = (
        sitk.GetImageFromArray(
            prediction
        )
    )

    cc_img = (
        sitk.ConnectedComponent(
            candidate_img
        )
    )

    cc = (
        sitk.GetArrayFromImage(
            cc_img
        )
    )

    stats = (
        sitk.LabelShapeStatisticsImageFilter()
    )

    stats.Execute(
        cc_img
    )

    final_mask = np.zeros_like(
        prediction,
        dtype=np.uint8
    )

    kept_components = []

    for label in stats.GetLabels():

        component = (
            cc == label
        )

        voxel_count = int(
            component.sum()
        )

        if (
            voxel_count
            < MIN_3D_VOXELS
        ):
            continue

        z_indices = np.where(
            component
        )[0]

        if len(z_indices) == 0:
            continue

        z_span = (
            int(
                z_indices.max()
            )
            - int(
                z_indices.min()
            )
            + 1
        )

        if (
            z_span
            < MIN_3D_SLICES
        ):
            continue

        final_mask[
            component
        ] = 1

        kept_components.append(
            {
                "label": int(label),
                "voxels": voxel_count,
                "z_start": int(
                    z_indices.min()
                ),
                "z_end": int(
                    z_indices.max()
                ),
                "z_span": z_span
            }
        )

    return (
        final_mask,
        kept_components
    )


# ============================================================
# RUN ALL THRESHOLDS
# ============================================================

results = []

all_selected_rows = []


for threshold in PET_THRESHOLDS:

    print()
    print("=" * 90)
    print(
        f"===== PET THRESHOLD = {threshold:.2f} ====="
    )
    print("=" * 90)

    # --------------------------------------------------------
    # Build raw PET-split prediction
    # --------------------------------------------------------

    (
        raw_prediction,
        raw_selected_components,
        selected_rows
    ) = build_pet_split_prediction(
        threshold
    )

    raw_metrics = calculate_metrics(
        raw_prediction
    )

    print()
    print("RAW PET-SPLIT RESULT")
    print(
        "Selected components:",
        raw_selected_components
    )
    print(
        "Raw prediction voxels:",
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
    # Apply V4 3D filtering
    # --------------------------------------------------------

    (
        final_prediction,
        kept_components
    ) = apply_3d_filter(
        raw_prediction
    )

    final_metrics = calculate_metrics(
        final_prediction
    )

    print()
    print("AFTER 3D FILTER")
    print(
        "Kept 3D components:",
        len(
            kept_components
        )
    )
    print(
        "Final prediction voxels:",
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
    # Save prediction
    # --------------------------------------------------------

    prediction_img = (
        sitk.GetImageFromArray(
            final_prediction
        )
    )

    prediction_img.CopyInformation(
        ct_img
    )

    prediction_path = (
        OUTPUT_DIR
        / (
            "case7_pet_split_"
            f"{threshold:.2f}.nii.gz"
        )
    )

    sitk.WriteImage(
        prediction_img,
        str(prediction_path)
    )

    # --------------------------------------------------------
    # Save selected component information
    # --------------------------------------------------------

    for row in selected_rows:

        row_copy = row.copy()

        row_copy.pop(
            "mask",
            None
        )

        row_copy[
            "final_kept"
        ] = None

        all_selected_rows.append(
            row_copy
        )

    # --------------------------------------------------------
    # Result row
    # --------------------------------------------------------

    results.append(
        {
            "threshold": threshold,

            "raw_selected_components":
                raw_selected_components,

            "raw_prediction_voxels":
                raw_metrics[
                    "prediction_voxels"
                ],

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

            "kept_3d_components":
                len(
                    kept_components
                ),

            "final_prediction_voxels":
                final_metrics[
                    "prediction_voxels"
                ],

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

            "prediction_path":
                str(
                    prediction_path
                ),
        }
    )


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(
    results
)

selected_df = pd.DataFrame(
    all_selected_rows
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 90)
print("===== FINAL PET-SPLIT THRESHOLD COMPARISON =====")
print("=" * 90)

print(
    results_df[
        [
            "threshold",
            "raw_selected_components",
            "raw_prediction_voxels",
            "raw_dice",
            "raw_iou",
            "raw_slice_recall",
            "raw_fpr",
            "kept_3d_components",
            "final_prediction_voxels",
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
# SAVE CSVs
# ============================================================

summary_path = (
    OUTPUT_DIR
    / "case7_pet_split_end_to_end_results.csv"
)

components_path = (
    OUTPUT_DIR
    / "case7_pet_split_selected_components.csv"
)

results_df.to_csv(
    summary_path,
    index=False
)

selected_df.to_csv(
    components_path,
    index=False
)


# ============================================================
# BEST END-TO-END RESULT
#
# This is ONLY a factual report of the measured metric.
# No threshold is declared as "best" in the research sense.
# ============================================================

best_row = results_df.loc[
    results_df[
        "final_dice"
    ].idxmax()
]

print()
print("=" * 90)
print("HIGHEST OBSERVED FINAL DICE IN THIS AUDIT")
print("=" * 90)

print(
    "Threshold:",
    best_row["threshold"]
)

print(
    "Final Dice:",
    f"{best_row['final_dice']:.4f}"
)

print(
    "Final IoU:",
    f"{best_row['final_iou']:.4f}"
)

print(
    "Final Slice Recall:",
    f"{best_row['final_slice_recall']:.4f}"
)

print(
    "Final FPR:",
    f"{best_row['final_fpr']:.4f}"
)

print(
    "Final prediction voxels:",
    int(
        best_row[
            "final_prediction_voxels"
        ]
    )
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 90)
print("===== PET SPLIT END-TO-END EXPERIMENT COMPLETE =====")
print("=" * 90)

print()
print("Saved:")
print(summary_path)
print(components_path)