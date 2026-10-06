from pathlib import Path
import runpy
import numpy as np
import pandas as pd
import cv2
import SimpleITK as sitk


ROOT = Path(".")


# =========================================================
# LOAD EXISTING CASE 7 PIPELINE
# =========================================================

ns = runpy.run_path(
    "src/v4_validate_case7.py",
    run_name="__case7_union_ablation__"
)


prepare_case7_original = ns["prepare_case7"]
build_prediction_with_rescue = ns[
    "build_prediction_with_rescue"
]


# =========================================================
# UNION BODY MASK
# =========================================================

def build_union_body_mask(ct):

    body = np.zeros_like(
        ct,
        dtype=np.uint8
    )

    kernel = np.ones(
        (11, 11),
        np.uint8
    )

    for z in range(ct.shape[0]):

        binary = (
            ct[z] > -900
        ).astype(np.uint8) * 255

        closed = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # IMPORTANT:
        # Keep ALL connected components instead of
        # selecting only the largest one.
        for contour in contours:

            cv2.drawContours(
                body[z],
                [contour],
                -1,
                1,
                -1
            )

    return body.astype(bool)


# =========================================================
# PATCH ONLY BODY-MASK FUNCTION
# =========================================================
#
# prepare_case7() resolves build_body_mask through the
# function object's own __globals__, not the runpy result
# dictionary. Patch that dictionary directly.
# =========================================================

prepare_case7_original.__globals__["build_body_mask"] = (
    build_union_body_mask
)


# =========================================================
# PREPARE CASE 7 WITH UNION BODY
# =========================================================

print("\n" + "=" * 80)
print("CASE 7 — BODY UNION FROZEN V4 ABLATION")
print("=" * 80)

prepared, ct_img, ct, union_body = (
    prepare_case7_original()
)


# =========================================================
# BASIC AUDIT
# =========================================================

gt = prepared[
    "gt"
]

candidate_masks = prepared[
    "candidate_masks"
]

reference = prepared[
    "reference"
]

print("\nUnion body voxels:", int(union_body.sum()))
print("GT voxels:", int(gt.sum()))
print("Candidates:", len(candidate_masks))

accepted = prepared[
    "accepted"
]

print(
    "Accepted candidates:",
    int(accepted.sum())
)

accepted_slices = sorted(
    {
        int(reference.iloc[i]["slice"])
        for i, flag in enumerate(accepted)
        if flag
    }
)

print(
    "Accepted slices:",
    accepted_slices
)


# =========================================================
# GT / BODY COVERAGE
# =========================================================

gt_inside = int(
    np.logical_and(
        gt,
        union_body
    ).sum()
)

print(
    "\nGT inside union body:",
    gt_inside
)

print(
    "Body coverage:",
    f"{100.0 * gt_inside / gt.sum():.2f}%"
)


# =========================================================
# RUN THE FROZEN FINAL PREDICTION
#
# ONLY BODY MASK HAS CHANGED.
#
# Everything else remains:
#   original core      = 2.00
#   recovered core     = 1.50
#   rescue PET         = 1.50
#   rescue mean SUV    = 2.25
#   rescue area        = 20
#   rescue distance    = 28
# =========================================================

prediction, rescue_count, rescue_rows = (
    build_prediction_with_rescue(
        prepared,
        ct,
        union_body,
        1.50,
        2.25
    )
)


# =========================================================
# METRICS
# =========================================================

pred = (
    prediction > 0
)

gt_bool = (
    gt > 0
)

intersection = int(
    np.logical_and(
        pred,
        gt_bool
    ).sum()
)

pred_voxels = int(
    pred.sum()
)

gt_voxels = int(
    gt_bool.sum()
)

union = int(
    np.logical_or(
        pred,
        gt_bool
    ).sum()
)

dice = (
    2.0 * intersection
    / (
        pred_voxels
        + gt_voxels
        + 1e-9
    )
)

iou = (
    intersection
    / (
        union
        + 1e-9
    )
)


gt_slices = np.any(
    gt_bool,
    axis=(1, 2)
)

pred_slices = np.any(
    pred,
    axis=(1, 2)
)

slice_recall = (
    float(
        np.logical_and(
            gt_slices,
            pred_slices
        ).sum()
    )
    / (
        gt_slices.sum()
        + 1e-9
    )
)


background = ~gt_bool

false_positive_voxels = int(
    np.logical_and(
        pred,
        background
    ).sum()
)

fpr = (
    false_positive_voxels
    / (
        background.sum()
        + 1e-9
    )
)


# =========================================================
# PRINT RESULT
# =========================================================

print("\n" + "=" * 80)
print("===== BODY UNION FROZEN V4 RESULT =====")
print("=" * 80)

print(
    "Dice              =",
    f"{dice:.6f}"
)

print(
    "IoU               =",
    f"{iou:.6f}"
)

print(
    "Slice Recall      =",
    f"{slice_recall:.6f}"
)

print(
    "FPR               =",
    f"{fpr:.6f}"
)

print(
    "Prediction Voxels =",
    pred_voxels
)

print(
    "GT Voxels         =",
    gt_voxels
)

print(
    "Intersection      =",
    intersection
)

print(
    "False Positive Voxels =",
    false_positive_voxels
)

print(
    "Rescue Slices     =",
    rescue_count
)


# =========================================================
# RESCUE SLICE DETAILS
# =========================================================

if rescue_rows:

    rescue_df = pd.DataFrame(
        rescue_rows
    )

    print("\n===== SELECTED RESCUE COMPONENTS =====")

    print(
        rescue_df.to_string(
            index=False
        )
    )

else:

    rescue_df = pd.DataFrame()

    print(
        "\nNo rescue components selected."
    )


# =========================================================
# SAVE PREDICTION
# =========================================================

out_dir = (
    ROOT
    / "validation_results"
    / "PETCT_0011f3deaf"
)

out_dir.mkdir(
    parents=True,
    exist_ok=True
)


prediction_img = (
    sitk.GetImageFromArray(
        prediction.astype(np.uint8)
    )
)

prediction_img.CopyInformation(
    ct_img
)

prediction_path = (
    out_dir
    / "case7_body_union_frozen_prediction.nii.gz"
)

sitk.WriteImage(
    prediction_img,
    str(prediction_path)
)


# =========================================================
# SAVE METRICS
# =========================================================

metrics = pd.DataFrame(
    [
        {
            "case": "PETCT_0011f3deaf",
            "body_mask": "ALL_COMPONENT_UNION",
            "dice": dice,
            "iou": iou,
            "slice_recall": slice_recall,
            "fpr": fpr,
            "prediction_voxels": pred_voxels,
            "gt_voxels": gt_voxels,
            "intersection": intersection,
            "false_positive_voxels":
                false_positive_voxels,
            "rescue_slices":
                rescue_count,
        }
    ]
)

metrics_path = (
    out_dir
    / "case7_body_union_frozen_metrics.csv"
)

metrics.to_csv(
    metrics_path,
    index=False
)


print("\nSaved:")
print(prediction_path)
print(metrics_path)

print(
    "\n===== BODY UNION ABLATION COMPLETE ====="
)
