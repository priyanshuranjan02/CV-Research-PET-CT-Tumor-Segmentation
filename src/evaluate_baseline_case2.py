from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk


ROOT = Path("development_cases/PETCT_04606080a0")

CT_UID = "1.3.6.1.4.1.14519.5.2.1.97311793319319996098681191493157652433"

GT_PATH = ROOT / "tumor_mask_ct_visible.nii.gz"


# =========================================================
# FROZEN BASELINE
# Same segmentation parameters as the original pipeline.py
# =========================================================

def baseline_segment(img):
    # Normalize CT slice to 8-bit
    normalized = cv2.normalize(
        img,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        3
    )

    # Morphological closing
    kernel = np.ones((5, 5), np.uint8)

    clean = cv2.morphologyEx(
        adaptive,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Dilation
    clean = cv2.dilate(
        clean,
        kernel,
        iterations=1
    )

    # Remove image border
    clean[:5, :] = 0
    clean[-5:, :] = 0
    clean[:, :5] = 0
    clean[:, -5:] = 0

    # External contours
    contours, _ = cv2.findContours(
        clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Same baseline contour filtering
    valid = [
        c for c in contours
        if 1000 < cv2.contourArea(c) < 50000
    ]

    prediction = np.zeros_like(normalized, dtype=np.uint8)

    if valid:
        best = max(valid, key=cv2.contourArea)
        cv2.drawContours(
            prediction,
            [best],
            -1,
            1,
            thickness=-1
        )

    elif contours:
        # Same fallback as baseline
        best = max(contours, key=cv2.contourArea)
        cv2.drawContours(
            prediction,
            [best],
            -1,
            1,
            thickness=-1
        )

    return prediction.astype(bool)


# =========================================================
# LOAD CT
# =========================================================

series_files = []

for path in ROOT.rglob("*.dcm"):
    try:
        import pydicom

        ds = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True
        )

        if getattr(ds, "SeriesInstanceUID", None) == CT_UID:
            series_files.append(path)

    except Exception:
        continue


if not series_files:
    raise RuntimeError("CT DICOM series not found.")

series_files = sorted(
    series_files,
    key=lambda p: p.name
)


reader = sitk.ImageSeriesReader()
reader.SetFileNames([str(p) for p in series_files])
ct_img = reader.Execute()

ct = sitk.GetArrayFromImage(ct_img).astype(np.float32)


# =========================================================
# LOAD GROUND TRUTH
# =========================================================

gt_img = sitk.ReadImage(str(GT_PATH))
gt = sitk.GetArrayFromImage(gt_img) > 0


if ct.shape != gt.shape:
    raise RuntimeError(
        f"Shape mismatch: CT={ct.shape}, GT={gt.shape}"
    )


print("===== CASE 2 BASELINE EVALUATION =====")
print("CT shape:", ct.shape)
print("GT shape:", gt.shape)


# =========================================================
# EVALUATION
# =========================================================

dice_scores = []
iou_scores = []

gt_positive_slices = 0
detected_positive_slices = 0
false_positive_slices = 0

intersection_total = 0
prediction_total = 0
gt_total = 0


for z in range(ct.shape[0]):

    prediction = baseline_segment(ct[z])
    truth = gt[z]

    pred_sum = int(prediction.sum())
    gt_sum = int(truth.sum())

    intersection = int(
        np.logical_and(prediction, truth).sum()
    )

    union = int(
        np.logical_or(prediction, truth).sum()
    )

    if gt_sum > 0:
        gt_positive_slices += 1

        if pred_sum > 0:
            detected_positive_slices += 1

    elif pred_sum > 0:
        false_positive_slices += 1

    # Per-slice Dice
    if pred_sum == 0 and gt_sum == 0:
        dice = 1.0
    elif pred_sum + gt_sum == 0:
        dice = 0.0
    else:
        dice = (
            2.0 * intersection /
            (pred_sum + gt_sum)
        )

    # Per-slice IoU
    if union == 0:
        iou = 1.0
    else:
        iou = intersection / union

    dice_scores.append(dice)
    iou_scores.append(iou)

    intersection_total += intersection
    prediction_total += pred_sum
    gt_total += gt_sum


# =========================================================
# RESULTS
# =========================================================

dice_scores = np.array(dice_scores)
iou_scores = np.array(iou_scores)

mean_dice = float(np.mean(dice_scores))
median_dice = float(np.median(dice_scores))

mean_iou = float(np.mean(iou_scores))
median_iou = float(np.median(iou_scores))

slice_recall = (
    detected_positive_slices / gt_positive_slices
    if gt_positive_slices
    else 0.0
)

negative_slices = ct.shape[0] - gt_positive_slices

fpr = (
    false_positive_slices / negative_slices
    if negative_slices
    else 0.0
)


print()
print("===== RESULTS =====")
print("Total CT slices:", ct.shape[0])
print("GT positive slices:", gt_positive_slices)
print("GT negative slices:", negative_slices)
print()
print("Detected positive slices:", detected_positive_slices)
print("False-positive slices:", false_positive_slices)
print()
print("Slice recall:", round(slice_recall, 4))
print("False-positive rate:", round(fpr, 4))
print()
print("Mean Dice:", round(mean_dice, 4))
print("Median Dice:", round(median_dice, 4))
print("Mean IoU:", round(mean_iou, 4))
print("Median IoU:", round(median_iou, 4))