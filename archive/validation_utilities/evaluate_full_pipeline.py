from pathlib import Path
import csv
import numpy as np
import cv2

CT_DIR = Path("verified_case/ct_2d")
MASK_DIR = Path("verified_case/masks_2d")
OUT_CSV = Path("verified_case/full_pipeline_metrics.csv")


def pipeline_prediction(img):
    # -------------------------------
    # PREPROCESSING — same as pipeline.py
    # -------------------------------
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

    # -------------------------------
    # SEGMENTATION — same parameters
    # -------------------------------
    adaptive_mask = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        3
    )

    # -------------------------------
    # MORPHOLOGY — same as pipeline.py
    # -------------------------------
    kernel = np.ones((5, 5), np.uint8)

    clean_mask = cv2.morphologyEx(
        adaptive_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    clean_mask = cv2.dilate(
        clean_mask,
        kernel,
        iterations=1
    )

    # -------------------------------
    # REMOVE BORDERS
    # -------------------------------
    h, w = clean_mask.shape
    margin = 5

    clean_mask[:margin, :] = 0
    clean_mask[-margin:, :] = 0
    clean_mask[:, :margin] = 0
    clean_mask[:, -margin:] = 0

    # -------------------------------
    # CONTOURS — same as pipeline.py
    # -------------------------------
    contours, _ = cv2.findContours(
        clean_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = []

    for cnt in contours:
        if cnt is None or len(cnt) < 3:
            continue

        area = cv2.contourArea(cnt)

        if 1000 < area < 50000:
            valid_contours.append(cnt)

    if len(valid_contours) > 0:
        tumor_contour = max(
            valid_contours,
            key=cv2.contourArea
        )
    elif len(contours) > 0:
        tumor_contour = max(
            contours,
            key=cv2.contourArea
        )
    else:
        tumor_contour = None

    # Convert selected contour to binary prediction mask
    prediction = np.zeros_like(clean_mask)

    if tumor_contour is not None:
        cv2.drawContours(
            prediction,
            [tumor_contour],
            -1,
            255,
            thickness=-1
        )

    return prediction


rows = []

for mask_path in sorted(MASK_DIR.glob("slice_*.png")):

    name = mask_path.stem
    ct_path = CT_DIR / f"{name}.tif"

    img = cv2.imread(
        str(ct_path),
        cv2.IMREAD_GRAYSCALE
    )

    gt = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise RuntimeError(f"Could not read CT: {ct_path}")

    if gt is None:
        raise RuntimeError(f"Could not read mask: {mask_path}")

    pred = pipeline_prediction(img)

    gt_bin = gt > 0
    pred_bin = pred > 0

    gt_pixels = int(gt_bin.sum())
    pred_pixels = int(pred_bin.sum())

    intersection = int(
        np.logical_and(gt_bin, pred_bin).sum()
    )

    union = int(
        np.logical_or(gt_bin, pred_bin).sum()
    )

    dice = (
        2.0 * intersection /
        (gt_pixels + pred_pixels)
        if gt_pixels + pred_pixels > 0
        else 1.0
    )

    iou = (
        intersection / union
        if union > 0
        else 1.0
    )

    rows.append({
        "slice": name,
        "ground_truth_pixels": gt_pixels,
        "predicted_pixels": pred_pixels,
        "intersection": intersection,
        "dice": dice,
        "iou": iou,
        "ground_truth_positive": int(gt_pixels > 0),
        "prediction_positive": int(pred_pixels > 0),
    })


with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(rows)


positive = [
    r for r in rows
    if r["ground_truth_positive"] == 1
]

negative = [
    r for r in rows
    if r["ground_truth_positive"] == 0
]

detected_positive = sum(
    r["prediction_positive"]
    for r in positive
)

false_positive_slices = sum(
    r["prediction_positive"]
    for r in negative
)

mean_dice = np.mean([
    r["dice"] for r in positive
])

median_dice = np.median([
    r["dice"] for r in positive
])

mean_iou = np.mean([
    r["iou"] for r in positive
])

median_iou = np.median([
    r["iou"] for r in positive
])

print("\n=== FULL PIPELINE EVALUATION ===")
print("Total CT slices:", len(rows))
print("Ground-truth positive slices:", len(positive))
print("Ground-truth negative slices:", len(negative))

print("\n--- Positive slices ---")
print(
    f"Detected positive slices: "
    f"{detected_positive}/{len(positive)}"
)

print(
    f"Slice recall: "
    f"{detected_positive / len(positive):.4f}"
)

print(f"Mean Dice: {mean_dice:.4f}")
print(f"Median Dice: {median_dice:.4f}")
print(f"Mean IoU: {mean_iou:.4f}")
print(f"Median IoU: {median_iou:.4f}")

print("\n--- Negative slices ---")
print(
    f"False-positive slices: "
    f"{false_positive_slices}/{len(negative)}"
)

print(
    f"False-positive slice rate: "
    f"{false_positive_slices / len(negative):.4f}"
)

print("\nSaved:", OUT_CSV)
