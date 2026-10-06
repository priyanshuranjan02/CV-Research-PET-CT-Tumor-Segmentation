from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from v4_validate_case7 import prepare_case7 as prepare_case7_original, normalize_ct

CASE_NAME = "PETCT_0011f3deaf"
MAX_CANDIDATE_AREA = 10000
PET_THRESHOLDS = [1.25, 1.50, 1.75, 2.00, 2.25]
MIN_COMPONENT_AREA = 20

OUTPUT_DIR = ROOT / "validation_results" / CASE_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "case7_pet_split_audit.csv"


def build_union_body_mask(ct):
    body = np.zeros_like(ct, dtype=np.uint8)
    kernel = np.ones((11, 11), np.uint8)
    for z in range(ct.shape[0]):
        binary = (ct[z] > -900).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 0:
                cv2.drawContours(body[z], [contour], -1, 1, -1)
    return body.astype(bool)


prepare_case7_original.__globals__["build_body_mask"] = build_union_body_mask

print("=" * 90)
print("===== CASE 7 — PET-GUIDED OVERSIZED CONTOUR SPLIT AUDIT =====")
print("=" * 90)
print("Case:", CASE_NAME)
print("PET thresholds:", PET_THRESHOLDS)
print("Minimum PET component area:", MIN_COMPONENT_AREA)
print("Oversized contour area >", MAX_CANDIDATE_AREA)

prepared, ct_img, ct, body = prepare_case7_original()
suv = prepared["suv"]

gt_path = ROOT / "verified_case" / "tumor_mask_ct_visible.nii.gz"
if not gt_path.exists():
    raise FileNotFoundError(f"GT not found: {gt_path}")
gt_img = sitk.ReadImage(str(gt_path))
gt = sitk.GetArrayFromImage(gt_img) > 0
gt_slices = np.where(np.any(gt, axis=(1, 2)))[0]

print("CT shape:", ct.shape)
print("SUV shape:", suv.shape)
print("Union-body voxels:", int(body.sum()))
print("GT-positive slices:", len(gt_slices))

ct_display = normalize_ct(ct)
opening_kernel = np.ones((5, 5), np.uint8)
rows = []

for z in gt_slices:
    body_slice = body[z]
    if not np.any(body_slice):
        continue

    adaptive = cv2.adaptiveThreshold(
        ct_display[z], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 3
    )
    opened = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, opening_kernel)
    opened = np.where(body_slice, opened, 0).astype(np.uint8)
    opened[:5, :] = 0
    opened[-5:, :] = 0
    opened[:, :5] = 0
    opened[:, -5:] = 0

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gt_slice = gt[z]
    gt_area = int(gt_slice.sum())

    for contour_idx, contour in enumerate(contours):
        contour_area = float(cv2.contourArea(contour))
        if contour_area <= MAX_CANDIDATE_AREA:
            continue

        contour_mask = np.zeros_like(opened, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 1, -1)
        contour_pixels = contour_mask > 0

        for threshold in PET_THRESHOLDS:
            hot = (contour_pixels & np.isfinite(suv[z]) & (suv[z] >= threshold)).astype(np.uint8)
            n, labels, stats, centroids = cv2.connectedComponentsWithStats(hot, connectivity=8)

            for component_id in range(1, n):
                area = int(stats[component_id, cv2.CC_STAT_AREA])
                if area < MIN_COMPONENT_AREA:
                    continue

                component = labels == component_id
                values = suv[z][component]
                if values.size == 0:
                    continue

                overlap = int(np.logical_and(component, gt_slice).sum())
                union = int(np.logical_or(component, gt_slice).sum())
                dice = 2.0 * overlap / (area + gt_area) if area + gt_area > 0 else 0.0
                iou = overlap / union if union > 0 else 0.0

                rows.append({
                    "slice": int(z),
                    "contour_index": int(contour_idx),
                    "contour_area": int(contour_pixels.sum()),
                    "threshold": float(threshold),
                    "component_id": int(component_id),
                    "component_area": area,
                    "mean_suv": float(np.mean(values)),
                    "median_suv": float(np.median(values)),
                    "max_suv": float(np.max(values)),
                    "p90_suv": float(np.percentile(values, 90)),
                    "p95_suv": float(np.percentile(values, 95)),
                    "frac_ge_1.5": float(np.mean(values >= 1.5)),
                    "frac_ge_2.25": float(np.mean(values >= 2.25)),
                    "overlap_gt": overlap,
                    "dice_gt": float(dice),
                    "iou_gt": float(iou),
                    "cx": float(centroids[component_id][0]),
                    "cy": float(centroids[component_id][1]),
                })

df = pd.DataFrame(rows)
if df.empty:
    print("No PET components survived the audit.")
    raise SystemExit(0)

best_geo = (df.sort_values(["slice", "dice_gt", "overlap_gt"], ascending=[True, False, False])
              .groupby("slice", as_index=False).first())

print("\n" + "=" * 90)
print("BEST PET-SPLIT COMPONENT BY GT DICE (AUDIT ONLY)")
print("=" * 90)
print(best_geo[["slice","threshold","contour_area","component_area","overlap_gt","dice_gt","mean_suv","median_suv","max_suv","p95_suv"]].to_string(index=False))

best_mean = (df.sort_values(["slice", "mean_suv", "component_area"], ascending=[True, False, True])
               .groupby("slice", as_index=False).first())

print("\n" + "=" * 90)
print("BEST COMPONENT BY MEAN SUV (NO GT USED FOR SELECTION)")
print("=" * 90)
print(best_mean[["slice","threshold","contour_area","component_area","mean_suv","median_suv","max_suv","p95_suv","overlap_gt","dice_gt"]].to_string(index=False))

summary_rows = []
for threshold in PET_THRESHOLDS:
    t = df[df["threshold"] == threshold]
    if t.empty:
        continue
    best_t = (t.sort_values(["slice", "dice_gt", "overlap_gt"], ascending=[True, False, False])
                .groupby("slice", as_index=False).first())
    mean_choice = (t.sort_values(["slice", "mean_suv", "component_area"], ascending=[True, False, True])
                     .groupby("slice", as_index=False).first())
    summary_rows.append({
        "threshold": threshold,
        "components": int(len(t)),
        "slices_with_components": int(best_t["slice"].nunique()),
        "best_dice_macro": float(best_t["dice_gt"].mean()),
        "best_dice_max": float(best_t["dice_gt"].max()),
        "best_overlap_slices": int((best_t["overlap_gt"] > 0).sum()),
        "mean_suv_choice_macro_dice": float(mean_choice["dice_gt"].mean()),
        "mean_suv_choice_overlap_slices": int((mean_choice["overlap_gt"] > 0).sum()),
    })

summary_df = pd.DataFrame(summary_rows)
print("\n" + "=" * 90)
print("THRESHOLD SUMMARY")
print("=" * 90)
print(summary_df.to_string(index=False))

print("\n" + "=" * 90)
print("TOP 20 PET COMPONENTS BY MEAN SUV")
print("=" * 90)
print(df.sort_values("mean_suv", ascending=False).head(20)[[
    "slice","threshold","contour_area","component_area","mean_suv","median_suv","max_suv","p95_suv","overlap_gt","dice_gt"
]].to_string(index=False))

df.to_csv(OUTPUT_CSV, index=False)
print("\nAUDIT COMPLETE")
print("Rows saved:", len(df))
print("Saved:", OUTPUT_CSV)