from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk


CASE = Path(
    "development_cases/PETCT_0168f65af8/"
    "manifest-1789661648760/FDG-PET-CT-Lesions/"
    "PETCT_0168f65af8"
)

GT_PATH = Path(
    "development_cases/PETCT_0168f65af8/"
    "tumor_mask_ct_visible.nii.gz"
)


def find_series(case_dir, text):
    matches = [
        p for p in case_dir.rglob("*")
        if p.is_dir() and text in p.name
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one series containing '{text}', "
            f"found {len(matches)}"
        )

    return matches[0]


def read_series(path):
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(path))

    if not files:
        raise RuntimeError(f"No DICOM files: {path}")

    reader.SetFileNames(files)
    return reader.Execute()


def normalize_ct(ct):
    ct = np.clip(ct, -1000, 1000)
    return ((ct + 1000) / 2000 * 255).astype(np.uint8)


def build_body(ct):
    body = np.zeros_like(ct, dtype=np.uint8)
    kernel = np.ones((11, 11), np.uint8)

    for z in range(ct.shape[0]):

        binary = (ct[z] > -900).astype(np.uint8) * 255

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

        if contours:
            c = max(contours, key=cv2.contourArea)
            cv2.drawContours(
                body[z],
                [c],
                -1,
                1,
                -1
            )

    return body.astype(bool)


def percentile01(values, value):
    values = np.asarray(values, dtype=float)

    if len(values) <= 1:
        return 0.5

    return float(
        np.mean(values <= value)
    )


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


print("Loading CT and PET...")

ct_dir = find_series(CASE, "GK p.v.3")
pet_dir = find_series(CASE, "PET corr.")

ct_img = read_series(ct_dir)
pet_img = read_series(pet_dir)

print("CT:", ct_img.GetSize())
print("PET:", pet_img.GetSize())

# PET -> CT physical space
pet_ct = sitk.Resample(
    pet_img,
    ct_img,
    sitk.Transform(),
    sitk.sitkLinear,
    0.0,
    pet_img.GetPixelID()
)

ct = sitk.GetArrayFromImage(ct_img).astype(np.float32)
pet = sitk.GetArrayFromImage(pet_ct).astype(np.float32)

body = build_body(ct)

ct_display = normalize_ct(ct)

kernel = np.ones((5, 5), np.uint8)

records = []

print("Generating candidates...")

for z in range(ct.shape[0]):

    body_slice = body[z]

    if not body_slice.any():
        continue

    adaptive = cv2.adaptiveThreshold(
        ct_display[z],
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        3
    )

    # Opening only.
    clean = cv2.morphologyEx(
        adaptive,
        cv2.MORPH_OPEN,
        kernel
    )

    clean = np.where(
        body_slice,
        clean,
        0
    ).astype(np.uint8)

    clean[:5, :] = 0
    clean[-5:, :] = 0
    clean[:, :5] = 0
    clean[:, -5:] = 0

    contours, _ = cv2.findContours(
        clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Positive PET reference values only.
    body_pet = pet[z][body_slice]
    positive_pet = body_pet[body_pet > 0]

    if positive_pet.size < 20:
        continue

    q25, q50, q75, q90 = np.percentile(
        positive_pet,
        [25, 50, 75, 90]
    )

    iqr = max(q75 - q25, 1.0)

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 100 or area > 15000:
            continue

        mask = np.zeros_like(ct_display[z], dtype=np.uint8)

        cv2.drawContours(
            mask,
            [contour],
            -1,
            1,
            -1
        )

        pixels = mask > 0

        if pixels.sum() == 0:
            continue

        ct_values = ct_display[z][pixels].astype(float)
        pet_values = pet[z][pixels].astype(float)

        perimeter = cv2.arcLength(
            contour,
            True
        )

        circularity = (
            4 * np.pi * area /
            (perimeter * perimeter + 1e-6)
        )

        # Stable PET contrast against the current body slice.
        pet_mean = float(np.mean(pet_values))
        uptake_contrast = (
            pet_mean - q50
        ) / iqr

        # How much of the candidate is in the hot-pixel region.
        hot_fraction = float(
            np.mean(pet_values >= q90)
        )

        # CT texture proxy.
        lap = cv2.Laplacian(
            ct_display[z],
            cv2.CV_32F
        )

        texture = float(
            np.var(lap[pixels])
        )

        x, y, w, h = cv2.boundingRect(contour)

        records.append({
            "slice": z,
            "area": float(area),
            "circularity": float(
                np.clip(circularity, 0, 1)
            ),
            "pet_mean": pet_mean,
            "uptake_contrast": float(uptake_contrast),
            "hot_fraction": hot_fraction,
            "texture": texture,
            "bbox_w": w,
            "bbox_h": h,
            "mask": mask
        })


df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError("No candidates generated.")

print("Candidate rows:", len(df))

# ---------------------------------------------------------
# GLOBAL scoring
# ---------------------------------------------------------

pet_values = df["uptake_contrast"].to_numpy()
hot_values = df["hot_fraction"].to_numpy()
texture_values = np.log1p(
    df["texture"].to_numpy()
)

# Global percentile ranks.
df["pet_score"] = [
    percentile01(pet_values, x)
    for x in pet_values
]

df["hot_score"] = [
    percentile01(hot_values, x)
    for x in hot_values
]

df["texture_score"] = [
    percentile01(texture_values, x)
    for x in texture_values
]

# Absolute shape score.
df["shape_score"] = df["circularity"]

# Hybrid score.
df["score"] = (
    0.55 * df["pet_score"]
    + 0.20 * df["hot_score"]
    + 0.15 * df["texture_score"]
    + 0.10 * df["shape_score"]
)

# ---------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------

prediction = np.zeros_like(
    ct,
    dtype=np.uint8
)

# Conservative absolute requirements.
eligible = df[
    (df["uptake_contrast"] >= 1.0)
    &
    (df["hot_fraction"] >= 0.20)
    &
    (df["score"] >= 0.80)
]

# At most one candidate per slice.
best = (
    eligible
    .sort_values("score", ascending=False)
    .drop_duplicates("slice")
)

for _, row in best.iterrows():

    z = int(row["slice"])
    mask = row["mask"]

    prediction[z][mask > 0] = 1


# ---------------------------------------------------------
# Save prediction
# ---------------------------------------------------------

prediction_img = sitk.GetImageFromArray(
    prediction.astype(np.uint8)
)

prediction_img.CopyInformation(ct_img)

output = CASE / "hybrid_segmentation_v2.nii.gz"

sitk.WriteImage(
    prediction_img,
    str(output)
)

# Remove numpy array objects before CSV.
csv_df = df.drop(
    columns=["mask"]
)

csv_path = CASE / "hybrid_v2_candidate_scores.csv"

csv_df.to_csv(
    csv_path,
    index=False
)

print("\nSaved prediction:", output)
print("Saved candidate table:", csv_path)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

gt_img = sitk.ReadImage(
    str(GT_PATH)
)

gt = (
    sitk.GetArrayFromImage(gt_img) > 0
)

pred = prediction > 0

intersection = np.logical_and(
    pred,
    gt
).sum()

union = np.logical_or(
    pred,
    gt
).sum()

dice = (
    2 * intersection /
    (pred.sum() + gt.sum())
    if pred.sum() + gt.sum() > 0
    else 0
)

iou = (
    intersection / union
    if union > 0
    else 0
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

print("\n===== HYBRID V2 DEVELOPMENT RESULT =====")
print(
    "Candidate rows:",
    len(df)
)

print(
    "Selected slices:",
    int(pred_positive.sum())
)

print(
    "GT positive slices:",
    int(gt_positive.sum())
)

print(
    "Slice recall:",
    round(
        float(
            tp /
            max(gt_positive.sum(), 1)
        ),
        4
    )
)

print(
    "Dice:",
    round(float(dice), 4)
)

print(
    "IoU:",
    round(float(iou), 4)
)

print(
    "False-positive slices:",
    int(fp)
)

print(
    "False-positive rate:",
    round(
        float(
            fp /
            max((~gt_positive).sum(), 1)
        ),
        4
    )
)

print(
    "\nSelected candidates:",
    len(best)
)

print(
    "Selected slices:",
    sorted(
        best["slice"].astype(int).tolist()
    )
)
