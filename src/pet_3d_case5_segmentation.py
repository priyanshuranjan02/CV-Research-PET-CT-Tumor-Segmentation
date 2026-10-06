from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
import pydicom


# =========================================================
# CASE
# =========================================================

CASE = Path(
    "development_cases/PETCT_11afab3485/"
    "manifest-1789921915902/FDG-PET-CT-Lesions/"
    "PETCT_11afab3485"
)

GT_PATH = Path(
    "development_cases/PETCT_11afab3485/"
    "tumor_mask_ct_visible.nii.gz"
)


# =========================================================
# DIAGNOSTIC PARAMETERS
# =========================================================

PET_THRESHOLD = 1.50
MIN_COMPONENT_VOXELS = 75
MIN_COMPONENT_SLICES = 2


# =========================================================
# HELPERS
# =========================================================

def find_series(case_dir, text):

    matches = [
        p
        for p in case_dir.rglob("*")
        if p.is_dir() and text in p.name
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one directory containing "
            f"'{text}', found {len(matches)}"
        )

    return matches[0]


def read_series(path):

    reader = sitk.ImageSeriesReader()

    files = reader.GetGDCMSeriesFileNames(
        str(path)
    )

    if not files:
        raise RuntimeError(
            f"No DICOM files found in {path}"
        )

    reader.SetFileNames(files)

    return reader.Execute()


def read_pet_metadata(pet_dir):

    reader = sitk.ImageSeriesReader()

    files = reader.GetGDCMSeriesFileNames(
        str(pet_dir)
    )

    if not files:
        raise RuntimeError(
            f"No PET files found in {pet_dir}"
        )

    weight = None
    dose = None
    units = None

    for path in files:

        ds = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True
        )

        if weight is None:

            value = getattr(
                ds,
                "PatientWeight",
                None
            )

            if value is not None:
                weight = float(value)

        if dose is None:

            seq = getattr(
                ds,
                "RadiopharmaceuticalInformationSequence",
                None
            )

            if seq:

                value = getattr(
                    seq[0],
                    "RadionuclideTotalDose",
                    None
                )

                if value is not None:
                    dose = float(value)

        if units is None:
            units = getattr(
                ds,
                "Units",
                None
            )

    if weight is None:
        raise RuntimeError(
            "PatientWeight not found."
        )

    if dose is None:
        raise RuntimeError(
            "Injected dose not found."
        )

    return weight, dose, units


def calculate_suv(
    activity_bqml,
    weight_kg,
    dose_bq
):

    return (
        activity_bqml
        * (weight_kg * 1000.0)
        / dose_bq
    )


def build_body_mask(ct):

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

        if not contours:
            continue

        contour = max(
            contours,
            key=cv2.contourArea
        )

        cv2.drawContours(
            body[z],
            [contour],
            -1,
            1,
            -1
        )

    return body.astype(bool)


# =========================================================
# LOAD
# =========================================================

print("Loading CT...")

ct_img = read_series(
    find_series(
        CASE,
        "GK p.v.3"
    )
)

print("Loading PET...")

pet_dir = find_series(
    CASE,
    "PET corr."
)

pet_img = read_series(
    pet_dir
)

(
    weight_kg,
    dose_bq,
    units
) = read_pet_metadata(
    pet_dir
)

print(
    "Patient weight (kg):",
    weight_kg
)

print(
    "Injected dose (Bq):",
    dose_bq
)

print(
    "PET units:",
    units
)

ct = sitk.GetArrayFromImage(
    ct_img
).astype(
    np.float32
)

pet_stored = sitk.GetArrayFromImage(
    pet_img
).astype(
    np.float64
)

suv_native = calculate_suv(
    pet_stored,
    weight_kg,
    dose_bq
)

suv_native_img = sitk.GetImageFromArray(
    suv_native.astype(
        np.float32
    )
)

suv_native_img.CopyInformation(
    pet_img
)

suv_ct_img = sitk.Resample(
    suv_native_img,
    ct_img,
    sitk.Transform(),
    sitk.sitkLinear,
    0.0,
    sitk.sitkFloat32
)

suv = sitk.GetArrayFromImage(
    suv_ct_img
).astype(
    np.float64
)

gt_img = sitk.ReadImage(
    str(GT_PATH)
)

gt = (
    sitk.GetArrayFromImage(
        gt_img
    ) > 0
)

body = build_body_mask(
    ct
)


# =========================================================
# PURE 3D PET SEGMENTATION
# =========================================================

print(
    "\n===== PURE 3D PET SEGMENTATION ====="
)

print(
    "PET threshold:",
    PET_THRESHOLD
)

print(
    "Minimum component voxels:",
    MIN_COMPONENT_VOXELS
)

print(
    "Minimum component slices:",
    MIN_COMPONENT_SLICES
)

pet_binary = (
    body
    & (
        suv >= PET_THRESHOLD
    )
).astype(
    np.uint8
)

pet_binary_img = sitk.GetImageFromArray(
    pet_binary
)

cc_img = sitk.ConnectedComponent(
    pet_binary_img
)

cc = sitk.GetArrayFromImage(
    cc_img
)

stats = sitk.LabelShapeStatisticsImageFilter()

stats.Execute(
    cc_img
)

num_components = (
    stats.GetNumberOfLabels()
)

print(
    "Total 3D PET components:",
    num_components
)

final_mask = np.zeros_like(
    pet_binary,
    dtype=np.uint8
)

kept = 0
kept_voxels = 0

for label in stats.GetLabels():

    component = (
        cc == label
    )

    voxel_count = int(
        component.sum()
    )

    if (
        voxel_count
        < MIN_COMPONENT_VOXELS
    ):
        continue

    z_indices = np.where(
        component
    )[0]

    if len(
        z_indices
    ) == 0:
        continue

    z_span = (
        int(z_indices.max())
        - int(z_indices.min())
        + 1
    )

    if (
        z_span
        < MIN_COMPONENT_SLICES
    ):
        continue

    final_mask[
        component
    ] = 1

    kept += 1
    kept_voxels += voxel_count

print(
    "Kept 3D PET components:",
    kept
)

print(
    "Prediction voxels:",
    int(
        final_mask.sum()
    )
)


# =========================================================
# EVALUATION
# =========================================================

pred = (
    final_mask > 0
)

intersection = np.logical_and(
    pred,
    gt
).sum()

union = np.logical_or(
    pred,
    gt
).sum()

dice = (
    2.0
    * intersection
    / (
        pred.sum()
        + gt.sum()
    )
    if (
        pred.sum()
        + gt.sum()
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

fpr = (
    fp
    / max(
        (~gt_positive).sum(),
        1
    )
)

print(
    "\n===== PURE 3D PET RESULT ====="
)

print(
    "GT positive slices:",
    int(gt_positive.sum())
)

print(
    "Predicted positive slices:",
    int(pred_positive.sum())
)

print(
    "Slice recall:",
    round(
        float(slice_recall),
        4
    )
)

print(
    "Dice:",
    round(
        float(dice),
        4
    )
)

print(
    "IoU:",
    round(
        float(iou),
        4
    )
)

print(
    "False-positive slices:",
    int(fp)
)

print(
    "False-negative slices:",
    int(fn)
)

print(
    "False-positive rate:",
    round(
        float(fpr),
        4
    )
)

print(
    "Prediction voxels:",
    int(pred.sum())
)

print(
    "GT voxels:",
    int(gt.sum())
)


# =========================================================
# SAVE
# =========================================================

prediction_img = sitk.GetImageFromArray(
    final_mask
)

prediction_img.CopyInformation(
    ct_img
)

prediction_path = (
    CASE
    / "pet_3d_case5_threshold_1p50.nii.gz"
)

sitk.WriteImage(
    prediction_img,
    str(prediction_path)
)

print(
    "\nPrediction saved:",
    prediction_path
)

print(
    "\nPure 3D PET diagnostic complete."
)