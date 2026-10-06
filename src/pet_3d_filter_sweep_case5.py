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

PET_THRESHOLD = 1.50

MIN_VOXEL_VALUES = [
    20,
    50,
    75,
    100,
    150,
    250,
    500,
    1000,
]

MIN_SLICE_VALUES = [
    1,
    2,
    3,
    4,
    5,
    6,
]


# =========================================================
# HELPERS
# =========================================================

def find_series(case_dir, text):

    matches = [
        p
        for p in case_dir.rglob("*")
        if p.is_dir()
        and text in p.name
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
        * weight_kg
        * 1000.0
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
# LOAD DATA
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

weight, dose, units = read_pet_metadata(
    pet_dir
)

print(
    "Patient weight (kg):",
    weight
)

print(
    "Injected dose (Bq):",
    dose
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
    weight,
    dose
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

gt = (
    sitk.GetArrayFromImage(
        sitk.ReadImage(
            str(GT_PATH)
        )
    ) > 0
)

body = build_body_mask(
    ct
)


# =========================================================
# 3D PET COMPONENTS
# =========================================================

print(
    f"\nBuilding PET >= {PET_THRESHOLD:.2f} components..."
)

pet_binary = (
    body
    & (
        suv >= PET_THRESHOLD
    )
).astype(
    np.uint8
)

cc_img = sitk.ConnectedComponent(
    sitk.GetImageFromArray(
        pet_binary
    )
)

cc = sitk.GetArrayFromImage(
    cc_img
)

stats = sitk.LabelShapeStatisticsImageFilter()
stats.Execute(
    cc_img
)

components = []

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

    if len(z_indices) == 0:
        continue

    z_span = (
        int(z_indices.max())
        - int(z_indices.min())
        + 1
    )

    components.append(
        (
            int(label),
            voxels,
            z_span,
            component
        )
    )

print(
    "Total 3D components:",
    len(components)
)


# =========================================================
# FILTER SWEEP
# =========================================================

results = []

for min_voxels in MIN_VOXEL_VALUES:

    for min_slices in MIN_SLICE_VALUES:

        pred = np.zeros_like(
            pet_binary,
            dtype=bool
        )

        kept_components = 0

        for (
            label,
            voxels,
            z_span,
            component
        ) in components:

            if voxels < min_voxels:
                continue

            if z_span < min_slices:
                continue

            pred[component] = True
            kept_components += 1

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
            if pred.sum() + gt.sum() > 0
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

        recall = (
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

        results.append(
            {
                "min_voxels": min_voxels,
                "min_slices": min_slices,
                "components": kept_components,
                "pred_voxels": int(
                    pred.sum()
                ),
                "dice": float(dice),
                "iou": float(iou),
                "slice_recall": float(recall),
                "fpr": float(fpr),
            }
        )


# =========================================================
# PRINT RESULTS
# =========================================================

results.sort(
    key=lambda x: (
        -x["dice"],
        -x["slice_recall"],
        x["fpr"]
    )
)

print(
    "\n===== TOP FILTER SETTINGS BY DICE ====="
)

print(
    "min_voxels  min_slices  components  "
    "pred_voxels  Dice     IoU      Recall   FPR"
)

for row in results[:15]:

    print(
        f"{row['min_voxels']:10d}  "
        f"{row['min_slices']:10d}  "
        f"{row['components']:10d}  "
        f"{row['pred_voxels']:11d}  "
        f"{row['dice']:.4f}  "
        f"{row['iou']:.4f}  "
        f"{row['slice_recall']:.4f}  "
        f"{row['fpr']:.4f}"
    )

print(
    "\n===== FILTER SWEEP COMPLETE ====="
)