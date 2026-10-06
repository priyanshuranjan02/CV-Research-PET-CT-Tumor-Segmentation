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

PET_THRESHOLDS = [
    1.0,
    1.25,
    1.5,
    1.75,
    2.0
]

MIN_COMPONENT_AREA = 20
MAX_COMPONENT_AREA = 10000


# =========================================================
# HELPERS
# =========================================================

def find_series(case_dir, text):

    matches = [
        p for p in case_dir.rglob("*")
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


def read_pet_parameters(pet_dir):

    reader = sitk.ImageSeriesReader()

    files = reader.GetGDCMSeriesFileNames(
        str(pet_dir)
    )

    if not files:
        raise RuntimeError(
            f"No PET DICOM files found in {pet_dir}"
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

                dose_value = getattr(
                    seq[0],
                    "RadionuclideTotalDose",
                    None
                )

                if dose_value is not None:
                    dose = float(
                        dose_value
                    )

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

    return (
        weight,
        dose,
        units
    )


def calculate_suv(
    activity_bqml,
    patient_weight_kg,
    injected_dose_bq
):

    weight_g = (
        patient_weight_kg
        * 1000.0
    )

    return (
        activity_bqml
        * weight_g
        / injected_dose_bq
    )


def best_component_metrics(
    component_mask,
    gt_slice
):

    area = int(
        component_mask.sum()
    )

    intersection = int(
        np.logical_and(
            component_mask,
            gt_slice
        ).sum()
    )

    denominator = (
        area
        + int(gt_slice.sum())
    )

    dice = (
        2.0 * intersection
        / denominator
        if denominator > 0
        else 0.0
    )

    return (
        area,
        intersection,
        dice
    )


# =========================================================
# LOAD CT / PET / GT
# =========================================================

print("Loading CT...")

ct_dir = find_series(
    CASE,
    "GK p.v.3"
)

ct_img = read_series(
    ct_dir
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
    PATIENT_WEIGHT_KG,
    INJECTED_DOSE_BQ,
    PET_UNITS
) = read_pet_parameters(
    pet_dir
)

print(
    "Patient weight (kg):",
    PATIENT_WEIGHT_KG
)

print(
    "Injected dose (Bq):",
    INJECTED_DOSE_BQ
)

print(
    "PET units:",
    PET_UNITS
)

print(
    "CT:",
    ct_img.GetSize(),
    ct_img.GetSpacing()
)

print(
    "PET:",
    pet_img.GetSize(),
    pet_img.GetSpacing()
)

pet_stored = sitk.GetArrayFromImage(
    pet_img
).astype(
    np.float64
)

suv_native = calculate_suv(
    pet_stored,
    PATIENT_WEIGHT_KG,
    INJECTED_DOSE_BQ
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

# Use the existing CT body definition from V4/V5,
# so the PET-only audit remains anatomically constrained.
ct = sitk.GetArrayFromImage(
    ct_img
).astype(
    np.float32
)

body = np.zeros_like(
    ct,
    dtype=bool
)

body_kernel = np.ones(
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
        body_kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        contour = max(
            contours,
            key=cv2.contourArea
        )

        cv2.drawContours(
            body[z].astype(np.uint8),
            [contour],
            -1,
            1,
            -1
        )

# Rebuild body correctly because astype() above creates a copy.
body = np.zeros_like(
    ct,
    dtype=np.uint8
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
        body_kernel,
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

body = body.astype(bool)

gt_slices = np.where(
    np.any(
        gt,
        axis=(1, 2)
    )
)[0]


# =========================================================
# PET-ONLY BODY COMPONENT AUDIT
# =========================================================

print(
    "\n===== PET-ONLY BODY COMPONENT AUDIT ====="
)

for z in gt_slices:

    gt_slice = gt[z]

    print(
        f"\nSlice {z}: "
        f"GT voxels={int(gt_slice.sum())}"
    )

    pet_body_values = suv[z][body[z]]

    print(
        f"  Body PET median="
        f"{np.median(pet_body_values):.4f}, "
        f"max="
        f"{np.max(pet_body_values):.4f}"
    )

    for threshold in PET_THRESHOLDS:

        pet_binary = (
            body[z]
            & (
                suv[z]
                >= threshold
            )
        ).astype(
            np.uint8
        )

        (
            num_labels,
            labels,
            stats,
            centroids
        ) = cv2.connectedComponentsWithStats(
            pet_binary,
            connectivity=8
        )

        components = []

        for label in range(
            1,
            num_labels
        ):

            component = (
                labels == label
            )

            area = int(
                component.sum()
            )

            if (
                area < MIN_COMPONENT_AREA
                or area > MAX_COMPONENT_AREA
            ):
                continue

            comp_area, overlap, dice = (
                best_component_metrics(
                    component,
                    gt_slice
                )
            )

            component_suv = suv[z][
                component
            ]

            components.append(
                {
                    "area": comp_area,
                    "overlap": overlap,
                    "dice": dice,
                    "median_suv": float(
                        np.median(
                            component_suv
                        )
                    ),
                    "max_suv": float(
                        np.max(
                            component_suv
                        )
                    )
                }
            )

        components.sort(
            key=lambda x: x["dice"],
            reverse=True
        )

        if not components:

            print(
                f"  PET >= {threshold:.2f}: "
                "no components"
            )

            continue

        best = components[0]

        print(
            f"  PET >= {threshold:.2f}: "
            f"components={len(components)}, "
            f"best area={best['area']}, "
            f"overlap={best['overlap']}, "
            f"Dice={best['dice']:.4f}, "
            f"median SUV={best['median_suv']:.4f}, "
            f"max SUV={best['max_suv']:.4f}"
        )

print(
    "\nPET-only audit complete."
)