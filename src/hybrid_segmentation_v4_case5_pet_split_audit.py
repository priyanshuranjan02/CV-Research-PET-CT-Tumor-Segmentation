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
# HELPERS
# =========================================================

def read_pet_parameters(pet_dir):
    reader = sitk.ImageSeriesReader()

    files = reader.GetGDCMSeriesFileNames(
        str(pet_dir)
    )

    if not files:
        raise RuntimeError(
            f"No PET DICOM files found in {pet_dir}"
        )

    slopes = []
    intercepts = []

    weight = None
    dose = None
    units = None

    for path in files:

        ds = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True
        )

        slopes.append(
            float(
                getattr(
                    ds,
                    "RescaleSlope",
                    1.0
                )
            )
        )

        intercepts.append(
            float(
                getattr(
                    ds,
                    "RescaleIntercept",
                    0.0
                )
            )
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
                    dose = float(dose_value)

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
        np.asarray(
            slopes,
            dtype=np.float64
        ),
        np.asarray(
            intercepts,
            dtype=np.float64
        ),
        weight,
        dose,
        units
    )


def find_series(case_dir, text):

    matches = [
        p for p in case_dir.rglob("*")
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


# =========================================================
# LOAD DATA
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
    PET_SLOPES,
    PET_INTERCEPTS,
    PATIENT_WEIGHT_KG,
    INJECTED_DOSE_BQ,
    PET_UNITS
) = read_pet_parameters(
    pet_dir
)

print(
    "PET slices:",
    len(PET_SLOPES)
)

print(
    "PET slope range:",
    float(PET_SLOPES.min()),
    "to",
    float(PET_SLOPES.max())
)

print(
    "PET intercept range:",
    float(PET_INTERCEPTS.min()),
    "to",
    float(PET_INTERCEPTS.max())
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


# =========================================================
# PET SUV
# =========================================================

pet_stored = sitk.GetArrayFromImage(
    pet_img
).astype(
    np.float64
)

print(
    "Native PET stored shape:",
    pet_stored.shape
)

if pet_stored.shape[0] != len(
    PET_SLOPES
):
    raise RuntimeError(
        "Native PET slice count does not "
        "match DICOM calibration count."
    )

# SimpleITK has already applied the PET
# DICOM RescaleSlope/RescaleIntercept.
suv_native = calculate_suv(
    pet_stored,
    PATIENT_WEIGHT_KG,
    INJECTED_DOSE_BQ
)

print(
    "Native SUV max:",
    round(
        float(
            np.nanmax(
                suv_native
            )
        ),
        4
    )
)

suv_native_img = sitk.GetImageFromArray(
    suv_native.astype(
        np.float32
    )
)

suv_native_img.CopyInformation(
    pet_img
)


# =========================================================
# RESAMPLE SUV TO CT SPACE
# =========================================================

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

ct = sitk.GetArrayFromImage(
    ct_img
).astype(
    np.float32
)

print(
    "Resampled SUV:",
    suv_ct_img.GetSize()
)

print(
    "SUV max:",
    round(
        float(
            np.nanmax(
                suv
            )
        ),
        4
    )
)


# =========================================================
# BODY ROI
# =========================================================

print(
    "\nBuilding body ROI..."
)

body = build_body_mask(
    ct
)

print(
    "Body voxels:",
    int(
        body.sum()
    )
)


# =========================================================
# GROUND TRUTH
# =========================================================

gt_img = sitk.ReadImage(
    str(GT_PATH)
)

gt = (
    sitk.GetArrayFromImage(
        gt_img
    ) > 0
)

gt_positive_slices = np.where(
    np.any(
        gt,
        axis=(1, 2)
    )
)[0]


# =========================================================
# PET COMPONENT SPLIT AUDIT
# =========================================================
#
# Diagnostic only.
#
# For each GT-positive CT slice:
# 1. Recreate V4's CT candidate mask.
# 2. Find CT contours larger than MAX_CANDIDATE_AREA.
# 3. Threshold PET SUV inside each large contour.
# 4. Compute connected components.
# 5. Report the best component Dice against GT.
#
# This script DOES NOT change V4 and DOES NOT
# write a prediction file.
# =========================================================

MAX_CANDIDATE_AREA = 10000

opening_kernel = np.ones(
    (5, 5),
    np.uint8
)

ct_display = (
    np.clip(
        ct,
        -1000,
        1000
    ) + 1000
) / 2000 * 255

ct_display = ct_display.astype(
    np.uint8
)

print(
    "\n===== PET COMPONENT SPLIT AUDIT ====="
)

for z in gt_positive_slices:

    gt_slice = gt[z]

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
        body[z],
        opened,
        0
    ).astype(
        np.uint8
    )

    opened[:5, :] = 0
    opened[-5:, :] = 0
    opened[:, :5] = 0
    opened[:, -5:] = 0

    contours, _ = cv2.findContours(
        opened,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    oversized_contours = 0

    for contour in contours:

        contour_mask = np.zeros(
            ct.shape[1:],
            dtype=np.uint8
        )

        cv2.drawContours(
            contour_mask,
            [contour],
            -1,
            1,
            -1
        )

        contour_mask = (
            contour_mask > 0
        )

        contour_area = int(
            contour_mask.sum()
        )

        if contour_area <= MAX_CANDIDATE_AREA:
            continue

        oversized_contours += 1

        print(
            f"\nSlice {z}: "
            f"large CT contour area={contour_area}"
        )

        for threshold in [
            1.0,
            1.25,
            1.5,
            1.75,
            2.0
        ]:

            pet_mask = (
                contour_mask
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
                pet_mask,
                connectivity=8
            )

            best_dice = 0.0
            best_overlap = 0
            best_area = 0

            for label in range(
                1,
                num_labels
            ):

                component = (
                    labels == label
                )

                area_component = int(
                    component.sum()
                )

                if area_component < 20:
                    continue

                intersection = np.logical_and(
                    component,
                    gt_slice
                ).sum()

                denominator = (
                    area_component
                    + int(
                        gt_slice.sum()
                    )
                )

                dice_component = (
                    2.0
                    * intersection
                    / denominator
                    if denominator > 0
                    else 0.0
                )

                if dice_component > best_dice:

                    best_dice = (
                        dice_component
                    )

                    best_overlap = int(
                        intersection
                    )

                    best_area = (
                        area_component
                    )

            print(
                f"  PET >= {threshold:.2f}: "
                f"best component area={best_area}, "
                f"GT overlap={best_overlap}, "
                f"Dice={best_dice:.4f}"
            )

    if oversized_contours == 0:

        print(
            f"\nSlice {z}: "
            "no oversized CT contour"
        )

print(
    "\nAudit complete."
)
