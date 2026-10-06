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
    1.50,
    1.75,
    2.00
]

MIN_COMPONENT_VOXELS = 20


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
# LOAD CT / PET / GT
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

ct = sitk.GetArrayFromImage(
    ct_img
).astype(
    np.float32
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
# 3D PET COMPONENT AUDIT
# =========================================================

print(
    "\n===== 3D PET COMPONENT AUDIT ====="
)

for threshold in PET_THRESHOLDS:

    pet_binary = (
        body
        & (
            suv
            >= threshold
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

    all_components = []
    gt_components = []

    for label in stats.GetLabels():

        component = (
            cc == label
        )

        voxel_count = int(
            component.sum()
        )

        if voxel_count < MIN_COMPONENT_VOXELS:
            continue

        z_indices = np.where(
            component
        )[0]

        if len(z_indices) == 0:
            continue

        z_start = int(
            z_indices.min()
        )

        z_end = int(
            z_indices.max()
        )

        z_span = (
            z_end
            - z_start
            + 1
        )

        overlap = int(
            np.logical_and(
                component,
                gt
            ).sum()
        )

        gt_voxels = int(
            gt.sum()
        )

        dice = (
            2.0 * overlap
            / (
                voxel_count
                + gt_voxels
            )
            if (
                voxel_count
                + gt_voxels
            ) > 0
            else 0.0
        )

        item = {
            "label": int(label),
            "voxels": voxel_count,
            "z_start": z_start,
            "z_end": z_end,
            "z_span": z_span,
            "gt_overlap": overlap,
            "dice": dice
        }

        all_components.append(
            item
        )

        if overlap > 0:
            gt_components.append(
                item
            )

    all_components.sort(
        key=lambda x: x["voxels"],
        reverse=True
    )

    gt_components.sort(
        key=lambda x: x["dice"],
        reverse=True
    )

    positive_component_voxels = sum(
        x["voxels"]
        for x in gt_components
    )

    print(
        f"\nPET >= {threshold:.2f}: "
        f"components >= {MIN_COMPONENT_VOXELS} voxels="
        f"{len(all_components)}, "
        f"GT-overlapping components="
        f"{len(gt_components)}, "
        f"voxels in GT-overlapping components="
        f"{positive_component_voxels}"
    )

    if gt_components:

        best = gt_components[0]

        print(
            "  Best GT-overlapping component: "
            f"voxels={best['voxels']}, "
            f"z={best['z_start']}-{best['z_end']}, "
            f"z-span={best['z_span']}, "
            f"GT overlap={best['gt_overlap']}, "
            f"Dice={best['dice']:.4f}"
        )

        print(
            "  GT-overlapping components:"
        )

        for item in gt_components[:10]:

            print(
                f"    label={item['label']}, "
                f"voxels={item['voxels']}, "
                f"z={item['z_start']}-{item['z_end']}, "
                f"span={item['z_span']}, "
                f"overlap={item['gt_overlap']}, "
                f"Dice={item['dice']:.4f}"
            )

    else:

        print(
            "  No component overlaps the GT."
        )


# =========================================================
# GT SLICE COVERAGE
# =========================================================

print(
    "\n===== PET THRESHOLD GT SLICE COVERAGE ====="
)

gt_slices = np.where(
    np.any(
        gt,
        axis=(1, 2)
    )
)[0]

for threshold in PET_THRESHOLDS:

    pet_binary = (
        body
        & (
            suv
            >= threshold
        )
    )

    pred_slices = np.any(
        pet_binary,
        axis=(1, 2)
    )

    covered = [
        int(z)
        for z in gt_slices
        if pred_slices[z]
    ]

    print(
        f"PET >= {threshold:.2f}: "
        f"GT slices with any PET threshold coverage="
        f"{covered} "
        f"({len(covered)}/{len(gt_slices)})"
    )

print(
    "\n3D PET audit complete."
)