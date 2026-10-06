from pathlib import Path
import cv2
import numpy as np
import pandas as pd
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
        units,
        files
    )


# =========================================================
# PROVISIONAL DEVELOPMENT PARAMETERS
# =========================================================

MIN_CANDIDATE_AREA = 100
MAX_CANDIDATE_AREA = 10000

LOCAL_RING_KERNEL = np.ones((31, 31), np.uint8)

MIN_LOCAL_RATIO = 1.6
MIN_LOCAL_DIFFERENCE = 0.5
MIN_CANDIDATE_SUV = 1.0

MIN_3D_VOXELS = 75
MIN_3D_SLICES = 2


# =========================================================
# HELPERS
# =========================================================

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

    for z in range(ct.shape[0]):

        binary = (
            ct[z] > -900
        ).astype(np.uint8) * 255

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


def normalize_ct(ct):
    clipped = np.clip(
        ct,
        -1000,
        1000
    )

    return (
        (clipped + 1000)
        / 2000
        * 255
    ).astype(np.uint8)


def calculate_suv(
    activity_bqml,
    patient_weight_kg,
    injected_dose_bq
):

    weight_g = (
        patient_weight_kg
        * 1000.0
    )

    suvbw = (
        activity_bqml
        * weight_g
        / injected_dose_bq
    )

    return suvbw


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
    PET_UNITS,
    PET_FILES
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
# PET SUV CALCULATION IN NATIVE PET SPACE
# =========================================================

pet_stored = sitk.GetArrayFromImage(
    pet_img
).astype(np.float64)

print(
    "Native PET stored shape:",
    pet_stored.shape
)

if pet_stored.shape[0] != len(PET_SLOPES):
    raise RuntimeError(
        "Native PET slice count does not match "
        "DICOM calibration count: "
        f"{pet_stored.shape[0]} vs "
        f"{len(PET_SLOPES)}"
    )

suv_native = calculate_suv(
    pet_stored,
    PATIENT_WEIGHT_KG,
    INJECTED_DOSE_BQ
)

print(
    "Native SUV max:",
    round(
        float(np.nanmax(suv_native)),
        4
    )
)

# ---------------------------------------------------------
# Create native PET-space SUV image
# ---------------------------------------------------------

suv_native_img = sitk.GetImageFromArray(
    suv_native.astype(np.float32)
)

suv_native_img.CopyInformation(
    pet_img
)

# ---------------------------------------------------------
# PET SUV -> CT physical space
# ---------------------------------------------------------

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
).astype(np.float64)

ct = sitk.GetArrayFromImage(
    ct_img
).astype(np.float32)

print(
    "Resampled SUV:",
    suv_ct_img.GetSize()
)

print(
    "SUV max:",
    round(
        float(np.nanmax(suv)),
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
    int(body.sum())
)


# =========================================================
# CANDIDATE GENERATION
# =========================================================

ct_display = normalize_ct(
    ct
)

opening_kernel = np.ones(
    (5, 5),
    np.uint8
)

candidate_rows = []

candidate_volume = np.zeros_like(
    ct,
    dtype=np.uint8
)


print(
    "\nGenerating local-PET candidates..."
)


for z in range(
    ct.shape[0]
):

    body_slice = body[z]

    if not np.any(
        body_slice
    ):
        continue

    # -----------------------------------------------------
    # CT candidate generation
    # -----------------------------------------------------

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
        body_slice,
        opened,
        0
    ).astype(np.uint8)

    opened[:5, :] = 0
    opened[-5:, :] = 0
    opened[:, :5] = 0
    opened[:, -5:] = 0

    contours, _ = cv2.findContours(
        opened,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # -----------------------------------------------------
    # Candidate features
    # -----------------------------------------------------

    for contour in contours:

        area = float(
            cv2.contourArea(
                contour
            )
        )

        if (
            area < MIN_CANDIDATE_AREA
            or area > MAX_CANDIDATE_AREA
        ):
            continue

        candidate_mask = np.zeros(
            ct.shape[1:],
            dtype=np.uint8
        )

        cv2.drawContours(
            candidate_mask,
            [contour],
            -1,
            1,
            -1
        )

        candidate_pixels = (
            candidate_mask > 0
        )

        if not np.any(
            candidate_pixels
        ):
            continue

        # -------------------------------------------------
        # Local PET neighborhood
        # -------------------------------------------------

        dilated = cv2.dilate(
            candidate_mask,
            LOCAL_RING_KERNEL,
            iterations=1
        )

        ring = (
            (dilated > 0)
            & (~candidate_pixels)
            & body_slice
        )

        if not np.any(ring):
            continue

        candidate_suv = suv[z][
            candidate_pixels
        ]

        ring_suv = suv[z][
            ring
        ]

        candidate_suv = (
            candidate_suv[
                np.isfinite(
                    candidate_suv
                )
            ]
        )

        ring_suv = (
            ring_suv[
                np.isfinite(
                    ring_suv
                )
            ]
        )

        if (
            len(candidate_suv) == 0
            or len(ring_suv) == 0
        ):
            continue

        candidate_median = float(
            np.median(
                candidate_suv
            )
        )

        ring_median = float(
            np.median(
                ring_suv
            )
        )

        local_difference = (
            candidate_median
            - ring_median
        )

        local_ratio = (
            candidate_median
            / (ring_median + 1e-6)
        )

        # -------------------------------------------------
        # CT texture
        # -------------------------------------------------

        lap = cv2.Laplacian(
            ct_display[z],
            cv2.CV_32F
        )

        texture = float(
            np.var(
                lap[
                    candidate_pixels
                ]
            )
        )

        # -------------------------------------------------
        # Shape
        # -------------------------------------------------

        perimeter = float(
            cv2.arcLength(
                contour,
                True
            )
        )

        circularity = (
            4.0
            * np.pi
            * area
            / (
                perimeter
                * perimeter
                + 1e-6
            )
        )

        # -------------------------------------------------
        # Candidate filters
        # -------------------------------------------------

        passes_pet = (
            local_ratio
            >= MIN_LOCAL_RATIO
        )

        passes_difference = (
            local_difference
            >= MIN_LOCAL_DIFFERENCE
        )

        passes_suv = (
            candidate_median
            >= MIN_CANDIDATE_SUV
        )

        accepted = (
            passes_pet
            and passes_difference
            and passes_suv
        )

        candidate_rows.append(
            {
                "slice": z,
                "area": area,
                "candidate_suv":
                    candidate_median,
                "background_suv":
                    ring_median,
                "local_difference":
                    local_difference,
                "local_ratio":
                    local_ratio,
                "texture":
                    texture,
                "circularity":
                    float(
                        np.clip(
                            circularity,
                            0.0,
                            1.0
                        )
                    ),
                "accepted":
                    int(accepted),
                "mask":
                    candidate_mask
            }
        )


df = pd.DataFrame(
    candidate_rows
)

print(
    "\nTotal candidates:",
    len(df)
)

print(
    "Accepted candidates:",
    int(
        df["accepted"].sum()
    )
)


# =========================================================
# BUILD 3D CANDIDATE MASK
# =========================================================

accepted_rows = df[
    df["accepted"] == 1
]

for _, row in accepted_rows.iterrows():

    z = int(
        row["slice"]
    )

    candidate_volume[z][
        row["mask"] > 0
    ] = 1


print(
    "Accepted candidate voxels:",
    int(candidate_volume.sum())
)

# =========================================================
# AUDIT: ACCEPTED CANDIDATE VS GROUND TRUTH
# =========================================================

gt_audit_img = sitk.ReadImage(
    str(GT_PATH)
)

gt_audit = (
    sitk.GetArrayFromImage(
        gt_audit_img
    ) > 0
)

candidate_bool = (
    candidate_volume > 0
)

candidate_gt_overlap = np.logical_and(
    candidate_bool,
    gt_audit
).sum()

gt_positive_slices = np.where(
    np.any(gt_audit, axis=(1, 2))
)[0]

candidate_positive_slices = np.where(
    np.any(candidate_bool, axis=(1, 2))
)[0]

candidate_slice_overlap = len(
    set(gt_positive_slices)
    & set(candidate_positive_slices)
)

print(
    "Accepted-candidate overlap with GT voxels:",
    int(candidate_gt_overlap)
)

print(
    "GT-positive slices:",
    gt_positive_slices.tolist()
)

print(
    "Accepted-candidate positive slices:",
    candidate_positive_slices.tolist()
)

print(
    "GT slices containing accepted candidates:",
    candidate_slice_overlap
)

# =========================================================
# 3D CONNECTED COMPONENTS
# =========================================================

candidate_img = sitk.GetImageFromArray(
    candidate_volume
)

cc_img = sitk.ConnectedComponent(
    candidate_img
)

cc = sitk.GetArrayFromImage(
    cc_img
)

stats = sitk.LabelShapeStatisticsImageFilter()

stats.Execute(
    cc_img
)

# =========================================================
# AUDIT: WHICH 3D COMPONENTS CONTAIN GT OVERLAP?
# =========================================================

overlap_components = []

for label in stats.GetLabels():

    component = (cc == label)

    overlap = np.logical_and(
        component,
        gt_audit
    ).sum()

    if overlap > 0:

        z_indices = np.where(
            component
        )[0]

        voxel_count = int(
            component.sum()
        )

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

        overlap_components.append(
            {
                "label": int(label),
                "component_voxels": voxel_count,
                "gt_overlap_voxels": int(overlap),
                "z_start": z_start,
                "z_end": z_end,
                "z_span": z_span,
                "passes_voxel_filter":
                    voxel_count >= MIN_3D_VOXELS,
                "passes_slice_filter":
                    z_span >= MIN_3D_SLICES
            }
        )

print("\n===== GT-OVERLAPPING 3D COMPONENTS =====")

for item in overlap_components:
    print(item)

print(
    "Number of GT-overlapping components:",
    len(overlap_components)
)

print(
    "3D candidate components:",
    stats.GetNumberOfLabels()
)


# =========================================================
# KEEP SPATIALLY COHERENT COMPONENTS
# =========================================================

final_mask = np.zeros_like(
    candidate_volume,
    dtype=np.uint8
)

kept_components = []

for label in stats.GetLabels():

    component = (
        cc == label
    )

    voxel_count = int(
        component.sum()
    )

    if voxel_count < MIN_3D_VOXELS:
        continue

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

    if z_span < MIN_3D_SLICES:
        continue

    final_mask[
        component
    ] = 1

    kept_components.append(
        {
            "label":
                int(label),
            "voxels":
                voxel_count,
            "z_start":
                int(z_indices.min()),
            "z_end":
                int(z_indices.max()),
            "z_span":
                z_span
        }
    )


print(
    "Kept 3D components:",
    len(kept_components)
)

print(
    "Final prediction voxels:",
    int(final_mask.sum())
)


# =========================================================
# SAVE RESULTS
# =========================================================

prediction_img = sitk.GetImageFromArray(
    final_mask
)

prediction_img.CopyInformation(
    ct_img
)

prediction_path = (
    CASE /
    "hybrid_segmentation_v4_balanced.nii.gz"
)

sitk.WriteImage(
    prediction_img,
    str(prediction_path)
)

# Candidate CSV
csv_path = (
    CASE /
    "hybrid_v4_balanced_candidate_scores.csv"
)

df.drop(
    columns=["mask"]
).to_csv(
    csv_path,
    index=False
)

print(
    "\nPrediction saved:",
    prediction_path
)

print(
    "Candidate table saved:",
    csv_path
)


# =========================================================
# DEVELOPMENT EVALUATION
# =========================================================

if GT_PATH.exists():

    gt_img = sitk.ReadImage(
        str(GT_PATH)
    )

    gt = (
        sitk.GetArrayFromImage(
            gt_img
        ) > 0
    )

    pred = final_mask > 0

    intersection = np.logical_and(
        pred,
        gt
    ).sum()

    union = np.logical_or(
        pred,
        gt
    ).sum()

    dice = (
        2.0 * intersection
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

    print(
        "\n===== HYBRID V4 DEVELOPMENT RESULT ====="
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
            float(
                tp /
                max(
                    gt_positive.sum(),
                    1
                )
            ),
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
            float(
                fp /
                max(
                    (~gt_positive).sum(),
                    1
                )
            ),
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


print(
    "\nV4 complete."
)
