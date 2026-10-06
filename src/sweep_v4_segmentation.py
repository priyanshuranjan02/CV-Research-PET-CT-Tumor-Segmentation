from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom
from itertools import product

# =========================================================
# FIXED V4 PARAMETERS
# =========================================================

MIN_CANDIDATE_AREA = 100
MAX_CANDIDATE_AREA = 10000

LOCAL_RING_KERNEL = np.ones((31, 31), np.uint8)

MIN_3D_VOXELS = 75
MIN_3D_SLICES = 2

# =========================================================
# THRESHOLDS TO SWEEP
# =========================================================

SUV_THRESHOLDS = [
    0.50,
    0.75,
    1.00,
    1.25,
]

DIFF_THRESHOLDS = [
    0.25,
    0.40,
    0.50,
]

RATIO_THRESHOLDS = [
    1.30,
    1.60,
    1.90,
]

# =========================================================
# DEVELOPMENT CASES
# =========================================================

CASES = [
    {
        "name": "PETCT_0168f65af8",
        "case": Path(
            "development_cases/PETCT_0168f65af8/"
            "manifest-1789661648760/FDG-PET-CT-Lesions/"
            "PETCT_0168f65af8"
        ),
    },
    {
        "name": "PETCT_04606080a0",
        "case": Path(
            "development_cases/PETCT_04606080a0/"
            "FDG-PET-CT-Lesions/"
            "PETCT_04606080a0"
        ),
    },
    {
        "name": "PETCT_04ab5c61c9",
        "case": Path(
            "development_cases/PETCT_04ab5c61c9/"
            "manifest-1789918980481/FDG-PET-CT-Lesions/"
            "PETCT_04ab5c61c9"
        ),
    },
    {
        "name": "PETCT_0b57b247b6",
        "case": Path(
            "development_cases/PETCT_0b57b247b6/"
            "manifest-1789920519449/FDG-PET-CT-Lesions/"
            "PETCT_0b57b247b6"
        ),
    },
    {
        "name": "PETCT_11afab3485",
        "case": Path(
            "development_cases/PETCT_11afab3485/"
            "manifest-1789921915902/FDG-PET-CT-Lesions/"
            "PETCT_11afab3485"
        ),
    },
    {
        "name": "PETCT_185da4c8b6",
        "case": Path(
            "development_cases/PETCT_185da4c8b6/"
            "manifest-1789923055836/FDG-PET-CT-Lesions/"
            "PETCT_185da4c8b6"
        ),
    },
]

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
            f"{case_dir}: expected exactly one directory "
            f"containing '{text}', found {len(matches)}"
        )

    return matches[0]


def read_series(path):
    reader = sitk.ImageSeriesReader()

    files = reader.GetGDCMSeriesFileNames(str(path))

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
            value = getattr(ds, "PatientWeight", None)

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
            units = getattr(ds, "Units", None)

    if weight is None:
        raise RuntimeError(
            f"PatientWeight not found in {pet_dir}"
        )

    if dose is None:
        raise RuntimeError(
            f"Injected dose not found in {pet_dir}"
        )

    return weight, dose, units


def calculate_suv(
    activity_bqml,
    patient_weight_kg,
    injected_dose_bq
):
    weight_g = patient_weight_kg * 1000.0

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


# =========================================================
# PREPARE ONE CASE
# =========================================================

def prepare_case(case_name, case_dir):

    print()
    print("=" * 70)
    print("Preparing:", case_name)
    print("=" * 70)

    # Find the GT mask by walking upward from the DICOM series.
    gt_path = None

    for parent in [case_dir] + list(case_dir.parents):
        candidate = parent / "tumor_mask_ct_visible.nii.gz"
        if candidate.exists():
            gt_path = candidate
            break

    if gt_path is None:
        raise FileNotFoundError(
            f"Could not find tumor_mask_ct_visible.nii.gz "
            f"for {case_name} starting from {case_dir}"
        )

    print("GT:", gt_path)

    # -----------------------------------------------------
    # Load CT
    # -----------------------------------------------------

    print("Loading CT...")
    ct_dir = find_series(
        case_dir,
        "GK p.v.3"
    )

    ct_img = read_series(ct_dir)

    # -----------------------------------------------------
    # Load PET
    # -----------------------------------------------------

    print("Loading PET...")
    pet_dir = find_series(
        case_dir,
        "PET corr."
    )

    pet_img = read_series(pet_dir)

    weight, dose, units = read_pet_parameters(
        pet_dir
    )

    print("CT size:", ct_img.GetSize())
    print("PET size:", pet_img.GetSize())
    print("Patient weight:", weight)
    print("Injected dose:", dose)
    print("PET units:", units)

    # -----------------------------------------------------
    # PET -> SUV
    # -----------------------------------------------------

    pet_stored = sitk.GetArrayFromImage(
        pet_img
    ).astype(np.float64)

    suv_native = calculate_suv(
        pet_stored,
        weight,
        dose
    )

    suv_native_img = sitk.GetImageFromArray(
        suv_native.astype(np.float32)
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
    ).astype(np.float64)

    ct = sitk.GetArrayFromImage(
        ct_img
    ).astype(np.float32)

    print(
        "SUV max:",
        round(float(np.nanmax(suv)), 4)
    )

    # -----------------------------------------------------
    # Body ROI
    # -----------------------------------------------------

    print("Building body ROI...")
    body = build_body_mask(ct)

    ct_display = normalize_ct(ct)

    opening_kernel = np.ones(
        (5, 5),
        np.uint8
    )

    # -----------------------------------------------------
    # Candidate label volume
    #
    # Every accepted candidate is represented by a unique
    # integer label. This lets us reuse the same candidates
    # for all threshold combinations without storing large
    # 2D masks for every candidate.
    # -----------------------------------------------------

    candidate_label_volume = np.zeros(
        ct.shape,
        dtype=np.int32
    )

    candidate_rows = []

    candidate_id = 1

    print("Generating candidates...")

    for z in range(ct.shape[0]):

        body_slice = body[z]

        if not np.any(body_slice):
            continue

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

        lap = cv2.Laplacian(
            ct_display[z],
            cv2.CV_32F
        )

        for contour in contours:

            area = float(
                cv2.contourArea(contour)
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

            if not np.any(candidate_pixels):
                continue

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

            ring_suv = suv[z][ring]

            candidate_suv = candidate_suv[
                np.isfinite(candidate_suv)
            ]

            ring_suv = ring_suv[
                np.isfinite(ring_suv)
            ]

            if (
                len(candidate_suv) == 0
                or len(ring_suv) == 0
            ):
                continue

            candidate_median = float(
                np.median(candidate_suv)
            )

            ring_median = float(
                np.median(ring_suv)
            )

            local_difference = (
                candidate_median
                - ring_median
            )

            local_ratio = (
                candidate_median
                / (ring_median + 1e-6)
            )

            texture = float(
                np.var(
                    lap[candidate_pixels]
                )
            )

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
                    perimeter * perimeter
                    + 1e-6
                )
            )

            # Assign unique label to this candidate.
            candidate_label_volume[z][
                candidate_mask > 0
            ] = candidate_id

            candidate_rows.append(
                {
                    "id": candidate_id,
                    "slice": z,
                    "area": area,
                    "candidate_suv": candidate_median,
                    "background_suv": ring_median,
                    "local_difference":
                        local_difference,
                    "local_ratio":
                        local_ratio,
                    "texture": texture,
                    "circularity":
                        float(
                            np.clip(
                                circularity,
                                0.0,
                                1.0
                            )
                        ),
                }
            )

            candidate_id += 1

    df = pd.DataFrame(candidate_rows)

    # -----------------------------------------------------
    # IMPORTANT:
    # Use the exact candidate feature values generated by
    # the original V4 implementation.
    #
    # The duplicated PET-feature calculation in this sweep
    # produced values inconsistent with the authoritative
    # V4 candidate CSVs. Candidate geometry/count is already
    # identical, so reuse the original V4 feature table.
    # -----------------------------------------------------

    reference_csvs = {
        "PETCT_0168f65af8":
            Path(
                "development_cases/PETCT_0168f65af8/"
                "manifest-1789661648760/FDG-PET-CT-Lesions/"
                "PETCT_0168f65af8/"
                "hybrid_v4_balanced_candidate_scores.csv"
            ),
        "PETCT_04606080a0":
            Path(
                "development_cases/PETCT_04606080a0/"
                "FDG-PET-CT-Lesions/"
                "PETCT_04606080a0/"
                "hybrid_v4_balanced_candidate_scores.csv"
            ),
        "PETCT_04ab5c61c9":
            Path(
                "development_cases/PETCT_04ab5c61c9/"
                "manifest-1789918980481/FDG-PET-CT-Lesions/"
                "PETCT_04ab5c61c9/"
                "hybrid_v4_balanced_candidate_scores.csv"
            ),
        "PETCT_0b57b247b6":
            Path(
                "development_cases/PETCT_0b57b247b6/"
                "manifest-1789920519449/FDG-PET-CT-Lesions/"
                "PETCT_0b57b247b6/"
                "hybrid_v4_balanced_candidate_scores.csv"
            ),
        "PETCT_11afab3485":
            Path(
                "development_cases/PETCT_11afab3485/"
                "manifest-1789921915902/FDG-PET-CT-Lesions/"
                "PETCT_11afab3485/"
                "hybrid_v4_balanced_candidate_scores.csv"
            ),
        "PETCT_185da4c8b6":
            Path(
                "development_cases/PETCT_185da4c8b6/"
                "manifest-1789923055836/FDG-PET-CT-Lesions/"
                "PETCT_185da4c8b6/"
                "hybrid_v4_balanced_candidate_scores.csv"
            ),
    }

    ref_path = reference_csvs[case_name]

    if not ref_path.exists():
        raise FileNotFoundError(
            f"Reference V4 candidate table not found: {ref_path}"
        )

    ref = pd.read_csv(ref_path)

    if len(ref) != len(df):
        raise RuntimeError(
            f"{case_name}: candidate count mismatch. "
            f"Sweep={len(df)}, V4={len(ref)}"
        )

    # Verify that the candidate generation order/geometry matches.
    if not np.array_equal(
        df["slice"].to_numpy(),
        ref["slice"].to_numpy()
    ):
        raise RuntimeError(
            f"{case_name}: candidate slice ordering differs "
            "from original V4."
        )

    if not np.allclose(
        df["area"].to_numpy(),
        ref["area"].to_numpy(),
        rtol=0.0,
        atol=1e-4
    ):
        raise RuntimeError(
            f"{case_name}: candidate areas differ "
            "from original V4."
        )

    for col in [
        "candidate_suv",
        "background_suv",
        "local_difference",
        "local_ratio",
        "texture",
        "circularity",
    ]:
        df[col] = ref[col].to_numpy()

    print(
        "Total candidates:",
        len(df)
    )

    print(
        "Reference V4 feature table:",
        ref_path
    )

    print(
        "Reference V4 accepted:",
        int(
            ref["accepted"].sum()
        )
    )

    return {
        "name": case_name,
        "ct_img": ct_img,
        "candidate_labels":
            candidate_label_volume,
        "candidates": df,
        "gt": (
            sitk.GetArrayFromImage(
                sitk.ReadImage(str(gt_path))
            ) > 0
        ),
    }


# =========================================================
# FINAL 3D FILTERING + METRICS
# =========================================================

def evaluate_selection(
    candidate_labels,
    candidate_df,
    gt,
    suv_threshold,
    diff_threshold,
    ratio_threshold
):

    selected = (
        (candidate_df["candidate_suv"].to_numpy() >= suv_threshold)
        & (
            candidate_df["local_difference"].to_numpy()
            >= diff_threshold
        )
        & (
            candidate_df["local_ratio"].to_numpy()
            >= ratio_threshold
        )
    )

    # TEMPORARY DEBUG
    if (
        suv_threshold == SUV_THRESHOLDS[0]
        and diff_threshold == DIFF_THRESHOLDS[0]
        and ratio_threshold == RATIO_THRESHOLDS[0]
    ):
        print("\n--- DEBUG CANDIDATE SELECTION ---")
        print("Candidates:", len(candidate_df))
        print(
            "SUV min/max:",
            float(candidate_df["candidate_suv"].min()),
            float(candidate_df["candidate_suv"].max())
        )
        print(
            "Difference min/max:",
            float(candidate_df["local_difference"].min()),
            float(candidate_df["local_difference"].max())
        )
        print(
            "Ratio min/max:",
            float(candidate_df["local_ratio"].min()),
            float(candidate_df["local_ratio"].max())
        )
        print(
            "SUV pass:",
            int(
                (
                    candidate_df["candidate_suv"]
                    >= suv_threshold
                ).sum()
            )
        )
        print(
            "Difference pass:",
            int(
                (
                    candidate_df["local_difference"]
                    >= diff_threshold
                ).sum()
            )
        )
        print(
            "Ratio pass:",
            int(
                (
                    candidate_df["local_ratio"]
                    >= ratio_threshold
                ).sum()
            )
        )
        print(
            "ALL pass:",
            int(selected.sum())
        )
        print(
            candidate_df[
                [
                    "id",
                    "candidate_suv",
                    "local_difference",
                    "local_ratio"
                ]
            ].head(5).to_string(index=False)
        )
        print("--- END DEBUG ---\n")

    selected_ids = candidate_df.loc[
        selected,
        "id"
    ].to_numpy(
        dtype=np.int32
    )

    if len(selected_ids) == 0:
        candidate_volume = np.zeros_like(
            candidate_labels,
            dtype=np.uint8
        )
    else:
        candidate_volume = np.isin(
            candidate_labels,
            selected_ids
        ).astype(np.uint8)

    # -----------------------------------------------------
    # Exact V4 3D connected-component filtering
    # -----------------------------------------------------

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
    stats.Execute(cc_img)

    final_mask = np.zeros_like(
        candidate_volume,
        dtype=np.uint8
    )

    for label in stats.GetLabels():

        component = (
            cc == label
        )

        voxel_count = int(
            component.sum()
        )

        if voxel_count < MIN_3D_VOXELS:
            continue

        z_indices = np.where(component)[0]

        if len(z_indices) == 0:
            continue

        z_span = (
            int(z_indices.max())
            - int(z_indices.min())
            + 1
        )

        if z_span < MIN_3D_SLICES:
            continue

        final_mask[component] = 1

    # -----------------------------------------------------
    # Voxel metrics
    # -----------------------------------------------------

    pred = final_mask > 0

    intersection = np.logical_and(
        pred,
        gt
    ).sum()

    union = np.logical_or(
        pred,
        gt
    ).sum()

    pred_voxels = int(pred.sum())
    gt_voxels = int(gt.sum())

    dice = (
        2.0 * intersection
        / (pred_voxels + gt_voxels)
        if (pred_voxels + gt_voxels) > 0
        else 0.0
    )

    iou = (
        intersection / union
        if union > 0
        else 0.0
    )

    # -----------------------------------------------------
    # Slice metrics
    # -----------------------------------------------------

    gt_positive = np.any(
        gt,
        axis=(1, 2)
    )

    pred_positive = np.any(
        pred,
        axis=(1, 2)
    )

    tp = int(
        np.logical_and(
            gt_positive,
            pred_positive
        ).sum()
    )

    fp = int(
        np.logical_and(
            ~gt_positive,
            pred_positive
        ).sum()
    )

    fn = int(
        np.logical_and(
            gt_positive,
            ~pred_positive
        ).sum()
    )

    gt_positive_count = int(
        gt_positive.sum()
    )

    negative_slice_count = int(
        (~gt_positive).sum()
    )

    slice_recall = (
        tp / gt_positive_count
        if gt_positive_count > 0
        else 0.0
    )

    fpr = (
        fp / negative_slice_count
        if negative_slice_count > 0
        else 0.0
    )

    return {
        "suv": suv_threshold,
        "difference": diff_threshold,
        "ratio": ratio_threshold,
        "selected_candidates":
            int(selected.sum()),
        "prediction_voxels":
            int(pred_voxels),
        "gt_voxels":
            int(gt_voxels),
        "dice": float(dice),
        "iou": float(iou),
        "slice_recall":
            float(slice_recall),
        "false_positive_slices":
            fp,
        "false_negative_slices":
            fn,
        "fpr":
            float(fpr),
    }


# =========================================================
# MAIN
# =========================================================

print()
print("=" * 70)
print("===== V4 SEGMENTATION-LEVEL THRESHOLD SWEEP =====")
print("=" * 70)

all_cases = []

for info in CASES:

    try:
        prepared = prepare_case(
            info["name"],
            info["case"]
        )

        all_cases.append(prepared)

    except Exception as exc:
        print()
        print(
            "ERROR preparing",
            info["name"]
        )
        print(exc)
        raise


# =========================================================
# SWEEP
# =========================================================

case_results = []
macro_results = []

combinations = list(
    product(
        SUV_THRESHOLDS,
        DIFF_THRESHOLDS,
        RATIO_THRESHOLDS
    )
)

print()
print(
    "Threshold combinations:",
    len(combinations)
)

for index, (
    suv_t,
    diff_t,
    ratio_t
) in enumerate(combinations, start=1):

    print(
        f"\n[{index}/{len(combinations)}] "
        f"SUV={suv_t:.2f}, "
        f"diff={diff_t:.2f}, "
        f"ratio={ratio_t:.2f}"
    )

    metrics_for_combo = []

    for prepared in all_cases:

        result = evaluate_selection(
            prepared["candidate_labels"],
            prepared["candidates"],
            prepared["gt"],
            suv_t,
            diff_t,
            ratio_t
        )

        result["case"] = prepared["name"]

        case_results.append(result)
        metrics_for_combo.append(result)

        print(
            f"  {prepared['name']}: "
            f"Dice={result['dice']:.4f}, "
            f"IoU={result['iou']:.4f}, "
            f"Recall={result['slice_recall']:.4f}, "
            f"FPR={result['fpr']:.4f}"
        )

    macro_results.append(
        {
            "suv": suv_t,
            "difference": diff_t,
            "ratio": ratio_t,
            "macro_dice":
                float(
                    np.mean([
                        r["dice"]
                        for r in metrics_for_combo
                    ])
                ),
            "macro_iou":
                float(
                    np.mean([
                        r["iou"]
                        for r in metrics_for_combo
                    ])
                ),
            "macro_slice_recall":
                float(
                    np.mean([
                        r["slice_recall"]
                        for r in metrics_for_combo
                    ])
                ),
            "macro_fpr":
                float(
                    np.mean([
                        r["fpr"]
                        for r in metrics_for_combo
                    ])
                ),
            "total_prediction_voxels":
                int(
                    sum(
                        r["prediction_voxels"]
                        for r in metrics_for_combo
                    )
                ),
        }
    )


# =========================================================
# SAVE
# =========================================================

case_df = pd.DataFrame(case_results)
macro_df = pd.DataFrame(macro_results)

case_output = Path(
    "development_cases/"
    "v4_segmentation_threshold_sweep_cases.csv"
)

macro_output = Path(
    "development_cases/"
    "v4_segmentation_threshold_sweep.csv"
)

case_df.to_csv(
    case_output,
    index=False
)

macro_df.to_csv(
    macro_output,
    index=False
)

# =========================================================
# REPORT
# =========================================================

print()
print("=" * 70)
print("===== TOP BY MACRO DICE =====")
print("=" * 70)

print(
    macro_df.sort_values(
        "macro_dice",
        ascending=False
    ).head(15).to_string(index=False)
)

print()
print("=" * 70)
print("===== TOP BY MACRO IOU =====")
print("=" * 70)

print(
    macro_df.sort_values(
        "macro_iou",
        ascending=False
    ).head(10).to_string(index=False)
)

print()
print("=" * 70)
print("===== CURRENT V4 BASELINE =====")
print("=" * 70)

baseline = macro_df[
    (macro_df["suv"] == 1.00)
    & (macro_df["difference"] == 0.50)
    & (macro_df["ratio"] == 1.60)
]

if len(baseline):
    print(
        baseline.to_string(index=False)
    )
else:
    print("Baseline combination not found.")

print()
print("Saved:")
print(case_output)
print(macro_output)

print()
print("===== SWEEP COMPLETE =====")
