from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom

# =========================================================
# FIXED V4 PARAMETERS
# =========================================================

MIN_CANDIDATE_AREA = 100
MAX_CANDIDATE_AREA = 10000

LOCAL_RING_KERNEL = np.ones((31, 31), np.uint8)

MIN_LOCAL_RATIO = 1.6
MIN_LOCAL_DIFFERENCE = 0.5
MIN_CANDIDATE_SUV = 1.0

MIN_3D_VOXELS = 75
MIN_3D_SLICES = 2

CORE_THRESHOLDS = [
    1.50,
    1.75,
    2.00,
]

CASES = [
    (
        "PETCT_0168f65af8",
        Path(
            "development_cases/PETCT_0168f65af8/"
            "manifest-1789661648760/FDG-PET-CT-Lesions/"
            "PETCT_0168f65af8"
        ),
    ),
    (
        "PETCT_04606080a0",
        Path(
            "development_cases/PETCT_04606080a0/"
            "FDG-PET-CT-Lesions/"
            "PETCT_04606080a0"
        ),
    ),
    (
        "PETCT_04ab5c61c9",
        Path(
            "development_cases/PETCT_04ab5c61c9/"
            "manifest-1789918980481/FDG-PET-CT-Lesions/"
            "PETCT_04ab5c61c9"
        ),
    ),
    (
        "PETCT_0b57b247b6",
        Path(
            "development_cases/PETCT_0b57b247b6/"
            "manifest-1789920519449/FDG-PET-CT-Lesions/"
            "PETCT_0b57b247b6"
        ),
    ),
    (
        "PETCT_11afab3485",
        Path(
            "development_cases/PETCT_11afab3485/"
            "manifest-1789921915902/FDG-PET-CT-Lesions/"
            "PETCT_11afab3485"
        ),
    ),
    (
        "PETCT_185da4c8b6",
        Path(
            "development_cases/PETCT_185da4c8b6/"
            "manifest-1789923055836/FDG-PET-CT-Lesions/"
            "PETCT_185da4c8b6"
        ),
    ),
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
            f"{case_dir}: expected one directory containing "
            f"'{text}', found {len(matches)}"
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

    weight = None
    dose = None

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
                    dose = float(dose_value)

    if weight is None:
        raise RuntimeError(
            f"PatientWeight not found in {pet_dir}"
        )

    if dose is None:
        raise RuntimeError(
            f"Injected dose not found in {pet_dir}"
        )

    return weight, dose


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


def find_gt(case_dir):
    for parent in [case_dir] + list(case_dir.parents):

        candidate = (
            parent
            / "tumor_mask_ct_visible.nii.gz"
        )

        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"GT not found for {case_dir}"
    )


def find_reference_csv(case_name):
    paths = {
        "PETCT_0168f65af8": Path(
            "development_cases/PETCT_0168f65af8/"
            "manifest-1789661648760/FDG-PET-CT-Lesions/"
            "PETCT_0168f65af8/"
            "hybrid_v4_balanced_candidate_scores.csv"
        ),
        "PETCT_04606080a0": Path(
            "development_cases/PETCT_04606080a0/"
            "FDG-PET-CT-Lesions/"
            "PETCT_04606080a0/"
            "hybrid_v4_balanced_candidate_scores.csv"
        ),
        "PETCT_04ab5c61c9": Path(
            "development_cases/PETCT_04ab5c61c9/"
            "manifest-1789918980481/FDG-PET-CT-Lesions/"
            "PETCT_04ab5c61c9/"
            "hybrid_v4_balanced_candidate_scores.csv"
        ),
        "PETCT_0b57b247b6": Path(
            "development_cases/PETCT_0b57b247b6/"
            "manifest-1789920519449/FDG-PET-CT-Lesions/"
            "PETCT_0b57b247b6/"
            "hybrid_v4_balanced_candidate_scores.csv"
        ),
        "PETCT_11afab3485": Path(
            "development_cases/PETCT_11afab3485/"
            "manifest-1789921915902/FDG-PET-CT-Lesions/"
            "PETCT_11afab3485/"
            "hybrid_v4_balanced_candidate_scores.csv"
        ),
        "PETCT_185da4c8b6": Path(
            "development_cases/PETCT_185da4c8b6/"
            "manifest-1789923055836/FDG-PET-CT-Lesions/"
            "PETCT_185da4c8b6/"
            "hybrid_v4_balanced_candidate_scores.csv"
        ),
    }

    return paths[case_name]


# =========================================================
# PREPARE CASE
# =========================================================

def prepare_case(case_name, case_dir):

    print()
    print("=" * 70)
    print("Preparing:", case_name)
    print("=" * 70)

    gt_path = find_gt(case_dir)
    reference_csv = find_reference_csv(case_name)

    print("GT:", gt_path)
    print("Reference:", reference_csv)

    # -----------------------------------------------------
    # Load CT
    # -----------------------------------------------------

    ct_dir = find_series(
        case_dir,
        "GK p.v.3"
    )

    ct_img = read_series(
        ct_dir
    )

    ct = sitk.GetArrayFromImage(
        ct_img
    ).astype(np.float32)

    # -----------------------------------------------------
    # Load PET
    # -----------------------------------------------------

    pet_dir = find_series(
        case_dir,
        "PET corr."
    )

    pet_img = read_series(
        pet_dir
    )

    weight, dose = read_pet_parameters(
        pet_dir
    )

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

    print(
        "CT:",
        ct_img.GetSize()
    )

    print(
        "PET:",
        pet_img.GetSize()
    )

    print(
        "SUV max:",
        round(
            float(np.nanmax(suv)),
            4
        )
    )

    # -----------------------------------------------------
    # Body ROI
    # -----------------------------------------------------

    body = build_body_mask(ct)

    # -----------------------------------------------------
    # Recreate V4 candidate geometry
    # -----------------------------------------------------

    ct_display = normalize_ct(ct)

    opening_kernel = np.ones(
        (5, 5),
        np.uint8
    )

    candidate_masks = []
    candidate_rows = []

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

            cand_suv = suv[z][
                candidate_pixels
            ]

            ring_suv = suv[z][ring]

            cand_suv = cand_suv[
                np.isfinite(cand_suv)
            ]

            ring_suv = ring_suv[
                np.isfinite(ring_suv)
            ]

            if (
                len(cand_suv) == 0
                or len(ring_suv) == 0
            ):
                continue

            cand_median = float(
                np.median(cand_suv)
            )

            ring_median = float(
                np.median(ring_suv)
            )

            local_difference = (
                cand_median
                - ring_median
            )

            local_ratio = (
                cand_median
                / (ring_median + 1e-6)
            )

            gate_count = (
                int(local_ratio >= MIN_LOCAL_RATIO)
                + int(
                    local_difference
                    >= MIN_LOCAL_DIFFERENCE
                )
                + int(
                    cand_median
                    >= MIN_CANDIDATE_SUV
                )
            )

            accepted = (
                gate_count >= 2
            )

            candidate_masks.append(
                (
                    z,
                    candidate_mask,
                    ring_median
                )
            )

            candidate_rows.append(
                {
                    "slice": z,
                    "area": area,
                    "candidate_suv":
                        cand_median,
                    "background_suv":
                        ring_median,
                    "local_difference":
                        local_difference,
                    "local_ratio":
                        local_ratio,
                    "accepted":
                        int(accepted),
                }
            )

    generated = pd.DataFrame(
        candidate_rows
    )

    reference = pd.read_csv(
        reference_csv
    )

    if len(generated) != len(reference):
        raise RuntimeError(
            f"{case_name}: candidate count mismatch. "
            f"generated={len(generated)}, "
            f"reference={len(reference)}"
        )

    if not np.array_equal(
        generated["slice"].to_numpy(),
        reference["slice"].to_numpy()
    ):
        raise RuntimeError(
            f"{case_name}: candidate ordering mismatch"
        )

    if not np.allclose(
        generated["area"].to_numpy(),
        reference["area"].to_numpy(),
        rtol=0.0,
        atol=1e-4
    ):
        raise RuntimeError(
            f"{case_name}: candidate geometry mismatch"
        )

    # Use authoritative original V4 acceptance decisions
    # and background SUV values.
    accepted_reference = (
        reference["accepted"].to_numpy() == 1
    )

    background_values = (
        reference["background_suv"].to_numpy()
    )

    # -----------------------------------------------------
    # GT
    # -----------------------------------------------------

    gt = (
        sitk.GetArrayFromImage(
            sitk.ReadImage(
                str(gt_path)
            )
        ) > 0
    )

    return {
        "name": case_name,
        "suv": suv,
        "gt": gt,
        "candidate_masks": candidate_masks,
        "accepted": accepted_reference,
        "background": background_values,
        "reference": reference,
    }


# =========================================================
# BUILD CORE MASK
# =========================================================

def build_prediction(
    prepared,
    rule
):

    candidate_masks = prepared["candidate_masks"]
    reference = prepared["reference"]

    # Original V4-accepted slices are used as the continuity anchors.
    accepted_reference = prepared["accepted"]

    accepted_slices = {
        int(reference.iloc[i]["slice"])
        for i, flag in enumerate(accepted_reference)
        if flag
    }
    background = prepared["background"]
    suv = prepared["suv"]

    shape = suv.shape

    candidate_volume = np.zeros(
        shape,
        dtype=np.uint8
    )

    for i, (
        z,
        candidate_mask,
        _generated_background
    ) in enumerate(candidate_masks):

        # -------------------------------------------------
        # REAL 2-of-3 V4 acceptance
        #
        # Use the authoritative feature values from the
        # original V4 candidate table. Do not use the
        # original 3-of-3 "accepted" column.
        # -------------------------------------------------

        candidate_suv = float(
            reference.iloc[i]["candidate_suv"]
        )

        local_difference = float(
            reference.iloc[i]["local_difference"]
        )

        local_ratio = float(
            reference.iloc[i]["local_ratio"]
        )

        gate_count = (
            int(candidate_suv >= MIN_CANDIDATE_SUV)
            + int(
                local_difference
                >= MIN_LOCAL_DIFFERENCE
            )
            + int(
                local_ratio
                >= MIN_LOCAL_RATIO
            )
        )

        # Keep every original V4 (3-of-3) accepted candidate.
        original_v4_accepted = bool(
            prepared["accepted"][i]
        )

        # Continuity-aware recovery:
        # allow a 2-of-3 candidate only when it lies within
        # +/- 2 slices of an original V4-accepted slice.
        accepted_neighbor = any(
            0 < abs(z - accepted_z) <= 2
            for accepted_z in accepted_slices
        )

        if not original_v4_accepted and not (
            gate_count >= 2 and accepted_neighbor
        ):
            continue

        candidate_pixels = (
            candidate_mask > 0
        )

        if rule == "FULL":

            core = candidate_pixels

        elif rule.startswith("SUV_CORE_"):

            threshold = float(
                rule.split("_")[-1]
            )

            core = (
                candidate_pixels
                & (suv[z] >= threshold)
            )

        elif rule == "RELATIVE_1.30":

            threshold = (
                background[i] * 1.30
            )

            core = (
                candidate_pixels
                & (suv[z] >= threshold)
            )

        elif rule == "DIFF_0.30":

            threshold = (
                background[i] + 0.30
            )

            core = (
                candidate_pixels
                & (suv[z] >= threshold)
            )

        else:
            raise ValueError(
                f"Unknown rule: {rule}"
            )

        candidate_volume[z][core] = 1

    # -----------------------------------------------------
    # Same V4 3D filtering
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

    return final_mask


# =========================================================
# METRICS
# =========================================================

def evaluate(
    prediction,
    gt
):

    pred = prediction > 0

    intersection = np.logical_and(
        pred,
        gt
    ).sum()

    union = np.logical_or(
        pred,
        gt
    ).sum()

    pred_voxels = int(
        pred.sum()
    )

    gt_voxels = int(
        gt.sum()
    )

    dice = (
        2.0 * intersection
        / (pred_voxels + gt_voxels)
        if pred_voxels + gt_voxels > 0
        else 0.0
    )

    iou = (
        intersection / union
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

    slice_recall = (
        tp / gt_positive.sum()
        if gt_positive.sum() > 0
        else 0.0
    )

    negative_slices = (
        ~gt_positive
    ).sum()

    fpr = (
        fp / negative_slices
        if negative_slices > 0
        else 0.0
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "slice_recall": float(slice_recall),
        "fpr": float(fpr),
        "prediction_voxels": pred_voxels,
    }


# =========================================================
# MAIN
# =========================================================

print()
print("=" * 70)
print("===== V4 PET-CORE EXPERIMENT =====")
print("=" * 70)

prepared_cases = []

for name, case_dir in CASES:

    prepared_cases.append(
        prepare_case(
            name,
            case_dir
        )
    )


results = []

for threshold in CORE_THRESHOLDS:

    rule = f"SUV_CORE_{threshold:.2f}"

    print()
    print("-" * 70)
    print("RULE:", rule)
    print("-" * 70)

    case_metrics = []

    for prepared in prepared_cases:

        prediction = build_prediction(
            prepared,
            rule
        )

        metrics = evaluate(
            prediction,
            prepared["gt"]
        )

        row = {
            "rule": rule,
            "case": prepared["name"],
            **metrics,
        }

        results.append(row)
        case_metrics.append(metrics)

        print(
            f"{prepared['name']}: "
            f"Dice={metrics['dice']:.4f}, "
            f"IoU={metrics['iou']:.4f}, "
            f"Recall={metrics['slice_recall']:.4f}, "
            f"FPR={metrics['fpr']:.4f}, "
            f"Voxels={metrics['prediction_voxels']}"
        )

    print(
        "MACRO:",
        f"Dice={np.mean([m['dice'] for m in case_metrics]):.4f}, "
        f"IoU={np.mean([m['iou'] for m in case_metrics]):.4f}, "
        f"Recall={np.mean([m['slice_recall'] for m in case_metrics]):.4f}, "
        f"FPR={np.mean([m['fpr'] for m in case_metrics]):.4f}"
    )


# =========================================================
# SAVE
# =========================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "development_cases/v4_neighbor_core_experiment_cases.csv",
    index=False
)

macro_df = (
    results_df
    .groupby("rule")
    .agg(
        macro_dice=("dice", "mean"),
        macro_iou=("iou", "mean"),
        macro_slice_recall=("slice_recall", "mean"),
        macro_fpr=("fpr", "mean"),
        total_prediction_voxels=("prediction_voxels", "sum"),
    )
    .reset_index()
)

macro_df.to_csv(
    "development_cases/v4_neighbor_core_experiment_macro.csv",
    index=False
)

print()
print("=" * 70)
print("===== MACRO RESULTS =====")
print("=" * 70)

print(
    macro_df.to_string(index=False)
)

print()
print("Saved:")
print(
    "development_cases/v4_neighbor_core_experiment_cases.csv"
)
print(
    "development_cases/v4_neighbor_core_experiment_macro.csv"
)

print()
print("===== EXPERIMENT COMPLETE =====")
