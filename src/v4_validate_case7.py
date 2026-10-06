from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import SimpleITK as sitk


# =========================================================
# LOAD THE EXACT FROZEN DISTANCE-28 DEFINITIONS
# WITHOUT RUNNING ITS MAIN SECTION
# =========================================================

source = Path(
    "src/v4_oversized_gap_rescue_distance28.py"
).read_text()

prefix = source.rsplit(
    "# MAIN",
    1
)[0]

exec(
    compile(
        prefix,
        "v4_oversized_gap_rescue_distance28.py",
        "exec"
    ),
    globals()
)


# =========================================================
# CASE 7: PETCT_0011f3deaf
# =========================================================

CASE_NAME = "PETCT_0011f3deaf"

CT_DIR = Path(
    "verified_case/ct/CT/manifest-1789580951552/"
    "FDG-PET-CT-Lesions/PETCT_0011f3deaf/"
    "03-23-2003-NA-PET-CT Ganzkoerper  primaer mit KM-10445/"
    "4.000000-GK p.v.3-58263"
)

PET_DIR = Path(
    "verified_case/pet_reference/PET_reference/manifest-1789650489488/"
    "FDG-PET-CT-Lesions/PETCT_0011f3deaf/"
    "03-23-2003-NA-PET-CT Ganzkoerper  primaer mit KM-10445/"
    "7.000000-PET corr.-78839"
)

GT_PATH = Path(
    "verified_case/tumor_mask_ct_visible.nii.gz"
)

OUTPUT_DIR = Path(
    "validation_results/PETCT_0011f3deaf"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# PREPARE CASE WITHOUT DEVELOPMENT REFERENCE CSV
# =========================================================

def prepare_case7():

    print()
    print("=" * 72)
    print("===== PREPARING CASE 7: PETCT_0011f3deaf =====")
    print("=" * 72)

    # -----------------------------------------------------
    # CT
    # -----------------------------------------------------

    ct_img = read_series(
        CT_DIR
    )

    ct = sitk.GetArrayFromImage(
        ct_img
    ).astype(np.float32)

    # -----------------------------------------------------
    # PET
    # -----------------------------------------------------

    pet_img = read_series(
        PET_DIR
    )

    weight, dose = read_pet_parameters(
        PET_DIR
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

    print("CT:", ct_img.GetSize())
    print("PET:", pet_img.GetSize())
    print("Patient weight:", weight)
    print("Injected dose:", dose)
    print(
        "SUV max:",
        round(float(np.nanmax(suv)), 4)
    )

    # -----------------------------------------------------
    # BODY ROI
    # -----------------------------------------------------

    body = build_body_mask(
        ct
    )

    # -----------------------------------------------------
    # RECREATE EXACT V4 CANDIDATE GEOMETRY
    # -----------------------------------------------------

    ct_display = normalize_ct(
        ct
    )

    opening_kernel = np.ones(
        (5, 5),
        np.uint8
    )

    candidate_masks = []
    candidate_rows = []

    for z in range(
        ct.shape[0]
    ):

        body_slice = body[z]

        if not np.any(
            body_slice
        ):
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

            if not np.any(
                ring
            ):
                continue

            cand_suv = suv[z][
                candidate_pixels
            ]

            ring_suv = suv[z][
                ring
            ]

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
                np.median(
                    cand_suv
                )
            )

            ring_median = float(
                np.median(
                    ring_suv
                )
            )

            local_difference = (
                cand_median
                - ring_median
            )

            local_ratio = (
                cand_median
                / (ring_median + 1e-6)
            )

            # IMPORTANT:
            # Original V4 anchor = 3-of-3 gates.
            gate_count = (
                int(
                    local_ratio
                    >= MIN_LOCAL_RATIO
                )
                + int(
                    local_difference
                    >= MIN_LOCAL_DIFFERENCE
                )
                + int(
                    cand_median
                    >= MIN_CANDIDATE_SUV
                )
            )

            original_v4_accepted = (
                gate_count == 3
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
                        int(
                            original_v4_accepted
                        ),
                }
            )

    reference = pd.DataFrame(
        candidate_rows
    )

    if reference.empty:
        raise RuntimeError(
            "Case 7: no candidates generated."
        )

    accepted_count = int(
        reference["accepted"].sum()
    )

    accepted_slices = int(
        reference.loc[
            reference["accepted"] == 1,
            "slice"
        ].nunique()
    )

    print(
        "Generated candidates:",
        len(reference)
    )

    print(
        "Original V4 3-of-3 accepted candidates:",
        accepted_count
    )

    print(
        "Original V4 accepted slices:",
        accepted_slices
    )

    reference.to_csv(
        OUTPUT_DIR
        / "case7_generated_candidate_scores.csv",
        index=False
    )

    # -----------------------------------------------------
    # GROUND TRUTH
    # -----------------------------------------------------

    gt = (
        sitk.GetArrayFromImage(
            sitk.ReadImage(
                str(GT_PATH)
            )
        ) > 0
    )

    if gt.shape != suv.shape:
        raise RuntimeError(
            f"Case 7: GT/SUV shape mismatch: "
            f"GT={gt.shape}, SUV={suv.shape}"
        )

    return {
        "name": CASE_NAME,
        "suv": suv,
        "gt": gt,
        "candidate_masks": candidate_masks,
        "accepted":
            reference["accepted"].to_numpy() == 1,
        "background":
            reference["background_suv"].to_numpy(),
        "reference": reference,
    }, ct_img, ct, body


# =========================================================
# RUN FROZEN FINAL CONFIGURATION
# =========================================================

prepared, ct_img, ct, body = (
    prepare_case7()
)

print()
print("=" * 72)
print("===== FROZEN FINAL CONFIGURATION =====")
print("=" * 72)
print("Dual core original      = 2.00")
print("Dual core recovered     = 1.50")
print("Rescue PET threshold    = 1.50")
print("Rescue mean SUV         = 2.25")
print("Rescue component area   = 20 px")
print("Rescue neighbor distance= 28 px")


prediction, rescue_count, rescue_rows = (
    build_prediction_with_rescue(
        prepared,
        ct,
        body,
        1.50,
        2.25
    )
)


# =========================================================
# EVALUATION
# =========================================================

metrics = evaluate(
    prediction,
    prepared["gt"]
)

print()
print("=" * 72)
print("===== CASE 7 VALIDATION RESULT =====")
print("=" * 72)

print(
    f"Case: {CASE_NAME}"
)

print(
    f"Dice          = {metrics['dice']:.6f}"
)

print(
    f"IoU           = {metrics['iou']:.6f}"
)

print(
    f"Slice Recall  = {metrics['slice_recall']:.6f}"
)

print(
    f"FPR           = {metrics['fpr']:.6f}"
)

print(
    f"Prediction Voxels = "
    f"{metrics['prediction_voxels']}"
)

print(
    f"Rescue Slices = {rescue_count}"
)


# =========================================================
# SAVE PREDICTION
# =========================================================

prediction_img = sitk.GetImageFromArray(
    prediction.astype(np.uint8)
)

prediction_img.CopyInformation(
    ct_img
)

sitk.WriteImage(
    prediction_img,
    str(
        OUTPUT_DIR
        / "case7_final_prediction_distance28.nii.gz"
    )
)


# =========================================================
# SAVE METRICS
# =========================================================

metrics_row = {
    "case": CASE_NAME,
    "configuration":
        "DUAL_CORE_2.00_1.50_RESCUE_1.50_MEAN_2.25_DISTANCE_28",
    "dice": metrics["dice"],
    "iou": metrics["iou"],
    "slice_recall":
        metrics["slice_recall"],
    "fpr": metrics["fpr"],
    "prediction_voxels":
        metrics["prediction_voxels"],
    "rescue_selected_slices":
        rescue_count,
}

pd.DataFrame(
    [metrics_row]
).to_csv(
    OUTPUT_DIR
    / "case7_validation_metrics.csv",
    index=False
)


# =========================================================
# SAVE SELECTED RESCUE COMPONENTS
# =========================================================

pd.DataFrame(
    rescue_rows
).to_csv(
    OUTPUT_DIR
    / "case7_selected_rescue_components.csv",
    index=False
)


print()
print("Saved:")
print(
    OUTPUT_DIR
    / "case7_generated_candidate_scores.csv"
)
print(
    OUTPUT_DIR
    / "case7_final_prediction_distance28.nii.gz"
)
print(
    OUTPUT_DIR
    / "case7_validation_metrics.csv"
)
print(
    OUTPUT_DIR
    / "case7_selected_rescue_components.csv"
)

print()
print("=" * 72)
print("===== CASE 7 VALIDATION COMPLETE =====")
print("=" * 72)
