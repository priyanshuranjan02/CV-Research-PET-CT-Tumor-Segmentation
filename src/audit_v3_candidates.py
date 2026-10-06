from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk


# =========================================================
# SIX DEVELOPMENT CASES
# =========================================================

CASES = [
    (
        "PETCT_0168f65af8",
        Path("development_cases/PETCT_0168f65af8"),
    ),
    (
        "PETCT_04606080a0",
        Path("development_cases/PETCT_04606080a0"),
    ),
    (
        "PETCT_04ab5c61c9",
        Path("development_cases/PETCT_04ab5c61c9"),
    ),
    (
        "PETCT_0b57b247b6",
        Path("development_cases/PETCT_0b57b247b6"),
    ),
    (
        "PETCT_11afab3485",
        Path("development_cases/PETCT_11afab3485"),
    ),
    (
        "PETCT_185da4c8b6",
        Path("development_cases/PETCT_185da4c8b6"),
    ),
]


# =========================================================
# EXACT V3 PARAMETERS USED FOR CANDIDATE GENERATION
# =========================================================

MIN_CANDIDATE_AREA = 100
MAX_CANDIDATE_AREA = 10000

OPENING_KERNEL = np.ones(
    (5, 5),
    np.uint8
)


# =========================================================
# HELPERS
# =========================================================

def find_series(case_dir, text):

    matches = [
        p
        for p in case_dir.rglob("*")
        if (
            p.is_dir()
            and text in p.name
            and "__MACOSX" not in p.parts
            and "verified_download" not in p.parts
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"{case_dir}: expected exactly one primary "
            f"directory containing '{text}', found "
            f"{len(matches)}"
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


# =========================================================
# AUDIT ONE CASE
# =========================================================

def audit_case(case_name, root):

    print()
    print("=" * 70)
    print("CASE:", case_name)
    print("=" * 70)

    # -----------------------------------------------------
    # Locate CT
    # -----------------------------------------------------

    ct_dir = find_series(
        root,
        "GK p.v.3"
    )

    # -----------------------------------------------------
    # Load CT
    # -----------------------------------------------------

    print("Loading CT...")

    ct_img = read_series(
        ct_dir
    )

    ct = sitk.GetArrayFromImage(
        ct_img
    ).astype(np.float32)

    # -----------------------------------------------------
    # Load ground truth
    # -----------------------------------------------------

    gt_path = (
        root /
        "tumor_mask_ct_visible.nii.gz"
    )

    if not gt_path.exists():
        raise RuntimeError(
            f"GT not found: {gt_path}"
        )

    gt_img = sitk.ReadImage(
        str(gt_path)
    )

    gt = (
        sitk.GetArrayFromImage(
            gt_img
        ) > 0
    )

    if gt.shape != ct.shape:
        raise RuntimeError(
            f"CT/GT shape mismatch: "
            f"{ct.shape} vs {gt.shape}"
        )

    # -----------------------------------------------------
    # Load ACTUAL V3 candidate table
    # -----------------------------------------------------

    csv_candidates = list(
        root.rglob(
            "hybrid_v3_candidate_scores.csv"
        )
    )

    csv_candidates = [
        p
        for p in csv_candidates
        if "__MACOSX" not in p.parts
    ]

    if len(csv_candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one V3 CSV, found "
            f"{len(csv_candidates)}"
        )

    csv_path = csv_candidates[0]

    df = pd.read_csv(
        csv_path
    )

    required_columns = [
        "slice",
        "area",
        "candidate_suv",
        "background_suv",
        "local_difference",
        "local_ratio",
        "texture",
        "circularity",
        "accepted",
    ]

    missing = [
        c
        for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing CSV columns: {missing}"
        )

    print(
        "Actual V3 CSV:",
        csv_path
    )

    print(
        "Actual V3 candidates:",
        len(df)
    )

    print(
        "Actual V3 accepted:",
        int(df["accepted"].sum())
    )

    # -----------------------------------------------------
    # Build EXACT CT candidate masks
    # -----------------------------------------------------

    print("Building body ROI...")

    body = build_body_mask(
        ct
    )

    print(
        "Body voxels:",
        int(body.sum())
    )

    ct_display = normalize_ct(
        ct
    )

    audited_rows = []

    row_index = 0

    print(
        "Reconstructing candidate masks..."
    )

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
            OPENING_KERNEL
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
            # Match reconstructed candidate to actual CSV row
            # -------------------------------------------------

            if row_index >= len(df):
                raise RuntimeError(
                    "Reconstructed more candidates than "
                    "the actual V3 CSV."
                )

            actual = df.iloc[
                row_index
            ]

            actual_slice = int(
                actual["slice"]
            )

            actual_area = float(
                actual["area"]
            )

            if actual_slice != z:
                raise RuntimeError(
                    f"Candidate ordering mismatch at row "
                    f"{row_index}: "
                    f"CSV slice={actual_slice}, "
                    f"reconstructed slice={z}"
                )

            if not np.isclose(
                actual_area,
                area,
                rtol=0,
                atol=1e-4
            ):
                raise RuntimeError(
                    f"Candidate area mismatch at row "
                    f"{row_index}: "
                    f"CSV={actual_area}, "
                    f"reconstructed={area}"
                )

            # -------------------------------------------------
            # Ground-truth overlap
            # -------------------------------------------------

            gt_slice = gt[z]

            candidate_pixels_count = int(
                candidate_pixels.sum()
            )

            gt_pixels_count = int(
                gt_slice.sum()
            )

            intersection = int(
                np.logical_and(
                    candidate_pixels,
                    gt_slice
                ).sum()
            )

            union = int(
                np.logical_or(
                    candidate_pixels,
                    gt_slice
                ).sum()
            )

            # Dice
            denominator = (
                candidate_pixels_count
                + gt_pixels_count
            )

            if denominator == 0:
                dice = 0.0
            else:
                dice = (
                    2.0
                    * intersection
                    / denominator
                )

            # IoU
            if union == 0:
                iou = 0.0
            else:
                iou = (
                    intersection
                    / union
                )

            # Candidate precision
            precision = (
                intersection
                / candidate_pixels_count
                if candidate_pixels_count > 0
                else 0.0
            )

            # GT coverage
            recall = (
                intersection
                / gt_pixels_count
                if gt_pixels_count > 0
                else 0.0
            )

            if intersection == 0:
                group = "zero"
            elif dice < 0.10:
                group = "low"
            elif dice < 0.30:
                group = "moderate"
            elif dice < 0.50:
                group = "strong"
            else:
                group = "very_strong"

            audited_rows.append(
                {
                    "case": case_name,
                    "csv_row": row_index,
                    "slice": z,
                    "area": actual_area,
                    "candidate_pixels":
                        candidate_pixels_count,
                    "gt_pixels":
                        gt_pixels_count,
                    "intersection":
                        intersection,
                    "candidate_dice":
                        dice,
                    "candidate_iou":
                        iou,
                    "candidate_precision":
                        precision,
                    "candidate_recall":
                        recall,
                    "candidate_suv":
                        actual["candidate_suv"],
                    "background_suv":
                        actual["background_suv"],
                    "local_difference":
                        actual["local_difference"],
                    "local_ratio":
                        actual["local_ratio"],
                    "texture":
                        actual["texture"],
                    "circularity":
                        actual["circularity"],
                    "accepted":
                        actual["accepted"],
                    "overlap_group":
                        group,
                }
            )

            row_index += 1

    # -----------------------------------------------------
    # Exact row-count validation
    # -----------------------------------------------------

    if row_index != len(df):
        raise RuntimeError(
            f"Candidate count mismatch: "
            f"reconstructed={row_index}, "
            f"CSV={len(df)}"
        )

    audit = pd.DataFrame(
        audited_rows
    )

    # -----------------------------------------------------
    # Sanity check Dice
    # -----------------------------------------------------

    if (
        audit["candidate_dice"]
        .max()
        > 1.000001
    ):
        raise RuntimeError(
            "Dice calculation produced a value > 1."
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    output = (
        root /
        "v3_candidate_gt_audit.csv"
    )

    audit.to_csv(
        output,
        index=False
    )

    print(
        "Saved:",
        output
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()
    print("----- AUDIT SUMMARY -----")

    print(
        "Candidates:",
        len(audit)
    )

    print(
        "Accepted:",
        int(audit["accepted"].sum())
    )

    print(
        "Zero-overlap:",
        int(
            (
                audit["intersection"]
                == 0
            ).sum()
        )
    )

    print(
        "Dice >= 0.10:",
        int(
            (
                audit["candidate_dice"]
                >= 0.10
            ).sum()
        )
    )

    print(
        "Dice >= 0.30:",
        int(
            (
                audit["candidate_dice"]
                >= 0.30
            ).sum()
        )
    )

    print(
        "Dice >= 0.50:",
        int(
            (
                audit["candidate_dice"]
                >= 0.50
            ).sum()
        )
    )

    accepted = audit[
        audit["accepted"] == 1
    ]

    print()
    print("----- ACCEPTED CANDIDATES -----")

    print(
        "Accepted:",
        len(accepted)
    )

    print(
        "Accepted Dice >= 0.10:",
        int(
            (
                accepted["candidate_dice"]
                >= 0.10
            ).sum()
        )
    )

    print(
        "Accepted Dice >= 0.30:",
        int(
            (
                accepted["candidate_dice"]
                >= 0.30
            ).sum()
        )
    )

    # -----------------------------------------------------
    # Best candidate
    # -----------------------------------------------------

    best_idx = audit[
        "candidate_dice"
    ].idxmax()

    best = audit.loc[
        best_idx
    ]

    print()
    print("----- BEST CANDIDATE -----")

    print(
        "Slice:",
        int(best["slice"])
    )

    print(
        "Dice:",
        round(
            float(best["candidate_dice"]),
            4
        )
    )

    print(
        "IoU:",
        round(
            float(best["candidate_iou"]),
            4
        )
    )

    print(
        "Area:",
        round(
            float(best["area"]),
            2
        )
    )

    print(
        "Candidate SUV:",
        round(
            float(best["candidate_suv"]),
            4
        )
    )

    print(
        "Background SUV:",
        round(
            float(best["background_suv"]),
            4
        )
    )

    print(
        "Local difference:",
        round(
            float(best["local_difference"]),
            4
        )
    )

    print(
        "Local ratio:",
        round(
            float(best["local_ratio"]),
            4
        )
    )

    return audit


# =========================================================
# RUN ALL SIX CASES
# =========================================================

all_audits = []

for case_name, root in CASES:

    audit = audit_case(
        case_name,
        root
    )

    all_audits.append(
        audit
    )


# =========================================================
# COMBINE
# =========================================================

combined = pd.concat(
    all_audits,
    ignore_index=True
)

combined_path = Path(
    "development_cases/"
    "v3_candidate_gt_audit_all_cases.csv"
)

combined.to_csv(
    combined_path,
    index=False
)


# =========================================================
# GLOBAL FEATURE ANALYSIS
# =========================================================

print()
print("=" * 70)
print("SIX-CASE FEATURE ANALYSIS")
print("=" * 70)

for group in [
    "zero",
    "low",
    "moderate",
    "strong",
    "very_strong"
]:

    subset = combined[
        combined["overlap_group"] == group
    ]

    print()
    print(
        group.upper(),
        "n =",
        len(subset)
    )

    if len(subset) == 0:
        continue

    features = [
        "area",
        "candidate_suv",
        "background_suv",
        "local_difference",
        "local_ratio",
        "texture",
        "circularity",
        "candidate_dice",
        "candidate_iou",
    ]

    print(
        subset[features]
        .median()
        .round(4)
        .to_string()
    )


# =========================================================
# ACCEPTED vs REJECTED
# =========================================================

print()
print("=" * 70)
print("ACCEPTED vs REJECTED")
print("=" * 70)

for value in [0, 1]:

    subset = combined[
        combined["accepted"] == value
    ]

    label = (
        "ACCEPTED"
        if value == 1
        else "REJECTED"
    )

    print()
    print(
        label,
        "n =",
        len(subset)
    )

    features = [
        "candidate_suv",
        "background_suv",
        "local_difference",
        "local_ratio",
        "texture",
        "circularity",
        "area",
        "candidate_dice",
        "candidate_iou",
    ]

    print(
        subset[features]
        .median()
        .round(4)
        .to_string()
    )


# =========================================================
# FINAL
# =========================================================

print()
print("=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print(
    "Total candidates:",
    len(combined)
)

print(
    "Total accepted:",
    int(combined["accepted"].sum())
)

print(
    "Maximum candidate Dice:",
    round(
        float(
            combined["candidate_dice"].max()
        ),
        4
    )
)

print(
    "Saved:",
    combined_path
)