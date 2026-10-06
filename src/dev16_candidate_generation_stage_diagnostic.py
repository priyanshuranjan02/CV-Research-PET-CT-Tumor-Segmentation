"""
DEV16 — Candidate Generation Stage Diagnostic

Purpose:
    Diagnose exactly where GT-positive slices are lost during
    Dev10 raw candidate generation.

Stages:
    1. Body mask
    2. CT adaptive threshold / opening
    3. CT contour existence
    4. Valid CT contour
    5. PET hot overlap inside CT contour
    6. PET connected component
    7. PET component area filter
    8. Final raw candidate

IMPORTANT:
    Dev10 is NOT modified.
    This script reproduces the exact Dev10 candidate-generation
    operations and records diagnostics against the GT.
"""

from pathlib import Path
import io
from contextlib import redirect_stdout

import cv2
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

DEV10_PATH = SRC / "dev10_lite_hot_core.py"


# ============================================================
# Load exact Dev10 definitions
# ============================================================

import sys

sys.path.insert(
    0,
    str(SRC)
)

import dev10_lite_hot_core as dev10


load_case = dev10.load_case
build_union_body_mask = dev10.build_union_body_mask
normalize_ct = dev10.normalize_ct

MIN_CONTOUR_AREA = dev10.MIN_CONTOUR_AREA
PET_THRESHOLD = dev10.PET_THRESHOLD
MIN_PET_COMPONENT_AREA = dev10.MIN_PET_COMPONENT_AREA


# ============================================================
# Development cases
# ============================================================

DEV_CASES = [
    "PETCT_0168f65af8",
    "PETCT_04606080a0",
    "PETCT_04ab5c61c9",
    "PETCT_0b57b247b6",
    "PETCT_11afab3485",
    "PETCT_185da4c8b6",
]


# ============================================================
# Output directory
# ============================================================

OUT_DIR = (
    ROOT
    / "results"
    / "development_cases"
    / "dev16_candidate_generation_stage_results"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Diagnostic function
# ============================================================

def diagnose_case(case):

    case_id = case["name"]

    ct = case["ct"]
    suv = case["suv"]
    gt_volume = case["gt"]

    print()
    print("=" * 100)
    print(f"===== LOADING {case_id} =====")
    print("=" * 100)

    print(
        "CT:",
        ct.shape
    )

    print(
        "PET:",
        suv.shape
    )

    print(
        "SUV max:",
        float(np.max(suv))
    )

    print(
        "GT voxels:",
        int(np.sum(gt_volume > 0))
    )

    gt_positive_slices = np.where(
        gt_volume.reshape(
            gt_volume.shape[0],
            -1
        ).any(axis=1)
    )[0]

    print(
        "GT-positive slices:",
        len(gt_positive_slices)
    )

    print(
        "PET_THRESHOLD:",
        PET_THRESHOLD
    )

    print(
        "MIN_CONTOUR_AREA:",
        MIN_CONTOUR_AREA
    )

    print(
        "MIN_PET_COMPONENT_AREA:",
        MIN_PET_COMPONENT_AREA
    )

    print()

    # ========================================================
    # Exact Dev10 preprocessing
    # ========================================================

    body = build_union_body_mask(
        ct
    )

    ct_display = normalize_ct(
        ct
    )

    opening_kernel = np.ones(
        (5, 5),
        np.uint8
    )

    # ========================================================
    # IMPORTANT
    #
    # PET and CT dimensions can differ.
    # Dev10 itself indexes suv[z], so we preserve the exact
    # Dev10 behaviour here.
    # ========================================================

    raw_volume = np.zeros(
        suv.shape,
        dtype=np.uint8
    )

    rows = []

    # ========================================================
    # Process every GT-positive slice
    # ========================================================

    for z in gt_positive_slices:

        gt = (
            gt_volume[z] > 0
        )

        gt_area = int(
            gt.sum()
        )

        row = {
            "case_id": case_id,
            "slice": int(z),
            "gt_area": gt_area,

            "body_present": False,

            "ct_contours_total": 0,
            "ct_contours_valid_area": 0,

            "contours_touch_gt": 0,
            "contours_with_pet_hot": 0,

            "pet_hot_pixels_total": 0,
            "pet_hot_pixels_inside_gt": 0,

            "pet_components_total": 0,
            "pet_components_valid_area": 0,

            "pet_valid_components_touch_gt": 0,

            "raw_candidate_pixels": 0,
            "raw_candidate_pixels_inside_gt": 0,

            "stage": "UNKNOWN",
        }

        # ----------------------------------------------------
        # Stage 1: body mask
        # ----------------------------------------------------

        body_slice = body[z]

        if not np.any(
            body_slice
        ):

            row["stage"] = (
                "NO_BODY_MASK"
            )

            rows.append(row)
            continue

        row["body_present"] = True

        # ----------------------------------------------------
        # Stage 2: exact Dev10 CT preprocessing
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Stage 3: CT contours
        # ----------------------------------------------------

        contours, _ = cv2.findContours(
            opened,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        row[
            "ct_contours_total"
        ] = len(contours)

        # ----------------------------------------------------
        # Process every contour
        # ----------------------------------------------------

        for contour in contours:

            area = float(
                cv2.contourArea(
                    contour
                )
            )

            if area < MIN_CONTOUR_AREA:
                continue

            row[
                "ct_contours_valid_area"
            ] += 1

            contour_mask = np.zeros(
                opened.shape,
                dtype=np.uint8
            )

            cv2.drawContours(
                contour_mask,
                [contour],
                -1,
                1,
                -1
            )

            contour_bool = (
                contour_mask > 0
            )

            # Does CT contour overlap GT?
            if np.any(
                contour_bool & gt
            ):
                row[
                    "contours_touch_gt"
                ] += 1

            # ------------------------------------------------
            # Stage 4: PET threshold inside contour
            # ------------------------------------------------

            pet_hot = (
                suv[z] >= PET_THRESHOLD
            )

            hot = (
                contour_mask
                &
                pet_hot.astype(
                    np.uint8
                )
            )

            hot_pixels = int(
                hot.sum()
            )

            row[
                "pet_hot_pixels_total"
            ] += hot_pixels

            row[
                "pet_hot_pixels_inside_gt"
            ] += int(
                np.sum(
                    hot.astype(bool)
                    & gt
                )
            )

            if not np.any(
                hot
            ):
                continue

            row[
                "contours_with_pet_hot"
            ] += 1

            # ------------------------------------------------
            # Stage 5: PET connected components
            # ------------------------------------------------

            (
                n_components,
                labels,
                stats,
                centroids
            ) = cv2.connectedComponentsWithStats(
                hot,
                connectivity=8
            )

            for component_id in range(
                1,
                n_components
            ):

                row[
                    "pet_components_total"
                ] += 1

                area = int(
                    stats[
                        component_id,
                        cv2.CC_STAT_AREA
                    ]
                )

                if area < MIN_PET_COMPONENT_AREA:
                    continue

                row[
                    "pet_components_valid_area"
                ] += 1

                component = (
                    labels == component_id
                )

                if np.any(
                    component & gt
                ):
                    row[
                        "pet_valid_components_touch_gt"
                    ] += 1

                # ------------------------------------------------
                # Exact Dev10 raw candidate assignment
                # ------------------------------------------------

                raw_volume[z][
                    component
                ] = 1

        # ----------------------------------------------------
        # Final raw candidate
        # ----------------------------------------------------

        raw = (
            raw_volume[z] > 0
        )

        raw_area = int(
            raw.sum()
        )

        raw_gt_overlap = int(
            np.sum(
                raw & gt
            )
        )

        row[
            "raw_candidate_pixels"
        ] = raw_area

        row[
            "raw_candidate_pixels_inside_gt"
        ] = raw_gt_overlap

        # ----------------------------------------------------
        # Determine FIRST failure stage
        # ----------------------------------------------------

        if raw_area > 0:

            if raw_gt_overlap > 0:
                row[
                    "stage"
                ] = "RAW_CANDIDATE_HIT"

            else:
                row[
                    "stage"
                ] = "RAW_CANDIDATE_MISLOCALIZED"

        elif row[
            "pet_valid_components_touch_gt"
        ] > 0:

            row[
                "stage"
            ] = (
                "PET_COMPONENT_EXISTS_BUT_RAW_MISALIGNED"
            )

        elif row[
            "contours_with_pet_hot"
        ] > 0:

            row[
                "stage"
            ] = (
                "PET_HOT_EXISTS_BUT_COMPONENT_FILTER"
            )

        elif row[
            "contours_touch_gt"
        ] > 0:

            row[
                "stage"
            ] = (
                "CT_CONTOUR_HAS_GT_BUT_NO_PET_HOT"
            )

        elif row[
            "ct_contours_valid_area"
        ] > 0:

            row[
                "stage"
            ] = (
                "VALID_CT_CONTOUR_BUT_NOT_GT_OVERLAP"
            )

        elif row[
            "ct_contours_total"
        ] > 0:

            row[
                "stage"
            ] = (
                "CT_CONTOURS_EXIST_BUT_AREA_FILTER"
            )

        else:

            row[
                "stage"
            ] = (
                "NO_VALID_CT_CONTOUR"
            )

        rows.append(row)

    return rows


# ============================================================
# MAIN
# ============================================================

print("=" * 100)
print("DEV16 — Candidate Generation Stage Diagnostic")
print("=" * 100)
print()

all_rows = []


for i, case_id in enumerate(
    DEV_CASES,
    start=1
):

    print(
        f"CASE {i}/{len(DEV_CASES)}: {case_id}"
    )

    case = load_case(
        case_id
    )

    rows = diagnose_case(
        case
    )

    all_rows.extend(
        rows
    )

    print()


# ============================================================
# DataFrame
# ============================================================

df = pd.DataFrame(
    all_rows
)


# ============================================================
# Save slice-level diagnostics
# ============================================================

slice_path = (
    OUT_DIR
    / "dev16_slice_stage_diagnostic.csv"
)

df.to_csv(
    slice_path,
    index=False
)


# ============================================================
# Stage counts
# ============================================================

stage_counts = (
    df[
        "stage"
    ]
    .value_counts()
    .rename_axis(
        "stage"
    )
    .reset_index(
        name="slice_count"
    )
)


stage_path = (
    OUT_DIR
    / "dev16_stage_summary.csv"
)

stage_counts.to_csv(
    stage_path,
    index=False
)


# ============================================================
# Case summary
# ============================================================

case_summary = []

for case_id, group in df.groupby(
    "case_id"
):

    case_summary.append(
        {
            "case_id": case_id,
            "gt_positive_slices": len(group),

            "raw_candidate_hits": int(
                (
                    group["stage"]
                    == "RAW_CANDIDATE_HIT"
                ).sum()
            ),

            "no_raw_candidate": int(
                (
                    group["stage"]
                    != "RAW_CANDIDATE_HIT"
                ).sum()
            ),

            "no_body_mask": int(
                (
                    group["stage"]
                    == "NO_BODY_MASK"
                ).sum()
            ),

            "no_valid_ct_contour": int(
                (
                    group["stage"]
                    == "NO_VALID_CT_CONTOUR"
                ).sum()
            ),

            "ct_contour_area_failure": int(
                (
                    group["stage"]
                    == "CT_CONTOURS_EXIST_BUT_AREA_FILTER"
                ).sum()
            ),

            "ct_contour_not_gt": int(
                (
                    group["stage"]
                    == "VALID_CT_CONTOUR_BUT_NOT_GT_OVERLAP"
                ).sum()
            ),

            "no_pet_hot": int(
                (
                    group["stage"]
                    == "CT_CONTOUR_HAS_GT_BUT_NO_PET_HOT"
                ).sum()
            ),

            "pet_component_filter": int(
                (
                    group["stage"]
                    == "PET_HOT_EXISTS_BUT_COMPONENT_FILTER"
                ).sum()
            ),

            "pet_component_exists_raw_misaligned": int(
                (
                    group["stage"]
                    == "PET_COMPONENT_EXISTS_BUT_RAW_MISALIGNED"
                ).sum()
            ),
        }
    )


case_df = pd.DataFrame(
    case_summary
)

case_path = (
    OUT_DIR
    / "dev16_case_summary.csv"
)

case_df.to_csv(
    case_path,
    index=False
)


# ============================================================
# Print results
# ============================================================

print("=" * 100)
print("DEV16 COMPLETE")
print("=" * 100)
print()

print(
    "Stage distribution:"
)

print(
    stage_counts.to_string(
        index=False
    )
)

print()

print(
    "Case summary:"
)

print(
    case_df.to_string(
        index=False
    )
)

print()

print(
    "Output files:"
)

print(
    f"Slice diagnostic : {slice_path}"
)

print(
    f"Stage summary    : {stage_path}"
)

print(
    f"Case summary     : {case_path}"
)

print()