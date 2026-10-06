"""
DEV17 — CT Contour Recovery Diagnostic

Purpose
-------
Investigate the CT-contour localization bottleneck identified by Dev16.

Only GT-positive slices where the current Dev10 CT contour stage
does NOT overlap the GT are the primary diagnostic targets.

Variants tested:

A. DEV10_BASELINE
   adaptive threshold + 5x5 opening + area >= 100

B. NO_OPENING
   adaptive threshold + no opening + area >= 100

C. OPEN_3X3
   adaptive threshold + 3x3 opening + area >= 100

D. AREA_50
   adaptive threshold + 5x5 opening + area >= 50

E. AREA_25
   adaptive threshold + 5x5 opening + area >= 25

This is a diagnostic only.
Dev10 is NOT modified.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import cv2
import io
from contextlib import redirect_stdout


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

DEV10_PATH = SRC / "dev10_lite_hot_core.py"


# ============================================================
# Import exact Dev10 definitions
# ============================================================

sys.path.insert(
    0,
    str(SRC)
)

with redirect_stdout(io.StringIO()):
    import dev10_lite_hot_core as dev10


load_case = dev10.load_case
build_union_body_mask = dev10.build_union_body_mask
normalize_ct = dev10.normalize_ct

BASE_MIN_CONTOUR_AREA = (
    dev10.MIN_CONTOUR_AREA
)


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
# Variants
# ============================================================

VARIANTS = [
    {
        "name": "DEV10_BASELINE",
        "kernel": 5,
        "min_area": BASE_MIN_CONTOUR_AREA,
    },
    {
        "name": "NO_OPENING",
        "kernel": None,
        "min_area": BASE_MIN_CONTOUR_AREA,
    },
    {
        "name": "OPEN_3X3",
        "kernel": 3,
        "min_area": BASE_MIN_CONTOUR_AREA,
    },
    {
        "name": "AREA_50",
        "kernel": 5,
        "min_area": 50,
    },
    {
        "name": "AREA_25",
        "kernel": 5,
        "min_area": 25,
    },
]


# ============================================================
# Output
# ============================================================

OUT_DIR = (
    ROOT
    / "results"
    / "development_cases"
    / "dev17_ct_contour_recovery_results"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Build CT contour mask for one slice
# ============================================================

def build_contours(
    ct_slice,
    body_slice,
    ct_display_slice,
    kernel_size,
    min_area
):

    # --------------------------------------------------------
    # Exact Dev10 adaptive threshold
    # --------------------------------------------------------

    adaptive = cv2.adaptiveThreshold(
        ct_display_slice,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        3
    )

    # --------------------------------------------------------
    # Opening variant
    # --------------------------------------------------------

    if kernel_size is None:

        opened = adaptive

    else:

        kernel = np.ones(
            (
                kernel_size,
                kernel_size
            ),
            np.uint8
        )

        opened = cv2.morphologyEx(
            adaptive,
            cv2.MORPH_OPEN,
            kernel
        )

    # --------------------------------------------------------
    # Exact Dev10 body restriction
    # --------------------------------------------------------

    opened = np.where(
        body_slice,
        opened,
        0
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Exact Dev10 border suppression
    # --------------------------------------------------------

    opened[:5, :] = 0
    opened[-5:, :] = 0
    opened[:, :5] = 0
    opened[:, -5:] = 0

    # --------------------------------------------------------
    # External contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        opened,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = []

    for contour in contours:

        area = float(
            cv2.contourArea(
                contour
            )
        )

        if area < min_area:
            continue

        valid_contours.append(
            (
                contour,
                area
            )
        )

    return (
        opened,
        contours,
        valid_contours
    )


# ============================================================
# Diagnose one case
# ============================================================

def diagnose_case(
    case_id
):

    print()
    print("=" * 100)
    print(
        f"===== CASE: {case_id} ====="
    )
    print("=" * 100)

    case = load_case(
        case_id
    )

    ct = case["ct"]
    gt_volume = case["gt"]

    body = build_union_body_mask(
        ct
    )

    ct_display = normalize_ct(
        ct
    )

    gt_positive_slices = np.where(
        gt_volume.reshape(
            gt_volume.shape[0],
            -1
        ).any(axis=1)
    )[0]

    rows = []

    for z in gt_positive_slices:

        gt = (
            gt_volume[z] > 0
        )

        gt_area = int(
            gt.sum()
        )

        # ----------------------------------------------------
        # Determine baseline status
        # ----------------------------------------------------

        variant_results = {}

        for variant in VARIANTS:

            (
                opened,
                all_contours,
                valid_contours
            ) = build_contours(
                ct[z],
                body[z],
                ct_display[z],
                variant["kernel"],
                variant["min_area"]
            )

            total_contours = len(
                all_contours
            )

            valid_count = len(
                valid_contours
            )

            contours_touching_gt = 0
            best_overlap = 0
            best_dice = 0.0
            best_iou = 0.0
            best_area = 0.0

            union_mask = np.zeros(
                opened.shape,
                dtype=np.uint8
            )

            for (
                contour,
                area
            ) in valid_contours:

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

                overlap = int(
                    np.sum(
                        contour_bool & gt
                    )
                )

                if overlap > 0:

                    contours_touching_gt += 1

                if overlap > best_overlap:

                    best_overlap = overlap
                    best_area = float(
                        contour_bool.sum()
                    )

                    denom = (
                        best_area
                        + gt_area
                    )

                    if denom > 0:

                        best_dice = (
                            2.0
                            * overlap
                            / denom
                        )

                    union = (
                        best_area
                        + gt_area
                        - overlap
                    )

                    if union > 0:

                        best_iou = (
                            overlap
                            / union
                        )

                union_mask[
                    contour_bool
                ] = 1

            union_overlap = int(
                np.sum(
                    (
                        union_mask > 0
                    )
                    & gt
                )
            )

            variant_results[
                variant["name"]
            ] = {
                "total_contours": total_contours,
                "valid_contours": valid_count,
                "contours_touching_gt": (
                    contours_touching_gt
                ),
                "union_gt_overlap": (
                    union_overlap
                ),
                "best_overlap": (
                    best_overlap
                ),
                "best_dice": (
                    best_dice
                ),
                "best_iou": (
                    best_iou
                ),
                "best_contour_area": (
                    best_area
                ),
            }

        # ----------------------------------------------------
        # Record row
        # ----------------------------------------------------

        row = {
            "case_id": case_id,
            "slice": int(z),
            "gt_area": gt_area,
        }

        for variant_name, result in (
            variant_results.items()
        ):

            prefix = (
                variant_name.lower()
            )

            row[
                f"{prefix}_total_contours"
            ] = result[
                "total_contours"
            ]

            row[
                f"{prefix}_valid_contours"
            ] = result[
                "valid_contours"
            ]

            row[
                f"{prefix}_touching_gt"
            ] = result[
                "contours_touching_gt"
            ]

            row[
                f"{prefix}_union_gt_overlap"
            ] = result[
                "union_gt_overlap"
            ]

            row[
                f"{prefix}_best_overlap"
            ] = result[
                "best_overlap"
            ]

            row[
                f"{prefix}_best_dice"
            ] = result[
                "best_dice"
            ]

            row[
                f"{prefix}_best_iou"
            ] = result[
                "best_iou"
            ]

            row[
                f"{prefix}_best_contour_area"
            ] = result[
                "best_contour_area"
            ]

        # ----------------------------------------------------
        # Recovery classification
        # ----------------------------------------------------

        baseline_hit = (
            variant_results[
                "DEV10_BASELINE"
            ]["union_gt_overlap"] > 0
        )

        recovered_variants = []

        for variant_name in (
            [
                "NO_OPENING",
                "OPEN_3X3",
                "AREA_50",
                "AREA_25",
            ]
        ):

            recovered = (
                variant_results[
                    variant_name
                ]["union_gt_overlap"] > 0
            )

            if (
                not baseline_hit
                and recovered
            ):
                recovered_variants.append(
                    variant_name
                )

        row[
            "baseline_hit"
        ] = baseline_hit

        row[
            "recovered_by_any_variant"
        ] = bool(
            len(
                recovered_variants
            ) > 0
        )

        row[
            "recovery_variants"
        ] = ",".join(
            recovered_variants
        )

        rows.append(
            row
        )

    return rows


# ============================================================
# MAIN
# ============================================================

print("=" * 100)
print("DEV17 — CT Contour Recovery Diagnostic")
print("=" * 100)
print()

print(
    "Dev10 baseline contour area:",
    BASE_MIN_CONTOUR_AREA
)

print()

all_rows = []

for case_id in DEV_CASES:

    rows = diagnose_case(
        case_id
    )

    all_rows.extend(
        rows
    )


# ============================================================
# DataFrame
# ============================================================

df = pd.DataFrame(
    all_rows
)


# ============================================================
# Save all slice results
# ============================================================

slice_path = (
    OUT_DIR
    / "dev17_slice_contour_recovery.csv"
)

df.to_csv(
    slice_path,
    index=False
)


# ============================================================
# Analyze only baseline failures
# ============================================================

baseline_failures = df[
    df["baseline_hit"] == False
].copy()


failure_path = (
    OUT_DIR
    / "dev17_baseline_failures.csv"
)

baseline_failures.to_csv(
    failure_path,
    index=False
)


# ============================================================
# Variant recovery summary
# ============================================================

summary_rows = []

total_baseline_failures = len(
    baseline_failures
)

for variant_name in (
    [
        "NO_OPENING",
        "OPEN_3X3",
        "AREA_50",
        "AREA_25",
    ]
):

    col = (
        variant_name.lower()
        + "_union_gt_overlap"
    )

    recovered = int(
        (
            baseline_failures[col]
            > 0
        ).sum()
    )

    recovery_rate = (
        recovered
        / total_baseline_failures
        if total_baseline_failures > 0
        else 0.0
    )

    summary_rows.append(
        {
            "variant": variant_name,
            "baseline_failed_slices": (
                total_baseline_failures
            ),
            "recovered_slices": recovered,
            "recovery_rate": recovery_rate,
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)

summary_path = (
    OUT_DIR
    / "dev17_variant_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# Case-level summary
# ============================================================

case_rows = []

for case_id, group in df.groupby(
    "case_id"
):

    baseline_fail = group[
        group["baseline_hit"] == False
    ]

    row = {
        "case_id": case_id,
        "gt_positive_slices": len(group),
        "baseline_hits": int(
            group["baseline_hit"].sum()
        ),
        "baseline_failures": len(
            baseline_fail
        ),
    }

    for variant_name in (
        [
            "NO_OPENING",
            "OPEN_3X3",
            "AREA_50",
            "AREA_25",
        ]
    ):

        col = (
            variant_name.lower()
            + "_union_gt_overlap"
        )

        row[
            variant_name
            + "_recovered"
        ] = int(
            (
                baseline_fail[col]
                > 0
            ).sum()
        )

    case_rows.append(
        row
    )


case_df = pd.DataFrame(
    case_rows
)

case_path = (
    OUT_DIR
    / "dev17_case_summary.csv"
)

case_df.to_csv(
    case_path,
    index=False
)


# ============================================================
# Print results
# ============================================================

print()
print("=" * 100)
print("DEV17 COMPLETE")
print("=" * 100)
print()

print(
    f"GT-positive slices: {len(df)}"
)

print(
    f"Dev10 baseline failures: "
    f"{total_baseline_failures}"
)

print()

print(
    "Variant recovery:"
)

print(
    summary_df.to_string(
        index=False
    )
)

print()

print(
    "Case-level recovery:"
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
    f"All slices     : {slice_path}"
)

print(
    f"Baseline fails : {failure_path}"
)

print(
    f"Variant summary: {summary_path}"
)

print(
    f"Case summary   : {case_path}"
)

print()