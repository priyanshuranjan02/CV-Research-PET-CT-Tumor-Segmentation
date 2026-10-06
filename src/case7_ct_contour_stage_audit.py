from pathlib import Path
import runpy
import numpy as np
import pandas as pd
import cv2


# ---------------------------------------------------------
# Load the exact Case 7 preparation utilities.
# ---------------------------------------------------------

ns = runpy.run_path(
    "src/v4_validate_case7.py",
    run_name="__case7_stage_audit__"
)

prepare_case7 = ns["prepare_case7"]
normalize_ct = ns["normalize_ct"]


prepared, ct_img, ct, body = prepare_case7()

gt = prepared["gt"]

ct_display = normalize_ct(ct)

opening_kernel = np.ones(
    (5, 5),
    np.uint8
)


# ---------------------------------------------------------
# Analyze every stage of CT contour generation.
# ---------------------------------------------------------

GT_SLICES = np.where(
    np.any(
        gt,
        axis=(1, 2)
    )
)[0]


rows = []


for z in GT_SLICES:

    gt_slice = gt[z]
    gt_area = int(gt_slice.sum())

    # =====================================================
    # STAGE 1 — RAW ADAPTIVE THRESHOLD
    # =====================================================

    adaptive = cv2.adaptiveThreshold(
        ct_display[z],
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        3
    )

    contours_raw, _ = cv2.findContours(
        adaptive,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    raw_best_overlap = 0
    raw_best_area = 0

    for contour in contours_raw:

        mask = np.zeros(
            gt_slice.shape,
            dtype=np.uint8
        )

        cv2.drawContours(
            mask,
            [contour],
            -1,
            1,
            -1
        )

        overlap = int(
            np.logical_and(
                mask > 0,
                gt_slice
            ).sum()
        )

        if overlap > raw_best_overlap:

            raw_best_overlap = overlap
            raw_best_area = int(mask.sum())


    # =====================================================
    # STAGE 2 — MORPHOLOGICAL OPENING
    # =====================================================

    opened = cv2.morphologyEx(
        adaptive,
        cv2.MORPH_OPEN,
        opening_kernel
    )

    contours_open, _ = cv2.findContours(
        opened,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    open_best_overlap = 0
    open_best_area = 0

    for contour in contours_open:

        mask = np.zeros(
            gt_slice.shape,
            dtype=np.uint8
        )

        cv2.drawContours(
            mask,
            [contour],
            -1,
            1,
            -1
        )

        overlap = int(
            np.logical_and(
                mask > 0,
                gt_slice
            ).sum()
        )

        if overlap > open_best_overlap:

            open_best_overlap = overlap
            open_best_area = int(mask.sum())


    # =====================================================
    # STAGE 3 — BODY MASK + BORDER REMOVAL
    # =====================================================

    filtered = np.where(
        body[z],
        opened,
        0
    ).astype(np.uint8)

    filtered[:5, :] = 0
    filtered[-5:, :] = 0
    filtered[:, :5] = 0
    filtered[:, -5:] = 0

    filtered[:, :5] = 0
    filtered[:, -5:] = 0

    contours_filtered, _ = cv2.findContours(
        filtered,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filtered_best_overlap = 0
    filtered_best_area = 0
    filtered_best_dice = 0.0

    eligible_count = 0

    for contour in contours_filtered:

        mask = np.zeros(
            gt_slice.shape,
            dtype=np.uint8
        )

        cv2.drawContours(
            mask,
            [contour],
            -1,
            1,
            -1
        )

        area = int(mask.sum())

        overlap = int(
            np.logical_and(
                mask > 0,
                gt_slice
            ).sum()
        )

        dice = (
            2.0 * overlap
            / (
                area
                + gt_area
                + 1e-9
            )
        )

        if (
            area >= 100
            and area <= 10000
        ):
            eligible_count += 1

        if dice > filtered_best_dice:

            filtered_best_dice = dice
            filtered_best_overlap = overlap
            filtered_best_area = area


    # =====================================================
    # RECORD
    # =====================================================

    rows.append(
        {
            "slice": int(z),
            "gt_area": gt_area,

            "raw_contours":
                len(contours_raw),
            "raw_best_overlap":
                raw_best_overlap,
            "raw_best_area":
                raw_best_area,

            "opened_contours":
                len(contours_open),
            "opened_best_overlap":
                open_best_overlap,
            "opened_best_area":
                open_best_area,

            "filtered_contours":
                len(contours_filtered),
            "filtered_best_overlap":
                filtered_best_overlap,
            "filtered_best_area":
                filtered_best_area,
            "filtered_best_dice":
                filtered_best_dice,

            "eligible_area_100_10000":
                eligible_count,
        }
    )


df = pd.DataFrame(rows)


print("\n" + "=" * 100)
print("CASE 7 — CT CONTOUR STAGE AUDIT")
print("=" * 100)

pd.set_option(
    "display.max_rows",
    100
)

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    240
)

print(
    df.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# Summary.
# ---------------------------------------------------------

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

print(
    "GT-positive slices:",
    len(df)
)

print(
    "Slices with raw adaptive contour overlap:",
    int(
        (df["raw_best_overlap"] > 0).sum()
    )
)

print(
    "Slices with overlap after opening:",
    int(
        (df["opened_best_overlap"] > 0).sum()
    )
)

print(
    "Slices with overlap after body/border filtering:",
    int(
        (df["filtered_best_overlap"] > 0).sum()
    )
)

print(
    "Best raw contour overlap:",
    int(
        df["raw_best_overlap"].max()
    )
)

print(
    "Best opened contour overlap:",
    int(
        df["opened_best_overlap"].max()
    )
)

print(
    "Best filtered contour Dice:",
    f"{df['filtered_best_dice'].max():.4f}"
)


# ---------------------------------------------------------
# Save.
# ---------------------------------------------------------

out = (
    Path("validation_results")
    / "PETCT_0011f3deaf"
    / "case7_ct_contour_stage_audit.csv"
)

out.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    out,
    index=False
)

print(
    "\nSaved:",
    out
)

print(
    "\n===== CT STAGE AUDIT COMPLETE ====="
)
