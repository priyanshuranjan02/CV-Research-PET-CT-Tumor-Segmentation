from pathlib import Path
import runpy
import numpy as np
import pandas as pd


ns = runpy.run_path(
    "src/v4_validate_case7.py",
    run_name="__case7_union_feature_audit__"
)

prepare_case7 = ns["prepare_case7"]

prepared, ct_img, ct, original_body = prepare_case7()

candidate_masks = prepared["candidate_masks"]
reference = prepared["reference"]
suv = prepared["suv"]
gt = prepared["gt"]


# =========================================================
# UNION BODY
# =========================================================

body = np.zeros_like(
    ct,
    dtype=bool
)

import cv2

kernel = np.ones(
    (11, 11),
    np.uint8
)

for z in range(ct.shape[0]):

    binary = (
        ct[z] > -900
    ).astype(np.uint8) * 255

    closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        cv2.drawContours(
            body[z].astype(np.uint8),
            [contour],
            -1,
            1,
            -1
        )

# The above drawing cannot modify the boolean temporary safely,
# so rebuild the union explicitly.
body = np.zeros_like(ct, dtype=np.uint8)

for z in range(ct.shape[0]):

    binary = (
        ct[z] > -900
    ).astype(np.uint8) * 255

    closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        cv2.drawContours(
            body[z],
            [contour],
            -1,
            1,
            -1
        )

body = body.astype(bool)


# =========================================================
# RE-GENERATE CANDIDATES USING UNION BODY
# =========================================================

ct_display = ns["normalize_ct"](ct)

opening_kernel = np.ones(
    (5, 5),
    np.uint8
)

rows = []

GT_SLICES = np.where(
    np.any(
        gt,
        axis=(1, 2)
    )
)[0]


for z in range(ct.shape[0]):

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
    ).astype(np.uint8)

    opened[:5, :] = 0
    opened[-5:, :] = 0
    opened[:, :5] = 0
    opened[:, -5:] = 0

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

        if area < 100:
            continue

        if area > 10000:
            # Keep oversized contours for diagnosis too.
            pass

        mask = np.zeros(
            ct.shape[1:],
            dtype=np.uint8
        )

        cv2.drawContours(
            mask,
            [contour],
            -1,
            1,
            -1
        )

        pixels = mask > 0

        vals = suv[z][pixels]

        vals = vals[
            np.isfinite(vals)
        ]

        if vals.size == 0:
            continue

        if gt[z].any():

            overlap = int(
                np.logical_and(
                    pixels,
                    gt[z]
                ).sum()
            )

            gt_area = int(
                gt[z].sum()
            )

            dice = (
                2.0 * overlap
                / (
                    int(pixels.sum())
                    + gt_area
                    + 1e-9
                )
            )

        else:

            overlap = 0
            dice = 0.0

        rows.append(
            {
                "slice": z,
                "area": int(pixels.sum()),
                "overlap_gt": overlap,
                "candidate_gt_dice": dice,

                "median_suv":
                    float(np.median(vals)),
                "mean_suv":
                    float(np.mean(vals)),
                "max_suv":
                    float(np.max(vals)),
                "p90_suv":
                    float(np.percentile(vals, 90)),
                "p95_suv":
                    float(np.percentile(vals, 95)),

                "frac_ge_1.0":
                    float(np.mean(vals >= 1.0)),
                "frac_ge_1.5":
                    float(np.mean(vals >= 1.5)),
                "frac_ge_2.25":
                    float(np.mean(vals >= 2.25)),
            }
        )


df = pd.DataFrame(rows)


# =========================================================
# ONLY GT-POSITIVE SLICES
# =========================================================

d = df[
    df["slice"].isin(
        GT_SLICES
    )
].copy()


print("\n" + "=" * 100)
print("CASE 7 — UNION-BODY CANDIDATE FEATURE AUDIT")
print("=" * 100)

print(
    "Total union-body candidates:",
    len(df)
)

print(
    "Candidates on GT-positive slices:",
    len(d)
)


# =========================================================
# BEST CANDIDATE PER GT SLICE
# =========================================================

best_overlap = (
    d.sort_values(
        [
            "slice",
            "overlap_gt",
            "candidate_gt_dice"
        ],
        ascending=[
            True,
            False,
            False
        ]
    )
    .groupby(
        "slice",
        as_index=False
    )
    .first()
)

print("\n" + "=" * 100)
print("BEST GEOMETRIC CANDIDATE PER GT SLICE")
print("=" * 100)

print(
    best_overlap.to_string(
        index=False
    )
)


# =========================================================
# BEST PET CANDIDATE PER GT SLICE
# =========================================================

best_pet = (
    d.sort_values(
        [
            "slice",
            "median_suv"
        ],
        ascending=[
            True,
            False
        ]
    )
    .groupby(
        "slice",
        as_index=False
    )
    .first()
)

print("\n" + "=" * 100)
print("BEST PET-MEDIAN CANDIDATE PER GT SLICE")
print("=" * 100)

print(
    best_pet[
        [
            "slice",
            "area",
            "overlap_gt",
            "candidate_gt_dice",
            "median_suv",
            "mean_suv",
            "max_suv",
            "p90_suv",
            "p95_suv",
            "frac_ge_1.5",
            "frac_ge_2.25"
        ]
    ].to_string(
        index=False
    )
)


# =========================================================
# TOP GEOMETRIC OVERLAP CANDIDATES
# =========================================================

print("\n" + "=" * 100)
print("TOP 20 GT-OVERLAPPING CANDIDATES")
print("=" * 100)

print(
    d.sort_values(
        [
            "candidate_gt_dice",
            "overlap_gt"
        ],
        ascending=[
            False,
            False
        ]
    )
    .head(20)
    .to_string(
        index=False
    )
)


# =========================================================
# THRESHOLD DIAGNOSTICS
# =========================================================

print("\n" + "=" * 100)
print("FEATURE PASS COUNTS ON GT-POSITIVE SLICES")
print("=" * 100)

for threshold in [
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    2.00
]:

    print(
        f"median SUV >= {threshold:.2f}:",
        int(
            (d["median_suv"] >= threshold).sum()
        )
    )

for threshold in [
    0.25,
    0.40,
    0.50,
    0.75,
    1.00
]:

    print(
        f"mean-difference proxy >= {threshold:.2f}:",
        int(
            (
                (
                    d["mean_suv"]
                    - d["median_suv"]
                )
                >= threshold
            ).sum()
        )
    )


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

print(
    "GT-positive slices:",
    len(GT_SLICES)
)

print(
    "Slices with ANY candidate overlap:",
    int(
        (best_overlap["overlap_gt"] > 0).sum()
    )
)

print(
    "Best candidate/GT Dice:",
    f"{best_overlap['candidate_gt_dice'].max():.4f}"
)

print(
    "Highest median SUV among candidates:",
    f"{d['median_suv'].max():.6f}"
)

print(
    "Highest mean SUV among candidates:",
    f"{d['mean_suv'].max():.6f}"
)

print(
    "Highest P95 SUV among candidates:",
    f"{d['p95_suv'].max():.4f}"
)

print(
    "Highest fraction SUV >=1.5:",
    f"{d['frac_ge_1.5'].max():.4f}"
)

print(
    "Highest fraction SUV >=2.25:",
    f"{d['frac_ge_2.25'].max():.4f}"
)


# =========================================================
# SAVE
# =========================================================

out = (
    Path("validation_results")
    / "PETCT_0011f3deaf"
    / "case7_union_candidate_feature_audit.csv"
)

out.parent.mkdir(
    parents=True,
    exist_ok=True
)

d.to_csv(
    out,
    index=False
)

print(
    "\nSaved:",
    out
)

print(
    "\n===== UNION FEATURE AUDIT COMPLETE ====="
)
