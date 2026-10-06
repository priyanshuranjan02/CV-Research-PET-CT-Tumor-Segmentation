from pathlib import Path
import runpy
import numpy as np
import cv2
import pandas as pd


ns = runpy.run_path(
    "src/v4_validate_case7.py",
    run_name="__case7_body_union_audit__"
)

prepare_case7 = ns["prepare_case7"]

prepared, ct_img, ct, current_body = prepare_case7()

gt = prepared["gt"]


# ---------------------------------------------------------
# Build diagnostic body mask:
# UNION of all connected components after the existing
# CT > -900 + closing operation.
#
# This is an ABLATION only.
# It does NOT modify the frozen pipeline.
# ---------------------------------------------------------

union_body = np.zeros_like(
    ct,
    dtype=bool
)

kernel = np.ones(
    (11, 11),
    np.uint8
)


rows = []


gt_slices = np.where(
    np.any(
        gt,
        axis=(1, 2)
    )
)[0]


for z in gt_slices:

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

    slice_union = np.zeros(
        ct.shape[1:],
        dtype=np.uint8
    )

    for contour in contours:

        cv2.drawContours(
            slice_union,
            [contour],
            -1,
            1,
            -1
        )

    union_body[z] = (
        slice_union > 0
    )

    gt_slice = gt[z]

    gt_area = int(
        gt_slice.sum()
    )

    current_inside = int(
        np.logical_and(
            current_body[z],
            gt_slice
        ).sum()
    )

    union_inside = int(
        np.logical_and(
            union_body[z],
            gt_slice
        ).sum()
    )

    rows.append(
        {
            "slice": int(z),
            "gt_area": gt_area,
            "current_body_overlap": current_inside,
            "current_body_coverage_pct":
                100.0
                * current_inside
                / gt_area,

            "union_body_overlap": union_inside,
            "union_body_coverage_pct":
                100.0
                * union_inside
                / gt_area,

            "newly_recovered_voxels":
                union_inside - current_inside
        }
    )


df = pd.DataFrame(rows)


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
    220
)


print("\n" + "=" * 90)
print("CASE 7 — BODY UNION ABLATION")
print("=" * 90)

print(
    df.to_string(
        index=False
    )
)


print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

total_gt = int(
    gt.sum()
)

current_overlap = int(
    np.logical_and(
        current_body,
        gt
    ).sum()
)

union_overlap = int(
    np.logical_and(
        union_body,
        gt
    ).sum()
)

print(
    "Total GT voxels:",
    total_gt
)

print(
    "Current largest-component body coverage:",
    f"{100.0 * current_overlap / total_gt:.2f}%"
)

print(
    "All-components union body coverage:",
    f"{100.0 * union_overlap / total_gt:.2f}%"
)

print(
    "Additional GT voxels recovered:",
    union_overlap - current_overlap
)

print(
    "Slices with current coverage <95%:",
    int(
        (df["current_body_coverage_pct"] < 95).sum()
    )
)

print(
    "Slices with union coverage <95%:",
    int(
        (df["union_body_coverage_pct"] < 95).sum()
    )
)

print(
    "\nPreviously problematic slices:"
)

print(
    df[
        df["current_body_coverage_pct"] < 95
    ].to_string(
        index=False
    )
)


out = (
    Path("validation_results")
    / "PETCT_0011f3deaf"
    / "case7_body_union_audit.csv"
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
    "\n===== BODY UNION ABLATION COMPLETE ====="
)
