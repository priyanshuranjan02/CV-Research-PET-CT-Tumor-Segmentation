from pathlib import Path
import runpy
import numpy as np
import pandas as pd
import SimpleITK as sitk


ROOT = Path(".")


print("\n" + "=" * 80)
print("CASE 7 — CT CANDIDATE vs GT GEOMETRY AUDIT")
print("=" * 80)


# ---------------------------------------------------------
# Load the existing validator.
# ---------------------------------------------------------

ns = runpy.run_path(
    "src/v4_validate_case7.py",
    run_name="__case7_audit__"
)

prepare_case7 = ns["prepare_case7"]


# ---------------------------------------------------------
# prepare_case7() returns:
#
#   prepared, ct_img, ct, body
# ---------------------------------------------------------

prepared, ct_img, ct, body = (
    prepare_case7()
)

candidate_masks = prepared[
    "candidate_masks"
]

reference = prepared[
    "reference"
]

gt = prepared[
    "gt"
]


print("\nCandidates:", len(candidate_masks))
print("Reference rows:", len(reference))

if len(candidate_masks) != len(reference):
    raise RuntimeError(
        f"Candidate/reference mismatch: "
        f"{len(candidate_masks)} vs {len(reference)}"
    )


# ---------------------------------------------------------
# GT-positive slices.
# ---------------------------------------------------------

gt_positive_slices = np.where(
    np.any(
        gt,
        axis=(1, 2)
    )
)[0]


# ---------------------------------------------------------
# Candidate vs GT overlap.
# ---------------------------------------------------------

rows = []


for i in range(len(reference)):

    row = reference.iloc[i]

    z = int(
        row["slice"]
    )

    if z < 0 or z >= gt.shape[0]:
        continue

    if not gt[z].any():
        continue

    z_from_mask, candidate_mask, _ = candidate_masks[i]

    candidate_mask = (
        np.asarray(
            candidate_mask
        ) > 0
    )

    gt_slice = gt[z]

    candidate_area = int(
        candidate_mask.sum()
    )

    gt_voxels = int(
        gt_slice.sum()
    )

    overlap = int(
        np.logical_and(
            candidate_mask,
            gt_slice
        ).sum()
    )

    union = int(
        np.logical_or(
            candidate_mask,
            gt_slice
        ).sum()
    )

    dice = (
        2.0 * overlap
        / (
            candidate_area
            + gt_voxels
            + 1e-9
        )
    )

    iou = (
        overlap
        / (
            union
            + 1e-9
        )
    )

    rows.append(
        {
            "slice": z,
            "gt_voxels": gt_voxels,
            "candidate_area": candidate_area,
            "overlap_voxels": overlap,
            "candidate_gt_dice": dice,
            "candidate_gt_iou": iou,
            "candidate_suv": float(
                row["candidate_suv"]
            ),
            "background_suv": float(
                row["background_suv"]
            ),
            "local_difference": float(
                row["local_difference"]
            ),
            "local_ratio": float(
                row["local_ratio"]
            ),
            "accepted": int(
                row["accepted"]
            ),
        }
    )


df = pd.DataFrame(rows)


if df.empty:
    raise RuntimeError(
        "No candidates found on GT-positive slices."
    )


# ---------------------------------------------------------
# Best candidate for each GT-positive slice.
# ---------------------------------------------------------

best = (
    df.sort_values(
        [
            "slice",
            "overlap_voxels",
            "candidate_gt_dice",
            "candidate_suv"
        ],
        ascending=[
            True,
            False,
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


# ---------------------------------------------------------
# Full table.
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("BEST CT CANDIDATE PER GT-POSITIVE SLICE")
print("=" * 80)

pd.set_option(
    "display.max_rows",
    200
)

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    220
)

print(
    best[
        [
            "slice",
            "gt_voxels",
            "candidate_area",
            "overlap_voxels",
            "candidate_gt_dice",
            "candidate_gt_iou",
            "candidate_suv",
            "background_suv",
            "local_difference",
            "local_ratio",
            "accepted"
        ]
    ].to_string(
        index=False
    )
)


# ---------------------------------------------------------
# Summary.
# ---------------------------------------------------------

covered = int(
    (
        best["overlap_voxels"]
        > 0
    ).sum()
)

dice_positive = int(
    (
        best["candidate_gt_dice"]
        > 0
    ).sum()
)

dice_010 = int(
    (
        best["candidate_gt_dice"]
        >= 0.10
    ).sum()
)

dice_025 = int(
    (
        best["candidate_gt_dice"]
        >= 0.25
    ).sum()
)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    "GT-positive slices:",
    len(gt_positive_slices)
)

print(
    "GT slices with ANY candidate overlap:",
    covered,
    "/",
    len(gt_positive_slices)
)

print(
    "GT slices with candidate Dice > 0:",
    dice_positive,
    "/",
    len(gt_positive_slices)
)

print(
    "GT slices with candidate Dice >= 0.10:",
    dice_010,
    "/",
    len(gt_positive_slices)
)

print(
    "GT slices with candidate Dice >= 0.25:",
    dice_025,
    "/",
    len(gt_positive_slices)
)

print(
    "Best candidate/GT Dice:",
    f"{best['candidate_gt_dice'].max():.6f}"
)

print(
    "Best candidate/GT IoU:",
    f"{best['candidate_gt_iou'].max():.6f}"
)

print(
    "Maximum candidate SUV on GT slices:",
    f"{best['candidate_suv'].max():.6f}"
)

print(
    "Maximum local difference on GT slices:",
    f"{best['local_difference'].max():.6f}"
)


# ---------------------------------------------------------
# Strong candidates.
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("GT SLICES WITH CANDIDATE DICE >= 0.10")
print("=" * 80)

strong = best[
    best["candidate_gt_dice"] >= 0.10
]

if strong.empty:
    print("None")
else:
    print(
        strong[
            [
                "slice",
                "gt_voxels",
                "candidate_area",
                "overlap_voxels",
                "candidate_gt_dice",
                "candidate_suv",
                "background_suv",
                "local_difference",
                "local_ratio"
            ]
        ].to_string(
            index=False
        )
    )


# ---------------------------------------------------------
# Save.
# ---------------------------------------------------------

out_dir = (
    ROOT
    / "validation_results"
    / "PETCT_0011f3deaf"
)

out_dir.mkdir(
    parents=True,
    exist_ok=True
)

out_path = (
    out_dir
    / "case7_candidate_gt_overlap_audit.csv"
)

best.to_csv(
    out_path,
    index=False
)

print(
    "\nSaved:",
    out_path
)

print(
    "\n===== GEOMETRY AUDIT COMPLETE ====="
)
