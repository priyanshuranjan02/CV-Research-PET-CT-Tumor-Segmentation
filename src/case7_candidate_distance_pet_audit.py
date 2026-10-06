from pathlib import Path
import runpy
import numpy as np
import pandas as pd


ROOT = Path(".")

ns = runpy.run_path(
    "src/v4_validate_case7.py",
    run_name="__case7_audit__"
)

prepare_case7 = ns["prepare_case7"]

prepared, ct_img, ct, body = prepare_case7()

candidate_masks = prepared["candidate_masks"]
reference = prepared["reference"]
suv = prepared["suv"]
gt = prepared["gt"]

gt_slices = np.where(
    np.any(gt, axis=(1, 2))
)[0]

rows = []

for z in gt_slices:

    gt_slice = gt[z]

    ys_gt, xs_gt = np.where(gt_slice)

    gt_cx = float(xs_gt.mean())
    gt_cy = float(ys_gt.mean())

    slice_candidates = []

    for i, (cz, candidate_mask, _) in enumerate(candidate_masks):

        if int(cz) != int(z):
            continue

        mask = np.asarray(candidate_mask) > 0

        if not np.any(mask):
            continue

        ys, xs = np.where(mask)

        cx = float(xs.mean())
        cy = float(ys.mean())

        centroid_distance = float(
            np.hypot(
                cx - gt_cx,
                cy - gt_cy
            )
        )

        overlap = int(
            np.logical_and(
                mask,
                gt_slice
            ).sum()
        )

        area = int(mask.sum())

        vals = suv[z][mask]
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            continue

        overlap_vals = suv[z][
            np.logical_and(
                mask,
                gt_slice
            )
        ]

        overlap_vals = overlap_vals[
            np.isfinite(overlap_vals)
        ]

        slice_candidates.append(
            {
                "slice": int(z),
                "candidate_index": int(i),
                "area": area,
                "overlap": overlap,
                "centroid_distance": centroid_distance,
                "candidate_median_suv": float(np.median(vals)),
                "candidate_mean_suv": float(np.mean(vals)),
                "candidate_max_suv": float(np.max(vals)),
                "candidate_p90_suv": float(np.percentile(vals, 90)),
                "candidate_p95_suv": float(np.percentile(vals, 95)),
                "frac_suv_ge_1.5": float(np.mean(vals >= 1.5)),
                "frac_suv_ge_2.25": float(np.mean(vals >= 2.25)),
                "overlap_median_suv": (
                    float(np.median(overlap_vals))
                    if overlap_vals.size > 0
                    else 0.0
                ),
            }
        )

    if not slice_candidates:
        continue

    cdf = pd.DataFrame(slice_candidates)

    # Best geometrical candidate:
    # maximum overlap first, then Dice-like overlap ratio.
    cdf["dice"] = (
        2.0 * cdf["overlap"]
        / (
            cdf["area"]
            + len(xs_gt)
            + 1e-9
        )
    )

    best_overlap = cdf.sort_values(
        ["overlap", "dice"],
        ascending=[False, False]
    ).iloc[0]

    # Closest candidate to GT centroid.
    closest = cdf.sort_values(
        "centroid_distance"
    ).iloc[0]

    rows.append(
        {
            "slice": int(z),
            "gt_area": len(xs_gt),

            "best_overlap":
                int(best_overlap["overlap"]),
            "best_overlap_dice":
                float(best_overlap["dice"]),
            "best_overlap_area":
                int(best_overlap["area"]),
            "best_overlap_centroid_distance":
                float(best_overlap["centroid_distance"]),
            "best_overlap_median_suv":
                float(best_overlap["candidate_median_suv"]),
            "best_overlap_mean_suv":
                float(best_overlap["candidate_mean_suv"]),
            "best_overlap_max_suv":
                float(best_overlap["candidate_max_suv"]),
            "best_overlap_p90_suv":
                float(best_overlap["candidate_p90_suv"]),
            "best_overlap_p95_suv":
                float(best_overlap["candidate_p95_suv"]),
            "best_overlap_frac_ge_1.5":
                float(best_overlap["frac_suv_ge_1.5"]),
            "best_overlap_frac_ge_2.25":
                float(best_overlap["frac_suv_ge_2.25"]),
            "best_overlap_region_median_suv":
                float(best_overlap["overlap_median_suv"]),

            "closest_distance":
                float(closest["centroid_distance"]),
            "closest_overlap":
                int(closest["overlap"]),
            "closest_area":
                int(closest["area"]),
            "closest_median_suv":
                float(closest["candidate_median_suv"]),
            "closest_max_suv":
                float(closest["candidate_max_suv"]),
        }
    )


df = pd.DataFrame(rows)

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)

print("\n" + "=" * 90)
print("CASE 7 — CANDIDATE DISTANCE + PET DILUTION AUDIT")
print("=" * 90)

print("\nPer GT-positive slice:")
print(df.to_string(index=False))

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

print(
    "GT-positive slices:",
    len(gt_slices)
)

print(
    "Slices with candidate overlap:",
    int((df["best_overlap"] > 0).sum()),
    "/",
    len(gt_slices)
)

print(
    "Median closest-candidate distance:",
    f"{df['closest_distance'].median():.2f} px"
)

print(
    "Maximum closest-candidate overlap:",
    int(df["closest_overlap"].max())
)

print(
    "Best geometric candidate Dice:",
    f"{df['best_overlap_dice'].max():.4f}"
)

print(
    "Best-overlap candidate max SUV:",
    f"{df['best_overlap_max_suv'].max():.4f}"
)

print(
    "Best-overlap candidate median SUV:",
    f"{df['best_overlap_median_suv'].max():.6f}"
)

print(
    "Best-overlap candidate P95 SUV:",
    f"{df['best_overlap_p95_suv'].max():.4f}"
)

print(
    "Best-overlap candidate fraction SUV >= 1.5:",
    f"{df['best_overlap_frac_ge_1.5'].max():.4f}"
)

out = (
    ROOT
    / "validation_results"
    / "PETCT_0011f3deaf"
    / "case7_candidate_distance_pet_audit.csv"
)

out.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    out,
    index=False
)

print("\nSaved:", out)
print("\n===== AUDIT COMPLETE =====")
