from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parent.parent

CSV = (
    ROOT
    / "validation_results"
    / "PETCT_0011f3deaf"
    / "case7_all_contours_pet_split_3d_components.csv"
)

GT_VOXELS = 9206

THRESHOLD = 2.25

# 3D volume caps to test.
VOLUME_CAPS = [
    1000,
    2000,
    3000,
    5000,
    7500,
    10000,
    15000,
    25000,
    50000,
    100000,
    200000,
]

SCORE_FEATURES = [
    "mean_suv",
    "median_suv",
    "max_suv",
    "suv_concentration",
]


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(CSV)

df = df[
    df["threshold"] == THRESHOLD
].copy()

df = df[
    df["kept"] == 1
].copy()


# ============================================================
# DERIVED FEATURE
# ============================================================

df["suv_concentration"] = (
    df["mean_suv"]
    * df["median_suv"]
)


# ============================================================
# METRICS FOR UNION OF SELECTED COMPONENTS
#
# Connected components are mutually disjoint, so component
# voxel counts and GT overlaps can be summed.
# ============================================================

def union_metrics(selected):

    if selected.empty:

        return {
            "components": 0,
            "voxels": 0,
            "overlap": 0,
            "dice": 0.0,
            "iou": 0.0,
        }

    voxels = int(
        selected["voxels"].sum()
    )

    overlap = int(
        selected["overlap_gt"].sum()
    )

    dice = (
        2.0
        * overlap
        / (
            voxels
            + GT_VOXELS
        )
    )

    union = (
        voxels
        + GT_VOXELS
        - overlap
    )

    iou = (
        overlap
        / union
        if union > 0
        else 0.0
    )

    return {
        "components": len(selected),
        "voxels": voxels,
        "overlap": overlap,
        "dice": float(dice),
        "iou": float(iou),
    }


# ============================================================
# AUDIT
# ============================================================

print()
print("=" * 110)
print("===== CASE 7 — SIZE-CONSTRAINED 3D COMPONENT SELECTION =====")
print("=" * 110)

print()
print("Threshold:", THRESHOLD)
print("Initial kept components:", len(df))
print("GT voxels:", GT_VOXELS)


results = []


for score_feature in SCORE_FEATURES:

    print()
    print("-" * 110)
    print(
        f"===== SCORE FEATURE: {score_feature.upper()} ====="
    )
    print("-" * 110)

    for cap in VOLUME_CAPS:

        eligible = df[
            df["voxels"] <= cap
        ].copy()

        if eligible.empty:

            print(
                f"Cap {cap:7d}: "
                f"NO ELIGIBLE COMPONENT"
            )

            continue

        ranked = eligible.sort_values(
            [
                score_feature,
                "median_suv",
                "mean_suv",
            ],
            ascending=[
                False,
                False,
                False,
            ]
        )

        selected = ranked.head(
            1
        )

        metrics = union_metrics(
            selected
        )

        row = {
            "score_feature":
                score_feature,
            "volume_cap":
                cap,
            "eligible_components":
                len(eligible),
            "selected_label":
                int(
                    selected.iloc[0]["label"]
                ),
            "selected_voxels":
                int(
                    selected.iloc[0]["voxels"]
                ),
            "selected_z_start":
                int(
                    selected.iloc[0]["z_start"]
                ),
            "selected_z_end":
                int(
                    selected.iloc[0]["z_end"]
                ),
            "selected_mean_suv":
                float(
                    selected.iloc[0]["mean_suv"]
                ),
            "selected_median_suv":
                float(
                    selected.iloc[0]["median_suv"]
                ),
            "selected_max_suv":
                float(
                    selected.iloc[0]["max_suv"]
                ),
            "selected_overlap_gt":
                int(
                    selected.iloc[0]["overlap_gt"]
                ),
            "selected_dice":
                float(
                    selected.iloc[0]["dice_gt"]
                ),
            "selected_iou":
                float(
                    selected.iloc[0]["iou_gt"]
                ),
        }

        results.append(
            row
        )

        print(
            f"Cap {cap:7d}: "
            f"label={row['selected_label']:3d}, "
            f"voxels={row['selected_voxels']:6d}, "
            f"z={row['selected_z_start']}-"
            f"{row['selected_z_end']}, "
            f"meanSUV={row['selected_mean_suv']:.3f}, "
            f"medianSUV={row['selected_median_suv']:.3f}, "
            f"Dice={row['selected_dice']:.4f}"
        )


# ============================================================
# TOP COMPONENTS UNDER THE MOST INTERESTING CAPS
# ============================================================

print()
print("=" * 110)
print("===== TOP COMPONENTS WITH VOLUME <= 10000 =====")
print("=" * 110)

eligible = df[
    df["voxels"] <= 10000
].copy()

print(
    eligible.sort_values(
        "mean_suv",
        ascending=False
    )[
        [
            "label",
            "voxels",
            "z_start",
            "z_end",
            "z_span",
            "mean_suv",
            "median_suv",
            "max_suv",
            "suv_concentration",
            "overlap_gt",
            "dice_gt",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# COMPARE TOP-1 VS TOP-2 VS TOP-3
#
# Only for the 10000 voxel cap.
# ============================================================

print()
print("=" * 110)
print("===== TOP-K WITH 10000-VOXEL CAP =====")
print("=" * 110)

eligible = df[
    df["voxels"] <= 10000
].copy()

for feature in SCORE_FEATURES:

    ranked = eligible.sort_values(
        feature,
        ascending=False
    )

    print()
    print(
        f"{feature}:"
    )

    for k in [1, 2, 3, 4, 5]:

        selected = ranked.head(
            k
        )

        metrics = union_metrics(
            selected
        )

        print(
            f"  Top {k}: "
            f"voxels={metrics['voxels']:6d}, "
            f"overlap={metrics['overlap']:5d}, "
            f"Dice={metrics['dice']:.4f}, "
            f"IoU={metrics['iou']:.4f}"
        )


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(
    results
)

OUTPUT = (
    ROOT
    / "validation_results"
    / "PETCT_0011f3deaf"
    / "case7_3d_size_constrained_selection.csv"
)

results_df.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 110)
print("===== SIZE-CONSTRAINED SELECTION AUDIT COMPLETE =====")
print("=" * 110)

print()
print("Saved:")
print(OUTPUT)
