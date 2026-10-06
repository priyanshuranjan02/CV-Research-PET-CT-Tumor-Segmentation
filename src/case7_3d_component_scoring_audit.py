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


df = pd.read_csv(CSV)

# ------------------------------------------------------------
# We start with the threshold that gave the cleanest
# all-contour result in the previous experiment.
# ------------------------------------------------------------

THRESHOLD = 2.25

df = df[
    df["threshold"] == THRESHOLD
].copy()

kept = df[
    df["kept"] == 1
].copy()

print()
print("=" * 110)
print("===== CASE 7 — 3D COMPONENT SCORING AUDIT =====")
print("=" * 110)

print()
print("Threshold:", THRESHOLD)
print("Kept 3D components:", len(kept))


# ============================================================
# DERIVED FEATURES
#
# None of these use GT.
# ============================================================

kept["mean_to_median"] = (
    kept["mean_suv"]
    / kept["median_suv"].clip(lower=1e-6)
)

kept["suv_concentration"] = (
    kept["mean_suv"]
    * kept["median_suv"]
)

kept["intensity_density"] = (
    kept["mean_suv"]
    / np.sqrt(
        kept["voxels"].clip(lower=1)
    )
)

kept["mean_x_volume"] = (
    kept["mean_suv"]
    * np.log1p(
        kept["voxels"]
    )
)

kept["median_x_volume"] = (
    kept["median_suv"]
    * np.log1p(
        kept["voxels"]
    )
)

# Size-normalized intensity score.
kept["mean_per_1000vox"] = (
    kept["mean_suv"]
    / (
        kept["voxels"]
        / 1000.0
    )
)


# ============================================================
# PRINT ALL COMPONENTS
# ============================================================

cols = [
    "label",
    "voxels",
    "z_start",
    "z_end",
    "z_span",
    "mean_suv",
    "median_suv",
    "max_suv",
    "mean_to_median",
    "suv_concentration",
    "mean_x_volume",
    "median_x_volume",
    "mean_per_1000vox",
    "overlap_gt",
    "dice_gt",
]


print()
print("-" * 110)
print("ALL KEPT COMPONENTS")
print("-" * 110)

print(
    kept.sort_values(
        "mean_suv",
        ascending=False
    )[cols].to_string(
        index=False
    )
)


# ============================================================
# RANKING AUDITS
#
# GT is shown only to see where the lesion components rank.
# It is NOT used in the ranking itself.
# ============================================================

ranking_features = [
    "mean_suv",
    "median_suv",
    "max_suv",
    "z_span",
    "voxels",
    "mean_to_median",
    "suv_concentration",
    "mean_x_volume",
    "median_x_volume",
    "mean_per_1000vox",
]


for feature in ranking_features:

    ranked = kept.sort_values(
        feature,
        ascending=False
    ).reset_index(
        drop=True
    )

    ranked["rank"] = (
        np.arange(
            len(ranked)
        )
        + 1
    )

    print()
    print("-" * 110)
    print(
        f"RANKING BY {feature.upper()}"
    )
    print("-" * 110)

    print(
        ranked[
            [
                "rank",
                "label",
                "voxels",
                "z_start",
                "z_end",
                "z_span",
                feature,
                "mean_suv",
                "median_suv",
                "max_suv",
                "overlap_gt",
                "dice_gt",
            ]
        ]
        .head(24)
        .to_string(
            index=False
        )
    )


# ============================================================
# TOP-K UNION AUDIT
#
# This asks:
# "If we selected the top K components according to a
# non-GT score, how would the resulting segmentation look?"
#
# GT is used ONLY to calculate the resulting metric.
# ============================================================

print()
print("=" * 110)
print("===== TOP-K COMPONENT UNION AUDIT =====")
print("=" * 110)


def union_metrics(
    selected
):

    pred_voxels = int(
        selected["voxels"].sum()
    )

    overlap = int(
        selected["overlap_gt"].sum()
    )

    gt_voxels = int(
        9206
    )

    dice = (
        2.0
        * overlap
        / (
            pred_voxels
            + gt_voxels
        )
        if (
            pred_voxels
            + gt_voxels
        ) > 0
        else 0.0
    )

    iou_den = (
        pred_voxels
        + gt_voxels
        - overlap
    )

    iou = (
        overlap
        / iou_den
        if iou_den > 0
        else 0.0
    )

    return (
        pred_voxels,
        overlap,
        dice,
        iou
    )


for feature in [
    "mean_suv",
    "median_suv",
    "max_suv",
    "suv_concentration",
    "mean_x_volume",
    "median_x_volume",
]:

    ranked = kept.sort_values(
        feature,
        ascending=False
    )

    print()
    print(
        f"{feature}:"
    )

    for k in [
        1,
        2,
        3,
        4,
        5,
        6,
        8,
        10,
    ]:

        selected = ranked.head(
            k
        )

        (
            voxels,
            overlap,
            dice,
            iou
        ) = union_metrics(
            selected
        )

        print(
            f"  Top {k:2d}: "
            f"voxels={voxels:6d}, "
            f"overlap={overlap:5d}, "
            f"Dice={dice:.4f}, "
            f"IoU={iou:.4f}"
        )


# ============================================================
# SPECIAL CHECK:
# COMPONENTS IN THE TWO DISEASE-LIKE Z REGIONS
#
# This does NOT select the prediction.
# It only shows which non-GT features characterize the
# components around the known GT-containing regions.
# ============================================================

print()
print("=" * 110)
print("===== COMPONENTS AROUND Z=277–293 AND Z=352–363 =====")
print("=" * 110)

region_df = kept[
    (
        (
            kept["z_end"] >= 277
        )
        &
        (
            kept["z_start"] <= 293
        )
    )
    |
    (
        (
            kept["z_end"] >= 352
        )
        &
        (
            kept["z_start"] <= 363
        )
    )
].copy()

print(
    region_df.sort_values(
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
            "mean_to_median",
            "suv_concentration",
            "mean_x_volume",
            "overlap_gt",
            "dice_gt",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 110)
print("===== AUDIT COMPLETE =====")
print("=" * 110)

print()
print(
    "The ranking tables show whether a non-GT component score "
    "can prioritize the lesion-containing structures."
)
