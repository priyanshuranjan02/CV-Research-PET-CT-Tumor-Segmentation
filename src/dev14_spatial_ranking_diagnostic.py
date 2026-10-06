from pathlib import Path
import sys
import io
import gc
from contextlib import redirect_stdout

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

DEV10_PATH = (
    SRC
    / "dev10_lite_hot_core.py"
)

DEV9_CSV = (
    ROOT
    / "development_cases"
    / "dev9_pet_intensity_distribution_results"
    / "dev9_pet_intensity_distribution_components.csv"
)

OUTPUT_DIR = (
    ROOT
    / "development_cases"
    / "dev14_spatial_ranking_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "dev14_spatial_ranking_summary.csv"
)

DETAILS_CSV = (
    OUTPUT_DIR
    / "dev14_spatial_ranking_details.csv"
)


# ============================================================
# SIX VERIFIED DEVELOPMENT CASES
# ============================================================

CASES = [
    "PETCT_0168f65af8",
    "PETCT_04606080a0",
    "PETCT_04ab5c61c9",
    "PETCT_0b57b247b6",
    "PETCT_11afab3485",
    "PETCT_185da4c8b6",
]


# ============================================================
# LOAD DEV10 DEFINITIONS ONLY
#
# IMPORTANT:
# Do not run Dev10 main section.
#
# rsplit() is intentional because Dev10 itself contains
# the string "# MAIN" inside its V4 loader.
# ============================================================

if not DEV10_PATH.exists():

    raise FileNotFoundError(
        f"Missing Dev10 file:\n{DEV10_PATH}"
    )

with redirect_stdout(io.StringIO()):

    source = DEV10_PATH.read_text(
        encoding="utf-8"
    )

    marker = "\n# MAIN"

    if marker not in source:

        raise RuntimeError(
            "Could not find Dev10 main-section marker."
        )

    prefix = source.rsplit(
        marker,
        1
    )[0]

    exec(
        compile(
            prefix,
            str(DEV10_PATH),
            "exec"
        ),
        globals()
    )


# ============================================================
# LOAD DEV9 COMPONENT TABLE
# ============================================================

if not DEV9_CSV.exists():

    raise FileNotFoundError(
        f"Missing Dev9 component CSV:\n{DEV9_CSV}"
    )

dev9 = pd.read_csv(
    DEV9_CSV
)

required_columns = {
    "case",
    "label",
    "voxels",
    "z_start",
    "z_end",
    "z_span",
    "median_suv",
    "mean_suv_hot3_a0p5",
    "overlap_gt",
    "dice_gt",
}

missing = [
    col
    for col in required_columns
    if col not in dev9.columns
]

if missing:

    raise RuntimeError(
        "Dev9 CSV is missing columns:\n"
        + "\n".join(missing)
    )


# ============================================================
# SPATIAL FEATURE CALCULATION
# ============================================================

def calculate_spatial_features(
    cc,
    components
):

    labels = (
        components["label"]
        .astype(int)
        .to_numpy()
    )

    n_labels = (
        len(labels)
    )

    label_to_index = {
        int(label): i
        for i, label in enumerate(labels)
    }

    voxels = (
        components["voxels"]
        .astype(float)
        .to_numpy()
    )

    sum_x = np.zeros(
        n_labels,
        dtype=np.float64
    )

    sum_y = np.zeros(
        n_labels,
        dtype=np.float64
    )

    sum_z = np.zeros(
        n_labels,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Calculate centroids without creating a mask for each
    # component.
    # --------------------------------------------------------

    for z in range(
        cc.shape[0]
    ):

        slice_labels = cc[z]

        ys, xs = np.nonzero(
            slice_labels > 0
        )

        if len(xs) == 0:
            continue

        current_labels = (
            slice_labels[ys, xs]
            .astype(int)
        )

        valid = np.isin(
            current_labels,
            labels
        )

        if not np.any(valid):
            continue

        current_labels = (
            current_labels[valid]
        )

        xs = (
            xs[valid]
            .astype(np.float64)
        )

        ys = (
            ys[valid]
            .astype(np.float64)
        )

        indices = np.array(
            [
                label_to_index[int(x)]
                for x in current_labels
            ],
            dtype=np.int64
        )

        np.add.at(
            sum_x,
            indices,
            xs
        )

        np.add.at(
            sum_y,
            indices,
            ys
        )

        np.add.at(
            sum_z,
            indices,
            float(z)
        )

    safe_voxels = np.maximum(
        voxels,
        1.0
    )

    centroid_x = (
        sum_x
        / safe_voxels
    )

    centroid_y = (
        sum_y
        / safe_voxels
    )

    centroid_z = (
        sum_z
        / safe_voxels
    )

    # --------------------------------------------------------
    # Z information
    # --------------------------------------------------------

    z_start = (
        components["z_start"]
        .astype(float)
        .to_numpy()
    )

    z_end = (
        components["z_end"]
        .astype(float)
        .to_numpy()
    )

    z_span = (
        components["z_span"]
        .astype(float)
        .to_numpy()
    )

    z_center = (
        (z_start + z_end)
        / 2.0
    )

    # --------------------------------------------------------
    # Neighbor calculations
    #
    # A neighbor is considered spatially close when:
    #
    #   1. z intervals are separated by <= 2 slices
    #   2. centroid distance in 3D is <= threshold
    #
    # This is only a ranking feature.
    # --------------------------------------------------------

    neighbor_count_50 = np.zeros(
        n_labels,
        dtype=np.float64
    )

    neighbor_count_100 = np.zeros(
        n_labels,
        dtype=np.float64
    )

    nearest_distance = np.full(
        n_labels,
        np.inf,
        dtype=np.float64
    )

    z_neighbor_count = np.zeros(
        n_labels,
        dtype=np.float64
    )

    for i in range(
        n_labels
    ):

        dx = (
            centroid_x
            - centroid_x[i]
        )

        dy = (
            centroid_y
            - centroid_y[i]
        )

        dz = (
            centroid_z
            - centroid_z[i]
        )

        distance = np.sqrt(
            dx * dx
            + dy * dy
            + dz * dz
        )

        distance[i] = np.inf

        nearest_distance[i] = (
            float(
                np.min(distance)
            )
            if n_labels > 1
            else np.inf
        )

        # ----------------------------------------------------
        # Z interval gap
        # ----------------------------------------------------

        gap = np.maximum(
            0.0,
            np.maximum(
                z_start - z_end[i] - 1.0,
                z_start[i] - z_end - 1.0
            )
        )

        z_close = (
            gap <= 2.0
        )

        z_close[i] = False

        z_neighbor_count[i] = (
            float(
                np.sum(z_close)
            )
        )

        close_50 = (
            z_close
            & (distance <= 50.0)
        )

        close_100 = (
            z_close
            & (distance <= 100.0)
        )

        neighbor_count_50[i] = (
            float(
                np.sum(close_50)
            )
        )

        neighbor_count_100[i] = (
            float(
                np.sum(close_100)
            )
        )

    # --------------------------------------------------------
    # Spatial density / continuity
    # --------------------------------------------------------

    z_density = (
        voxels
        / np.maximum(
            z_span,
            1.0
        )
    )

    features = pd.DataFrame(
        {
            "label": labels,

            "centroid_x":
                centroid_x,

            "centroid_y":
                centroid_y,

            "centroid_z":
                centroid_z,

            "z_center":
                z_center,

            "z_span":
                z_span,

            "z_density":
                z_density,

            "nearest_component_distance":
                nearest_distance,

            "neighbor_count_50":
                neighbor_count_50,

            "neighbor_count_100":
                neighbor_count_100,

            "z_neighbor_count":
                z_neighbor_count,
        }
    )

    return features


# ============================================================
# PERCENTILE RANK
# ============================================================

def percentile_rank(
    series,
    higher_is_better=True
):

    values = (
        series
        .astype(float)
        .to_numpy()
    )

    order = np.argsort(
        values
    )

    ranks = np.empty(
        len(values),
        dtype=np.float64
    )

    ranks[order] = (
        np.arange(
            len(values)
        )
        / max(
            len(values) - 1,
            1
        )
    )

    if not higher_is_better:

        ranks = 1.0 - ranks

    return ranks


# ============================================================
# ADD RANKING SCORE
# ============================================================

def add_score(
    df,
    configuration
):

    result = df.copy()

    # --------------------------------------------------------
    # PET baseline
    # --------------------------------------------------------

    median_pet = percentile_rank(
        result["median_suv"],
        higher_is_better=True
    )

    # --------------------------------------------------------
    # Spatial features
    # --------------------------------------------------------

    neighbor50 = percentile_rank(
        result["neighbor_count_50"],
        higher_is_better=True
    )

    neighbor100 = percentile_rank(
        result["neighbor_count_100"],
        higher_is_better=True
    )

    z_neighbors = percentile_rank(
        result["z_neighbor_count"],
        higher_is_better=True
    )

    z_span = percentile_rank(
        result["z_span"],
        higher_is_better=True
    )

    # Smaller nearest distance is better.
    nearest = percentile_rank(
        result["nearest_component_distance"],
        higher_is_better=False
    )

    z_density = percentile_rank(
        result["z_density"],
        higher_is_better=True
    )

    # --------------------------------------------------------
    # Configurations
    # --------------------------------------------------------

    if configuration == "median_only":

        score = median_pet

    elif configuration == "spatial_neighbor50":

        score = neighbor50

    elif configuration == "spatial_neighbor100":

        score = neighbor100

    elif configuration == "spatial_z":

        score = z_neighbors

    elif configuration == "spatial_distance":

        score = nearest

    elif configuration == "spatial_continuity":

        score = (
            0.5 * z_neighbors
            + 0.5 * z_span
        )

    elif configuration == "pet_plus_neighbor50":

        score = (
            0.5 * median_pet
            + 0.5 * neighbor50
        )

    elif configuration == "pet_plus_neighbor100":

        score = (
            0.5 * median_pet
            + 0.5 * neighbor100
        )

    elif configuration == "pet_plus_spatial":

        score = (
            0.5 * median_pet
            + 0.25 * neighbor100
            + 0.25 * nearest
        )

    elif configuration == "pet_plus_continuity":

        score = (
            0.5 * median_pet
            + 0.25 * z_neighbors
            + 0.25 * z_span
        )

    elif configuration == "pet_plus_all_spatial":

        score = (
            0.40 * median_pet
            + 0.20 * neighbor100
            + 0.15 * z_neighbors
            + 0.15 * nearest
            + 0.10 * z_density
        )

    else:

        raise ValueError(
            f"Unknown configuration: {configuration}"
        )

    result["ranking_score"] = (
        np.asarray(
            score,
            dtype=np.float64
        )
    )

    return result


# ============================================================
# CONFIGURATIONS
# ============================================================

CONFIGURATIONS = [

    "median_only",

    "spatial_neighbor50",
    "spatial_neighbor100",
    "spatial_z",
    "spatial_distance",
    "spatial_continuity",

    "pet_plus_neighbor50",
    "pet_plus_neighbor100",
    "pet_plus_spatial",
    "pet_plus_continuity",
    "pet_plus_all_spatial",
]


TOP_K_VALUES = [
    3,
    5,
    10,
]


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 115)
print("===== DEV14 SPATIAL RANKING DIAGNOSTIC =====")
print("=" * 115)
print()

print(
    "Dev9 components:",
    len(dev9)
)

print(
    "Cases:",
    len(CASES)
)

print(
    "Candidate construction: EXACT DEV10-LITE"
)

print()


# ============================================================
# PROCESS ONE CASE AT A TIME
# ============================================================

for case_name in CASES:

    print()
    print("=" * 115)
    print(
        f"===== {case_name} ====="
    )
    print("=" * 115)

    case_dev9 = (
        dev9[
            dev9["case"]
            == case_name
        ]
        .copy()
    )

    print(
        "Dev9 components:",
        len(case_dev9)
    )

    # --------------------------------------------------------
    # Load exact Dev10 case
    # --------------------------------------------------------

    case = load_case(
        case_name
    )

    print(
        "Generating exact Dev10-Lite candidates..."
    )

    raw_volume = (
        generate_candidate_volume(
            case
        )
    )

    print(
        "Building exact Dev10-Lite 3D components..."
    )

    components, cc = (
        build_3d_component_table(
            case,
            raw_volume
        )
    )

    print(
        "Dev10 eligible components:",
        len(components)
    )

    # --------------------------------------------------------
    # Exact label integrity check
    # --------------------------------------------------------

    dev9_labels = set(
        case_dev9[
            "label"
        ]
        .astype(int)
        .tolist()
    )

    dev10_labels = set(
        components[
            "label"
        ]
        .astype(int)
        .tolist()
    )

    missing = (
        dev9_labels
        - dev10_labels
    )

    extra = (
        dev10_labels
        - dev9_labels
    )

    exact_match = (
        len(missing) == 0
        and len(extra) == 0
    )

    print(
        "Exact Dev9/Dev10 label match:",
        exact_match
    )

    print(
        "Missing Dev9 labels:",
        len(missing)
    )

    print(
        "Extra Dev10 labels:",
        len(extra)
    )

    if not exact_match:

        raise RuntimeError(
            f"{case_name}: component labels do not match."
        )

    # --------------------------------------------------------
    # Calculate spatial features
    # --------------------------------------------------------

    print(
        "Calculating spatial features..."
    )

    spatial = calculate_spatial_features(
        cc,
        components
    )

    # --------------------------------------------------------
    # Join exact Dev9 metrics with spatial features
    # --------------------------------------------------------

    case_df = (
        case_dev9
        .merge(
            spatial,
            on="label",
            how="inner",
            suffixes=(
                "",
                "_spatial"
            )
        )
    )

    print(
        "Matched components:",
        len(case_df)
    )

    gt_count = int(
        (
            case_df["overlap_gt"]
            > 0
        )
        .sum()
    )

    print(
        "GT-overlapping components:",
        gt_count
    )

    # --------------------------------------------------------
    # Test configurations
    # --------------------------------------------------------

    for configuration in CONFIGURATIONS:

        ranked = add_score(
            case_df,
            configuration
        )

        ranked = (
            ranked
            .sort_values(
                [
                    "ranking_score",
                    "median_suv",
                ],
                ascending=[
                    False,
                    False,
                ]
            )
            .reset_index(
                drop=True
            )
        )

        ranked["rank"] = (
            np.arange(
                len(ranked)
            )
            + 1
        )

        gt_ranked = ranked[
            ranked["overlap_gt"] > 0
        ]

        if gt_ranked.empty:

            first_gt_rank = np.nan

        else:

            first_gt_rank = int(
                gt_ranked.iloc[0]["rank"]
            )

        for k in TOP_K_VALUES:

            top = ranked.head(k)

            top_gt = top[
                top["overlap_gt"] > 0
            ]

            if top_gt.empty:

                hit = 0
                best_dice = 0.0

            else:

                hit = 1

                best_dice = float(
                    top_gt[
                        "dice_gt"
                    ].max()
                )

            results.append(
                {
                    "case":
                        case_name,

                    "configuration":
                        configuration,

                    "top_k":
                        k,

                    "first_gt_rank":
                        first_gt_rank,

                    "gt_hit":
                        hit,

                    "best_gt_component_dice":
                        best_dice,
                }
            )

        # ----------------------------------------------------
        # Compact K=5 output
        # ----------------------------------------------------

        top5 = ranked.head(5)

        top5_gt = top5[
            top5["overlap_gt"] > 0
        ]

        if top5_gt.empty:

            top5_hit = 0
            top5_dice = 0.0

        else:

            top5_hit = 1

            top5_dice = float(
                top5_gt[
                    "dice_gt"
                ].max()
            )

        print(
            f"{configuration:28s}"
            f" GT rank={first_gt_rank:4d}"
            f" | Top5={top5_hit}"
            f" | Top5 Dice={top5_dice:.4f}"
        )

    # --------------------------------------------------------
    # Save case details
    # --------------------------------------------------------

    case_details_path = (
        OUTPUT_DIR
        / f"{case_name}_spatial_features.csv"
    )

    case_df.to_csv(
        case_details_path,
        index=False
    )

    # --------------------------------------------------------
    # Release large arrays
    # --------------------------------------------------------

    del raw_volume
    del cc
    del spatial
    del components
    del case
    del case_df

    gc.collect()


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(
    results
)

summary_rows = []

for configuration in CONFIGURATIONS:

    config_df = results_df[
        results_df[
            "configuration"
        ]
        == configuration
    ]

    # Top-3
    top3 = config_df[
        config_df["top_k"] == 3
    ]

    # Top-5
    top5 = config_df[
        config_df["top_k"] == 5
    ]

    # Top-10
    top10 = config_df[
        config_df["top_k"] == 10
    ]

    summary_rows.append(
        {
            "configuration":
                configuration,

            "mean_first_gt_rank":
                float(
                    config_df[
                        "first_gt_rank"
                    ].mean()
                ),

            "median_first_gt_rank":
                float(
                    config_df[
                        "first_gt_rank"
                    ].median()
                ),

            "top3_retrieval_rate":
                float(
                    top3["gt_hit"].mean()
                ),

            "top5_retrieval_rate":
                float(
                    top5["gt_hit"].mean()
                ),

            "top10_retrieval_rate":
                float(
                    top10["gt_hit"].mean()
                ),

            "mean_top5_best_dice":
                float(
                    top5[
                        "best_gt_component_dice"
                    ].mean()
                ),
        }
    )

summary_df = pd.DataFrame(
    summary_rows
)

summary_df = (
    summary_df
    .sort_values(
        [
            "mean_top5_best_dice",
            "top5_retrieval_rate",
            "top3_retrieval_rate",
        ],
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    DETAILS_CSV,
    index=False
)

summary_df.to_csv(
    SUMMARY_CSV,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 115)
print("===== DEV14 SUMMARY =====")
print("=" * 115)
print()

print(
    summary_df.to_string(
        index=False
    )
)

print()
print("=" * 115)
print("===== DEV14 COMPLETE =====")
print("=" * 115)

print()
print(
    "Summary CSV:"
)

print(
    SUMMARY_CSV
)

print()
print(
    "Details CSV:"
)

print(
    DETAILS_CSV
)