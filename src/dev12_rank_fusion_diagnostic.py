from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    ROOT
    / "development_cases"
    / "dev9_pet_intensity_distribution_results"
    / "dev9_pet_intensity_distribution_components.csv"
)

OUTPUT_DIR = (
    ROOT
    / "development_cases"
    / "dev12_rank_fusion_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANKING_CSV = (
    OUTPUT_DIR
    / "dev12_rank_fusion_diagnostic.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "dev12_rank_fusion_summary.csv"
)


# ============================================================
# CASES
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
# BASE FEATURES
# ============================================================

BASE_FEATURES = [
    "mean_suv",
    "median_suv",
    "p90_suv",
    "p95_suv",
    "max_suv",
    "frac_ge_3",
    "frac_ge_4",
    "frac_ge_5",
    "mean_excess_suv",
]


# ============================================================
# RANK-FUSION CONFIGURATIONS
#
# Each tuple is:
#
#     configuration name
#     features
#
# Feature values are converted into percentile ranks inside
# each case before being combined.
# ============================================================

FUSION_CONFIGS = [

    (
        "median_only",
        [
            "median_suv",
        ]
    ),

    (
        "mean_median",
        [
            "mean_suv",
            "median_suv",
        ]
    ),

    (
        "median_p95",
        [
            "median_suv",
            "p95_suv",
        ]
    ),

    (
        "median_hot3",
        [
            "median_suv",
            "frac_ge_3",
        ]
    ),

    (
        "median_p95_hot3",
        [
            "median_suv",
            "p95_suv",
            "frac_ge_3",
        ]
    ),

    (
        "robust_intensity",
        [
            "mean_suv",
            "median_suv",
            "p90_suv",
            "p95_suv",
        ]
    ),

    (
        "intensity_hot",
        [
            "mean_suv",
            "median_suv",
            "frac_ge_3",
        ]
    ),

    (
        "intensity_hot4",
        [
            "mean_suv",
            "median_suv",
            "frac_ge_3",
            "frac_ge_4",
        ]
    ),

    (
        "intensity_distribution",
        [
            "mean_suv",
            "median_suv",
            "p90_suv",
            "p95_suv",
            "frac_ge_3",
        ]
    ),

    (
        "all_robust",
        [
            "mean_suv",
            "median_suv",
            "p90_suv",
            "p95_suv",
            "frac_ge_3",
            "frac_ge_4",
            "mean_excess_suv",
        ]
    ),
]


# ============================================================
# TOP-K
# ============================================================

TOP_K_VALUES = [
    1,
    3,
    5,
    10,
]


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 115)
print(
    "===== DEV12 RANK-FUSION DIAGNOSTIC ====="
)
print("=" * 115)

print()

if not INPUT_CSV.exists():

    raise FileNotFoundError(
        f"Missing Dev9 component CSV:\n{INPUT_CSV}"
    )

df = pd.read_csv(
    INPUT_CSV
)

print(
    "Components:",
    len(df)
)

print(
    "Cases:",
    df["case"].nunique()
)


# ============================================================
# CHECK
# ============================================================

required = {
    "case",
    "label",
    "voxels",
    "overlap_gt",
    "dice_gt",
}

required.update(
    BASE_FEATURES
)

missing = [
    x
    for x in sorted(required)
    if x not in df.columns
]

if missing:

    raise RuntimeError(
        "Missing columns:\n"
        + "\n".join(
            f"  {x}"
            for x in missing
        )
    )


# ============================================================
# FUSION FUNCTION
# ============================================================

def percentile_rank(
    series
):
    """
    Convert a feature into [0, 1] percentile ranks.

    Highest feature value gets a value close to 1.
    """

    return (
        series.rank(
            method="average",
            pct=True
        )
    )


def build_fusion_score(
    case_df,
    features
):

    rank_columns = []

    for feature in features:

        rank_col = (
            f"rank__{feature}"
        )

        case_df[
            rank_col
        ] = percentile_rank(
            case_df[
                feature
            ]
        )

        rank_columns.append(
            rank_col
        )

    case_df[
        "fusion_score"
    ] = case_df[
        rank_columns
    ].mean(
        axis=1
    )

    return case_df


# ============================================================
# DIAGNOSTIC STORAGE
# ============================================================

ranking_rows = []


# ============================================================
# CASE LOOP
# ============================================================

for case_name in CASES:

    case_original = (
        df[
            df["case"]
            == case_name
        ]
        .copy()
    )

    if case_original.empty:

        print(
            f"WARNING: {case_name} missing."
        )

        continue

    print()
    print("=" * 115)
    print(
        f"===== {case_name} ====="
    )
    print("=" * 115)

    print(
        "Components:",
        len(case_original)
    )

    print(
        "GT-overlapping:",
        int(
            (
                case_original[
                    "overlap_gt"
                ] > 0
            ).sum()
        )
    )

    # --------------------------------------------------------
    # Test every fusion configuration
    # --------------------------------------------------------

    for (
        fusion_name,
        features
    ) in FUSION_CONFIGS:

        case_df = (
            case_original
            .copy()
        )

        case_df = build_fusion_score(
            case_df,
            features
        )

        ranked = (
            case_df
            .sort_values(
                [
                    "fusion_score",
                    "median_suv",
                    "mean_suv",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ]
            )
            .reset_index(
                drop=True
            )
        )

        ranked[
            "rank"
        ] = (
            np.arange(
                len(ranked)
            )
            + 1
        )

        # ----------------------------------------------------
        # GT component ranks
        # ----------------------------------------------------

        gt_ranked = ranked[
            ranked["overlap_gt"] > 0
        ]

        if gt_ranked.empty:

            first_gt_rank = np.nan
            first_gt_dice = 0.0

        else:

            first_gt_rank = int(
                gt_ranked.iloc[0]["rank"]
            )

            first_gt_dice = float(
                gt_ranked.iloc[0]["dice_gt"]
            )

        # ----------------------------------------------------
        # Top-K diagnostics
        # ----------------------------------------------------

        topk = {}

        for k in TOP_K_VALUES:

            top = ranked.head(
                k
            )

            top_gt = top[
                top["overlap_gt"] > 0
            ]

            if top_gt.empty:

                best_dice = 0.0
                hit = 0

            else:

                best_dice = float(
                    top_gt[
                        "dice_gt"
                    ].max()
                )

                hit = 1

            topk[k] = {
                "hit": hit,
                "dice": best_dice,
            }

        ranking_rows.append(
            {
                "case":
                    case_name,

                "fusion_name":
                    fusion_name,

                "features":
                    "+".join(features),

                "first_gt_rank":
                    first_gt_rank,

                "first_gt_dice":
                    first_gt_dice,

                "top1_hit":
                    topk[1]["hit"],

                "top3_hit":
                    topk[3]["hit"],

                "top5_hit":
                    topk[5]["hit"],

                "top10_hit":
                    topk[10]["hit"],

                "top1_best_dice":
                    topk[1]["dice"],

                "top3_best_dice":
                    topk[3]["dice"],

                "top5_best_dice":
                    topk[5]["dice"],

                "top10_best_dice":
                    topk[10]["dice"],
            }
        )

        print(
            f"{fusion_name:<24}",
            f"GT rank={str(first_gt_rank):>4}",
            f"| Top3={topk[3]['hit']}",
            f"| Top5={topk[5]['hit']}",
            f"| Top5 Dice="
            f"{topk[5]['dice']:.4f}"
        )


# ============================================================
# DATAFRAME
# ============================================================

ranking_df = pd.DataFrame(
    ranking_rows
)


# ============================================================
# FEATURE-FUSION SUMMARY
# ============================================================

summary_rows = []

for fusion_name in (
    ranking_df[
        "fusion_name"
    ]
    .unique()
):

    temp = ranking_df[
        ranking_df[
            "fusion_name"
        ]
        == fusion_name
    ]

    finite_ranks = (
        temp[
            "first_gt_rank"
        ]
        .dropna()
    )

    summary_rows.append(
        {
            "fusion_name":
                fusion_name,

            "features":
                temp[
                    "features"
                ].iloc[0],

            "mean_first_gt_rank":
                float(
                    finite_ranks.mean()
                )
                if not finite_ranks.empty
                else np.nan,

            "median_first_gt_rank":
                float(
                    finite_ranks.median()
                )
                if not finite_ranks.empty
                else np.nan,

            "top1_retrieval_rate":
                float(
                    temp[
                        "top1_hit"
                    ].mean()
                ),

            "top3_retrieval_rate":
                float(
                    temp[
                        "top3_hit"
                    ].mean()
                ),

            "top5_retrieval_rate":
                float(
                    temp[
                        "top5_hit"
                    ].mean()
                ),

            "top10_retrieval_rate":
                float(
                    temp[
                        "top10_hit"
                    ].mean()
                ),

            "mean_top3_best_dice":
                float(
                    temp[
                        "top3_best_dice"
                    ].mean()
                ),

            "mean_top5_best_dice":
                float(
                    temp[
                        "top5_best_dice"
                    ].mean()
                ),

            "mean_top10_best_dice":
                float(
                    temp[
                        "top10_best_dice"
                    ].mean()
                ),
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# SUMMARY OUTPUT
# ============================================================

print()
print("=" * 115)
print(
    "===== DEV12 RANK-FUSION SUMMARY ====="
)
print("=" * 115)

summary_sorted = (
    summary_df
    .sort_values(
        [
            "top5_retrieval_rate",
            "mean_top5_best_dice",
            "top3_retrieval_rate",
        ],
        ascending=[
            False,
            False,
            False,
        ]
    )
)

print()

print(
    summary_sorted.to_string(
        index=False
    )
)


# ============================================================
# BEST FUSION CONFIGURATION
# ============================================================

print()
print("=" * 115)
print(
    "===== TOP RANK-FUSION CONFIGURATIONS ====="
)
print("=" * 115)

print()

print(
    summary_sorted
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# CASE × BEST FUSION
# ============================================================

print()
print("=" * 115)
print(
    "===== BEST FUSION PER CASE ====="
)
print("=" * 115)

best_case_rows = []

for case_name in CASES:

    temp = ranking_df[
        ranking_df[
            "case"
        ]
        == case_name
    ]

    if temp.empty:
        continue

    best = (
        temp
        .sort_values(
            [
                "top5_best_dice",
                "top3_best_dice",
                "top5_hit",
                "first_gt_rank",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ]
        )
        .iloc[0]
    )

    best_case_rows.append(
        best.to_dict()
    )

best_case_df = pd.DataFrame(
    best_case_rows
)

print()

print(
    best_case_df[
        [
            "case",
            "fusion_name",
            "first_gt_rank",
            "top1_hit",
            "top3_hit",
            "top5_hit",
            "top10_hit",
            "top3_best_dice",
            "top5_best_dice",
            "top10_best_dice",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# COMPARE AGAINST DEV11 BASELINES
# ============================================================

print()
print("=" * 115)
print(
    "===== DEV12 VS SINGLE-FEATURE BASELINES ====="
)
print("=" * 115)

print()

baseline_names = [
    "median_only",
    "mean_median",
]

comparison = (
    summary_df[
        summary_df[
            "fusion_name"
        ].isin(
            baseline_names
        )
        |
        (
            summary_df[
                "top5_retrieval_rate"
            ]
            ==
            summary_df[
                "top5_retrieval_rate"
            ].max()
        )
    ]
    .sort_values(
        "top5_retrieval_rate",
        ascending=False
    )
)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

ranking_df.to_csv(
    RANKING_CSV,
    index=False
)

summary_df.to_csv(
    SUMMARY_CSV,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 115)
print(
    "===== DEV12 COMPLETE ====="
)
print("=" * 115)

print()

print(
    "Ranking diagnostic:"
)

print(
    RANKING_CSV
)

print()

print(
    "Summary:"
)

print(
    SUMMARY_CSV
)