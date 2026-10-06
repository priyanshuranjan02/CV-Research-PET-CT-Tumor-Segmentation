from pathlib import Path
import pandas as pd
import numpy as np


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
    / "dev11_component_rank_diagnostic_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANKING_CSV = (
    OUTPUT_DIR
    / "dev11_component_ranking_diagnostic.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "dev11_component_ranking_summary.csv"
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
# FEATURES TO DIAGNOSE
# ============================================================

SCORE_FEATURES = [
    "mean_suv",
    "median_suv",
    "p90_suv",
    "p95_suv",
    "max_suv",
    "mean_excess_suv",
    "mean_suv_hot3_a0p5",
    "mean_suv_hot4_a0p5",
    "mean_suv_hot5_a0p5",
    "p95_hot4_a0p5",
    "p95_hot5_a0p5",
]


# ============================================================
# TOP-K LEVELS
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
    "===== DEV11 COMPONENT RANKING DIAGNOSTIC ====="
)
print("=" * 115)

print()
print(
    "Input:"
)
print(
    INPUT_CSV
)

if not INPUT_CSV.exists():

    raise FileNotFoundError(
        f"\nDev9 component CSV not found:\n{INPUT_CSV}\n\n"
        "Run Dev9 first."
    )

df = pd.read_csv(
    INPUT_CSV
)

print()
print(
    "Components loaded:",
    len(df)
)

print(
    "Cases:",
    df["case"].nunique()
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = {
    "case",
    "label",
    "voxels",
    "overlap_gt",
    "dice_gt",
}

required_columns.update(
    SCORE_FEATURES
)

missing = [
    column
    for column in sorted(
        required_columns
    )
    if column not in df.columns
]

if missing:

    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(
            f"  - {x}"
            for x in missing
        )
    )


# ============================================================
# GT FLAG
#
# ONLY FOR DIAGNOSTIC ANALYSIS.
# This is NOT used to select components in the actual model.
# ============================================================

df["gt_overlap"] = (
    df["overlap_gt"] > 0
)


# ============================================================
# RANK DIAGNOSTIC
# ============================================================

ranking_rows = []


for case_name in CASES:

    case_df = (
        df[
            df["case"]
            == case_name
        ]
        .copy()
    )

    if case_df.empty:

        print(
            f"\nWARNING: {case_name} not found."
        )

        continue

    total_components = len(
        case_df
    )

    gt_components = (
        case_df[
            case_df["gt_overlap"]
        ]
    )

    print()
    print("=" * 115)
    print(
        f"===== {case_name} ====="
    )
    print("=" * 115)

    print(
        "Total components:",
        total_components
    )

    print(
        "GT-overlapping components:",
        len(gt_components)
    )

    print(
        "Maximum oracle component Dice:",
        round(
            float(
                case_df[
                    "dice_gt"
                ].max()
            ),
            6
        )
    )

    for feature in SCORE_FEATURES:

        ranked = (
            case_df
            .sort_values(
                [
                    feature,
                    "median_suv",
                    "mean_suv",
                    "voxels",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    True,
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

        # ----------------------------------------------------
        # Rank of first GT-overlapping component
        # ----------------------------------------------------

        gt_ranked = ranked[
            ranked["gt_overlap"]
        ]

        if gt_ranked.empty:

            first_gt_rank = np.nan
            first_gt_label = np.nan
            first_gt_dice = 0.0

        else:

            first_gt = (
                gt_ranked
                .iloc[0]
            )

            first_gt_rank = int(
                first_gt["rank"]
            )

            first_gt_label = int(
                first_gt["label"]
            )

            first_gt_dice = float(
                first_gt["dice_gt"]
            )

        # ----------------------------------------------------
        # Best component Dice available within Top-K
        #
        # This answers:
        #
        # "Does the current ranking put a useful component
        # inside Top-1 / Top-3 / Top-5 / Top-10?"
        # ----------------------------------------------------

        topk_values = {}

        for k in TOP_K_VALUES:

            topk = ranked.head(
                k
            )

            gt_topk = topk[
                topk["gt_overlap"]
            ]

            if gt_topk.empty:

                best_dice = 0.0
                gt_hit = 0

            else:

                best_dice = float(
                    gt_topk[
                        "dice_gt"
                    ].max()
                )

                gt_hit = 1

            topk_values[
                k
            ] = (
                best_dice,
                gt_hit
            )

        # ----------------------------------------------------
        # Top-1 component
        # ----------------------------------------------------

        top1 = ranked.iloc[0]

        top1_is_gt = int(
            bool(
                top1["gt_overlap"]
            )
        )

        top1_dice = float(
            top1["dice_gt"]
        )

        ranking_rows.append(
            {
                "case":
                    case_name,

                "score_feature":
                    feature,

                "total_components":
                    total_components,

                "gt_components":
                    len(gt_components),

                "first_gt_rank":
                    first_gt_rank,

                "first_gt_label":
                    first_gt_label,

                "first_gt_dice":
                    first_gt_dice,

                "top1_label":
                    int(
                        top1["label"]
                    ),

                "top1_is_gt_overlap":
                    top1_is_gt,

                "top1_component_dice":
                    top1_dice,

                "top1_best_gt_dice":
                    topk_values[1][0],

                "top3_best_gt_dice":
                    topk_values[3][0],

                "top5_best_gt_dice":
                    topk_values[5][0],

                "top10_best_gt_dice":
                    topk_values[10][0],

                "top1_gt_hit":
                    topk_values[1][1],

                "top3_gt_hit":
                    topk_values[3][1],

                "top5_gt_hit":
                    topk_values[5][1],

                "top10_gt_hit":
                    topk_values[10][1],
            }
        )

        # ----------------------------------------------------
        # Print the important result
        # ----------------------------------------------------

        print()
        print(
            f"{feature:<28}",
            f"first GT rank = "
            f"{first_gt_rank}",
            f"| top1 dice = "
            f"{top1_dice:.4f}",
            f"| top5 best dice = "
            f"{topk_values[5][0]:.4f}"
        )


# ============================================================
# DATAFRAME
# ============================================================

ranking_df = pd.DataFrame(
    ranking_rows
)


# ============================================================
# CASE-LEVEL DIAGNOSTIC SUMMARY
#
# For each case and feature:
#
#   first_gt_rank
#   top1 / top3 / top5 / top10 retrieval
#   best available component Dice
# ============================================================

print()
print("=" * 115)
print(
    "===== CASE-LEVEL SUMMARY ====="
)
print("=" * 115)

print()

case_summary = (
    ranking_df[
        [
            "case",
            "score_feature",
            "first_gt_rank",
            "first_gt_dice",
            "top1_component_dice",
            "top1_best_gt_dice",
            "top3_best_gt_dice",
            "top5_best_gt_dice",
            "top10_best_gt_dice",
            "top1_gt_hit",
            "top3_gt_hit",
            "top5_gt_hit",
            "top10_gt_hit",
        ]
    ]
    .sort_values(
        [
            "case",
            "first_gt_rank",
        ],
        na_position="last"
    )
)

print(
    case_summary.to_string(
        index=False
    )
)


# ============================================================
# FEATURE-LEVEL SUMMARY
#
# This is the most important section.
#
# A good ranking feature should:
#
#   - put GT-overlapping components near the top
#   - retrieve a useful component with small K
#   - preserve high oracle Dice
# ============================================================

feature_summary_rows = []

for feature in SCORE_FEATURES:

    feature_df = ranking_df[
        ranking_df[
            "score_feature"
        ]
        == feature
    ]

    finite_ranks = (
        feature_df[
            "first_gt_rank"
        ]
        .dropna()
    )

    feature_summary_rows.append(
        {
            "score_feature":
                feature,

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
                    feature_df[
                        "top1_gt_hit"
                    ].mean()
                ),

            "top3_retrieval_rate":
                float(
                    feature_df[
                        "top3_gt_hit"
                    ].mean()
                ),

            "top5_retrieval_rate":
                float(
                    feature_df[
                        "top5_gt_hit"
                    ].mean()
                ),

            "top10_retrieval_rate":
                float(
                    feature_df[
                        "top10_gt_hit"
                    ].mean()
                ),

            "mean_top1_best_gt_dice":
                float(
                    feature_df[
                        "top1_best_gt_dice"
                    ].mean()
                ),

            "mean_top3_best_gt_dice":
                float(
                    feature_df[
                        "top3_best_gt_dice"
                    ].mean()
                ),

            "mean_top5_best_gt_dice":
                float(
                    feature_df[
                        "top5_best_gt_dice"
                    ].mean()
                ),

            "mean_top10_best_gt_dice":
                float(
                    feature_df[
                        "top10_best_gt_dice"
                    ].mean()
                ),

            "max_oracle_component_dice":
                float(
                    feature_df[
                        "first_gt_dice"
                    ].max()
                ),
        }
    )


feature_summary_df = pd.DataFrame(
    feature_summary_rows
)


# ============================================================
# FEATURE SUMMARY OUTPUT
# ============================================================

print()
print("=" * 115)
print(
    "===== FEATURE-LEVEL RANKING SUMMARY ====="
)
print("=" * 115)

print()

print(
    feature_summary_df
    .sort_values(
        [
            "top5_retrieval_rate",
            "mean_top5_best_gt_dice",
        ],
        ascending=[
            False,
            False,
        ]
    )
    .to_string(
        index=False
    )
)


# ============================================================
# ORACLE ANALYSIS
#
# What is the maximum component Dice already present in the
# candidate set?
#
# If oracle component Dice is good but rank is poor:
#       ranking is the bottleneck.
#
# If oracle Dice itself is poor:
#       candidate generation is the bottleneck.
# ============================================================

print()
print("=" * 115)
print(
    "===== ORACLE COMPONENT ANALYSIS ====="
)
print("=" * 115)

oracle_rows = []

for case_name in CASES:

    case_df = df[
        df["case"]
        == case_name
    ]

    if case_df.empty:
        continue

    best = (
        case_df
        .sort_values(
            [
                "dice_gt",
                "overlap_gt",
            ],
            ascending=[
                False,
                False,
            ]
        )
        .iloc[0]
    )

    oracle_rows.append(
        {
            "case":
                case_name,

            "total_components":
                len(case_df),

            "best_component_label":
                int(
                    best["label"]
                ),

            "best_component_dice":
                float(
                    best["dice_gt"]
                ),

            "best_component_iou":
                float(
                    best["iou_gt"]
                ),

            "best_component_voxels":
                int(
                    best["voxels"]
                ),

            "best_component_mean_suv":
                float(
                    best["mean_suv"]
                ),

            "best_component_z_start":
                int(
                    best["z_start"]
                ),

            "best_component_z_end":
                int(
                    best["z_end"]
                ),
        }
    )

oracle_df = pd.DataFrame(
    oracle_rows
)

print()

print(
    oracle_df.to_string(
        index=False
    )
)

print()

print(
    "Mean oracle component Dice:",
    round(
        float(
            oracle_df[
                "best_component_dice"
            ].mean()
        ),
        6
    )
)


# ============================================================
# FINAL INTERPRETATION
# ============================================================

mean_oracle_dice = float(
    oracle_df[
        "best_component_dice"
    ].mean()
)

best_feature_row = (
    feature_summary_df
    .sort_values(
        [
            "top5_retrieval_rate",
            "mean_top5_best_gt_dice",
        ],
        ascending=[
            False,
            False,
        ]
    )
    .iloc[0]
)

print()
print("=" * 115)
print(
    "===== DEV11 DIAGNOSTIC INTERPRETATION ====="
)
print("=" * 115)

print()

print(
    "Mean oracle component Dice:",
    round(
        mean_oracle_dice,
        6
    )
)

print(
    "Best Top-5 retrieval feature:",
    best_feature_row[
        "score_feature"
    ]
)

print(
    "Top-5 retrieval rate:",
    round(
        float(
            best_feature_row[
                "top5_retrieval_rate"
            ]
        ),
        4
    )
)

print(
    "Mean best GT Dice in Top-5:",
    round(
        float(
            best_feature_row[
                "mean_top5_best_gt_dice"
            ]
        ),
        6
    )
)

print()

if mean_oracle_dice >= 0.30:

    print(
        "DIAGNOSTIC:"
    )

    print(
        "The candidate set contains reasonably "
        "useful tumor-overlapping components."
    )

    print(
        "Focus should therefore move toward "
        "component ranking/localization."
    )

else:

    print(
        "DIAGNOSTIC:"
    )

    print(
        "The candidate set itself has limited "
        "overlap with the GT."
    )

    print(
        "Candidate generation should therefore "
        "be improved before further ranking tuning."
    )


# ============================================================
# SAVE
# ============================================================

ranking_df.to_csv(
    RANKING_CSV,
    index=False
)

feature_summary_df.to_csv(
    SUMMARY_CSV,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 115)
print(
    "===== DEV11 COMPLETE ====="
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
    "Feature summary:"
)

print(
    SUMMARY_CSV
)