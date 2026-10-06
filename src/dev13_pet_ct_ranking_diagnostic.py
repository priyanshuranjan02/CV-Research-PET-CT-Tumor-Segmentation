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

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ============================================================
# LOAD EXACT DEV10-LITE DEFINITIONS
#
# IMPORTANT:
# We load ONLY the function/constant definitions above
# "# MAIN".
#
# Therefore Dev10-Lite's own experiment is NOT executed.
#
# This gives Dev13 the exact same:
#   - load_case()
#   - generate_candidate_volume()
#   - build_3d_component_table()
#   - V4 PET/SUV helpers
#
# used to produce the Dev9/Dev10 component labels.
# ============================================================

DEV10_PATH = (
    SRC
    / "dev10_lite_hot_core.py"
)


if not DEV10_PATH.exists():

    raise FileNotFoundError(
        f"Missing Dev10-Lite:\n{DEV10_PATH}"
    )


with redirect_stdout(io.StringIO()):

    source = DEV10_PATH.read_text(
        encoding="utf-8"
    )


    if "# MAIN" not in source:

        raise RuntimeError(
            "Could not find '# MAIN' in Dev10-Lite."
        )


    prefix = source.rsplit(
        "\n# MAIN",
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
# INPUT
# ============================================================

INPUT_CSV = (
    ROOT
    / "development_cases"
    / "dev9_pet_intensity_distribution_results"
    / "dev9_pet_intensity_distribution_components.csv"
)


if not INPUT_CSV.exists():

    raise FileNotFoundError(
        "Dev9 component CSV not found:\n"
        f"{INPUT_CSV}"
    )


dev9 = pd.read_csv(
    INPUT_CSV
)


# ============================================================
# REQUIRED DEV9 COLUMNS
# ============================================================

REQUIRED_COLUMNS = {
    "case",
    "label",
    "voxels",
    "overlap_gt",
    "dice_gt",
    "median_suv",
    "mean_suv_hot3_a0p5",
}


missing = [
    column
    for column in REQUIRED_COLUMNS
    if column not in dev9.columns
]


if missing:

    raise RuntimeError(
        "Missing columns in Dev9 CSV:\n"
        + "\n".join(
            missing
        )
    )


# ============================================================
# DEV13 PARAMETERS
# ============================================================

TOP_K = [
    3,
    5,
    10,
]


FUSIONS = [
    "median_only",
    "mean_hot_only",
    "median_plus_soft",
    "mean_hot_plus_soft",
    "median_plus_nonair",
    "mean_hot_plus_nonair",
]


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "development_cases"
    / "dev13_pet_ct_ranking_results"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SUMMARY_CSV = (
    OUTPUT_DIR
    / "dev13_pet_ct_ranking_summary.csv"
)


DETAIL_CSV = (
    OUTPUT_DIR
    / "dev13_pet_ct_ranking_details.csv"
)


# ============================================================
# CT FEATURE CALCULATION
#
# IMPORTANT:
# This operates on the EXACT 3D connected-component label
# volume produced by Dev10-Lite.
#
# No candidate reconstruction is performed here.
# ============================================================

def calculate_ct_features(
    ct,
    cc
):

    max_label = int(
        cc.max()
    )


    if max_label == 0:

        return pd.DataFrame()


    # --------------------------------------------------------
    # Per-label voxel counts
    # --------------------------------------------------------

    voxel_counts = np.bincount(
        cc.ravel(),
        minlength=max_label + 1
    )


    # --------------------------------------------------------
    # Accumulators
    # --------------------------------------------------------

    ct_sums = np.zeros(
        max_label + 1,
        dtype=np.float64
    )


    soft_counts = np.zeros(
        max_label + 1,
        dtype=np.int64
    )


    nonair_counts = np.zeros(
        max_label + 1,
        dtype=np.int64
    )


    # --------------------------------------------------------
    # Slice-wise processing
    #
    # This avoids creating one giant mask per component.
    # --------------------------------------------------------

    for z in range(
        cc.shape[0]
    ):

        labels = cc[z]


        positive = (
            labels > 0
        )


        if not np.any(
            positive
        ):

            continue


        labels_z = labels[
            positive
        ]


        ct_values = ct[z][
            positive
        ]


        # ----------------------------------------------------
        # Mean CT
        # ----------------------------------------------------

        ct_sums += np.bincount(
            labels_z,
            weights=ct_values,
            minlength=max_label + 1
        )


        # ----------------------------------------------------
        # Broad soft-tissue range
        #
        # [-150, 250] HU
        # ----------------------------------------------------

        soft = (
            (ct_values >= -150)
            &
            (ct_values <= 250)
        )


        if np.any(
            soft
        ):

            soft_counts += np.bincount(
                labels_z[soft],
                minlength=max_label + 1
            )


        # ----------------------------------------------------
        # Non-air fraction
        #
        # CT > -300 HU
        # ----------------------------------------------------

        nonair = (
            ct_values > -300
        )


        if np.any(
            nonair
        ):

            nonair_counts += np.bincount(
                labels_z[nonair],
                minlength=max_label + 1
            )


    # --------------------------------------------------------
    # Build feature table
    # --------------------------------------------------------

    rows = []


    for label in range(
        1,
        max_label + 1
    ):

        voxels = int(
            voxel_counts[label]
        )


        if voxels <= 0:

            continue


        rows.append(
            {
                "label":
                    int(label),

                "mean_ct_hu":
                    float(
                        ct_sums[label]
                        / voxels
                    ),

                "ct_soft_fraction":
                    float(
                        soft_counts[label]
                        / voxels
                    ),

                "ct_nonair_fraction":
                    float(
                        nonair_counts[label]
                        / voxels
                    ),
            }
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# PERCENTILE RANK
# ============================================================

def rank_feature(
    series
):

    return (
        series
        .rank(
            method="average",
            pct=True
        )
    )


# ============================================================
# BUILD FUSION SCORE
# ============================================================

def add_fusion_score(
    df,
    fusion
):

    ranked = df.copy()


    # --------------------------------------------------------
    # PET
    # --------------------------------------------------------

    ranked[
        "rank_median"
    ] = rank_feature(
        ranked["median_suv"]
    )


    ranked[
        "rank_mean_hot"
    ] = rank_feature(
        ranked[
            "mean_suv_hot3_a0p5"
        ]
    )


    # --------------------------------------------------------
    # CT
    # --------------------------------------------------------

    ranked[
        "rank_soft"
    ] = rank_feature(
        ranked[
            "ct_soft_fraction"
        ]
    )


    ranked[
        "rank_nonair"
    ] = rank_feature(
        ranked[
            "ct_nonair_fraction"
        ]
    )


    # --------------------------------------------------------
    # Fusion
    # --------------------------------------------------------

    if fusion == "median_only":

        ranked[
            "fusion_score"
        ] = ranked[
            "rank_median"
        ]


    elif fusion == "mean_hot_only":

        ranked[
            "fusion_score"
        ] = ranked[
            "rank_mean_hot"
        ]


    elif fusion == "median_plus_soft":

        ranked[
            "fusion_score"
        ] = (
            ranked["rank_median"]
            + ranked["rank_soft"]
        ) / 2.0


    elif fusion == "mean_hot_plus_soft":

        ranked[
            "fusion_score"
        ] = (
            ranked["rank_mean_hot"]
            + ranked["rank_soft"]
        ) / 2.0


    elif fusion == "median_plus_nonair":

        ranked[
            "fusion_score"
        ] = (
            ranked["rank_median"]
            + ranked["rank_nonair"]
        ) / 2.0


    elif fusion == "mean_hot_plus_nonair":

        ranked[
            "fusion_score"
        ] = (
            ranked["rank_mean_hot"]
            + ranked["rank_nonair"]
        ) / 2.0


    else:

        raise ValueError(
            f"Unknown fusion: {fusion}"
        )


    return ranked


# ============================================================
# DEV13
# ============================================================

results = []


print()
print("=" * 115)
print(
    "===== DEV13 PET + CT RANKING DIAGNOSTIC ====="
)
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
    "Candidate construction:",
    "EXACT DEV10-LITE"
)


# ============================================================
# CASE LOOP
# ============================================================

for case_name in CASES:

    print()
    print("=" * 115)
    print(
        f"===== {case_name} ====="
    )
    print("=" * 115)


    case = None
    raw_volume = None
    components = None
    cc = None


    try:

        # ====================================================
        # LOAD EXACT DEV10 CASE
        # ====================================================

        case = load_case(
            case_name
        )


        # ====================================================
        # GENERATE EXACT DEV10 CANDIDATE VOLUME
        # ====================================================

        print(
            "Generating exact Dev10-Lite candidates..."
        )


        raw_volume = (
            generate_candidate_volume(
                case
            )
        )


        # ====================================================
        # BUILD EXACT DEV10 3D COMPONENT TABLE
        #
        # This is the SAME operation used by Dev10-Lite
        # to create the labels contained in Dev9's CSV.
        # ====================================================

        print(
            "Building exact Dev10-Lite 3D components..."
        )


        (
            components,
            cc
        ) = build_3d_component_table(
            case,
            raw_volume
        )


        raw_volume = None


        print(
            "Dev10 eligible components:",
            len(components)
        )


        # ====================================================
        # DEV9 COMPONENTS FOR THIS CASE
        # ====================================================

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


        # ====================================================
        # EXACT LABEL MATCH CHECK
        #
        # This is the most important diagnostic.
        # ====================================================

        dev10_labels = set(
            components[
                "label"
            ]
            .astype(int)
            .tolist()
        )


        dev9_labels = set(
            case_dev9[
                "label"
            ]
            .astype(int)
            .tolist()
        )


        missing_from_dev10 = (
            dev9_labels
            - dev10_labels
        )


        extra_in_dev10 = (
            dev10_labels
            - dev9_labels
        )


        exact_match = (
            len(
                missing_from_dev10
            ) == 0
            and
            len(
                extra_in_dev10
            ) == 0
        )


        print(
            "Exact Dev9/Dev10 label match:",
            exact_match
        )


        print(
            "Missing Dev9 labels:",
            len(
                missing_from_dev10
            )
        )


        print(
            "Extra Dev10 labels:",
            len(
                extra_in_dev10
            )
        )


        # ----------------------------------------------------
        # Do not silently compare different component sets.
        # ----------------------------------------------------

        if not exact_match:

            print()
            print(
                "WARNING:"
            )

            print(
                "Dev9 and current Dev10 component labels "
                "do not match."
            )

            print(
                "Skipping ranking for this case."
            )

            continue


        # ====================================================
        # CT FEATURES ON EXACT DEV10 COMPONENTS
        # ====================================================

        print(
            "Calculating CT features..."
        )


        ct_features = calculate_ct_features(
            case["ct"],
            cc
        )


        # ====================================================
        # MERGE
        # ====================================================

        case_df = case_dev9.merge(
            ct_features,
            on="label",
            how="inner"
        )


        print(
            "Matched components:",
            len(case_df)
        )


        # ====================================================
        # GT OVERLAP AUDIT
        # ====================================================

        gt_overlap_count = int(
            (
                case_df[
                    "overlap_gt"
                ]
                > 0
            )
            .sum()
        )


        print(
            "GT-overlapping components:",
            gt_overlap_count
        )


        # ====================================================
        # FUSION LOOP
        # ====================================================

        for fusion in FUSIONS:

            ranked = add_fusion_score(
                case_df,
                fusion
            )


            # Highest score first.
            # PET median is used as the tie-breaker
            # to keep ordering deterministic.
            ranked = (
                ranked
                .sort_values(
                    [
                        "fusion_score",
                        "median_suv"
                    ],
                    ascending=[
                        False,
                        False
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


            # ------------------------------------------------
            # First GT-overlapping component
            # ------------------------------------------------

            gt_ranked = ranked[
                ranked[
                    "overlap_gt"
                ]
                > 0
            ]


            if gt_ranked.empty:

                first_gt_rank = np.nan

            else:

                first_gt_rank = int(
                    gt_ranked.iloc[0][
                        "rank"
                    ]
                )


            # ------------------------------------------------
            # Top-K analysis
            # ------------------------------------------------

            for k in TOP_K:

                top = ranked.head(
                    k
                )


                top_gt = top[
                    top[
                        "overlap_gt"
                    ]
                    > 0
                ]


                if top_gt.empty:

                    hit = 0
                    best_dice = 0.0

                else:

                    hit = 1

                    best_dice = float(
                        top_gt[
                            "dice_gt"
                        ]
                        .max()
                    )


                results.append(
                    {
                        "case":
                            case_name,

                        "fusion":
                            fusion,

                        "top_k":
                            k,

                        "first_gt_rank":
                            first_gt_rank,

                        "gt_hit":
                            hit,

                        "best_gt_component_dice":
                            best_dice,

                        "matched_components":
                            len(case_df),

                        "gt_overlap_components":
                            gt_overlap_count,
                    }
                )


            # ------------------------------------------------
            # Compact Top-5 console output
            # ------------------------------------------------

            top5 = ranked.head(
                5
            )


            top5_gt = top5[
                top5[
                    "overlap_gt"
                ]
                > 0
            ]


            if top5_gt.empty:

                top5_hit = 0
                top5_dice = 0.0

            else:

                top5_hit = 1

                top5_dice = float(
                    top5_gt[
                        "dice_gt"
                    ]
                    .max()
                )


            print(
                f"{fusion:<24}"
                f"GT rank={str(first_gt_rank):>4} "
                f"| Top5={top5_hit} "
                f"| Top5 Dice={top5_dice:.4f}"
            )


        # ====================================================
        # CLEAN CASE
        # ====================================================

        del case_df
        del case_dev9
        del ct_features


    finally:

        if case is not None:
            del case

        if raw_volume is not None:
            del raw_volume

        if components is not None:
            del components

        if cc is not None:
            del cc

        gc.collect()


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


if results_df.empty:

    raise RuntimeError(
        "Dev13 produced no ranking results."
    )


# ============================================================
# SAVE DETAILS
# ============================================================

results_df.to_csv(
    DETAIL_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


for fusion in FUSIONS:

    fusion_df = results_df[
        results_df["fusion"]
        == fusion
    ]


    top3 = fusion_df[
        fusion_df["top_k"]
        == 3
    ]


    top5 = fusion_df[
        fusion_df["top_k"]
        == 5
    ]


    top10 = fusion_df[
        fusion_df["top_k"]
        == 10
    ]


    summary_rows.append(
        {
            "fusion":
                fusion,

            "mean_first_gt_rank":
                float(
                    top3[
                        "first_gt_rank"
                    ]
                    .mean()
                ),

            "median_first_gt_rank":
                float(
                    top3[
                        "first_gt_rank"
                    ]
                    .median()
                ),

            "top3_retrieval_rate":
                float(
                    top3[
                        "gt_hit"
                    ]
                    .mean()
                ),

            "top5_retrieval_rate":
                float(
                    top5[
                        "gt_hit"
                    ]
                    .mean()
                ),

            "top10_retrieval_rate":
                float(
                    top10[
                        "gt_hit"
                    ]
                    .mean()
                ),

            "mean_top5_best_dice":
                float(
                    top5[
                        "best_gt_component_dice"
                    ]
                    .mean()
                ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


summary = (
    summary
    .sort_values(
        [
            "top5_retrieval_rate",
            "mean_top5_best_dice"
        ],
        ascending=[
            False,
            False
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 115)
print(
    "===== DEV13 SUMMARY ====="
)
print("=" * 115)

print()

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary.to_csv(
    SUMMARY_CSV,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 115)
print(
    "===== DEV13 COMPLETE ====="
)
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
    "Detailed CSV:"
)

print(
    DETAIL_CSV
)