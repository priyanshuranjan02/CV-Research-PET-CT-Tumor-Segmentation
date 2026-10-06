from pathlib import Path
import sys
import importlib.util

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# DEV58 — DEVELOPMENT-COHORT SPATIAL SEPARATION AUDIT
#
# PURPOSE:
#   Determine whether spatial/anatomical position and component
#   size provide a consistent signal across ALL development
#   cases.
#
# IMPORTANT:
#   - Diagnostic only.
#   - Does NOT modify Dev10.
#   - Does NOT modify Dev39.
#   - Does NOT change candidate generation.
#   - Does NOT change ranking.
#   - Does NOT create a new final prediction.
#
# QUESTION:
#
#   Is the Dev57 observation about Component 84 a generalizable
#   cohort-level pattern, or only a validation-case artifact?
# ============================================================


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ============================================================
# DEVELOPMENT CASES
# ============================================================

DEVELOPMENT_CASES = [
    "PETCT_0168f65af8",
    "PETCT_04606080a0",
    "PETCT_04ab5c61c9",
    "PETCT_0b57b247b6",
    "PETCT_11afab3485",
    "PETCT_185da4c8b6",
]


# ============================================================
# FROZEN DEV39 PARAMETERS
# ============================================================

PET_THRESHOLD = 2.25
HOT_THRESHOLD = 3.0

MIN_3D_VOXELS = 75
MIN_3D_SLICES = 2

HOT_FRACTION_GATE = 0.40


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "results"
    / "development_cases"
    / "dev58_cohort_spatial_separation_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# IMPORT DEV10
# ============================================================

DEV10_PATH = SRC / "dev10_lite_hot_core.py"

spec10 = importlib.util.spec_from_file_location(
    "dev10_lite_hot_core",
    DEV10_PATH
)

dev10 = importlib.util.module_from_spec(
    spec10
)

spec10.loader.exec_module(
    dev10
)


# ============================================================
# IMPORT DEV39
# ============================================================

DEV39_PATH = SRC / "dev39_pet_quality_fpr_control.py"

spec39 = importlib.util.spec_from_file_location(
    "dev39_pet_quality_fpr_control",
    DEV39_PATH
)

dev39 = importlib.util.module_from_spec(
    spec39
)

spec39.loader.exec_module(
    dev39
)


# ============================================================
# BODY GEOMETRY
# ============================================================

def compute_body_geometry(body):

    body = body.astype(bool)

    coords = np.argwhere(body)

    if len(coords) == 0:

        return None

    centroid = coords.mean(
        axis=0
    )

    mins = coords.min(
        axis=0
    )

    maxs = coords.max(
        axis=0
    )

    distances = np.linalg.norm(
        coords - centroid,
        axis=1
    )

    return {

        "centroid_z": float(
            centroid[0]
        ),

        "centroid_y": float(
            centroid[1]
        ),

        "centroid_x": float(
            centroid[2]
        ),

        "z_min": int(
            mins[0]
        ),

        "z_max": int(
            maxs[0]
        ),

        "y_min": int(
            mins[1]
        ),

        "y_max": int(
            maxs[1]
        ),

        "x_min": int(
            mins[2]
        ),

        "x_max": int(
            maxs[2]
        ),

        "z_span": int(
            maxs[0] - mins[0] + 1
        ),

        "y_span": int(
            maxs[1] - mins[1] + 1
        ),

        "x_span": int(
            maxs[2] - mins[2] + 1
        ),

        "radius": float(
            np.max(distances)
        ),

        "body_voxels": int(
            np.count_nonzero(body)
        ),
    }


# ============================================================
# SPATIAL FEATURES
# ============================================================

def compute_spatial_features(
    cc,
    label,
    geometry
):

    mask = (
        cc == int(label)
    )

    coords = np.argwhere(
        mask
    )

    if len(coords) == 0:
        return {}

    centroid = coords.mean(
        axis=0
    )

    z, y, x = centroid

    z_min, y_min, x_min = (
        coords.min(
            axis=0
        )
    )

    z_max, y_max, x_max = (
        coords.max(
            axis=0
        )
    )

    bbox_z = int(
        z_max - z_min + 1
    )

    bbox_y = int(
        y_max - y_min + 1
    )

    bbox_x = int(
        x_max - x_min + 1
    )

    bbox_volume = (
        bbox_z
        * bbox_y
        * bbox_x
    )

    voxels = len(coords)

    if bbox_volume > 0:

        bbox_fill = (
            voxels
            / bbox_volume
        )

    else:

        bbox_fill = np.nan

    body_centroid = np.array([
        geometry["centroid_z"],
        geometry["centroid_y"],
        geometry["centroid_x"],
    ])

    component_centroid = np.array([
        z,
        y,
        x,
    ])

    radial_distance = np.linalg.norm(
        component_centroid
        - body_centroid
    )

    if geometry["radius"] > 0:

        normalized_radial = (
            radial_distance
            / geometry["radius"]
        )

    else:

        normalized_radial = np.nan

    normalized_z = (
        z - geometry["z_min"]
    ) / geometry["z_span"]

    normalized_y = (
        y - geometry["y_min"]
    ) / geometry["y_span"]

    normalized_x = (
        x - geometry["x_min"]
    ) / geometry["x_span"]

    # --------------------------------------------------------
    # Distance to nearest body bounding-box boundary
    # --------------------------------------------------------

    boundary_distance = min(

        z - geometry["z_min"],

        geometry["z_max"] - z,

        y - geometry["y_min"],

        geometry["y_max"] - y,

        x - geometry["x_min"],

        geometry["x_max"] - x,
    )

    # --------------------------------------------------------
    # Component / body volume fraction
    # --------------------------------------------------------

    body_fraction = (
        voxels
        / geometry["body_voxels"]
    )

    # --------------------------------------------------------
    # Aspect ratios
    # --------------------------------------------------------

    dims = sorted(
        [
            bbox_z,
            bbox_y,
            bbox_x,
        ],
        reverse=True
    )

    if dims[1] > 0:

        longest_middle = (
            dims[0]
            / dims[1]
        )

    else:

        longest_middle = np.nan

    if dims[2] > 0:

        longest_shortest = (
            dims[0]
            / dims[2]
        )

    else:

        longest_shortest = np.nan

    return {

        "centroid_z": float(z),
        "centroid_y": float(y),
        "centroid_x": float(x),

        "normalized_z": float(
            normalized_z
        ),

        "normalized_y": float(
            normalized_y
        ),

        "normalized_x": float(
            normalized_x
        ),

        "radial_distance": float(
            radial_distance
        ),

        "normalized_radial": float(
            normalized_radial
        ),

        "boundary_distance": float(
            boundary_distance
        ),

        "bbox_z": bbox_z,
        "bbox_y": bbox_y,
        "bbox_x": bbox_x,

        "bbox_volume": int(
            bbox_volume
        ),

        "bbox_fill": float(
            bbox_fill
        ),

        "component_voxels": int(
            voxels
        ),

        "body_fraction": float(
            body_fraction
        ),

        "longest_middle_ratio": float(
            longest_middle
        ),

        "longest_shortest_ratio": float(
            longest_shortest
        ),
    }


# ============================================================
# GT METRICS — POST HOC ONLY
# ============================================================

def compute_gt_metrics(
    cc,
    label,
    gt
):

    component = (
        cc == int(label)
    )

    gt = gt.astype(bool)

    component_voxels = int(
        np.count_nonzero(
            component
        )
    )

    gt_voxels = int(
        np.count_nonzero(
            gt
        )
    )

    overlap = int(
        np.count_nonzero(
            component & gt
        )
    )

    if (
        component_voxels
        + gt_voxels
    ) > 0:

        dice = (
            2.0 * overlap
            / (
                component_voxels
                + gt_voxels
            )
        )

    else:

        dice = 0.0

    if component_voxels > 0:

        precision = (
            overlap
            / component_voxels
        )

    else:

        precision = 0.0

    if gt_voxels > 0:

        recall = (
            overlap
            / gt_voxels
        )

    else:

        recall = 0.0

    return {

        "gt_overlap": overlap,

        "gt_overlap_flag": bool(
            overlap > 0
        ),

        "gt_dice": float(
            dice
        ),

        "gt_precision": float(
            precision
        ),

        "gt_recall": float(
            recall
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 100)
    print(
        "DEV58 — DEVELOPMENT-COHORT "
        "SPATIAL SEPARATION AUDIT"
    )
    print("=" * 100)

    print()
    print(
        "Cases:",
        len(DEVELOPMENT_CASES)
    )

    print(
        "PET threshold:",
        PET_THRESHOLD
    )

    print(
        "Hot threshold:",
        HOT_THRESHOLD
    )

    print(
        "3D filter:",
        f">={MIN_3D_VOXELS} voxels, "
        f">={MIN_3D_SLICES} slices"
    )

    print(
        "Hot fraction gate:",
        HOT_FRACTION_GATE
    )

    all_rows = []

    # ========================================================
    # CASE LOOP
    # ========================================================

    for case_name in DEVELOPMENT_CASES:

        print()
        print("=" * 100)
        print(
            f"CASE: {case_name}"
        )
        print("=" * 100)

        # ----------------------------------------------------
        # Load exact Dev10 case
        # ----------------------------------------------------

        case = dev10.load_case(
            case_name
        )

        ct = case[
            "ct"
        ].astype(
            np.float32
        )

        suv = case[
            "suv"
        ].astype(
            np.float32
        )

        gt = case[
            "gt"
        ].astype(
            bool
        )

        # ----------------------------------------------------
        # Exact Dev10 body mask
        # ----------------------------------------------------

        body = (
            dev10.build_union_body_mask(
                ct
            )
        )

        geometry = (
            compute_body_geometry(
                body
            )
        )

        # ----------------------------------------------------
        # Exact Dev39 PET-only candidate
        # ----------------------------------------------------

        candidate = (
            suv >= PET_THRESHOLD
        ).astype(
            np.uint8
        )

        candidate_voxels = int(
            np.count_nonzero(
                candidate
            )
        )

        print(
            "Candidate voxels:",
            candidate_voxels
        )

        # ----------------------------------------------------
        # Exact Dev10 3D components
        # ----------------------------------------------------

        components, cc = (
            dev10.build_3d_component_table(
                case,
                candidate
            )
        )

        print(
            "3D components:",
            len(components)
        )

        # ----------------------------------------------------
        # Exact Dev39 features
        # ----------------------------------------------------

        components = (
            dev39.add_component_features(
                components,
                cc,
                suv
            )
        )

        components = (
            dev39.add_pet_quality_score(
                components
            )
        )

        # ----------------------------------------------------
        # Exact Dev39 hot-fraction gate
        # ----------------------------------------------------

        eligible = components[
            components[
                "dev39_hot_fraction"
            ] >= HOT_FRACTION_GATE
        ].copy()

        print(
            "Eligible components:",
            len(eligible)
        )

        # ----------------------------------------------------
        # Component loop
        # ----------------------------------------------------

        for _, component_row in (
            eligible.iterrows()
        ):

            label = int(
                component_row[
                    "label"
                ]
            )

            spatial = (
                compute_spatial_features(
                    cc,
                    label,
                    geometry
                )
            )

            gt_metrics = (
                compute_gt_metrics(
                    cc,
                    label,
                    gt
                )
            )

            row = {

                "case": case_name,

                "label": label,

                # ------------------------------------------
                # PET / Dev39
                # ------------------------------------------

                "voxels": int(
                    component_row[
                        "dev39_voxels"
                    ]
                ),

                "z_span": int(
                    component_row[
                        "dev39_z_span"
                    ]
                ),

                "slice_count": int(
                    component_row[
                        "dev39_slice_count"
                    ]
                ),

                "longest_run": int(
                    component_row[
                        "dev39_longest_run"
                    ]
                ),

                "persistence": float(
                    component_row[
                        "dev39_persistence"
                    ]
                ),

                "mean_suv": float(
                    component_row[
                        "dev39_mean_suv"
                    ]
                ),

                "max_suv": float(
                    component_row[
                        "dev39_max_suv"
                    ]
                ),

                "hot_fraction": float(
                    component_row[
                        "dev39_hot_fraction"
                    ]
                ),

                "compactness": float(
                    component_row[
                        "dev39_compactness"
                    ]
                ),

                "pet_quality": float(
                    component_row[
                        "dev39_pet_quality"
                    ]
                ),
            }

            row.update(
                spatial
            )

            row.update(
                gt_metrics
            )

            # ------------------------------------------------
            # GT quality classes
            # ------------------------------------------------

            row[
                "gt_dice_ge_010"
            ] = bool(
                row["gt_dice"] >= 0.10
            )

            row[
                "gt_dice_ge_025"
            ] = bool(
                row["gt_dice"] >= 0.25
            )

            row[
                "gt_dice_ge_050"
            ] = bool(
                row["gt_dice"] >= 0.50
            )

            all_rows.append(
                row
            )

    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        all_rows
    )

    full_path = (
        OUTPUT_DIR
        / "dev58_all_eligible_components.csv"
    )

    df.to_csv(
        full_path,
        index=False
    )

    print()
    print(
        "Saved:",
        full_path
    )

    # ========================================================
    # COHORT SUMMARY
    # ========================================================

    print()
    print("=" * 100)
    print(
        "DEV58 — COHORT SUMMARY"
    )
    print("=" * 100)

    print()
    print(
        "Total eligible components:",
        len(df)
    )

    print(
        "GT-overlapping components:",
        int(
            df[
                "gt_overlap_flag"
            ].sum()
        )
    )

    print(
        "GT Dice >= 0.10:",
        int(
            df[
                "gt_dice_ge_010"
            ].sum()
        )
    )

    print(
        "GT Dice >= 0.25:",
        int(
            df[
                "gt_dice_ge_025"
            ].sum()
        )
    )

    print(
        "GT Dice >= 0.50:",
        int(
            df[
                "gt_dice_ge_050"
            ].sum()
        )
    )

    # ========================================================
    # GROUP COMPARISON
    # ========================================================

    feature_columns = [

        "voxels",

        "z_span",
        "slice_count",

        "mean_suv",
        "max_suv",
        "hot_fraction",
        "persistence",
        "compactness",
        "pet_quality",

        "normalized_z",
        "normalized_y",
        "normalized_x",

        "radial_distance",
        "normalized_radial",

        "boundary_distance",

        "bbox_z",
        "bbox_y",
        "bbox_x",

        "bbox_volume",
        "bbox_fill",

        "body_fraction",

        "longest_middle_ratio",
        "longest_shortest_ratio",
    ]

    summary_rows = []

    # --------------------------------------------------------
    # Three groups:
    #
    #   ALL
    #   GT-overlap
    #   GT-Dice >= 0.25
    #   GT-Dice < 0.25
    # --------------------------------------------------------

    groups = {

        "ALL": df,

        "GT_OVERLAP": df[
            df[
                "gt_overlap_flag"
            ]
        ],

        "GT_DICE_GE_025": df[
            df[
                "gt_dice_ge_025"
            ]
        ],

        "GT_DICE_LT_025": df[
            ~df[
                "gt_dice_ge_025"
            ]
        ],
    }

    for group_name, group in groups.items():

        if len(group) == 0:
            continue

        for feature in feature_columns:

            summary_rows.append({

                "group": group_name,

                "n": len(group),

                "feature": feature,

                "mean": float(
                    group[
                        feature
                    ].mean()
                ),

                "median": float(
                    group[
                        feature
                    ].median()
                ),

                "min": float(
                    group[
                        feature
                    ].min()
                ),

                "max": float(
                    group[
                        feature
                    ].max()
                ),
            })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        OUTPUT_DIR
        / "dev58_group_feature_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print()
    print(
        "Saved:",
        summary_path
    )

    # ========================================================
    # PRINT IMPORTANT FEATURES
    # ========================================================

    important_features = [

        "voxels",
        "body_fraction",
        "normalized_z",
        "normalized_radial",
        "boundary_distance",
        "bbox_volume",
        "bbox_fill",
        "mean_suv",
        "hot_fraction",
        "pet_quality",
    ]

    print()
    print("=" * 100)
    print(
        "DEV58 — IMPORTANT FEATURE COMPARISON"
    )
    print("=" * 100)

    print()

    for feature in important_features:

        print()
        print(
            f"--- {feature} ---"
        )

        for group_name in [
            "GT_OVERLAP",
            "GT_DICE_GE_025",
            "GT_DICE_LT_025",
        ]:

            subset = df

            if group_name == "GT_OVERLAP":

                subset = df[
                    df[
                        "gt_overlap_flag"
                    ]
                ]

            elif group_name == "GT_DICE_GE_025":

                subset = df[
                    df[
                        "gt_dice_ge_025"
                    ]
                ]

            elif group_name == "GT_DICE_LT_025":

                subset = df[
                    ~df[
                        "gt_dice_ge_025"
                    ]
                ]

            if len(subset) == 0:
                continue

            print(
                f"{group_name:18s}"
                f" n={len(subset):3d}"
                f" mean={subset[feature].mean():.6f}"
                f" median={subset[feature].median():.6f}"
                f" min={subset[feature].min():.6f}"
                f" max={subset[feature].max():.6f}"
            )

    # ========================================================
    # CASE-LEVEL SUMMARY
    # ========================================================

    case_rows = []

    for case_name in DEVELOPMENT_CASES:

        case_df = df[
            df[
                "case"
            ] == case_name
        ]

        case_rows.append({

            "case": case_name,

            "eligible_components": len(
                case_df
            ),

            "gt_overlap_components": int(
                case_df[
                    "gt_overlap_flag"
                ].sum()
            ),

            "gt_dice_ge_025": int(
                case_df[
                    "gt_dice_ge_025"
                ].sum()
            ),

            "best_gt_dice": float(
                case_df[
                    "gt_dice"
                ].max()
            ),

            "largest_component_voxels": int(
                case_df[
                    "voxels"
                ].max()
            ),

            "largest_component_radial": float(
                case_df.loc[
                    case_df[
                        "voxels"
                    ].idxmax(),
                    "normalized_radial"
                ]
            ),
        })

    case_df = pd.DataFrame(
        case_rows
    )

    case_path = (
        OUTPUT_DIR
        / "dev58_case_summary.csv"
    )

    case_df.to_csv(
        case_path,
        index=False
    )

    print()
    print(
        "Saved:",
        case_path
    )

    # ========================================================
    # PLOT 1:
    # SIZE VS RADIAL POSITION
    # ========================================================

    fig = plt.figure(
        figsize=(9, 7)
    )

    false_positive = df[
        ~df[
            "gt_dice_ge_025"
        ]
    ]

    true_positive = df[
        df[
            "gt_dice_ge_025"
        ]
    ]

    plt.scatter(
        false_positive[
            "normalized_radial"
        ],
        false_positive[
            "voxels"
        ],
        alpha=0.65,
        label="GT Dice < 0.25"
    )

    plt.scatter(
        true_positive[
            "normalized_radial"
        ],
        true_positive[
            "voxels"
        ],
        alpha=0.85,
        label="GT Dice >= 0.25"
    )

    plt.yscale(
        "log"
    )

    plt.xlabel(
        "Normalized radial distance"
    )

    plt.ylabel(
        "Component voxels"
    )

    plt.title(
        "DEV58 — Cohort Component Size vs Anatomical Position"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    path1 = (
        OUTPUT_DIR
        / "dev58_size_vs_radial_cohort.png"
    )

    plt.savefig(
        path1,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ========================================================
    # PLOT 2:
    # SIZE VS NORMALIZED Z
    # ========================================================

    fig = plt.figure(
        figsize=(9, 7)
    )

    plt.scatter(
        false_positive[
            "normalized_z"
        ],
        false_positive[
            "voxels"
        ],
        alpha=0.65,
        label="GT Dice < 0.25"
    )

    plt.scatter(
        true_positive[
            "normalized_z"
        ],
        true_positive[
            "voxels"
        ],
        alpha=0.85,
        label="GT Dice >= 0.25"
    )

    plt.yscale(
        "log"
    )

    plt.xlabel(
        "Normalized z position"
    )

    plt.ylabel(
        "Component voxels"
    )

    plt.title(
        "DEV58 — Cohort Component Size vs Z Position"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    path2 = (
        OUTPUT_DIR
        / "dev58_size_vs_z_cohort.png"
    )

    plt.savefig(
        path2,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ========================================================
    # PLOT 3:
    # PET QUALITY VS RADIAL POSITION
    # ========================================================

    fig = plt.figure(
        figsize=(9, 7)
    )

    plt.scatter(
        false_positive[
            "normalized_radial"
        ],
        false_positive[
            "pet_quality"
        ],
        alpha=0.65,
        label="GT Dice < 0.25"
    )

    plt.scatter(
        true_positive[
            "normalized_radial"
        ],
        true_positive[
            "pet_quality"
        ],
        alpha=0.85,
        label="GT Dice >= 0.25"
    )

    plt.xlabel(
        "Normalized radial distance"
    )

    plt.ylabel(
        "Dev39 PET quality"
    )

    plt.title(
        "DEV58 — PET Quality vs Anatomical Position"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    path3 = (
        OUTPUT_DIR
        / "dev58_pet_quality_vs_radial_cohort.png"
    )

    plt.savefig(
        path3,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ========================================================
    # PLOT 4:
    # PET QUALITY VS COMPONENT SIZE
    # ========================================================

    fig = plt.figure(
        figsize=(9, 7)
    )

    plt.scatter(
        false_positive[
            "voxels"
        ],
        false_positive[
            "pet_quality"
        ],
        alpha=0.65,
        label="GT Dice < 0.25"
    )

    plt.scatter(
        true_positive[
            "voxels"
        ],
        true_positive[
            "pet_quality"
        ],
        alpha=0.85,
        label="GT Dice >= 0.25"
    )

    plt.xscale(
        "log"
    )

    plt.xlabel(
        "Component voxels"
    )

    plt.ylabel(
        "Dev39 PET quality"
    )

    plt.title(
        "DEV58 — PET Quality vs Component Size"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    path4 = (
        OUTPUT_DIR
        / "dev58_pet_quality_vs_size_cohort.png"
    )

    plt.savefig(
        path4,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 100)
    print(
        "DEV58 COMPLETE"
    )
    print("=" * 100)

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Files:"
    )

    print(
        " - dev58_all_eligible_components.csv"
    )

    print(
        " - dev58_group_feature_summary.csv"
    )

    print(
        " - dev58_case_summary.csv"
    )

    print(
        " - dev58_size_vs_radial_cohort.png"
    )

    print(
        " - dev58_size_vs_z_cohort.png"
    )

    print(
        " - dev58_pet_quality_vs_radial_cohort.png"
    )

    print(
        " - dev58_pet_quality_vs_size_cohort.png"
    )


if __name__ == "__main__":
    main()