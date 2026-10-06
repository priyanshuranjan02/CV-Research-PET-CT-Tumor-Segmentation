from pathlib import Path
import sys
import numpy as np
import pandas as pd

# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import dev10_lite_hot_core as dev10


# ============================================================
# DEV50 CONFIGURATION
# ============================================================

CASES = [
    "PETCT_0168f65af8",
    "PETCT_04606080a0",
    "PETCT_04ab5c61c9",
    "PETCT_0b57b247b6",
    "PETCT_11afab3485",
    "PETCT_185da4c8b6",
]

PET_THRESHOLD = 2.25
HOT_THRESHOLD = 3.0

HOT_FRACTION_GATE = 0.40

TOP_K = 4
CORE_POLICY = "P70"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "development_cases"
    / "dev50_ct_context_ranking"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAFE CORE
# ============================================================

def safe_build_core(
    cc,
    suv,
    label,
    policy
):

    result = dev10.build_core(
        cc,
        suv,
        int(label),
        policy
    )

    if isinstance(
        result,
        tuple
    ):

        for item in result:

            if isinstance(
                item,
                np.ndarray
            ):

                return item.astype(
                    bool
                )

    return result.astype(
        bool
    )


# ============================================================
# NORMALIZATION
# ============================================================

def minmax_normalize(values):

    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) == 0:
        return values

    vmin = np.min(values)
    vmax = np.max(values)

    if vmax - vmin < 1e-12:
        return np.ones_like(
            values
        )

    return (
        values - vmin
    ) / (
        vmax - vmin
    )


# ============================================================
# LONGEST CONSECUTIVE Z RUN
# ============================================================

def longest_consecutive_run(
    z_values
):

    if len(z_values) == 0:
        return 0

    z_values = sorted(
        set(
            int(z)
            for z in z_values
        )
    )

    best = 1
    current = 1

    for i in range(
        1,
        len(z_values)
    ):

        if (
            z_values[i]
            == z_values[i - 1] + 1
        ):

            current += 1

            best = max(
                best,
                current
            )

        else:

            current = 1

    return best


# ============================================================
# PET + CT COMPONENT FEATURES
# ============================================================

def calculate_component_features(
    components,
    cc,
    suv,
    ct
):

    rows = []

    for _, component in components.iterrows():

        label = int(
            component["label"]
        )

        mask = (
            cc == label
        )

        coords = np.argwhere(
            mask
        )

        if len(coords) == 0:
            continue

        z = coords[:, 0]
        y = coords[:, 1]
        x = coords[:, 2]

        pet_values = suv[
            mask
        ]

        ct_values = ct[
            mask
        ]

        voxels = len(
            coords
        )

        # ----------------------------------------------------
        # PET features
        # ----------------------------------------------------

        mean_suv = float(
            np.mean(
                pet_values
            )
        )

        max_suv = float(
            np.max(
                pet_values
            )
        )

        hot_fraction = float(
            np.mean(
                pet_values
                >= HOT_THRESHOLD
            )
        )

        z_unique = np.unique(
            z
        )

        slice_count = len(
            z_unique
        )

        persistence = (
            slice_count
            / suv.shape[0]
        )

        longest_run = (
            longest_consecutive_run(
                z_unique
            )
        )

        # ----------------------------------------------------
        # Shape
        # ----------------------------------------------------

        z_span = (
            int(z.max())
            - int(z.min())
            + 1
        )

        bbox_volume = (
            (z.max() - z.min() + 1)
            * (y.max() - y.min() + 1)
            * (x.max() - x.min() + 1)
        )

        compactness = (
            voxels
            / max(
                bbox_volume,
                1
            )
        )

        # ----------------------------------------------------
        # CT context
        # ----------------------------------------------------

        ct_mean = float(
            np.mean(
                ct_values
            )
        )

        ct_median = float(
            np.median(
                ct_values
            )
        )

        ct_std = float(
            np.std(
                ct_values
            )
        )

        ct_min = float(
            np.min(
                ct_values
            )
        )

        ct_max = float(
            np.max(
                ct_values
            )
        )

        # ----------------------------------------------------
        # CT tissue fractions
        #
        # These are deliberately broad contextual features,
        # NOT anatomical labels.
        # ----------------------------------------------------

        fraction_air = float(
            np.mean(
                ct_values < -500
            )
        )

        fraction_soft_tissue = float(
            np.mean(
                (
                    ct_values >= -150
                )
                &
                (
                    ct_values < 200
                )
            )
        )

        fraction_dense = float(
            np.mean(
                ct_values >= 200
            )
        )

        fraction_very_dense = float(
            np.mean(
                ct_values >= 500
            )
        )

        # ----------------------------------------------------
        # CT spatial variation
        # ----------------------------------------------------

        ct_high_variation = float(
            np.mean(
                np.abs(
                    ct_values
                    - ct_median
                ) > 100
            )
        )

        # ----------------------------------------------------
        # Centroid
        # ----------------------------------------------------

        centroid_z = float(
            np.mean(z)
        )

        centroid_y = float(
            np.mean(y)
        )

        centroid_x = float(
            np.mean(x)
        )

        rows.append({

            "label": label,

            "voxels": voxels,

            "z_start": int(
                z.min()
            ),

            "z_end": int(
                z.max()
            ),

            "z_span": z_span,

            "slice_count":
                slice_count,

            "persistence":
                persistence,

            "longest_run":
                longest_run,

            "compactness":
                compactness,

            "centroid_z":
                centroid_z,

            "centroid_y":
                centroid_y,

            "centroid_x":
                centroid_x,

            "mean_suv":
                mean_suv,

            "max_suv":
                max_suv,

            "hot_fraction":
                hot_fraction,

            "ct_mean":
                ct_mean,

            "ct_median":
                ct_median,

            "ct_std":
                ct_std,

            "ct_min":
                ct_min,

            "ct_max":
                ct_max,

            "ct_fraction_air":
                fraction_air,

            "ct_fraction_soft_tissue":
                fraction_soft_tissue,

            "ct_fraction_dense":
                fraction_dense,

            "ct_fraction_very_dense":
                fraction_very_dense,

            "ct_high_variation":
                ct_high_variation,

        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# DEV39 PET QUALITY
# ============================================================

def add_dev39_score(df):

    df = df.copy()

    df["n_mean_suv"] = (
        minmax_normalize(
            df["mean_suv"]
        )
    )

    df["n_max_suv"] = (
        minmax_normalize(
            df["max_suv"]
        )
    )

    df["n_hot_fraction"] = (
        minmax_normalize(
            df["hot_fraction"]
        )
    )

    df["n_persistence"] = (
        minmax_normalize(
            df["persistence"]
        )
    )

    df["n_longest_run"] = (
        minmax_normalize(
            df["longest_run"]
        )
    )

    df["n_compactness"] = (
        minmax_normalize(
            df["compactness"]
        )
    )

    df["dev39_score"] = (
        0.25 * df["n_mean_suv"]
        + 0.15 * df["n_max_suv"]
        + 0.15 * df["n_hot_fraction"]
        + 0.20 * df["n_persistence"]
        + 0.15 * df["n_longest_run"]
        + 0.10 * df["n_compactness"]
    )

    return df


# ============================================================
# CT CONTEXT SCORE
# ============================================================

def add_ct_scores(df):

    df = df.copy()

    # --------------------------------------------------------
    # Context features
    # --------------------------------------------------------

    df["n_ct_soft"] = (
        minmax_normalize(
            df["ct_fraction_soft_tissue"]
        )
    )

    df["n_ct_air"] = (
        minmax_normalize(
            df["ct_fraction_air"]
        )
    )

    df["n_ct_dense"] = (
        minmax_normalize(
            df["ct_fraction_dense"]
        )
    )

    df["n_ct_variation"] = (
        minmax_normalize(
            df["ct_high_variation"]
        )
    )

    df["n_ct_std"] = (
        minmax_normalize(
            df["ct_std"]
        )
    )

    # --------------------------------------------------------
    # Context compatibility
    #
    # Tumor-like candidates are expected to be primarily
    # soft-tissue rather than predominantly air or very dense
    # material.
    #
    # This is a ranking heuristic, NOT a tumor classifier.
    # --------------------------------------------------------

    df["ct_context_score"] = (
        0.50 * df["n_ct_soft"]
        + 0.20 * df["n_ct_variation"]
        + 0.15 * df["n_ct_std"]
        - 0.10 * df["n_ct_air"]
        - 0.05 * df["n_ct_dense"]
    )

    return df


# ============================================================
# RANKING POLICIES
# ============================================================

def add_ranking_scores(df):

    df = df.copy()

    # --------------------------------------------------------
    # Policy 1:
    # Frozen Dev39
    # --------------------------------------------------------

    df["rank_dev39"] = (
        df["dev39_score"]
    )

    # --------------------------------------------------------
    # Policy 2:
    # Mild CT integration
    #
    # Keep PET dominant.
    # --------------------------------------------------------

    df["rank_ct_mild"] = (
        0.80 * df["dev39_score"]
        + 0.20 * df["ct_context_score"]
    )

    # --------------------------------------------------------
    # Policy 3:
    # Moderate CT integration
    # --------------------------------------------------------

    df["rank_ct_moderate"] = (
        0.65 * df["dev39_score"]
        + 0.35 * df["ct_context_score"]
    )

    return df


# ============================================================
# GT COMPONENT AUDIT
# ============================================================

def add_gt_metrics(
    df,
    cc,
    gt
):

    gt_voxels = int(
        gt.sum()
    )

    rows = []

    for _, row in df.iterrows():

        label = int(
            row["label"]
        )

        mask = (
            cc == label
        )

        overlap = int(
            np.logical_and(
                mask,
                gt
            ).sum()
        )

        component_voxels = int(
            mask.sum()
        )

        dice = (
            2.0 * overlap
            / max(
                component_voxels
                + gt_voxels,
                1
            )
        )

        union = (
            component_voxels
            + gt_voxels
            - overlap
        )

        iou = (
            overlap
            / max(
                union,
                1
            )
        )

        rows.append({

            "label":
                label,

            "overlap_gt":
                overlap,

            "dice_gt":
                dice,

            "iou_gt":
                iou,

        })

    return df.merge(
        pd.DataFrame(
            rows
        ),
        on="label",
        how="left"
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_policy(
    case,
    cc,
    df,
    ranking_column,
    policy_name
):

    ranked = df.sort_values(
        [
            ranking_column,
            "mean_suv"
        ],
        ascending=False
    ).copy()

    ranked["rank"] = (
        np.arange(
            len(ranked)
        ) + 1
    )

    selected = ranked.head(
        TOP_K
    ).copy()

    metrics = (
        dev10.evaluate_selection(
            case,
            selected,
            cc,
            ranking_column,
            CORE_POLICY,
            TOP_K
        )
    )

    return {

        "method":
            policy_name,

        "dice":
            float(
                metrics["dice"]
            ),

        "iou":
            float(
                metrics["iou"]
            ),

        "slice_recall":
            float(
                metrics["slice_recall"]
            ),

        "fpr":
            float(
                metrics["fpr"]
            ),

        "prediction_voxels":
            int(
                metrics[
                    "prediction_voxels"
                ]
            ),

        "selected_labels":
            ",".join(
                str(
                    int(x)
                )
                for x in selected[
                    "label"
                ]
            ),

    }


# ============================================================
# CASE AUDIT
# ============================================================

def audit_case(
    case_name
):

    print()
    print("=" * 110)
    print(
        f"DEV50 — {case_name}"
    )
    print("=" * 110)

    case = dev10.load_case(
        case_name
    )

    ct = case["ct"]
    suv = case["suv"]
    gt = case["gt"]

    candidate_volume = (
        suv >= PET_THRESHOLD
    ).astype(
        np.uint8
    )

    components, cc = (
        dev10.build_3d_component_table(
            case,
            candidate_volume
        )
    )

    print(
        "Candidate voxels:",
        int(
            candidate_volume.sum()
        )
    )

    print(
        "3D components:",
        len(
            components
        )
    )

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    df = calculate_component_features(
        components,
        cc,
        suv,
        ct
    )

    df = add_dev39_score(
        df
    )

    df = add_ct_scores(
        df
    )

    df = add_ranking_scores(
        df
    )

    df = add_gt_metrics(
        df,
        cc,
        gt
    )

    # --------------------------------------------------------
    # Frozen Dev39 gate
    # --------------------------------------------------------

    df["eligible"] = (
        df["hot_fraction"]
        >= HOT_FRACTION_GATE
    )

    eligible = df[
        df["eligible"]
    ].copy()

    print(
        "Eligible components:",
        len(
            eligible
        )
    )

    # --------------------------------------------------------
    # Evaluate all policies
    # --------------------------------------------------------

    policies = [

        (
            "Dev39",
            "rank_dev39"
        ),

        (
            "Dev50_A_CT_Mild",
            "rank_ct_mild"
        ),

        (
            "Dev50_B_CT_Moderate",
            "rank_ct_moderate"
        ),

    ]

    results = []

    for method_name, ranking_column in policies:

        result = evaluate_policy(
            case,
            cc,
            eligible,
            ranking_column,
            method_name
        )

        result["case"] = (
            case_name
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Component ranking audit
    # --------------------------------------------------------

    audit = eligible.copy()

    audit = audit.sort_values(
        "rank_dev39",
        ascending=False
    ).copy()

    audit[
        "dev39_rank"
    ] = (
        np.arange(
            len(audit)
        ) + 1
    )

    # Print top components under each policy.
    for method_name, ranking_column in policies:

        ranked = eligible.sort_values(
            [
                ranking_column,
                "mean_suv"
            ],
            ascending=False
        ).copy()

        print()
        print(
            method_name,
            "TOP COMPONENTS"
        )

        print(
            ranked[
                [
                    "label",
                    "voxels",
                    "mean_suv",
                    "hot_fraction",
                    "persistence",
                    "ct_mean",
                    "ct_median",
                    "ct_std",
                    "ct_fraction_air",
                    "ct_fraction_soft_tissue",
                    "ct_fraction_dense",
                    "ct_high_variation",
                    "dice_gt",
                    ranking_column,
                ]
            ].head(
                TOP_K
            ).to_string(
                index=False
            )
        )

    return (
        df,
        results
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 110)
    print(
        "DEV50 — CT ANATOMICAL CONTEXT RANKING"
    )
    print("=" * 110)

    print()
    print(
        "Frozen candidate:"
        " PET-only SUV >= 2.25"
    )

    print(
        "Frozen hot-fraction gate:",
        HOT_FRACTION_GATE
    )

    print(
        "Top-K:",
        TOP_K
    )

    print(
        "Core:",
        CORE_POLICY
    )

    print()
    print(
        "Dev39 parameters remain frozen."
    )

    print(
        "dev10_lite_hot_core.py remains untouched."
    )

    all_components = []
    all_results = []

    for case_name in CASES:

        components, results = (
            audit_case(
                case_name
            )
        )

        components[
            "case"
        ] = case_name

        all_components.append(
            components
        )

        all_results.extend(
            results
        )

    components_df = pd.concat(
        all_components,
        ignore_index=True
    )

    results_df = pd.DataFrame(
        all_results
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    components_path = (
        OUTPUT_DIR
        / "dev50_component_audit.csv"
    )

    results_path = (
        OUTPUT_DIR
        / "dev50_case_results.csv"
    )

    components_df.to_csv(
        components_path,
        index=False
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    # --------------------------------------------------------
    # Macro
    # --------------------------------------------------------

    macro = (
        results_df
        .groupby("method")
        .agg(
            macro_dice=(
                "dice",
                "mean"
            ),
            macro_iou=(
                "iou",
                "mean"
            ),
            macro_slice_recall=(
                "slice_recall",
                "mean"
            ),
            macro_fpr=(
                "fpr",
                "mean"
            ),
            total_prediction_voxels=(
                "prediction_voxels",
                "sum"
            ),
            mean_prediction_voxels=(
                "prediction_voxels",
                "mean"
            ),
        )
        .reset_index()
    )

    macro_path = (
        OUTPUT_DIR
        / "dev50_macro_summary.csv"
    )

    macro.to_csv(
        macro_path,
        index=False
    )

    # --------------------------------------------------------
    # Print macro
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print(
        "DEV50 MACRO RESULTS"
    )
    print("=" * 110)

    print(
        macro.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Deltas versus Dev39
    # --------------------------------------------------------

    baseline = macro[
        macro["method"]
        == "Dev39"
    ].iloc[0]

    print()
    print(
        "DELTA VS DEV39"
    )

    for _, row in macro.iterrows():

        if row["method"] == "Dev39":
            continue

        print()
        print(
            row["method"]
        )

        print(
            "Dice:",
            f"{row['macro_dice'] - baseline['macro_dice']:+.6f}"
        )

        print(
            "IoU:",
            f"{row['macro_iou'] - baseline['macro_iou']:+.6f}"
        )

        print(
            "Slice recall:",
            f"{row['macro_slice_recall'] - baseline['macro_slice_recall']:+.6f}"
        )

        print(
            "FPR:",
            f"{row['macro_fpr'] - baseline['macro_fpr']:+.6f}"
        )

        print(
            "Prediction voxels:",
            f"{int(row['total_prediction_voxels'] - baseline['total_prediction_voxels']):+d}"
        )

    print()
    print(
        "Saved:"
    )

    print(
        components_path
    )

    print(
        results_path
    )

    print(
        macro_path
    )

    print()
    print("=" * 110)
    print(
        "DEV50 COMPLETE"
    )
    print("=" * 110)

if __name__ == "__main__":
    main()