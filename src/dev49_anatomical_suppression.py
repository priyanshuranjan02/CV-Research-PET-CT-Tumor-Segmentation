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
# DEV49 CONFIGURATION
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

# ------------------------------------------------------------
# Anatomical suppression hypotheses
# ------------------------------------------------------------
#
# These are RELATIVE z positions, not absolute slice numbers.
#
# A = aggressive:
#     suppress components whose centroid lies in the upper
#     30% of the aligned volume AND which are large/persistent.
#
# B = conservative:
#     suppress only extremely large AND highly persistent
#     components in the upper 20%.
#
# ------------------------------------------------------------

A_Z_THRESHOLD = 0.30
A_MIN_VOLUME_FRACTION = 0.10
A_MIN_PERSISTENCE = 0.05

B_Z_THRESHOLD = 0.20
B_MIN_VOLUME_FRACTION = 0.20
B_MIN_PERSISTENCE = 0.10


OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "development_cases"
    / "dev49_anatomical_suppression"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
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
        return np.ones_like(values)

    return (
        values - vmin
    ) / (
        vmax - vmin
    )


# ============================================================
# LONGEST Z RUN
# ============================================================

def longest_consecutive_run(z_values):

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
# COMPONENT FEATURES
# ============================================================

def calculate_features(
    components,
    cc,
    suv
):

    rows = []

    total_voxels = np.prod(
        suv.shape
    )

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

        suv_values = suv[
            mask
        ]

        voxels = len(coords)

        z_start = int(
            z.min()
        )

        z_end = int(
            z.max()
        )

        z_span = (
            z_end
            - z_start
            + 1
        )

        unique_z = np.unique(
            z
        )

        slice_count = len(
            unique_z
        )

        persistence = (
            slice_count
            / suv.shape[0]
        )

        longest_run = (
            longest_consecutive_run(
                unique_z
            )
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

        mean_suv = float(
            np.mean(
                suv_values
            )
        )

        max_suv = float(
            np.max(
                suv_values
            )
        )

        hot_fraction = float(
            np.mean(
                suv_values
                >= HOT_THRESHOLD
            )
        )

        centroid_z = float(
            np.mean(z)
        )

        centroid_y = float(
            np.mean(y)
        )

        centroid_x = float(
            np.mean(x)
        )

        # Relative axial position.
        #
        # 0 = beginning of volume
        # 1 = end of volume
        #
        relative_z = (
            centroid_z
            / max(
                suv.shape[0] - 1,
                1
            )
        )

        volume_fraction = (
            voxels
            / total_voxels
        )

        rows.append({

            "label": label,

            "voxels": voxels,

            "volume_fraction":
                volume_fraction,

            "z_start": z_start,
            "z_end": z_end,
            "z_span": z_span,

            "slice_count":
                slice_count,

            "persistence":
                persistence,

            "longest_run":
                longest_run,

            "centroid_z":
                centroid_z,

            "centroid_y":
                centroid_y,

            "centroid_x":
                centroid_x,

            "relative_z":
                relative_z,

            "mean_suv":
                mean_suv,

            "max_suv":
                max_suv,

            "hot_fraction":
                hot_fraction,

            "compactness":
                compactness,

        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# PET QUALITY SCORE
# ============================================================

def add_pet_quality_score(df):

    if df.empty:
        return df

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

    # EXACT Dev39 score.
    df["pet_quality_score"] = (
        0.25 * df["n_mean_suv"]
        + 0.15 * df["n_max_suv"]
        + 0.15 * df["n_hot_fraction"]
        + 0.20 * df["n_persistence"]
        + 0.15 * df["n_longest_run"]
        + 0.10 * df["n_compactness"]
    )

    return df


# ============================================================
# GT METRICS
# ============================================================

def add_gt_metrics(
    df,
    cc,
    gt
):

    rows = []

    gt_voxels = int(
        gt.sum()
    )

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

            "label": label,

            "overlap_gt":
                overlap,

            "dice_gt":
                dice,

            "iou_gt":
                iou,

        })

    return df.merge(
        pd.DataFrame(rows),
        on="label",
        how="left"
    )


# ============================================================
# SUPPRESSION RULES
# ============================================================

def apply_suppression_flags(
    df
):

    df = df.copy()

    # --------------------------------------------------------
    # Dev49-A: aggressive
    # --------------------------------------------------------

    df["suppress_A"] = (
        (df["relative_z"] <= A_Z_THRESHOLD)
        &
        (
            df["volume_fraction"]
            >= A_MIN_VOLUME_FRACTION
        )
        &
        (
            df["persistence"]
            >= A_MIN_PERSISTENCE
        )
    )

    # --------------------------------------------------------
    # Dev49-B: conservative
    # --------------------------------------------------------

    df["suppress_B"] = (
        (df["relative_z"] <= B_Z_THRESHOLD)
        &
        (
            df["volume_fraction"]
            >= B_MIN_VOLUME_FRACTION
        )
        &
        (
            df["persistence"]
            >= B_MIN_PERSISTENCE
        )
    )

    return df


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
# EXACT DEV10 EVALUATION
# ============================================================

def evaluate_method(
    case,
    cc,
    eligible,
    method_name
):

    if len(eligible) == 0:

        return {

            "method": method_name,

            "dice": 0.0,
            "iou": 0.0,
            "slice_recall": 0.0,
            "fpr": 0.0,
            "prediction_voxels": 0,

            "selected_labels": "",

        }

    ranked = eligible.sort_values(
        [
            "pet_quality_score",
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
    )

    # IMPORTANT:
    # Use exact Dev10 evaluator downstream.
    metrics = dev10.evaluate_selection(
        case,
        selected,
        cc,
        "pet_quality_score",
        CORE_POLICY,
        TOP_K
    )

    labels = [
        int(x)
        for x in selected[
            "label"
        ].tolist()
    ]

    return {

        "method": method_name,

        "dice": float(
            metrics["dice"]
        ),

        "iou": float(
            metrics["iou"]
        ),

        "slice_recall": float(
            metrics["slice_recall"]
        ),

        "fpr": float(
            metrics["fpr"]
        ),

        "prediction_voxels": int(
            metrics["prediction_voxels"]
        ),

        "selected_labels":
            ",".join(
                map(
                    str,
                    labels
                )
            ),

    }


# ============================================================
# SINGLE CASE
# ============================================================

def audit_case(
    case_name
):

    print()
    print("=" * 110)
    print(
        f"DEV49 — {case_name}"
    )
    print("=" * 110)

    case = dev10.load_case(
        case_name
    )

    suv = case["suv"]

    gt = case["gt"]

    # --------------------------------------------------------
    # PET-only candidate
    # --------------------------------------------------------

    candidate_volume = (
        suv >= PET_THRESHOLD
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Exact Dev10 3D components
    # --------------------------------------------------------

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
    # Features
    # --------------------------------------------------------

    df = calculate_features(
        components,
        cc,
        suv
    )

    df = add_pet_quality_score(
        df
    )

    df = add_gt_metrics(
        df,
        cc,
        gt
    )

    # --------------------------------------------------------
    # Dev39 gate
    # --------------------------------------------------------

    df["eligible_dev39"] = (
        df["hot_fraction"]
        >= HOT_FRACTION_GATE
    )

    eligible = df[
        df["eligible_dev39"]
    ].copy()

    # --------------------------------------------------------
    # Suppression flags
    # --------------------------------------------------------

    df = apply_suppression_flags(
        df
    )

    # Recreate eligible subsets after flags.
    eligible = df[
        df["eligible_dev39"]
    ].copy()

    eligible_A = eligible[
        ~eligible["suppress_A"]
    ].copy()

    eligible_B = eligible[
        ~eligible["suppress_B"]
    ].copy()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    baseline = evaluate_method(
        case,
        cc,
        eligible,
        "Dev39"
    )

    method_A = evaluate_method(
        case,
        cc,
        eligible_A,
        "Dev49_A"
    )

    method_B = evaluate_method(
        case,
        cc,
        eligible_B,
        "Dev49_B"
    )

    results = [
        baseline,
        method_A,
        method_B,
    ]

    # --------------------------------------------------------
    # Suppression audit
    # --------------------------------------------------------

    ranked = eligible.sort_values(
        [
            "pet_quality_score",
            "mean_suv"
        ],
        ascending=False
    ).copy()

    ranked["dev39_rank"] = (
        np.arange(
            len(ranked)
        ) + 1
    )

    selected_dev39 = ranked.head(
        TOP_K
    ).copy()

    selected_dev39[
        "suppressed_A"
    ] = selected_dev39[
        "label"
    ].isin(
        eligible_A[
            "label"
        ]
    ) == False

    selected_dev39[
        "suppressed_B"
    ] = selected_dev39[
        "label"
    ].isin(
        eligible_B[
            "label"
        ]
    ) == False

    print()
    print(
        "DEV39 SELECTED COMPONENTS"
    )

    print(
        selected_dev39[
            [
                "dev39_rank",
                "label",
                "voxels",
                "relative_z",
                "volume_fraction",
                "persistence",
                "mean_suv",
                "hot_fraction",
                "pet_quality_score",
                "overlap_gt",
                "dice_gt",
                "suppress_A",
                "suppress_B",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "RESULTS"
    )

    for result in results:

        print(
            f"{result['method']:10s} "
            f"Dice={result['dice']:.6f} "
            f"IoU={result['iou']:.6f} "
            f"Recall={result['slice_recall']:.6f} "
            f"FPR={result['fpr']:.6f} "
            f"Pred={result['prediction_voxels']} "
            f"Labels=[{result['selected_labels']}]"
        )

    return (
        df,
        selected_dev39,
        results
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 110)
    print(
        "DEV49 — ANATOMICAL SUPPRESSION EXPERIMENT"
    )
    print("=" * 110)

    print()
    print(
        "Dev39 is frozen."
    )

    print(
        "dev10_lite_hot_core.py is untouched."
    )

    print()
    print(
        "Dev49-A:"
    )

    print(
        f"relative_z <= {A_Z_THRESHOLD}, "
        f"volume_fraction >= "
        f"{A_MIN_VOLUME_FRACTION}, "
        f"persistence >= "
        f"{A_MIN_PERSISTENCE}"
    )

    print()
    print(
        "Dev49-B:"
    )

    print(
        f"relative_z <= {B_Z_THRESHOLD}, "
        f"volume_fraction >= "
        f"{B_MIN_VOLUME_FRACTION}, "
        f"persistence >= "
        f"{B_MIN_PERSISTENCE}"
    )

    all_components = []
    all_selected = []
    all_results = []

    for case_name in CASES:

        (
            components,
            selected,
            results
        ) = audit_case(
            case_name
        )

        components[
            "case"
        ] = case_name

        selected[
            "case"
        ] = case_name

        for result in results:

            result["case"] = (
                case_name
            )

        all_components.append(
            components
        )

        all_selected.append(
            selected
        )

        all_results.extend(
            results
        )

    # --------------------------------------------------------
    # Save component audit
    # --------------------------------------------------------

    components_df = pd.concat(
        all_components,
        ignore_index=True
    )

    selected_df = pd.concat(
        all_selected,
        ignore_index=True
    )

    results_df = pd.DataFrame(
        all_results
    )

    components_path = (
        OUTPUT_DIR
        / "dev49_component_audit.csv"
    )

    selected_path = (
        OUTPUT_DIR
        / "dev49_selected_components.csv"
    )

    results_path = (
        OUTPUT_DIR
        / "dev49_case_results.csv"
    )

    components_df.to_csv(
        components_path,
        index=False
    )

    selected_df.to_csv(
        selected_path,
        index=False
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    # --------------------------------------------------------
    # Macro summary
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
        / "dev49_macro_summary.csv"
    )

    macro.to_csv(
        macro_path,
        index=False
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print(
        "DEV49 MACRO RESULTS"
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
            "Dice delta:",
            f"{row['macro_dice'] - baseline['macro_dice']:+.6f}"
        )

        print(
            "IoU delta:",
            f"{row['macro_iou'] - baseline['macro_iou']:+.6f}"
        )

        print(
            "Recall delta:",
            f"{row['macro_slice_recall'] - baseline['macro_slice_recall']:+.6f}"
        )

        print(
            "FPR delta:",
            f"{row['macro_fpr'] - baseline['macro_fpr']:+.6f}"
        )

        print(
            "Prediction voxel delta:",
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
        selected_path
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
        "DEV49 COMPLETE"
    )
    print("=" * 110)

if __name__ == "__main__":
    main()