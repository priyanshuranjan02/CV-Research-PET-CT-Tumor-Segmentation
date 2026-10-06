# src/dev39_pet_quality_fpr_control.py

import os
import sys
import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import dev10_lite_hot_core as dev10


# ============================================================================
# CONFIGURATION
# ============================================================================

PET_THRESHOLD = 2.25

HOT_THRESHOLD = 3.0

CORE_POLICY = "P70"

TOP_K = 4

DEVELOPMENT_CASES = [
    "PETCT_0168f65af8",
    "PETCT_04606080a0",
    "PETCT_04ab5c61c9",
    "PETCT_0b57b247b6",
    "PETCT_11afab3485",
    "PETCT_185da4c8b6",
]


# ============================================================================
# OUTPUT
# ============================================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "development_cases",
    "dev39_pet_quality_fpr_control",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================================
# PET-ONLY CANDIDATE
# ============================================================================

def generate_pet_only_candidate(case):

    suv = case["suv"]

    return (
        suv >= PET_THRESHOLD
    ).astype(np.uint8)


# ============================================================================
# NORMALIZATION
# ============================================================================

def minmax_normalize(series):

    series = series.astype(float)

    minimum = series.min()
    maximum = series.max()

    if (
        not np.isfinite(minimum)
        or not np.isfinite(maximum)
        or maximum <= minimum
    ):
        return pd.Series(
            np.zeros(len(series)),
            index=series.index,
        )

    return (
        (series - minimum)
        / (maximum - minimum)
    )


# ============================================================================
# COMPONENT FEATURES
# ============================================================================

def add_component_features(
    components,
    cc,
    suv,
):

    df = components.copy()

    feature_rows = []

    for _, row in df.iterrows():

        label = int(
            row["label"]
        )

        mask = (
            cc == label
        )

        coords = np.where(mask)

        voxel_count = int(
            np.count_nonzero(mask)
        )

        if voxel_count == 0:

            feature_rows.append({
                "dev39_voxels": 0,
                "dev39_z_span": 0,
                "dev39_slice_count": 0,
                "dev39_longest_run": 0,
                "dev39_persistence": 0.0,
                "dev39_compactness": 0.0,
                "dev39_mean_suv": 0.0,
                "dev39_max_suv": 0.0,
                "dev39_hot_fraction": 0.0,
            })

            continue

        z_values = coords[0]

        z_start = int(
            z_values.min()
        )

        z_end = int(
            z_values.max()
        )

        z_span = (
            z_end - z_start + 1
        )

        unique_z = np.unique(
            z_values
        )

        slice_count = len(
            unique_z
        )

        # --------------------------------------------------------------
        # Longest consecutive run
        # --------------------------------------------------------------

        longest_run = 1
        current_run = 1

        for i in range(
            1,
            len(unique_z),
        ):

            if (
                unique_z[i]
                == unique_z[i - 1] + 1
            ):
                current_run += 1
            else:
                current_run = 1

            longest_run = max(
                longest_run,
                current_run,
            )

        persistence = (
            longest_run
            / max(z_span, 1)
        )

        # --------------------------------------------------------------
        # SUV statistics
        # --------------------------------------------------------------

        component_suv = suv[mask]

        mean_suv = float(
            np.mean(component_suv)
        )

        max_suv = float(
            np.max(component_suv)
        )

        hot_fraction = float(
            np.mean(
                component_suv
                >= HOT_THRESHOLD
            )
        )

        # --------------------------------------------------------------
        # Bounding-box compactness
        # --------------------------------------------------------------

        x_values = coords[1]
        y_values = coords[2]

        x_span = (
            int(x_values.max())
            - int(x_values.min())
            + 1
        )

        y_span = (
            int(y_values.max())
            - int(y_values.min())
            + 1
        )

        bbox_volume = (
            z_span
            * x_span
            * y_span
        )

        compactness = (
            voxel_count
            / max(
                bbox_volume,
                1,
            )
        )

        feature_rows.append({
            "dev39_voxels": voxel_count,
            "dev39_z_span": z_span,
            "dev39_slice_count": slice_count,
            "dev39_longest_run": longest_run,
            "dev39_persistence": persistence,
            "dev39_compactness": compactness,
            "dev39_mean_suv": mean_suv,
            "dev39_max_suv": max_suv,
            "dev39_hot_fraction": hot_fraction,
        })

    feature_df = pd.DataFrame(
        feature_rows,
        index=df.index,
    )

    return pd.concat(
        [
            df.reset_index(drop=True),
            feature_df.reset_index(drop=True),
        ],
        axis=1,
    )


# ============================================================================
# DEV38 PET QUALITY SCORE
# ============================================================================

def add_pet_quality_score(df):

    result = df.copy()

    result["n_mean_suv"] = (
        minmax_normalize(
            result["dev39_mean_suv"]
        )
    )

    result["n_max_suv"] = (
        minmax_normalize(
            result["dev39_max_suv"]
        )
    )

    result["n_hot_fraction"] = (
        minmax_normalize(
            result["dev39_hot_fraction"]
        )
    )

    result["n_persistence"] = (
        minmax_normalize(
            result["dev39_persistence"]
        )
    )

    result["n_longest_run"] = (
        minmax_normalize(
            np.log1p(
                result["dev39_longest_run"]
            )
        )
    )

    result["n_compactness"] = (
        minmax_normalize(
            result["dev39_compactness"]
        )
    )

    # --------------------------------------------------------------
    # Exact winning Dev38 PET-quality formulation
    # --------------------------------------------------------------

    result["dev39_pet_quality"] = (
        0.25 * result["n_mean_suv"]
        + 0.15 * result["n_max_suv"]
        + 0.15 * result["n_hot_fraction"]
        + 0.20 * result["n_persistence"]
        + 0.15 * result["n_longest_run"]
        + 0.10 * result["n_compactness"]
    )

    return result


# ============================================================================
# FPR CONTROL GATES
# ============================================================================

def apply_gate(
    df,
    gate_name,
):

    result = df.copy()

    if gate_name == "baseline":
        return result

    if gate_name == "persistence_0p50":

        return result[
            result["dev39_persistence"]
            >= 0.50
        ].copy()

    if gate_name == "persistence_0p60":

        return result[
            result["dev39_persistence"]
            >= 0.60
        ].copy()

    if gate_name == "hot_fraction_0p30":

        return result[
            result["dev39_hot_fraction"]
            >= 0.30
        ].copy()

    if gate_name == "hot_fraction_0p40":

        return result[
            result["dev39_hot_fraction"]
            >= 0.40
        ].copy()

    if gate_name == "mean_suv_2p50":

        return result[
            result["dev39_mean_suv"]
            >= 2.50
        ].copy()

    if gate_name == "mean_suv_3p00":

        return result[
            result["dev39_mean_suv"]
            >= 3.00
        ].copy()

    if gate_name == "persistence_hot":

        return result[
            (
                result["dev39_persistence"]
                >= 0.50
            )
            &
            (
                result["dev39_hot_fraction"]
                >= 0.30
            )
        ].copy()

    if gate_name == "persistence_suv":

        return result[
            (
                result["dev39_persistence"]
                >= 0.50
            )
            &
            (
                result["dev39_mean_suv"]
                >= 2.50
            )
        ].copy()

    if gate_name == "hot_suv":

        return result[
            (
                result["dev39_hot_fraction"]
                >= 0.30
            )
            &
            (
                result["dev39_mean_suv"]
                >= 2.50
            )
        ].copy()

    if gate_name == "conservative":

        return result[
            (
                result["dev39_persistence"]
                >= 0.50
            )
            &
            (
                result["dev39_hot_fraction"]
                >= 0.30
            )
            &
            (
                result["dev39_mean_suv"]
                >= 2.50
            )
        ].copy()

    raise ValueError(
        f"Unknown gate: {gate_name}"
    )


# ============================================================================
# GATE LIST
# ============================================================================

GATES = [
    "baseline",
    "persistence_0p50",
    "persistence_0p60",
    "hot_fraction_0p30",
    "hot_fraction_0p40",
    "mean_suv_2p50",
    "mean_suv_3p00",
    "persistence_hot",
    "persistence_suv",
    "hot_suv",
    "conservative",
]


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_gate(
    case,
    components,
    cc,
    gate_name,
):

    filtered = apply_gate(
        components,
        gate_name,
    )

    if len(filtered) == 0:

        return {
            "dice": 0.0,
            "iou": 0.0,
            "slice_recall": 0.0,
            "fpr": 0.0,
            "prediction_voxels": 0,
            "eligible_components": 0,
            "selected_labels": [],
        }

    # --------------------------------------------------------------
    # Rank by winning Dev38 PET quality score
    # --------------------------------------------------------------

    ranked = (
        filtered
        .sort_values(
            by=[
                "dev39_pet_quality",
                "dev39_mean_suv",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------------
    # Exact Dev10 downstream evaluation
    # --------------------------------------------------------------

    metrics = dev10.evaluate_selection(
        case,
        ranked,
        cc,
        "dev39_pet_quality",
        CORE_POLICY,
        TOP_K,
    )

    selected_labels = (
        ranked
        .head(TOP_K)["label"]
        .astype(int)
        .tolist()
    )

    return {
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
        "eligible_components": int(
            len(filtered)
        ),
        "selected_labels":
            selected_labels,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 100)
    print("DEV39 - PET QUALITY FPR CONTROL")
    print("=" * 100)

    print()
    print("Configuration")
    print("-" * 80)

    print(
        f"PET threshold : {PET_THRESHOLD}"
    )

    print(
        f"Hot threshold : {HOT_THRESHOLD}"
    )

    print(
        f"Core policy   : {CORE_POLICY}"
    )

    print(
        f"Top-K         : {TOP_K}"
    )

    print()
    print("Gates:")

    for gate in GATES:
        print(
            f"  - {gate}"
        )

    all_results = []
    component_audits = []

    # ==================================================================
    # CASE LOOP
    # ==================================================================

    for case_name in DEVELOPMENT_CASES:

        print()
        print("=" * 100)
        print(
            f"CASE: {case_name}"
        )
        print("=" * 100)

        case = dev10.load_case(
            case_name
        )

        print(
            f"CT : {case['ct'].shape}"
        )

        print(
            f"PET: {case['suv'].shape}"
        )

        print(
            f"SUV max: "
            f"{case['suv'].max():.4f}"
        )

        print(
            f"GT voxels: "
            f"{int(np.count_nonzero(case['gt']))}"
        )

        # --------------------------------------------------------------
        # PET-only candidate
        # --------------------------------------------------------------

        candidate = (
            generate_pet_only_candidate(
                case
            )
        )

        print(
            f"PET candidate voxels: "
            f"{int(np.count_nonzero(candidate))}"
        )

        # --------------------------------------------------------------
        # Exact Dev10 3D component extraction
        # --------------------------------------------------------------

        components, cc = (
            dev10.build_3d_component_table(
                case,
                candidate,
            )
        )

        print(
            f"3D components: "
            f"{len(components)}"
        )

        # --------------------------------------------------------------
        # Features
        # --------------------------------------------------------------

        components = (
            add_component_features(
                components,
                cc,
                case["suv"],
            )
        )

        components = (
            add_pet_quality_score(
                components
            )
        )

        # --------------------------------------------------------------
        # Component audit
        # --------------------------------------------------------------

        gt = case["gt"].astype(bool)

        for _, row in components.iterrows():

            label = int(
                row["label"]
            )

            mask = (
                cc == label
            )

            overlap = int(
                np.count_nonzero(
                    mask & gt
                )
            )

            component_audits.append({
                "case":
                    case_name,
                "label":
                    label,
                "voxels":
                    int(
                        row[
                            "dev39_voxels"
                        ]
                    ),
                "mean_suv":
                    float(
                        row[
                            "dev39_mean_suv"
                        ]
                    ),
                "max_suv":
                    float(
                        row[
                            "dev39_max_suv"
                        ]
                    ),
                "hot_fraction":
                    float(
                        row[
                            "dev39_hot_fraction"
                        ]
                    ),
                "persistence":
                    float(
                        row[
                            "dev39_persistence"
                        ]
                    ),
                "longest_run":
                    int(
                        row[
                            "dev39_longest_run"
                        ]
                    ),
                "compactness":
                    float(
                        row[
                            "dev39_compactness"
                        ]
                    ),
                "pet_quality":
                    float(
                        row[
                            "dev39_pet_quality"
                        ]
                    ),
                "gt_overlap":
                    overlap,
            })

        # --------------------------------------------------------------
        # Evaluate all gates
        # --------------------------------------------------------------

        for gate_name in GATES:

            metrics = evaluate_gate(
                case,
                components,
                cc,
                gate_name,
            )

            print()
            print(
                f"  Gate: {gate_name}"
            )

            print(
                f"    Eligible components: "
                f"{metrics['eligible_components']}"
            )

            print(
                f"    Selected labels: "
                f"{metrics['selected_labels']}"
            )

            print(
                f"    Dice: "
                f"{metrics['dice']:.6f}"
            )

            print(
                f"    IoU: "
                f"{metrics['iou']:.6f}"
            )

            print(
                f"    Slice recall: "
                f"{metrics['slice_recall']:.6f}"
            )

            print(
                f"    FPR: "
                f"{metrics['fpr']:.6f}"
            )

            print(
                f"    Prediction voxels: "
                f"{metrics['prediction_voxels']}"
            )

            all_results.append({
                "case":
                    case_name,
                "gate":
                    gate_name,
                "core_policy":
                    CORE_POLICY,
                "top_k":
                    TOP_K,
                "dice":
                    metrics["dice"],
                "iou":
                    metrics["iou"],
                "slice_recall":
                    metrics["slice_recall"],
                "fpr":
                    metrics["fpr"],
                "prediction_voxels":
                    metrics[
                        "prediction_voxels"
                    ],
                "eligible_components":
                    metrics[
                        "eligible_components"
                    ],
                "selected_labels":
                    str(
                        metrics[
                            "selected_labels"
                        ]
                    ),
            })

    # ==================================================================
    # DATAFRAMES
    # ==================================================================

    results_df = pd.DataFrame(
        all_results
    )

    component_df = pd.DataFrame(
        component_audits
    )

    # ==================================================================
    # MACRO
    # ==================================================================

    macro_df = (
        results_df
        .groupby(
            [
                "gate",
                "core_policy",
                "top_k",
            ],
            as_index=False,
        )
        .agg(
            macro_dice=(
                "dice",
                "mean",
            ),
            macro_iou=(
                "iou",
                "mean",
            ),
            macro_slice_recall=(
                "slice_recall",
                "mean",
            ),
            macro_fpr=(
                "fpr",
                "mean",
            ),
            total_prediction_voxels=(
                "prediction_voxels",
                "sum",
            ),
            mean_prediction_voxels=(
                "prediction_voxels",
                "mean",
            ),
            mean_eligible_components=(
                "eligible_components",
                "mean",
            ),
        )
        .sort_values(
            "macro_dice",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # ==================================================================
    # TRADEOFF TABLE
    # ==================================================================

    macro_df["dice_delta_vs_dev10"] = (
        macro_df["macro_dice"]
        - 0.100912
    )

    macro_df["fpr_delta_vs_dev10"] = (
        macro_df["macro_fpr"]
        - 0.081641
    )

    macro_df["dice_delta_vs_dev38"] = (
        macro_df["macro_dice"]
        - 0.120468
    )

    macro_df["fpr_delta_vs_dev38"] = (
        macro_df["macro_fpr"]
        - 0.115093
    )

    # A practical target:
    # retain most of Dev38 Dice while reducing FPR.
    macro_df["passes_target"] = (
        (macro_df["macro_dice"] >= 0.115)
        &
        (macro_df["macro_fpr"] < 0.10)
    )

    # ==================================================================
    # SAVE
    # ==================================================================

    results_path = os.path.join(
        OUTPUT_DIR,
        "dev39_case_results.csv",
    )

    macro_path = os.path.join(
        OUTPUT_DIR,
        "dev39_macro_summary.csv",
    )

    component_path = os.path.join(
        OUTPUT_DIR,
        "dev39_component_audit.csv",
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    macro_df.to_csv(
        macro_path,
        index=False,
    )

    component_df.to_csv(
        component_path,
        index=False,
    )

    # ==================================================================
    # PRINT MACRO
    # ==================================================================

    print()
    print("=" * 100)
    print("DEV39 MACRO RESULTS")
    print("=" * 100)

    print(
        macro_df.to_string(
            index=False
        )
    )

    # ==================================================================
    # BEST DICE
    # ==================================================================

    print()
    print("=" * 100)
    print("BEST DICE CONFIGURATION")
    print("=" * 100)

    best_dice = (
        macro_df
        .sort_values(
            "macro_dice",
            ascending=False,
        )
        .iloc[0]
    )

    print(
        f"Gate         : "
        f"{best_dice['gate']}"
    )

    print(
        f"Dice         : "
        f"{best_dice['macro_dice']:.6f}"
    )

    print(
        f"IoU          : "
        f"{best_dice['macro_iou']:.6f}"
    )

    print(
        f"Slice Recall : "
        f"{best_dice['macro_slice_recall']:.6f}"
    )

    print(
        f"FPR          : "
        f"{best_dice['macro_fpr']:.6f}"
    )

    # ==================================================================
    # BEST FPR-CONSTRAINED CONFIGURATION
    # ==================================================================

    print()
    print("=" * 100)
    print("BEST FPR-CONSTRAINED CONFIGURATION")
    print("=" * 100)

    constrained = macro_df[
        macro_df["macro_fpr"] < 0.10
    ].copy()

    if len(constrained) > 0:

        best_constrained = (
            constrained
            .sort_values(
                "macro_dice",
                ascending=False,
            )
            .iloc[0]
        )

        print(
            f"Gate         : "
            f"{best_constrained['gate']}"
        )

        print(
            f"Dice         : "
            f"{best_constrained['macro_dice']:.6f}"
        )

        print(
            f"IoU          : "
            f"{best_constrained['macro_iou']:.6f}"
        )

        print(
            f"Slice Recall : "
            f"{best_constrained['macro_slice_recall']:.6f}"
        )

        print(
            f"FPR          : "
            f"{best_constrained['macro_fpr']:.6f}"
        )

        print(
            f"Passes target: "
            f"{bool(best_constrained['passes_target'])}"
        )

    else:

        print(
            "No configuration achieved "
            "FPR < 0.10."
        )

    # ==================================================================
    # BENCHMARKS
    # ==================================================================

    print()
    print("=" * 100)
    print("BENCHMARKS")
    print("=" * 100)

    print(
        "Dev10 P70 K4:"
    )

    print(
        "  Dice = 0.100912"
    )

    print(
        "  FPR  = 0.081641"
    )

    print()

    print(
        "Dev38 PET Quality P70 K4:"
    )

    print(
        "  Dice = 0.120468"
    )

    print(
        "  FPR  = 0.115093"
    )

    # ==================================================================
    # COMPLETE
    # ==================================================================

    print()
    print("=" * 100)
    print("DEV39 COMPLETE")
    print("=" * 100)

    print()
    print("Saved:")

    print(results_path)
    print(macro_path)
    print(component_path)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()