from pathlib import Path
import sys
import numpy as np
import pandas as pd
import cv2


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import dev10_lite_hot_core as dev10


# ============================================================
# DEV51 CONFIGURATION
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
    / "dev51_ct_structural_ranking"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAFE CORE HANDLING
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

    if isinstance(result, tuple):

        for item in result:

            if isinstance(
                item,
                np.ndarray
            ):

                return item.astype(bool)

    return result.astype(bool)


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

        if z_values[i] == z_values[i - 1] + 1:

            current += 1

            best = max(
                best,
                current
            )

        else:

            current = 1

    return best


# ============================================================
# 2D CT STRUCTURAL MAP
# ============================================================

def compute_ct_structural_maps(
    ct
):

    # --------------------------------------------------------
    # Normalize CT only for gradient calculation.
    # The original HU values remain untouched.
    # --------------------------------------------------------

    ct_float = ct.astype(
        np.float32
    )

    # Robust clipping prevents extreme CT values from
    # dominating gradient computation.
    clipped = np.clip(
        ct_float,
        -1000,
        1000
    )

    normalized = (
        clipped + 1000.0
    ) / 2000.0

    gradient_magnitude = np.zeros_like(
        normalized,
        dtype=np.float32
    )

    edge_map = np.zeros_like(
        normalized,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Process each axial slice.
    # --------------------------------------------------------

    for z in range(
        normalized.shape[0]
    ):

        image = normalized[z]

        gx = cv2.Sobel(
            image,
            cv2.CV_32F,
            1,
            0,
            ksize=3
        )

        gy = cv2.Sobel(
            image,
            cv2.CV_32F,
            0,
            1,
            ksize=3
        )

        magnitude = cv2.magnitude(
            gx,
            gy
        )

        gradient_magnitude[z] = (
            magnitude
        )

        # Adaptive threshold based on the
        # slice's gradient distribution.
        threshold = np.percentile(
            magnitude,
            85
        )

        edges = (
            magnitude >= threshold
        )

        edge_map[z] = (
            edges.astype(
                np.uint8
            )
        )

    return (
        gradient_magnitude,
        edge_map
    )


# ============================================================
# PET COMPONENT BOUNDARY
# ============================================================

def get_component_boundary(
    mask
):

    boundary = np.zeros_like(
        mask,
        dtype=bool
    )

    z_values = np.where(
        mask.any(axis=(1, 2))
    )[0]

    for z in z_values:

        slice_mask = (
            mask[z].astype(
                np.uint8
            )
        )

        if slice_mask.sum() == 0:
            continue

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        eroded = cv2.erode(
            slice_mask,
            kernel,
            iterations=1
        )

        slice_boundary = (
            slice_mask > 0
        ) & (
            eroded == 0
        )

        boundary[z] = (
            slice_boundary
        )

    return boundary


# ============================================================
# COMPONENT FEATURES
# ============================================================

def calculate_component_features(
    components,
    cc,
    suv,
    ct,
    gradient_map,
    edge_map
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

        gradient_values = gradient_map[
            mask
        ]

        edge_values = edge_map[
            mask
        ]

        voxels = len(
            coords
        )

        # ====================================================
        # PET FEATURES
        # ====================================================

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

        # ====================================================
        # PET SHAPE
        # ====================================================

        z_span = (
            int(z.max())
            - int(z.min())
            + 1
        )

        bbox_volume = (
            (z.max() - z.min() + 1)
            *
            (y.max() - y.min() + 1)
            *
            (x.max() - x.min() + 1)
        )

        compactness = (
            voxels
            / max(
                bbox_volume,
                1
            )
        )

        # ====================================================
        # CT BASIC FEATURES
        # ====================================================

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

        # ====================================================
        # CT STRUCTURAL FEATURES
        # ====================================================

        gradient_mean = float(
            np.mean(
                gradient_values
            )
        )

        gradient_median = float(
            np.median(
                gradient_values
            )
        )

        gradient_p90 = float(
            np.percentile(
                gradient_values,
                90
            )
        )

        edge_density = float(
            np.mean(
                edge_values > 0
            )
        )

        # ====================================================
        # PET / CT BOUNDARY AGREEMENT
        # ====================================================

        boundary = (
            get_component_boundary(
                mask
            )
        )

        boundary_gradient = gradient_map[
            boundary
        ]

        boundary_edges = edge_map[
            boundary
        ]

        if boundary_gradient.size > 0:

            boundary_gradient_mean = float(
                np.mean(
                    boundary_gradient
                )
            )

            boundary_gradient_p75 = float(
                np.percentile(
                    boundary_gradient,
                    75
                )
            )

            boundary_edge_density = float(
                np.mean(
                    boundary_edges > 0
                )
            )

        else:

            boundary_gradient_mean = 0.0
            boundary_gradient_p75 = 0.0
            boundary_edge_density = 0.0

        # ====================================================
        # LOCAL CT VARIATION
        # ====================================================

        local_variation = float(
            np.mean(
                np.abs(
                    ct_values
                    - ct_median
                )
            )
        )

        high_variation_fraction = float(
            np.mean(
                np.abs(
                    ct_values
                    - ct_median
                ) > 100
            )
        )

        # ====================================================
        # CT TISSUE FRACTIONS
        # ====================================================

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

        rows.append({

            "label": label,

            "voxels": voxels,

            "z_start":
                int(z.min()),

            "z_end":
                int(z.max()),

            "z_span":
                z_span,

            "slice_count":
                slice_count,

            "persistence":
                persistence,

            "longest_run":
                longest_run,

            "compactness":
                compactness,

            "centroid_z":
                float(np.mean(z)),

            "centroid_y":
                float(np.mean(y)),

            "centroid_x":
                float(np.mean(x)),

            # PET
            "mean_suv":
                mean_suv,

            "max_suv":
                max_suv,

            "hot_fraction":
                hot_fraction,

            # CT
            "ct_mean":
                ct_mean,

            "ct_median":
                ct_median,

            "ct_std":
                ct_std,

            "ct_fraction_air":
                fraction_air,

            "ct_fraction_soft_tissue":
                fraction_soft_tissue,

            "ct_fraction_dense":
                fraction_dense,

            # CT structural
            "ct_gradient_mean":
                gradient_mean,

            "ct_gradient_median":
                gradient_median,

            "ct_gradient_p90":
                gradient_p90,

            "ct_edge_density":
                edge_density,

            "boundary_gradient_mean":
                boundary_gradient_mean,

            "boundary_gradient_p75":
                boundary_gradient_p75,

            "boundary_edge_density":
                boundary_edge_density,

            "ct_local_variation":
                local_variation,

            "ct_high_variation_fraction":
                high_variation_fraction,

        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# DEV39 PET QUALITY SCORE
# ============================================================

def add_dev39_score(
    df
):

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
        0.25
        * df["n_mean_suv"]

        + 0.15
        * df["n_max_suv"]

        + 0.15
        * df["n_hot_fraction"]

        + 0.20
        * df["n_persistence"]

        + 0.15
        * df["n_longest_run"]

        + 0.10
        * df["n_compactness"]
    )

    return df


# ============================================================
# DEV51 CT STRUCTURAL SCORE
# ============================================================

def add_ct_structural_score(
    df
):

    df = df.copy()

    # --------------------------------------------------------
    # Normalize CT structural features
    # --------------------------------------------------------

    df["n_gradient"] = (
        minmax_normalize(
            df["ct_gradient_mean"]
        )
    )

    df["n_gradient_p90"] = (
        minmax_normalize(
            df["ct_gradient_p90"]
        )
    )

    df["n_edge_density"] = (
        minmax_normalize(
            df["ct_edge_density"]
        )
    )

    df["n_boundary_gradient"] = (
        minmax_normalize(
            df["boundary_gradient_mean"]
        )
    )

    df["n_boundary_edge"] = (
        minmax_normalize(
            df["boundary_edge_density"]
        )
    )

    df["n_local_variation"] = (
        minmax_normalize(
            df["ct_local_variation"]
        )
    )

    # --------------------------------------------------------
    # Structural score
    #
    # Boundary information receives more weight than
    # interior CT intensity.
    # --------------------------------------------------------

    df["ct_structural_score"] = (

        0.25
        * df["n_boundary_gradient"]

        + 0.20
        * df["n_boundary_edge"]

        + 0.20
        * df["n_gradient"]

        + 0.15
        * df["n_gradient_p90"]

        + 0.10
        * df["n_edge_density"]

        + 0.10
        * df["n_local_variation"]

    )

    # --------------------------------------------------------
    # PET/CT structural agreement
    #
    # A component gets stronger structural support when
    # both its boundary gradient and boundary edge density
    # are meaningful.
    # --------------------------------------------------------

    df["pet_ct_boundary_agreement"] = (
        np.sqrt(
            np.clip(
                df["n_boundary_gradient"],
                0,
                1
            )
            *
            np.clip(
                df["n_boundary_edge"],
                0,
                1
            )
        )
    )

    return df


# ============================================================
# DEV51 RANKING POLICIES
# ============================================================

def add_dev51_rankings(
    df
):

    df = df.copy()

    # --------------------------------------------------------
    # Frozen Dev39
    # --------------------------------------------------------

    df["rank_dev39"] = (
        df["dev39_score"]
    )

    # --------------------------------------------------------
    # Dev51-A
    #
    # 80% PET quality
    # 20% CT structure
    # --------------------------------------------------------

    df["rank_dev51_a"] = (

        0.80
        * df["dev39_score"]

        + 0.20
        * df["ct_structural_score"]

    )

    # --------------------------------------------------------
    # Dev51-B
    #
    # 65% PET quality
    # 35% CT structure
    # --------------------------------------------------------

    df["rank_dev51_b"] = (

        0.65
        * df["dev39_score"]

        + 0.35
        * df["ct_structural_score"]

    )

    return df


# ============================================================
# GT AUDIT
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

        component_voxels = int(
            mask.sum()
        )

        overlap = int(
            np.logical_and(
                mask,
                gt
            ).sum()
        )

        dice = (
            2.0
            * overlap
            /
            max(
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
            /
            max(
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
        pd.DataFrame(rows),
        on="label",
        how="left"
    )


# ============================================================
# EVALUATE ONE RANKING
# ============================================================

def evaluate_policy(
    case,
    cc,
    eligible,
    ranking_column,
    method_name
):

    ranked = (
        eligible
        .sort_values(
            [
                ranking_column,
                "mean_suv"
            ],
            ascending=False
        )
        .copy()
    )

    ranked["rank"] = (
        np.arange(
            len(ranked)
        ) + 1
    )

    selected = ranked.head(
        TOP_K
    ).copy()

    # --------------------------------------------------------
    # IMPORTANT:
    # Use the exact Dev10 evaluator.
    # --------------------------------------------------------

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
            method_name,

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
                    int(label)
                )
                for label in selected[
                    "label"
                ]
            ),

    }


# ============================================================
# ONE CASE
# ============================================================

def audit_case(
    case_name
):

    print()
    print("=" * 110)
    print(
        f"DEV51 — {case_name}"
    )
    print("=" * 110)

    case = dev10.load_case(
        case_name
    )

    ct = case["ct"]
    suv = case["suv"]
    gt = case["gt"]

    # --------------------------------------------------------
    # EXACT PET-ONLY DEV39 CANDIDATE
    # --------------------------------------------------------

    candidate_volume = (
        suv >= PET_THRESHOLD
    ).astype(
        np.uint8
    )

    print(
        "Candidate voxels:",
        int(
            candidate_volume.sum()
        )
    )

    # --------------------------------------------------------
    # EXACT DEV10 3D COMPONENT BUILD
    # --------------------------------------------------------

    components, cc = (
        dev10.build_3d_component_table(
            case,
            candidate_volume
        )
    )

    print(
        "3D components:",
        len(
            components
        )
    )

    # --------------------------------------------------------
    # CT STRUCTURAL MAPS
    # --------------------------------------------------------

    print(
        "Computing CT structural maps..."
    )

    (
        gradient_map,
        edge_map
    ) = compute_ct_structural_maps(
        ct
    )

    # --------------------------------------------------------
    # COMPONENT FEATURES
    # --------------------------------------------------------

    df = calculate_component_features(
        components,
        cc,
        suv,
        ct,
        gradient_map,
        edge_map
    )

    # --------------------------------------------------------
    # DEV39 + DEV51 FEATURES
    # --------------------------------------------------------

    df = add_dev39_score(
        df
    )

    df = add_ct_structural_score(
        df
    )

    df = add_dev51_rankings(
        df
    )

    # --------------------------------------------------------
    # GT AUDIT
    # --------------------------------------------------------

    df = add_gt_metrics(
        df,
        cc,
        gt
    )

    # --------------------------------------------------------
    # EXACT DEV39 HOT-FRACTION GATE
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
    # POLICIES
    # --------------------------------------------------------

    policies = [

        (
            "Dev39",
            "rank_dev39"
        ),

        (
            "Dev51_A_CT_20pct",
            "rank_dev51_a"
        ),

        (
            "Dev51_B_CT_35pct",
            "rank_dev51_b"
        ),

    ]

    results = []

    # --------------------------------------------------------
    # Print ranking audit
    # --------------------------------------------------------

    for method_name, ranking_column in policies:

        ranked = (
            eligible
            .sort_values(
                [
                    ranking_column,
                    "mean_suv"
                ],
                ascending=False
            )
            .copy()
        )

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
                    "compactness",
                    "ct_gradient_mean",
                    "ct_gradient_p90",
                    "ct_edge_density",
                    "boundary_gradient_mean",
                    "boundary_edge_density",
                    "ct_local_variation",
                    "dice_gt",
                    ranking_column,
                ]
            ]
            .head(TOP_K)
            .to_string(
                index=False
            )
        )

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
        "DEV51 — CT STRUCTURAL BOUNDARY RANKING"
    )
    print("=" * 110)

    print()
    print(
        "Frozen PET threshold:",
        PET_THRESHOLD
    )

    print(
        "Frozen hot threshold:",
        HOT_THRESHOLD
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
        "dev10_lite_hot_core.py will NOT be modified."
    )

    print(
        "Dev39 PET ranking remains the baseline."
    )

    all_components = []
    all_results = []

    for case_name in CASES:

        (
            components,
            results
        ) = audit_case(
            case_name
        )

        components["case"] = (
            case_name
        )

        all_components.append(
            components
        )

        all_results.extend(
            results
        )

    # ========================================================
    # SAVE COMPONENT AUDIT
    # ========================================================

    components_df = pd.concat(
        all_components,
        ignore_index=True
    )

    results_df = pd.DataFrame(
        all_results
    )

    components_path = (
        OUTPUT_DIR
        / "dev51_component_audit.csv"
    )

    results_path = (
        OUTPUT_DIR
        / "dev51_case_results.csv"
    )

    components_df.to_csv(
        components_path,
        index=False
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    # ========================================================
    # MACRO RESULTS
    # ========================================================

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
        / "dev51_macro_summary.csv"
    )

    macro.to_csv(
        macro_path,
        index=False
    )

    # ========================================================
    # PRINT MACRO
    # ========================================================

    print()
    print("=" * 110)
    print(
        "DEV51 MACRO RESULTS"
    )
    print("=" * 110)

    print(
        macro.to_string(
            index=False
        )
    )

    # ========================================================
    # DEV39 BASELINE
    # ========================================================

    baseline = macro[
        macro["method"]
        == "Dev39"
    ]

    if len(baseline) == 0:

        raise RuntimeError(
            "Dev39 baseline was not produced."
        )

    baseline = baseline.iloc[0]

    # ========================================================
    # DELTAS
    # ========================================================

    print()
    print(
        "=" * 110
    )

    print(
        "DEV51 DELTA VS INTERNAL DEV39 BASELINE"
    )

    print(
        "=" * 110
    )

    for _, row in macro.iterrows():

        if (
            row["method"]
            == "Dev39"
        ):
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

    # ========================================================
    # SAVE FINAL REPORT
    # ========================================================

    report_path = (
        OUTPUT_DIR
        / "dev51_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "DEV51 — CT STRUCTURAL BOUNDARY RANKING\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            "Configuration\n"
        )

        f.write(
            f"PET threshold: {PET_THRESHOLD}\n"
        )

        f.write(
            f"Hot threshold: {HOT_THRESHOLD}\n"
        )

        f.write(
            f"Hot fraction gate: {HOT_FRACTION_GATE}\n"
        )

        f.write(
            f"Top-K: {TOP_K}\n"
        )

        f.write(
            f"Core policy: {CORE_POLICY}\n\n"
        )

        f.write(
            "Macro results\n"
        )

        f.write(
            macro.to_string(
                index=False
            )
        )

        f.write(
            "\n\n"
        )

        f.write(
            "IMPORTANT:\n"
        )

        f.write(
            "Dev39 is the frozen baseline.\n"
        )

        f.write(
            "dev10_lite_hot_core.py was not modified.\n"
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print(
        "=" * 110
    )

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

    print(
        report_path
    )

    print()
    print(
        "=" * 110
    )

    print(
        "DEV51 COMPLETE"
    )

    print(
        "=" * 110
    )


if __name__ == "__main__":
    main()