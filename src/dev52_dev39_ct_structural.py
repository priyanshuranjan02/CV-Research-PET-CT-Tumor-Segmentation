# src/dev52_dev39_ct_structural.py

import os
import sys
import numpy as np
import pandas as pd
import cv2


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
import dev39_pet_quality_fpr_control as dev39


# ============================================================================
# CONFIGURATION
# ============================================================================

PET_THRESHOLD = dev39.PET_THRESHOLD
HOT_THRESHOLD = dev39.HOT_THRESHOLD

CORE_POLICY = dev39.CORE_POLICY
TOP_K = dev39.TOP_K

# Dev51-B weighting
PET_WEIGHT = 0.65
CT_WEIGHT = 0.35

DEVELOPMENT_CASES = dev39.DEVELOPMENT_CASES


# ============================================================================
# OUTPUT
# ============================================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "development_cases",
    "dev52_dev39_ct_structural",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================================
# PET-ONLY CANDIDATE
# Exact Dev39 candidate generation
# ============================================================================

def generate_pet_only_candidate(case):

    suv = case["suv"]

    return (
        suv >= PET_THRESHOLD
    ).astype(np.uint8)


# ============================================================================
# CT STRUCTURAL MAPS
# ============================================================================

def build_ct_structural_maps(ct):

    z_count = ct.shape[0]

    gradient_maps = np.zeros_like(
        ct,
        dtype=np.float32,
    )

    edge_maps = np.zeros_like(
        ct,
        dtype=np.uint8,
    )

    variation_maps = np.zeros_like(
        ct,
        dtype=np.float32,
    )

    for z in range(z_count):

        image = ct[z].astype(
            np.float32
        )

        # --------------------------------------------------------------
        # Robust normalization for Sobel
        # --------------------------------------------------------------

        finite = np.isfinite(image)

        if not np.any(finite):
            continue

        values = image[finite]

        low = np.percentile(
            values,
            1,
        )

        high = np.percentile(
            values,
            99,
        )

        if high <= low:
            normalized = np.zeros_like(
                image,
                dtype=np.float32,
            )
        else:
            normalized = np.clip(
                (image - low)
                / (high - low),
                0.0,
                1.0,
            )

        # --------------------------------------------------------------
        # Sobel gradient magnitude
        # --------------------------------------------------------------

        gx = cv2.Sobel(
            normalized,
            cv2.CV_32F,
            1,
            0,
            ksize=3,
        )

        gy = cv2.Sobel(
            normalized,
            cv2.CV_32F,
            0,
            1,
            ksize=3,
        )

        gradient = np.sqrt(
            gx * gx
            + gy * gy
        )

        gradient_maps[z] = gradient

        # --------------------------------------------------------------
        # Adaptive structural edge map
        # Exact Dev51 idea:
        # threshold using the 85th percentile
        # --------------------------------------------------------------

        threshold = np.percentile(
            gradient,
            85,
        )

        edge_maps[z] = (
            gradient >= threshold
        ).astype(np.uint8)

        # --------------------------------------------------------------
        # Local CT variation
        # --------------------------------------------------------------

        mean = cv2.GaussianBlur(
            image,
            (5, 5),
            0,
        )

        sq_mean = cv2.GaussianBlur(
            image * image,
            (5, 5),
            0,
        )

        variance = np.maximum(
            sq_mean
            - mean * mean,
            0.0,
        )

        variation = np.sqrt(
            variance
        )

        variation_maps[z] = (
            variation
        )

    return (
        gradient_maps,
        edge_maps,
        variation_maps,
    )


# ============================================================================
# COMPONENT CT FEATURES
# ============================================================================

def add_ct_structural_features(
    components,
    cc,
    ct,
    gradient_maps,
    edge_maps,
    variation_maps,
):

    df = components.copy()

    rows = []

    for _, row in df.iterrows():

        label = int(
            row["label"]
        )

        mask = (
            cc == label
        )

        if not np.any(mask):

            rows.append({
                "ct_gradient_mean": 0.0,
                "ct_gradient_median": 0.0,
                "ct_gradient_p90": 0.0,
                "ct_edge_density": 0.0,
                "ct_boundary_gradient_mean": 0.0,
                "ct_boundary_gradient_p75": 0.0,
                "ct_boundary_edge_density": 0.0,
                "ct_local_variation": 0.0,
                "ct_high_variation_fraction": 0.0,
                "ct_tissue_fraction": 0.0,
            })

            continue

        gradient_values = gradient_maps[
            mask
        ]

        edge_values = edge_maps[
            mask
        ]

        variation_values = variation_maps[
            mask
        ]

        # --------------------------------------------------------------
        # Component gradient statistics
        # --------------------------------------------------------------

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
                90,
            )
        )

        edge_density = float(
            np.mean(
                edge_values
            )
        )

        local_variation = float(
            np.mean(
                variation_values
            )
        )

        variation_threshold = (
            np.percentile(
                variation_values,
                75,
            )
        )

        high_variation_fraction = float(
            np.mean(
                variation_values
                >= variation_threshold
            )
        )

        # --------------------------------------------------------------
        # Boundary extraction
        # --------------------------------------------------------------

        boundary_gradients = []
        boundary_edges = []

        z_values = np.unique(
            np.where(mask)[0]
        )

        for z in z_values:

            slice_mask = mask[z]

            if not np.any(slice_mask):
                continue

            kernel = np.ones(
                (3, 3),
                dtype=np.uint8,
            )

            eroded = cv2.erode(
                slice_mask.astype(
                    np.uint8
                ),
                kernel,
                iterations=1,
            )

            boundary = (
                slice_mask
                & (eroded == 0)
            )

            if not np.any(boundary):
                continue

            boundary_gradient = (
                gradient_maps[z][boundary]
            )

            boundary_edge = (
                edge_maps[z][boundary]
            )

            boundary_gradients.extend(
                boundary_gradient.tolist()
            )

            boundary_edges.extend(
                boundary_edge.tolist()
            )

        if len(boundary_gradients) == 0:

            boundary_gradient_mean = 0.0
            boundary_gradient_p75 = 0.0
            boundary_edge_density = 0.0

        else:

            boundary_gradients = np.asarray(
                boundary_gradients,
                dtype=np.float32,
            )

            boundary_edges = np.asarray(
                boundary_edges,
                dtype=np.float32,
            )

            boundary_gradient_mean = float(
                np.mean(
                    boundary_gradients
                )
            )

            boundary_gradient_p75 = float(
                np.percentile(
                    boundary_gradients,
                    75,
                )
            )

            boundary_edge_density = float(
                np.mean(
                    boundary_edges
                )
            )

        # --------------------------------------------------------------
        # CT tissue fraction
        #
        # Use a broad non-air CT condition. This is only a structural
        # contextual feature and does not alter the candidate mask.
        # --------------------------------------------------------------

        ct_values = ct[mask]

        tissue_fraction = float(
            np.mean(
                ct_values > -500
            )
        )

        rows.append({
            "ct_gradient_mean":
                gradient_mean,

            "ct_gradient_median":
                gradient_median,

            "ct_gradient_p90":
                gradient_p90,

            "ct_edge_density":
                edge_density,

            "ct_boundary_gradient_mean":
                boundary_gradient_mean,

            "ct_boundary_gradient_p75":
                boundary_gradient_p75,

            "ct_boundary_edge_density":
                boundary_edge_density,

            "ct_local_variation":
                local_variation,

            "ct_high_variation_fraction":
                high_variation_fraction,

            "ct_tissue_fraction":
                tissue_fraction,
        })

    feature_df = pd.DataFrame(
        rows,
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
# NORMALIZATION
# ============================================================================

def minmax_normalize(series):

    series = series.astype(
        float
    )

    minimum = series.min()
    maximum = series.max()

    if (
        not np.isfinite(minimum)
        or not np.isfinite(maximum)
        or maximum <= minimum
    ):

        return pd.Series(
            np.zeros(
                len(series)
            ),
            index=series.index,
        )

    return (
        (series - minimum)
        / (maximum - minimum)
    )


# ============================================================================
# DEV51-B CT STRUCTURAL SCORE
# ============================================================================

def add_ct_structural_score(df):

    result = df.copy()

    result["n_ct_boundary_gradient"] = (
        minmax_normalize(
            result[
                "ct_boundary_gradient_mean"
            ]
        )
    )

    result["n_ct_boundary_edge"] = (
        minmax_normalize(
            result[
                "ct_boundary_edge_density"
            ]
        )
    )

    result["n_ct_gradient"] = (
        minmax_normalize(
            result[
                "ct_gradient_mean"
            ]
        )
    )

    result["n_ct_gradient_p90"] = (
        minmax_normalize(
            result[
                "ct_gradient_p90"
            ]
        )
    )

    result["n_ct_edge_density"] = (
        minmax_normalize(
            result[
                "ct_edge_density"
            ]
        )
    )

    result["n_ct_local_variation"] = (
        minmax_normalize(
            result[
                "ct_local_variation"
            ]
        )
    )

    # --------------------------------------------------------------
    # Dev51 CT structural score
    # --------------------------------------------------------------

    result["dev52_ct_structural"] = (
        0.25
        * result[
            "n_ct_boundary_gradient"
        ]

        + 0.20
        * result[
            "n_ct_boundary_edge"
        ]

        + 0.20
        * result[
            "n_ct_gradient"
        ]

        + 0.15
        * result[
            "n_ct_gradient_p90"
        ]

        + 0.10
        * result[
            "n_ct_edge_density"
        ]

        + 0.10
        * result[
            "n_ct_local_variation"
        ]
    )

    return result


# ============================================================================
# DEV52 COMBINED RANKING
# ============================================================================

def add_dev52_score(df):

    result = df.copy()

    result["dev52_score"] = (
        PET_WEIGHT
        * result[
            "dev39_pet_quality"
        ]

        + CT_WEIGHT
        * result[
            "dev52_ct_structural"
        ]
    )

    return result


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_dev52(
    case,
    components,
    cc,
):

    # --------------------------------------------------------------
    # Exact Dev39 winning gate
    # --------------------------------------------------------------

    filtered = (
        components[
            components[
                "dev39_hot_fraction"
            ]
            >= 0.40
        ]
        .copy()
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
    # ONLY CHANGE FROM DEV39:
    # ranking score = 0.65 PET + 0.35 CT
    # --------------------------------------------------------------

    ranked = (
        filtered
        .sort_values(
            by=[
                "dev52_score",
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
        "dev52_score",
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
    print("DEV52 - DEV39 + CT STRUCTURAL RANKING")
    print("=" * 100)

    print()
    print("Configuration")
    print("-" * 80)

    print(
        f"PET threshold : "
        f"{PET_THRESHOLD}"
    )

    print(
        f"Hot threshold : "
        f"{HOT_THRESHOLD}"
    )

    print(
        f"Core policy   : "
        f"{CORE_POLICY}"
    )

    print(
        f"Top-K         : "
        f"{TOP_K}"
    )

    print(
        f"PET weight    : "
        f"{PET_WEIGHT}"
    )

    print(
        f"CT weight     : "
        f"{CT_WEIGHT}"
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
            f"CT : "
            f"{case['ct'].shape}"
        )

        print(
            f"PET: "
            f"{case['suv'].shape}"
        )

        print(
            f"SUV max: "
            f"{case['suv'].max():.4f}"
        )

        # --------------------------------------------------------------
        # Exact Dev39 PET-only candidate
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
        # Exact Dev39 component features
        # --------------------------------------------------------------

        components = (
            dev39.add_component_features(
                components,
                cc,
                case["suv"],
            )
        )

        # --------------------------------------------------------------
        # Exact Dev39 PET quality score
        # --------------------------------------------------------------

        components = (
            dev39.add_pet_quality_score(
                components
            )
        )

        # --------------------------------------------------------------
        # CT structural maps
        # --------------------------------------------------------------

        (
            gradient_maps,
            edge_maps,
            variation_maps,
        ) = build_ct_structural_maps(
            case["ct"]
        )

        # --------------------------------------------------------------
        # CT structural features
        # --------------------------------------------------------------

        components = (
            add_ct_structural_features(
                components,
                cc,
                case["ct"],
                gradient_maps,
                edge_maps,
                variation_maps,
            )
        )

        # --------------------------------------------------------------
        # CT structural score
        # --------------------------------------------------------------

        components = (
            add_ct_structural_score(
                components
            )
        )

        # --------------------------------------------------------------
        # Combined Dev52 score
        # --------------------------------------------------------------

        components = (
            add_dev52_score(
                components
            )
        )

        # --------------------------------------------------------------
        # Audit
        # --------------------------------------------------------------

        gt = case["gt"].astype(
            bool
        )

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

                "pet_quality":
                    float(
                        row[
                            "dev39_pet_quality"
                        ]
                    ),

                "ct_structural":
                    float(
                        row[
                            "dev52_ct_structural"
                        ]
                    ),

                "dev52_score":
                    float(
                        row[
                            "dev52_score"
                        ]
                    ),

                "ct_gradient_mean":
                    float(
                        row[
                            "ct_gradient_mean"
                        ]
                    ),

                "ct_boundary_gradient_mean":
                    float(
                        row[
                            "ct_boundary_gradient_mean"
                        ]
                    ),

                "ct_boundary_edge_density":
                    float(
                        row[
                            "ct_boundary_edge_density"
                        ]
                    ),

                "gt_overlap":
                    overlap,
            })

        # --------------------------------------------------------------
        # Evaluate Dev52
        # --------------------------------------------------------------

        metrics = evaluate_dev52(
            case,
            components,
            cc,
        )

        print()
        print(
            "DEV52 RESULT"
        )

        print(
            f"  Eligible components: "
            f"{metrics['eligible_components']}"
        )

        print(
            f"  Selected labels: "
            f"{metrics['selected_labels']}"
        )

        print(
            f"  Dice: "
            f"{metrics['dice']:.6f}"
        )

        print(
            f"  IoU: "
            f"{metrics['iou']:.6f}"
        )

        print(
            f"  Slice recall: "
            f"{metrics['slice_recall']:.6f}"
        )

        print(
            f"  FPR: "
            f"{metrics['fpr']:.6f}"
        )

        print(
            f"  Prediction voxels: "
            f"{metrics['prediction_voxels']}"
        )

        all_results.append({
            "case":
                case_name,

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

    macro = {
        "macro_dice":
            results_df[
                "dice"
            ].mean(),

        "macro_iou":
            results_df[
                "iou"
            ].mean(),

        "macro_slice_recall":
            results_df[
                "slice_recall"
            ].mean(),

        "macro_fpr":
            results_df[
                "fpr"
            ].mean(),

        "total_prediction_voxels":
            results_df[
                "prediction_voxels"
            ].sum(),

        "mean_prediction_voxels":
            results_df[
                "prediction_voxels"
            ].mean(),

        "mean_eligible_components":
            results_df[
                "eligible_components"
            ].mean(),
    }

    macro_df = pd.DataFrame(
        [macro]
    )

    # ==================================================================
    # COMPARISON WITH EXACT DEV39
    # ==================================================================

    DEV39_DICE = 0.120468
    DEV39_IOU = 0.070577
    DEV39_RECALL = 0.376596
    DEV39_FPR = 0.085242

    macro_df[
        "dice_delta_vs_dev39"
    ] = (
        macro_df[
            "macro_dice"
        ]
        - DEV39_DICE
    )

    macro_df[
        "iou_delta_vs_dev39"
    ] = (
        macro_df[
            "macro_iou"
        ]
        - DEV39_IOU
    )

    macro_df[
        "recall_delta_vs_dev39"
    ] = (
        macro_df[
            "macro_slice_recall"
        ]
        - DEV39_RECALL
    )

    macro_df[
        "fpr_delta_vs_dev39"
    ] = (
        macro_df[
            "macro_fpr"
        ]
        - DEV39_FPR
    )

    # ==================================================================
    # SAVE
    # ==================================================================

    results_path = os.path.join(
        OUTPUT_DIR,
        "dev52_case_results.csv",
    )

    macro_path = os.path.join(
        OUTPUT_DIR,
        "dev52_macro_summary.csv",
    )

    audit_path = os.path.join(
        OUTPUT_DIR,
        "dev52_component_audit.csv",
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
        audit_path,
        index=False,
    )

    # ==================================================================
    # FINAL REPORT
    # ==================================================================

    print()
    print("=" * 100)
    print("DEV52 MACRO RESULTS")
    print("=" * 100)

    print(
        macro_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("EXACT DEV39 BENCHMARK")
    print("=" * 100)

    print(
        f"Dice         : "
        f"{DEV39_DICE:.6f}"
    )

    print(
        f"IoU          : "
        f"{DEV39_IOU:.6f}"
    )

    print(
        f"Slice Recall : "
        f"{DEV39_RECALL:.6f}"
    )

    print(
        f"FPR          : "
        f"{DEV39_FPR:.6f}"
    )

    print()
    print("=" * 100)
    print("DEV52 DECISION")
    print("=" * 100)

    dice = macro_df.iloc[0][
        "macro_dice"
    ]

    fpr = macro_df.iloc[0][
        "macro_fpr"
    ]

    if (
        dice > DEV39_DICE
        and fpr <= DEV39_FPR + 0.01
    ):

        print(
            "DEV52 IMPROVES OVER DEV39"
        )

        print(
            "Candidate for further validation."
        )

    else:

        print(
            "DEV52 DOES NOT BEAT DEV39"
        )

        print(
            "Freeze Dev39 unless a "
            "specific methodological reason "
            "justifies further experimentation."
        )

    print()
    print("Saved:")
    print(results_path)
    print(macro_path)
    print(audit_path)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()