from pathlib import Path
import sys
import numpy as np
import pandas as pd
import cv2


# ============================================================
# PATH SETUP
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import dev10_lite_hot_core as dev10


# ============================================================
# CONFIGURATION
# ============================================================

PET_THRESHOLD = 2.25

MIN_COMPONENT_AREA = 100
MIN_MAX_SUV = 4.0

CORE_POLICY = "P70"
TOP_K = 4
RANKING_FEATURE = "mean_suv_hot3_a0p5"


CASES = [
    "PETCT_0168f65af8",
    "PETCT_04606080a0",
    "PETCT_04ab5c61c9",
    "PETCT_0b57b247b6",
    "PETCT_11afab3485",
    "PETCT_185da4c8b6",
]


# ============================================================
# OUTPUT
# ============================================================

OUT_DIR = (
    ROOT
    / "results"
    / "development_cases"
    / "dev34_pet_ct_gate_ablation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# BODY MASK
# ============================================================

def get_body_mask(case):
    """
    Use the exact Dev10 body-mask implementation.
    """

    return dev10.build_union_body_mask(
        case["ct"]
    )


# ============================================================
# PET COMPONENT FILTER
# ============================================================

def filtered_pet_mask(
    suv_slice
):
    """
    PET-only candidate generation.

    PET >= 2.25
    area >= 100
    max SUV >= 4.0
    """

    hot = (
        suv_slice >= PET_THRESHOLD
    ).astype(np.uint8)

    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            hot,
            connectivity=8
        )
    )

    output = np.zeros_like(
        hot,
        dtype=np.uint8
    )

    accepted = []

    for label in range(
        1,
        n_labels
    ):

        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA
            ]
        )

        if area < MIN_COMPONENT_AREA:
            continue

        component = (
            labels == label
        )

        max_suv = float(
            np.max(
                suv_slice[
                    component
                ]
            )
        )

        if max_suv < MIN_MAX_SUV:
            continue

        output[
            component
        ] = 1

        accepted.append({
            "label": int(label),
            "area": area,
            "max_suv": max_suv,
        })

    return (
        output,
        accepted
    )


# ============================================================
# CANDIDATE BUILDERS
# ============================================================

def build_variants(case):

    ct = case["ct"]
    suv = case["suv"]

    # --------------------------------------------------------
    # EXACT DEV10 BASELINE
    # --------------------------------------------------------

    baseline = (
        dev10.generate_candidate_volume(
            case
        )
    )

    # --------------------------------------------------------
    # BODY MASK
    # --------------------------------------------------------

    body = get_body_mask(
        case
    )

    # Make sure dimensions agree.
    if body.shape != suv.shape:
        raise ValueError(
            f"Body/SUV shape mismatch: "
            f"{body.shape} vs {suv.shape}"
        )

    # --------------------------------------------------------
    # PET ONLY
    # --------------------------------------------------------

    pet_only = (
        suv >= PET_THRESHOLD
    ).astype(np.uint8)

    # --------------------------------------------------------
    # PET + BODY
    # --------------------------------------------------------

    pet_body = (
        pet_only
        & body.astype(np.uint8)
    ).astype(np.uint8)

    # --------------------------------------------------------
    # PET FILTERED
    # --------------------------------------------------------

    pet_filtered = np.zeros_like(
        pet_only,
        dtype=np.uint8
    )

    for z in range(
        suv.shape[0]
    ):

        mask, _ = (
            filtered_pet_mask(
                suv[z]
            )
        )

        pet_filtered[z] = mask

    # --------------------------------------------------------
    # HYBRID
    # --------------------------------------------------------

    hybrid = baseline.copy()

    for z in range(
        suv.shape[0]
    ):

        # Only add PET fallback when
        # Dev10 candidate is empty.
        if np.count_nonzero(
            baseline[z]
        ) != 0:
            continue

        hybrid[z] = np.maximum(
            hybrid[z],
            pet_filtered[z]
        )

    return {
        "baseline": baseline,
        "pet_only": pet_only,
        "pet_body": pet_body,
        "pet_filtered": pet_filtered,
        "hybrid": hybrid,
    }


# ============================================================
# CANDIDATE-LEVEL METRICS
# ============================================================

def dice_binary(
    prediction,
    target
):

    prediction = (
        prediction > 0
    )

    target = (
        target > 0
    )

    intersection = np.count_nonzero(
        prediction & target
    )

    p = np.count_nonzero(
        prediction
    )

    t = np.count_nonzero(
        target
    )

    if p + t == 0:
        return 1.0

    return (
        2.0 * intersection
        / (p + t)
    )


def iou_binary(
    prediction,
    target
):

    prediction = (
        prediction > 0
    )

    target = (
        target > 0
    )

    intersection = np.count_nonzero(
        prediction & target
    )

    union = np.count_nonzero(
        prediction | target
    )

    if union == 0:
        return 1.0

    return (
        intersection
        / union
    )


def candidate_metrics(
    candidate,
    gt
):

    candidate = (
        candidate > 0
    )

    gt = (
        gt > 0
    )

    gt_slices = np.any(
        gt,
        axis=(1, 2)
    )

    candidate_slices = np.any(
        candidate,
        axis=(1, 2)
    )

    gt_positive = int(
        np.count_nonzero(
            gt_slices
        )
    )

    hits = int(
        np.count_nonzero(
            gt_slices
            & candidate_slices
        )
    )

    misses = (
        gt_positive
        - hits
    )

    coverage_values = []

    dice_values = []

    for z in range(
        gt.shape[0]
    ):

        gt_slice = gt[z]

        if not np.any(
            gt_slice
        ):
            continue

        candidate_slice = (
            candidate[z]
        )

        gt_count = np.count_nonzero(
            gt_slice
        )

        overlap = np.count_nonzero(
            candidate_slice
            & gt_slice
        )

        coverage_values.append(
            overlap / gt_count
        )

        dice_values.append(
            dice_binary(
                candidate_slice,
                gt_slice
            )
        )

    return {
        "gt_positive_slices":
            gt_positive,

        "candidate_hit_slices":
            hits,

        "candidate_missed_slices":
            misses,

        "candidate_slice_hit_rate":
            (
                hits / gt_positive
                if gt_positive
                else 0.0
            ),

        "mean_gt_slice_coverage":
            (
                float(
                    np.mean(
                        coverage_values
                    )
                )
                if coverage_values
                else 0.0
            ),

        "mean_gt_slice_dice":
            (
                float(
                    np.mean(
                        dice_values
                    )
                )
                if dice_values
                else 0.0
            ),

        "global_candidate_dice":
            dice_binary(
                candidate,
                gt
            ),

        "global_candidate_iou":
            iou_binary(
                candidate,
                gt
            ),

        "candidate_voxels":
            int(
                np.count_nonzero(
                    candidate
                )
            ),
    }


# ============================================================
# FINAL DEV10 EVALUATION
# ============================================================

def evaluate_final(
    case,
    volume
):

    components, cc = (
        dev10.build_3d_component_table(
            case,
            volume
        )
    )

    metrics = dev10.evaluate_selection(
        case,
        components,
        cc,
        RANKING_FEATURE,
        CORE_POLICY,
        TOP_K
    )

    return (
        metrics,
        components
    )


# ============================================================
# CASE ANALYSIS
# ============================================================

def analyze_case(
    case_name
):

    print("\n" + "=" * 90)
    print(case_name)
    print("=" * 90)

    case = dev10.load_case(
        case_name
    )

    gt = case["gt"]

    variants = build_variants(
        case
    )

    rows = []

    for name, volume in variants.items():

        # ----------------------------------------------------
        # Candidate-level analysis
        # ----------------------------------------------------

        candidate = candidate_metrics(
            volume,
            gt
        )

        # ----------------------------------------------------
        # Final Dev10 evaluation
        # ----------------------------------------------------

        final_metrics, components = (
            evaluate_final(
                case,
                volume
            )
        )

        row = {
            "case": case_name,
            "variant": name,

            # Candidate metrics
            "gt_positive_slices":
                candidate[
                    "gt_positive_slices"
                ],

            "candidate_hit_slices":
                candidate[
                    "candidate_hit_slices"
                ],

            "candidate_missed_slices":
                candidate[
                    "candidate_missed_slices"
                ],

            "candidate_slice_hit_rate":
                candidate[
                    "candidate_slice_hit_rate"
                ],

            "mean_gt_slice_coverage":
                candidate[
                    "mean_gt_slice_coverage"
                ],

            "mean_gt_slice_dice":
                candidate[
                    "mean_gt_slice_dice"
                ],

            "candidate_dice":
                candidate[
                    "global_candidate_dice"
                ],

            "candidate_iou":
                candidate[
                    "global_candidate_iou"
                ],

            "candidate_voxels":
                candidate[
                    "candidate_voxels"
                ],

            # Final Dev10 metrics
            "final_dice":
                float(
                    final_metrics[
                        "dice"
                    ]
                ),

            "final_iou":
                float(
                    final_metrics[
                        "iou"
                    ]
                ),

            "final_slice_recall":
                float(
                    final_metrics[
                        "slice_recall"
                    ]
                ),

            "final_fpr":
                float(
                    final_metrics[
                        "fpr"
                    ]
                ),

            "final_prediction_voxels":
                int(
                    final_metrics[
                        "prediction_voxels"
                    ]
                ),

            "components_3d":
                int(
                    len(components)
                ),
        }

        rows.append(row)

        print(
            f"\n{name}"
        )

        print(
            f"  Candidate: "
            f"hit={row['candidate_hit_slices']}/"
            f"{row['gt_positive_slices']} "
            f"("
            f"{row['candidate_slice_hit_rate']:.4f}"
            f") "
            f"coverage="
            f"{row['mean_gt_slice_coverage']:.4f} "
            f"Dice="
            f"{row['candidate_dice']:.4f}"
        )

        print(
            f"  Final: "
            f"Dice={row['final_dice']:.6f} "
            f"IoU={row['final_iou']:.6f} "
            f"Recall={row['final_slice_recall']:.6f} "
            f"FPR={row['final_fpr']:.6f} "
            f"Pred={row['final_prediction_voxels']}"
        )

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    all_rows = []

    for case_name in CASES:

        rows = analyze_case(
            case_name
        )

        all_rows.extend(
            rows
        )

    df = pd.DataFrame(
        all_rows
    )

    # ========================================================
    # MACRO SUMMARY
    # ========================================================

    summary = (
        df
        .groupby("variant")
        .agg(
            macro_candidate_dice=(
                "candidate_dice",
                "mean"
            ),

            macro_candidate_iou=(
                "candidate_iou",
                "mean"
            ),

            macro_candidate_hit_rate=(
                "candidate_slice_hit_rate",
                "mean"
            ),

            macro_candidate_coverage=(
                "mean_gt_slice_coverage",
                "mean"
            ),

            macro_final_dice=(
                "final_dice",
                "mean"
            ),

            macro_final_iou=(
                "final_iou",
                "mean"
            ),

            macro_final_slice_recall=(
                "final_slice_recall",
                "mean"
            ),

            macro_final_fpr=(
                "final_fpr",
                "mean"
            ),

            total_prediction_voxels=(
                "final_prediction_voxels",
                "sum"
            ),

            total_candidate_voxels=(
                "candidate_voxels",
                "sum"
            ),
        )
        .reset_index()
    )

    # ========================================================
    # BASELINE DELTAS
    # ========================================================

    baseline = summary[
        summary["variant"]
        == "baseline"
    ].iloc[0]

    summary[
        "delta_candidate_dice"
    ] = (
        summary[
            "macro_candidate_dice"
        ]
        - baseline[
            "macro_candidate_dice"
        ]
    )

    summary[
        "delta_candidate_coverage"
    ] = (
        summary[
            "macro_candidate_coverage"
        ]
        - baseline[
            "macro_candidate_coverage"
        ]
    )

    summary[
        "delta_final_dice"
    ] = (
        summary[
            "macro_final_dice"
        ]
        - baseline[
            "macro_final_dice"
        ]
    )

    summary[
        "delta_final_iou"
    ] = (
        summary[
            "macro_final_iou"
        ]
        - baseline[
            "macro_final_iou"
        ]
    )

    summary[
        "delta_final_recall"
    ] = (
        summary[
            "macro_final_slice_recall"
        ]
        - baseline[
            "macro_final_slice_recall"
        ]
    )

    summary[
        "delta_final_fpr"
    ] = (
        summary[
            "macro_final_fpr"
        ]
        - baseline[
            "macro_final_fpr"
        ]
    )

    # ========================================================
    # SAVE
    # ========================================================

    case_path = (
        OUT_DIR
        / "dev34_case_results.csv"
    )

    summary_path = (
        OUT_DIR
        / "dev34_policy_summary.csv"
    )

    df.to_csv(
        case_path,
        index=False
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # PRINT
    # ========================================================

    print("\n" + "=" * 110)
    print("DEV34 MACRO SUMMARY")
    print("=" * 110)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    print("\nSaved:")
    print(case_path)
    print(summary_path)


if __name__ == "__main__":
    main()