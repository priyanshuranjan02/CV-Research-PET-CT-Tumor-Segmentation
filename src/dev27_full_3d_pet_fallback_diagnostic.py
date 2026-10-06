"""
Dev27 — Full 3D PET-Fallback Integration Diagnostic

Purpose
-------
Test whether the selective PET fallback discovered in Dev26
actually improves the FINAL Dev10 3D pipeline.

Baseline:
    Exact Dev10 candidate volume
    -> exact Dev10 3D filtering
    -> exact Dev10 core/ranking/Top-K evaluation

Experiment:
    Dev10 candidate volume
    +
    selective PET fallback
    ->
    same downstream Dev10 3D pipeline

Fallback:
    PET threshold = 2.25
    minimum component area = 100
    minimum component max SUV = 4.0

Dev10 is NOT modified.
"""

from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

RESULT_DIR = (
    ROOT
    / "results"
    / "development_cases"
    / "dev27_full_3d_pet_fallback_results"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

sys.path.insert(0, str(SRC))

import dev10_lite_hot_core as dev10


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
# FALLBACK PARAMETERS FROM DEV26
# ============================================================

PET_THRESHOLD = 2.25
MIN_COMPONENT_AREA = 100
MIN_MAX_SUV = 4.0


# ============================================================
# DEV10 CT CONTOUR
# ============================================================

def get_dev10_ct_contours(
    ct_slice,
    body_slice
):
    """
    Exact Dev10 CT contour stage.
    """

    ct_display = dev10.normalize_ct(
        ct_slice
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    opened = cv2.morphologyEx(
        ct_display,
        cv2.MORPH_OPEN,
        kernel
    )

    binary = cv2.adaptiveThreshold(
        opened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        3
    )

    binary[
        body_slice == 0
    ] = 0

    binary[:5, :] = 0
    binary[-5:, :] = 0
    binary[:, :5] = 0
    binary[:, -5:] = 0

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return [
        c
        for c in contours
        if cv2.contourArea(c) >= 100
    ]


# ============================================================
# DEV10 2D CANDIDATE
# ============================================================

def build_dev10_slice_candidate(
    ct_slice,
    suv_slice,
    body_slice
):
    """
    Reproduce Dev10 2D candidate generation.
    """

    contours = get_dev10_ct_contours(
        ct_slice,
        body_slice
    )

    contour_mask = np.zeros(
        ct_slice.shape,
        dtype=np.uint8
    )

    if contours:

        cv2.drawContours(
            contour_mask,
            contours,
            -1,
            1,
            -1
        )

    pet_hot = (
        suv_slice >= PET_THRESHOLD
    )

    gated = np.logical_and(
        contour_mask > 0,
        pet_hot
    )

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            gated.astype(np.uint8),
            connectivity=8
        )
    )

    candidate = np.zeros(
        ct_slice.shape,
        dtype=np.uint8
    )

    components = []

    for label in range(
        1,
        num_labels
    ):

        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA
            ]
        )

        if area < 20:
            continue

        component = (
            labels == label
        )

        candidate[component] = 1

        components.append(
            {
                "label": label,
                "area": area,
            }
        )

    return (
        candidate,
        components
    )


# ============================================================
# PET FALLBACK
# ============================================================

def build_pet_fallback(
    suv_slice
):
    """
    Selective PET-only fallback from Dev26.

    Requirements:
        PET >= 2.25
        component area >= 100
        component max SUV >= 4.0
    """

    pet_hot = (
        suv_slice >= PET_THRESHOLD
    )

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            pet_hot.astype(np.uint8),
            connectivity=8
        )
    )

    fallback = np.zeros(
        suv_slice.shape,
        dtype=np.uint8
    )

    accepted = []

    for label in range(
        1,
        num_labels
    ):

        component = (
            labels == label
        )

        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA
            ]
        )

        if area < MIN_COMPONENT_AREA:
            continue

        values = suv_slice[
            component
        ]

        if values.size == 0:
            continue

        max_suv = float(
            np.max(values)
        )

        if max_suv < MIN_MAX_SUV:
            continue

        fallback[component] = 1

        accepted.append(
            {
                "label": label,
                "area": area,
                "max_suv": max_suv,
            }
        )

    return (
        fallback,
        accepted
    )


# ============================================================
# CANDIDATE VOLUME
# ============================================================

def build_baseline_and_fallback_volumes(
    case
):
    """
    Build:

        baseline_volume
        experimental_volume

    Experimental volume differs ONLY when the
    Dev10 CT-gated candidate is empty on a slice.
    """

    ct = case["ct"]
    suv = case["suv"]

    body = dev10.build_union_body_mask(
        ct
    )

    baseline_volume = np.zeros(
        ct.shape,
        dtype=np.uint8
    )

    experimental_volume = np.zeros(
        ct.shape,
        dtype=np.uint8
    )

    fallback_slice_count = 0
    fallback_component_count = 0

    for z in range(
        ct.shape[0]
    ):

        (
            baseline_slice,
            baseline_components
        ) = build_dev10_slice_candidate(
            ct[z],
            suv[z],
            body[z]
        )

        baseline_volume[z] = (
            baseline_slice
        )

        experimental_volume[z] = (
            baseline_slice
        )

        # ----------------------------------------------------
        # PET fallback ONLY if CT-gated candidate is empty
        # ----------------------------------------------------

        if len(
            baseline_components
        ) == 0:

            (
                fallback,
                accepted
            ) = build_pet_fallback(
                suv[z]
            )

            if accepted:

                experimental_volume[z] = (
                    np.maximum(
                        experimental_volume[z],
                        fallback
                    )
                )

                fallback_slice_count += 1
                fallback_component_count += (
                    len(accepted)
                )

    return (
        baseline_volume,
        experimental_volume,
        fallback_slice_count,
        fallback_component_count
    )


# ============================================================
# 3D COMPONENT FILTER
# ============================================================

def apply_dev10_3d_filter(
    case,
    raw_volume
):
    """
    Apply the exact Dev10 3D component construction.

    Dev10 returns:
        components, cc

    Both are required for the downstream
    evaluate_selection() stage.
    """

    raw_volume = raw_volume.astype(
        np.uint8,
        copy=False
    )

    components, cc = (
        dev10.build_3d_component_table(
            case,
            raw_volume
        )
    )

    return components, cc

# baseline_raw_volume = dev10.generate_candidate_volume(case)

# baseline_components, baseline_cc = (
#     dev10.build_3d_component_table(
#         case,
#         baseline_raw_volume
#     )
# )

# baseline_metrics = dev10.evaluate_selection(
#     case,
#     baseline_components,
#     baseline_cc,
#     "mean_suv_hot3_a0p5",
#     "P70",
#     4
# )

# print(
#     f"BASELINE CHECK {case_name}: "
#     f"{len(baseline_components)} components"
# )


# ============================================================
# FINAL PREDICTION
# ============================================================

# def evaluate_final_prediction(
#     case,
#     component_table,
#     top_k
# ):
#     """
#     Reconstruct a final prediction using the
#     same ranking/core policy used by Dev10.

#     P70 is used because Dev10's best development
#     configuration is P70 / K4.

#     Ranking:
#         mean_suv_hot3_a0p5
#     """

#     # --------------------------------------------------------
#     # IMPORTANT:
#     # Try to use Dev10's existing final-prediction helper
#     # if available.
#     # --------------------------------------------------------

#     candidate_functions = [
#         "build_final_prediction",
#         "generate_final_prediction",
#         "select_top_components",
#         "build_prediction_from_components",
#     ]

#     function = None

#     for name in candidate_functions:

#         if hasattr(
#             dev10,
#             name
#         ):

#             function = getattr(
#                 dev10,
#                 name
#             )

#             break

#     if function is None:

#         raise RuntimeError(
#             "Could not locate Dev10 final prediction helper. "
#             "Please inspect dev10_lite_hot_core.py for the "
#             "function used to apply P70, ranking and Top-K."
#         )

#     # Try common signatures.
#     attempts = [
#         (
#             case,
#             component_table,
#             "P70",
#             top_k,
#             "mean_suv_hot3_a0p5",
#         ),
#         (
#             case,
#             component_table,
#             "P70",
#             top_k,
#         ),
#         (
#             component_table,
#             case,
#             "P70",
#             top_k,
#         ),
#     ]

#     last_error = None

#     for args in attempts:

#         try:

#             result = function(
#                 *args
#             )

#             if isinstance(
#                 result,
#                 tuple
#             ):

#                 return result[0]

#             return result

#         except TypeError as exc:

#             last_error = exc

#     raise RuntimeError(
#         "Unable to call Dev10 final prediction helper. "
#         f"Last error: {last_error}"
#     )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction,
    gt
):
    """
    Calculate final 3D metrics.
    """

    prediction = (
        prediction.astype(bool)
    )

    gt = (
        gt.astype(bool)
    )

    gt_voxels = int(
        gt.sum()
    )

    pred_voxels = int(
        prediction.sum()
    )

    overlap = int(
        np.logical_and(
            prediction,
            gt
        ).sum()
    )

    union = int(
        np.logical_or(
            prediction,
            gt
        ).sum()
    )

    dice = (
        2.0 * overlap /
        (pred_voxels + gt_voxels)
        if pred_voxels + gt_voxels > 0
        else 0.0
    )

    iou = (
        overlap / union
        if union > 0
        else 0.0
    )

    false_positive = int(
        np.logical_and(
            prediction,
            ~gt
        ).sum()
    )

    fpr = (
        false_positive /
        prediction.size
    )

    gt_slice_indices = np.where(
        np.any(
            gt,
            axis=(1, 2)
        )
    )[0]

    if len(
        gt_slice_indices
    ) > 0:

        recalls = []

        for z in gt_slice_indices:

            gt_slice = gt[z]
            pred_slice = prediction[z]

            recalls.append(
                np.logical_and(
                    gt_slice,
                    pred_slice
                ).any()
            )

        slice_recall = (
            np.mean(
                recalls
            )
        )

    else:

        slice_recall = 0.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        "slice_recall": float(
            slice_recall
        ),
        "fpr": float(fpr),
        "prediction_voxels": pred_voxels,
        "gt_voxels": gt_voxels,
        "overlap_voxels": overlap,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 110)
    print(
        "DEV27 — FULL 3D PET FALLBACK INTEGRATION DIAGNOSTIC"
    )
    print("=" * 110)

    print(
        "\nFallback configuration:"
    )

    print(
        f"PET threshold: {PET_THRESHOLD}"
    )

    print(
        f"Minimum area: {MIN_COMPONENT_AREA}"
    )

    print(
        f"Minimum max SUV: {MIN_MAX_SUV}"
    )

    print(
        "\nBaseline downstream configuration:"
    )

    print(
        "Core policy: P70"
    )

    print(
        "Top-K: 4"
    )

    print(
        "Ranking: mean_suv_hot3_a0p5"
    )

    case_rows = []

    # all_baseline_predictions = []
    # all_experimental_predictions = []
    # all_gt = []

    for case_name in CASES:

        print("\n" + "=" * 110)
        print(
            f"===== CASE: {case_name} ====="
        )
        print("=" * 110)

        case = dev10.load_case(
            case_name
        )

        print(
            f"CT: {case['ct'].shape}"
        )

        print(
            f"PET: {case['suv'].shape}"
        )

        # ----------------------------------------------------
        # EXACT DEV10 BASELINE
        # ----------------------------------------------------

        baseline_volume = dev10.generate_candidate_volume(
            case
        )

        # Experimental starts as an EXACT copy of Dev10.
        experimental_volume = baseline_volume.copy()

        fallback_slices = 0
        fallback_components = 0

        # ----------------------------------------------------
        # SELECTIVE PET FALLBACK
        #
        # Add fallback ONLY on slices where the exact
        # Dev10 CT-gated candidate is completely empty.
        # ----------------------------------------------------

        for z in range(case["ct"].shape[0]):

            if np.count_nonzero(
                baseline_volume[z]
            ) != 0:
                continue

            fallback, accepted = build_pet_fallback(
                case["suv"][z]
            )

            if accepted:

                experimental_volume[z] = np.maximum(
                    experimental_volume[z],
                    fallback
                )

                fallback_slices += 1
                fallback_components += len(
                    accepted
                )

        print(
            f"Fallback slices added: "
            f"{fallback_slices}"
        )

        print(
            f"Fallback components added: "
            f"{fallback_components}"
        )

        # ----------------------------------------------------
        # Same Dev10 3D filtering
        # ----------------------------------------------------

        baseline_components, baseline_cc = (
            apply_dev10_3d_filter(
                case,
                baseline_volume
            )
        )

        experimental_components, experimental_cc = (
            apply_dev10_3d_filter(
                case,
                experimental_volume
            )
        )

        print(
            "Baseline eligible 3D components: "
            f"{len(baseline_components)}"
        )

        print(
            "Experimental eligible 3D components: "
            f"{len(experimental_components)}"
        )

        print(
            f"BASELINE CHECK {case_name}: "
            f"{len(baseline_components)} components"
        )

        # ----------------------------------------------------
        # Same final prediction
        # ----------------------------------------------------

        baseline_metrics = (
            dev10.evaluate_selection(
                case,
                baseline_components,
                baseline_cc,
                "mean_suv_hot3_a0p5",
                "P70",
                4
            )
        )

        experimental_metrics = (
            dev10.evaluate_selection(
                case,
                experimental_components,
                experimental_cc,
                "mean_suv_hot3_a0p5",
                "P70",
                4
            )
        )

        # baseline_prediction = (
        #     np.asarray(
        #         baseline_prediction
        #     ).astype(bool)
        # )

        # experimental_prediction = (
        #     np.asarray(
        #         experimental_prediction
        #     ).astype(bool)
        # )

        # gt = (
        #     case["gt"]
        #     .astype(bool)
        # )

        # # ----------------------------------------------------
        # # Metrics
        # # ----------------------------------------------------

        # baseline_metrics = (
        #     calculate_metrics(
        #         baseline_prediction,
        #         gt
        #     )
        # )

        # experimental_metrics = (
        #     calculate_metrics(
        #         experimental_prediction,
        #         gt
        #     )
        # )

        print(
            "\nBaseline:"
        )

        print(
            f"  Dice: "
            f"{baseline_metrics['dice']:.6f}"
        )

        print(
            f"  IoU: "
            f"{baseline_metrics['iou']:.6f}"
        )

        print(
            f"  Slice recall: "
            f"{baseline_metrics['slice_recall']:.6f}"
        )

        print(
            f"  FPR: "
            f"{baseline_metrics['fpr']:.8f}"
        )

        print(
            f"  Prediction voxels: "
            f"{baseline_metrics['prediction_voxels']}"
        )

        print(
            "\nExperimental:"
        )

        print(
            f"  Dice: "
            f"{experimental_metrics['dice']:.6f}"
        )

        print(
            f"  IoU: "
            f"{experimental_metrics['iou']:.6f}"
        )

        print(
            f"  Slice recall: "
            f"{experimental_metrics['slice_recall']:.6f}"
        )

        print(
            f"  FPR: "
            f"{experimental_metrics['fpr']:.8f}"
        )

        print(
            f"  Prediction voxels: "
            f"{experimental_metrics['prediction_voxels']}"
        )

        case_rows.append(
            {
                "case_id": case_name,

                "fallback_slices":
                    fallback_slices,

                "fallback_components":
                    fallback_components,

                "baseline_3d_components":
                    len(baseline_components),

                "experimental_3d_components":
                    len(experimental_components),

                "baseline_dice":
                    baseline_metrics[
                        "dice"
                    ],

                "experimental_dice":
                    experimental_metrics[
                        "dice"
                    ],

                "delta_dice":
                    (
                        experimental_metrics[
                            "dice"
                        ]
                        -
                        baseline_metrics[
                            "dice"
                        ]
                    ),

                "baseline_iou":
                    baseline_metrics[
                        "iou"
                    ],

                "experimental_iou":
                    experimental_metrics[
                        "iou"
                    ],

                "delta_iou":
                    (
                        experimental_metrics[
                            "iou"
                        ]
                        -
                        baseline_metrics[
                            "iou"
                        ]
                    ),

                "baseline_slice_recall":
                    baseline_metrics[
                        "slice_recall"
                    ],

                "experimental_slice_recall":
                    experimental_metrics[
                        "slice_recall"
                    ],

                "delta_slice_recall":
                    (
                        experimental_metrics[
                            "slice_recall"
                        ]
                        -
                        baseline_metrics[
                            "slice_recall"
                        ]
                    ),

                "baseline_fpr":
                    baseline_metrics[
                        "fpr"
                    ],

                "experimental_fpr":
                    experimental_metrics[
                        "fpr"
                    ],

                "delta_fpr":
                    (
                        experimental_metrics[
                            "fpr"
                        ]
                        -
                        baseline_metrics[
                            "fpr"
                        ]
                    ),

                "baseline_prediction_voxels":
                    baseline_metrics[
                        "prediction_voxels"
                    ],

                "experimental_prediction_voxels":
                    experimental_metrics[
                        "prediction_voxels"
                    ],
            }
        )

        # all_baseline_predictions.append(
        #     baseline_prediction
        # )

        # all_experimental_predictions.append(
        #     experimental_prediction
        # )

        # all_gt.append(
        #     gt
        # )

    # ========================================================
    # CASE SUMMARY
    # ========================================================

    case_summary = pd.DataFrame(
        case_rows
    )

    print("\n" + "=" * 110)
    print(
        "DEV27 CASE SUMMARY"
    )
    print("=" * 110)

    print(
        case_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # MACRO METRICS
    # ========================================================

    macro = {
        "baseline_macro_dice":
            case_summary[
                "baseline_dice"
            ].mean(),

        "experimental_macro_dice":
            case_summary[
                "experimental_dice"
            ].mean(),

        "delta_macro_dice":
            case_summary[
                "delta_dice"
            ].mean(),

        "baseline_macro_iou":
            case_summary[
                "baseline_iou"
            ].mean(),

        "experimental_macro_iou":
            case_summary[
                "experimental_iou"
            ].mean(),

        "delta_macro_iou":
            case_summary[
                "delta_iou"
            ].mean(),

        "baseline_macro_slice_recall":
            case_summary[
                "baseline_slice_recall"
            ].mean(),

        "experimental_macro_slice_recall":
            case_summary[
                "experimental_slice_recall"
            ].mean(),

        "delta_macro_slice_recall":
            case_summary[
                "delta_slice_recall"
            ].mean(),

        "baseline_macro_fpr":
            case_summary[
                "baseline_fpr"
            ].mean(),

        "experimental_macro_fpr":
            case_summary[
                "experimental_fpr"
            ].mean(),

        "delta_macro_fpr":
            case_summary[
                "delta_fpr"
            ].mean(),

        "baseline_total_prediction_voxels":
            case_summary[
                "baseline_prediction_voxels"
            ].sum(),

        "experimental_total_prediction_voxels":
            case_summary[
                "experimental_prediction_voxels"
            ].sum(),
    }

    macro_df = pd.DataFrame(
        [macro]
    )

    print("\n" + "=" * 110)
    print(
        "DEV27 MACRO COMPARISON"
    )
    print("=" * 110)

    print(
        macro_df.to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    case_path = (
        RESULT_DIR /
        "dev27_case_summary.csv"
    )

    macro_path = (
        RESULT_DIR /
        "dev27_macro_comparison.csv"
    )

    case_summary.to_csv(
        case_path,
        index=False
    )

    macro_df.to_csv(
        macro_path,
        index=False
    )

    print("\n" + "=" * 110)
    print(
        "DEV27 COMPLETE"
    )
    print("=" * 110)

    print(
        f"Case summary:\n{case_path}"
    )

    print(
        f"Macro comparison:\n{macro_path}"
    )

    print(
        "\nDev10 was NOT modified."
    )


if __name__ == "__main__":
    main()