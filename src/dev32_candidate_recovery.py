from pathlib import Path
import sys
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage

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


# ============================================================
# DEVELOPMENT CASES
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
# OUTPUT DIRECTORY
# ============================================================

OUT_DIR = (
    ROOT
    / "results"
    / "development_cases"
    / "dev32_candidate_recovery"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXACT DEV27 PET FALLBACK
# ============================================================

def build_pet_fallback(suv_slice):
    """
    Exact Dev27 fallback candidate generation.

    PET >= 2.25
    connected components
    area >= 100
    max SUV >= 4.0
    """

    hot = (suv_slice >= PET_THRESHOLD).astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        hot,
        connectivity=8
    )

    fallback = np.zeros_like(hot, dtype=np.uint8)
    accepted = []

    for label in range(1, n_labels):

        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < MIN_COMPONENT_AREA:
            continue

        component = labels == label

        max_suv = float(
            np.max(suv_slice[component])
        )

        if max_suv < MIN_MAX_SUV:
            continue

        fallback[component] = 1

        accepted.append({
            "label": int(label),
            "area": area,
            "max_suv": max_suv,
        })

    return fallback, accepted


# ============================================================
# DEV32 RECOVERY OPERATIONS
# ============================================================

def dilate_2d(mask, kernel_size):
    """
    Conservative in-plane dilation.
    """

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    return cv2.dilate(
        mask.astype(np.uint8),
        kernel,
        iterations=1
    )


def close_2d(mask, kernel_size):
    """
    Conservative slice-wise morphological closing.
    """

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    return cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel
    )


def close_3d(volume):
    """
    Very conservative 3D closing.

    Only closes tiny spatial gaps.
    """

    structure = np.zeros(
        (3, 3, 3),
        dtype=np.uint8
    )

    structure[:, :, 1] = 1
    structure[1, :, :] = 1
    structure[:, 1, :] = 1

    return ndimage.binary_closing(
        volume.astype(bool),
        structure=structure,
        iterations=1
    ).astype(np.uint8)


# ============================================================
# BUILD BASELINE + RECOVERY VARIANTS
# ============================================================

def build_variants(case):

    baseline = dev10.generate_candidate_volume(case)

    suv = case["suv"]

    variants = {}

    # --------------------------------------------------------
    # Variant 0: exact Dev27 fallback
    # --------------------------------------------------------

    fallback_volume = np.zeros_like(
        baseline,
        dtype=np.uint8
    )

    fallback_slices = 0
    fallback_components = 0

    for z in range(suv.shape[0]):

        # Only recover where Dev10 candidate is empty.
        if np.count_nonzero(baseline[z]) != 0:
            continue

        fallback, accepted = build_pet_fallback(
            suv[z]
        )

        if np.count_nonzero(fallback) == 0:
            continue

        fallback_volume[z] = fallback

        fallback_slices += 1
        fallback_components += len(accepted)

    exact_fallback = np.maximum(
        baseline,
        fallback_volume
    )

    variants["baseline"] = baseline
    variants["fallback"] = exact_fallback

    # --------------------------------------------------------
    # Variant 1: fallback + 3x3 dilation
    # --------------------------------------------------------

    dilated_3 = exact_fallback.copy()

    for z in range(dilated_3.shape[0]):

        if np.count_nonzero(
            fallback_volume[z]
        ) == 0:
            continue

        expanded = dilate_2d(
            fallback_volume[z],
            3
        )

        dilated_3[z] = np.maximum(
            dilated_3[z],
            expanded
        )

    variants["fallback_dilate3"] = dilated_3

    # --------------------------------------------------------
    # Variant 2: fallback + 5x5 dilation
    # --------------------------------------------------------

    dilated_5 = exact_fallback.copy()

    for z in range(dilated_5.shape[0]):

        if np.count_nonzero(
            fallback_volume[z]
        ) == 0:
            continue

        expanded = dilate_2d(
            fallback_volume[z],
            5
        )

        dilated_5[z] = np.maximum(
            dilated_5[z],
            expanded
        )

    variants["fallback_dilate5"] = dilated_5

    # --------------------------------------------------------
    # Variant 3: fallback + 3x3 closing
    # --------------------------------------------------------

    closed_3 = exact_fallback.copy()

    for z in range(closed_3.shape[0]):

        if np.count_nonzero(
            fallback_volume[z]
        ) == 0:
            continue

        closed = close_2d(
            fallback_volume[z],
            3
        )

        closed_3[z] = np.maximum(
            closed_3[z],
            closed
        )

    variants["fallback_close3"] = closed_3

    # --------------------------------------------------------
    # Variant 4: fallback + 5x5 closing
    # --------------------------------------------------------

    closed_5 = exact_fallback.copy()

    for z in range(closed_5.shape[0]):

        if np.count_nonzero(
            fallback_volume[z]
        ) == 0:
            continue

        closed = close_2d(
            fallback_volume[z],
            5
        )

        closed_5[z] = np.maximum(
            closed_5[z],
            closed
        )

    variants["fallback_close5"] = closed_5

    # --------------------------------------------------------
    # Variant 5: controlled 3D closing
    # --------------------------------------------------------

    closed_3d = close_3d(
        exact_fallback
    )

    # Do not allow 3D closing to modify ordinary
    # baseline candidate regions.
    new_voxels = (
        closed_3d > 0
    ) & (
        exact_fallback == 0
    )

    controlled_3d = exact_fallback.copy()
    controlled_3d[new_voxels] = 1

    variants["fallback_close3d"] = controlled_3d

    # --------------------------------------------------------
    # Variant 6: fallback + 3x3 dilation + closing
    # --------------------------------------------------------

    combined = exact_fallback.copy()

    for z in range(combined.shape[0]):

        if np.count_nonzero(
            fallback_volume[z]
        ) == 0:
            continue

        expanded = dilate_2d(
            fallback_volume[z],
            3
        )

        expanded = close_2d(
            expanded,
            3
        )

        combined[z] = np.maximum(
            combined[z],
            expanded
        )

    variants["fallback_dilate3_close3"] = combined

    return (
        variants,
        fallback_volume,
        fallback_slices,
        fallback_components,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_variant(case, volume):

    components, cc = dev10.build_3d_component_table(
        case,
        volume
    )

    metrics = dev10.evaluate_selection(
        case,
        components,
        cc,
        RANKING_FEATURE,
        CORE_POLICY,
        TOP_K
    )

    return metrics, components, cc


# ============================================================
# PER-CASE ANALYSIS
# ============================================================

def analyze_case(case_name):

    print("\n" + "=" * 80)
    print(case_name)
    print("=" * 80)

    case = dev10.load_case(case_name)

    (
        variants,
        fallback_volume,
        fallback_slices,
        fallback_components,
    ) = build_variants(case)

    print(
        f"Fallback slices: {fallback_slices}"
    )

    print(
        f"Fallback components: {fallback_components}"
    )

    rows = []

    for variant_name, volume in variants.items():

        metrics, components, cc = evaluate_variant(
            case,
            volume
        )

        prediction_voxels = int(
            metrics["prediction_voxels"]
        )

        row = {
            "case": case_name,
            "variant": variant_name,
            "dice": float(metrics["dice"]),
            "iou": float(metrics["iou"]),
            "slice_recall": float(
                metrics["slice_recall"]
            ),
            "fpr": float(metrics["fpr"]),
            "prediction_voxels": prediction_voxels,
            "candidate_voxels": int(
                np.count_nonzero(volume)
            ),
            "components_3d": int(
                len(components)
            ),
        }

        rows.append(row)

        print(
            f"{variant_name:28s} "
            f"Dice={row['dice']:.6f} "
            f"IoU={row['iou']:.6f} "
            f"Recall={row['slice_recall']:.6f} "
            f"FPR={row['fpr']:.6f} "
            f"Pred={prediction_voxels}"
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

        all_rows.extend(rows)

    df = pd.DataFrame(
        all_rows
    )

    # --------------------------------------------------------
    # Macro summary
    # --------------------------------------------------------

    summary = (
        df
        .groupby("variant")
        .agg(
            macro_dice=("dice", "mean"),
            macro_iou=("iou", "mean"),
            macro_slice_recall=(
                "slice_recall",
                "mean"
            ),
            macro_fpr=("fpr", "mean"),
            total_prediction_voxels=(
                "prediction_voxels",
                "sum"
            ),
            total_candidate_voxels=(
                "candidate_voxels",
                "sum"
            ),
        )
        .reset_index()
    )

    baseline = summary[
        summary["variant"] == "baseline"
    ].iloc[0]

    summary["delta_dice"] = (
        summary["macro_dice"]
        - baseline["macro_dice"]
    )

    summary["delta_iou"] = (
        summary["macro_iou"]
        - baseline["macro_iou"]
    )

    summary["delta_slice_recall"] = (
        summary["macro_slice_recall"]
        - baseline["macro_slice_recall"]
    )

    summary["delta_fpr"] = (
        summary["macro_fpr"]
        - baseline["macro_fpr"]
    )

    summary["delta_prediction_voxels"] = (
        summary["total_prediction_voxels"]
        - baseline["total_prediction_voxels"]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    case_path = (
        OUT_DIR
        / "dev32_case_results.csv"
    )

    summary_path = (
        OUT_DIR
        / "dev32_policy_summary.csv"
    )

    df.to_csv(
        case_path,
        index=False
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("DEV32 MACRO SUMMARY")
    print("=" * 100)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print("\nSaved:")
    print(case_path)
    print(summary_path)


if __name__ == "__main__":
    main()