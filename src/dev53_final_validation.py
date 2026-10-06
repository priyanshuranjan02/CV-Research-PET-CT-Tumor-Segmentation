# src/dev53_final_validation.py

from pathlib import Path
import sys
import runpy

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ============================================================================
# IMPORT EXACT FROZEN IMPLEMENTATIONS
# ============================================================================

import dev10_lite_hot_core as dev10
import dev39_pet_quality_fpr_control as dev39


# ============================================================================
# VALIDATION CASE LOADER
# ============================================================================

CASE7_SCRIPT = SRC / "v4_validate_case7.py"

case7_ns = runpy.run_path(
    str(CASE7_SCRIPT),
    run_name="__dev53_case7_loader__",
)

prepare_case7 = case7_ns["prepare_case7"]


# ============================================================================
# CONFIGURATION — EXACT CURRENT DEV39
# ============================================================================

CASE_NAME = "PETCT_0011f3deaf"

PET_THRESHOLD = dev39.PET_THRESHOLD
HOT_THRESHOLD = dev39.HOT_THRESHOLD
CORE_POLICY = dev39.CORE_POLICY
TOP_K = dev39.TOP_K

HOT_FRACTION_GATE = 0.40

# These are only for reporting.
# They are NOT used during inference.
USE_GT_FOR_EVALUATION_ONLY = True


# ============================================================================
# OUTPUT
# ============================================================================

OUTPUT_DIR = (
    ROOT
    / "results"
    / "validation"
    / "dev53_final_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# HELPERS
# ============================================================================

def safe_core(
    cc,
    suv,
    label,
    policy,
):
    """
    Call the exact Dev10 build_core() and safely extract its mask.

    Dev10 may return a tuple containing the binary core mask.
    """

    result = dev10.build_core(
        cc,
        suv,
        int(label),
        policy,
    )

    if isinstance(result, tuple):

        for item in result:

            if isinstance(
                item,
                np.ndarray,
            ):

                return item.astype(bool)

        raise RuntimeError(
            "build_core() returned a tuple "
            "but no ndarray mask was found."
        )

    return result.astype(bool)


def dice_score(
    prediction,
    ground_truth,
):
    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    intersection = np.count_nonzero(
        prediction & ground_truth
    )

    pred_sum = np.count_nonzero(
        prediction
    )

    gt_sum = np.count_nonzero(
        ground_truth
    )

    denominator = (
        pred_sum + gt_sum
    )

    if denominator == 0:
        return 1.0

    return (
        2.0 * intersection
        / denominator
    )


def iou_score(
    prediction,
    ground_truth,
):
    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    intersection = np.count_nonzero(
        prediction & ground_truth
    )

    union = np.count_nonzero(
        prediction | ground_truth
    )

    if union == 0:
        return 1.0

    return (
        intersection
        / union
    )


def slice_recall_score(
    prediction,
    ground_truth,
):
    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    gt_positive_slices = np.any(
        ground_truth,
        axis=(1, 2),
    )

    predicted_positive_slices = np.any(
        prediction,
        axis=(1, 2),
    )

    total_gt_slices = int(
        np.count_nonzero(
            gt_positive_slices
        )
    )

    if total_gt_slices == 0:
        return 0.0

    detected = np.count_nonzero(
        gt_positive_slices
        & predicted_positive_slices
    )

    return (
        detected
        / total_gt_slices
    )


def false_positive_rate(
    prediction,
    ground_truth,
):
    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    false_positive = np.count_nonzero(
        prediction
        & ~ground_truth
    )

    true_negative = np.count_nonzero(
        ~prediction
        & ~ground_truth
    )

    denominator = (
        false_positive
        + true_negative
    )

    if denominator == 0:
        return 0.0

    return (
        false_positive
        / denominator
    )


def save_axial_visualization(
    ct,
    suv,
    prediction,
    ground_truth,
    output_path,
):
    """
    Save a representative axial slice.

    The selected slice is the middle of the union of
    predicted and GT-positive slices when available.
    """

    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    pred_slices = np.where(
        np.any(
            prediction,
            axis=(1, 2),
        )
    )[0]

    gt_slices = np.where(
        np.any(
            ground_truth,
            axis=(1, 2),
        )
    )[0]

    all_slices = np.unique(
        np.concatenate(
            [
                pred_slices,
                gt_slices,
            ]
        )
    )

    if len(all_slices) == 0:
        z = ct.shape[0] // 2
    else:
        z = int(
            all_slices[
                len(all_slices) // 2
            ]
        )

    fig = plt.figure(
        figsize=(8, 8)
    )

    plt.imshow(
        ct[z],
        cmap="gray",
    )

    # Prediction outline
    pred = prediction[z]

    if np.any(pred):
        plt.contour(
            pred,
            levels=[0.5],
            linewidths=1.5,
        )

    # GT outline
    gt = ground_truth[z]

    if np.any(gt):
        plt.contour(
            gt,
            levels=[0.5],
            linewidths=1.5,
            linestyles="--",
        )

    plt.title(
        f"{CASE_NAME} — axial slice {z}"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================================
# START
# ============================================================================

print()
print("=" * 100)
print("DEV53 — FINAL INDEPENDENT VALIDATION")
print("=" * 100)

print()
print("This script uses:")
print("  • Exact current Dev39 implementation")
print("  • Exact Dev10 component construction")
print("  • GT-free component selection")
print("  • GT only for post-hoc evaluation")

print()
print("Configuration")
print("-" * 80)

print(
    f"Case                : {CASE_NAME}"
)

print(
    f"PET threshold       : {PET_THRESHOLD}"
)

print(
    f"Hot threshold       : {HOT_THRESHOLD}"
)

print(
    f"3D minimum voxels   : 75"
)

print(
    f"3D minimum slices   : 2"
)

print(
    f"Hot fraction gate   : {HOT_FRACTION_GATE}"
)

print(
    f"Top-K               : {TOP_K}"
)

print(
    f"Core policy         : {CORE_POLICY}"
)


# ============================================================================
# STEP 1 — LOAD VALIDATION CASE
# ============================================================================

print()
print("=" * 100)
print("STEP 1 — LOAD VALIDATION CASE")
print("=" * 100)

prepared, ct_img, ct, body = (
    prepare_case7()
)

suv = prepared["suv"]

ct = np.asarray(
    ct,
    dtype=np.float32,
)

suv = np.asarray(
    suv,
    dtype=np.float32,
)

print(
    f"CT shape            : {ct.shape}"
)

print(
    f"SUV shape           : {suv.shape}"
)

print(
    f"SUV maximum         : {float(np.max(suv)):.6f}"
)


# ============================================================================
# STEP 2 — LOAD GT FOR POST-HOC EVALUATION ONLY
# ============================================================================

print()
print("=" * 100)
print("STEP 2 — LOAD GROUND TRUTH FOR POST-HOC EVALUATION")
print("=" * 100)

gt = prepared["gt"]

gt = np.asarray(
    gt,
    dtype=bool,
)

print(
    f"GT shape            : {gt.shape}"
)

print(
    f"GT voxels           : "
    f"{int(np.count_nonzero(gt))}"
)

if gt.shape != suv.shape:

    raise RuntimeError(
        "GT and SUV shapes do not match: "
        f"GT={gt.shape}, SUV={suv.shape}"
    )


# ============================================================================
# STEP 3 — CREATE DEV10 COMPATIBILITY OBJECT
# ============================================================================

print()
print("=" * 100)
print("STEP 3 — PREPARE DEV10 COMPATIBILITY OBJECT")
print("=" * 100)

# IMPORTANT:
# The GT supplied here is a dummy zero mask.
# It is NOT used for component selection.
#
# The real GT remains separate in `gt` and is only used
# after the final prediction has been generated.

dummy_gt = np.zeros_like(
    suv,
    dtype=bool,
)

case = {
    "name": CASE_NAME,
    "ct": ct,
    "suv": suv,
    "gt": dummy_gt,
}


# ============================================================================
# STEP 4 — EXACT DEV39 PET-ONLY CANDIDATE
# ============================================================================

print()
print("=" * 100)
print("STEP 4 — PET-ONLY CANDIDATE")
print("=" * 100)

candidate_volume = (
    suv >= PET_THRESHOLD
).astype(np.uint8)

candidate_voxels = int(
    np.count_nonzero(
        candidate_volume
    )
)

print(
    f"Candidate voxels   : {candidate_voxels}"
)


# ============================================================================
# STEP 5 — EXACT DEV10 3D COMPONENT CONSTRUCTION
# ============================================================================

print()
print("=" * 100)
print("STEP 5 — EXACT DEV10 3D COMPONENTS")
print("=" * 100)

components, cc = (
    dev10.build_3d_component_table(
        case,
        candidate_volume,
    )
)

print(
    f"3D components      : {len(components)}"
)

if len(components) == 0:

    raise RuntimeError(
        "No valid 3D components were generated."
    )


# ============================================================================
# STEP 6 — EXACT DEV39 FEATURES
# ============================================================================

print()
print("=" * 100)
print("STEP 6 — EXACT DEV39 PET-QUALITY FEATURES")
print("=" * 100)

components = (
    dev39.add_component_features(
        components,
        cc,
        suv,
    )
)

components = (
    dev39.add_pet_quality_score(
        components,
    )
)

print(
    "PET-quality features calculated."
)


# ============================================================================
# STEP 7 — EXACT DEV39 HOT-FRACTION GATE
# ============================================================================

print()
print("=" * 100)
print("STEP 7 — DEV39 HOT-FRACTION GATE")
print("=" * 100)

eligible = (
    components[
        components[
            "dev39_hot_fraction"
        ]
        >= HOT_FRACTION_GATE
    ]
    .copy()
)

print(
    f"Components before gate : "
    f"{len(components)}"
)

print(
    f"Eligible components    : "
    f"{len(eligible)}"
)

if eligible.empty:

    raise RuntimeError(
        "No component passed the Dev39 "
        "hot-fraction gate."
    )


# ============================================================================
# STEP 8 — EXACT DEV39 RANKING
# ============================================================================

print()
print("=" * 100)
print("STEP 8 — EXACT DEV39 RANKING")
print("=" * 100)

ranked = (
    eligible
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
    .reset_index(
        drop=True,
    )
)

selected = (
    ranked
    .head(TOP_K)
    .copy()
)

selected_labels = (
    selected[
        "label"
    ]
    .astype(int)
    .tolist()
)

print(
    "Selected labels:",
    selected_labels,
)

print()
print(
    selected[
        [
            "label",
            "dev39_voxels",
            "dev39_mean_suv",
            "dev39_max_suv",
            "dev39_hot_fraction",
            "dev39_persistence",
            "dev39_longest_run",
            "dev39_compactness",
            "dev39_pet_quality",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================================
# STEP 9 — P70 CORE EXTRACTION
# ============================================================================

print()
print("=" * 100)
print("STEP 9 — P70 CORE EXTRACTION")
print("=" * 100)

prediction = np.zeros_like(
    suv,
    dtype=bool,
)

selected_core_rows = []

for _, row in selected.iterrows():

    label = int(
        row["label"]
    )

    core = safe_core(
        cc,
        suv,
        label,
        CORE_POLICY,
    )

    prediction |= core

    selected_core_rows.append({
        "label":
            label,

        "component_voxels":
            int(
                np.count_nonzero(
                    cc == label
                )
            ),

        "core_voxels":
            int(
                np.count_nonzero(
                    core
                )
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
    })

prediction_voxels = int(
    np.count_nonzero(
        prediction
    )
)

print(
    f"Prediction voxels : {prediction_voxels}"
)


# ============================================================================
# STEP 10 — POST-HOC EVALUATION
# ============================================================================

print()
print("=" * 100)
print("STEP 10 — POST-HOC VALIDATION METRICS")
print("=" * 100)

dice = dice_score(
    prediction,
    gt,
)

iou = iou_score(
    prediction,
    gt,
)

slice_recall = (
    slice_recall_score(
        prediction,
        gt,
    )
)

fpr = false_positive_rate(
    prediction,
    gt,
)

overlap = int(
    np.count_nonzero(
        prediction & gt
    )
)

pred_slices = int(
    np.count_nonzero(
        np.any(
            prediction,
            axis=(1, 2),
        )
    )
)

gt_slices = int(
    np.count_nonzero(
        np.any(
            gt,
            axis=(1, 2),
        )
    )
)

detected_gt_slices = int(
    np.count_nonzero(
        np.any(
            prediction,
            axis=(1, 2)
        )
        &
        np.any(
            gt,
            axis=(1, 2)
        )
    )
)

print(
    f"Dice               : {dice:.6f}"
)

print(
    f"IoU                : {iou:.6f}"
)

print(
    f"Slice recall       : {slice_recall:.6f}"
)

print(
    f"FPR                : {fpr:.6f}"
)

print(
    f"Prediction voxels  : {prediction_voxels}"
)

print(
    f"GT overlap voxels  : {overlap}"
)

print(
    f"Predicted slices   : {pred_slices}"
)

print(
    f"GT-positive slices : {gt_slices}"
)

print(
    f"Detected GT slices : {detected_gt_slices}"
)


# ============================================================================
# STEP 11 — SAVE PREDICTED MASK
# ============================================================================

print()
print("=" * 100)
print("STEP 11 — SAVE FINAL MASK")
print("=" * 100)

mask_img = sitk.GetImageFromArray(
    prediction.astype(np.uint8)
)

mask_img.CopyInformation(
    ct_img
)

mask_path = (
    OUTPUT_DIR
    / "tumor_mask.nii.gz"
)

sitk.WriteImage(
    mask_img,
    str(mask_path),
)

print(
    "Saved:",
    mask_path,
)


# ============================================================================
# STEP 12 — SAVE SELECTED COMPONENTS
# ============================================================================

selected_path = (
    OUTPUT_DIR
    / "selected_components.csv"
)

selected.to_csv(
    selected_path,
    index=False,
)

core_summary = pd.DataFrame(
    selected_core_rows
)

core_path = (
    OUTPUT_DIR
    / "selected_core_summary.csv"
)

core_summary.to_csv(
    core_path,
    index=False,
)

all_components_path = (
    OUTPUT_DIR
    / "all_component_scores.csv"
)

components.to_csv(
    all_components_path,
    index=False,
)

print(
    "Saved:",
    selected_path,
)

print(
    "Saved:",
    core_path,
)

print(
    "Saved:",
    all_components_path,
)


# ============================================================================
# STEP 13 — SAVE METRICS
# ============================================================================

metrics_df = pd.DataFrame(
    [
        {
            "case":
                CASE_NAME,

            "pet_threshold":
                PET_THRESHOLD,

            "hot_threshold":
                HOT_THRESHOLD,

            "hot_fraction_gate":
                HOT_FRACTION_GATE,

            "top_k":
                TOP_K,

            "core_policy":
                CORE_POLICY,

            "candidate_voxels":
                candidate_voxels,

            "components":
                len(components),

            "eligible_components":
                len(eligible),

            "selected_labels":
                str(
                    selected_labels
                ),

            "prediction_voxels":
                prediction_voxels,

            "gt_voxels":
                int(
                    np.count_nonzero(
                        gt
                    )
                ),

            "gt_overlap":
                overlap,

            "dice":
                dice,

            "iou":
                iou,

            "slice_recall":
                slice_recall,

            "fpr":
                fpr,

            "predicted_slices":
                pred_slices,

            "gt_positive_slices":
                gt_slices,

            "detected_gt_slices":
                detected_gt_slices,
        }
    ]
)

metrics_path = (
    OUTPUT_DIR
    / "validation_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False,
)

print(
    "Saved:",
    metrics_path,
)


# ============================================================================
# STEP 14 — VISUALIZATION
# ============================================================================

print()
print("=" * 100)
print("STEP 12 — VISUALIZATION")
print("=" * 100)

visualization_path = (
    OUTPUT_DIR
    / "axial_validation_overlay.png"
)

save_axial_visualization(
    ct,
    suv,
    prediction,
    gt,
    visualization_path,
)

print(
    "Saved:",
    visualization_path,
)


# ============================================================================
# STEP 15 — FINAL REPORT
# ============================================================================

report_path = (
    OUTPUT_DIR
    / "validation_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "DEV53 — FINAL INDEPENDENT VALIDATION\n"
    )

    f.write(
        "=" * 80
        + "\n\n"
    )

    f.write(
        f"Case: {CASE_NAME}\n\n"
    )

    f.write(
        "Frozen Dev39 configuration:\n"
    )

    f.write(
        f"PET threshold: {PET_THRESHOLD}\n"
    )

    f.write(
        f"Hot threshold: {HOT_THRESHOLD}\n"
    )

    f.write(
        "3D minimum voxels: 75\n"
    )

    f.write(
        "3D minimum slices: 2\n"
    )

    f.write(
        f"Hot fraction gate: "
        f"{HOT_FRACTION_GATE}\n"
    )

    f.write(
        f"Top-K: {TOP_K}\n"
    )

    f.write(
        f"Core policy: {CORE_POLICY}\n\n"
    )

    f.write(
        "Inference is GT-free.\n"
    )

    f.write(
        "GT is used only for post-hoc evaluation.\n\n"
    )

    f.write(
        "RESULTS\n"
    )

    f.write(
        "-" * 80
        + "\n"
    )

    f.write(
        f"Dice: {dice:.6f}\n"
    )

    f.write(
        f"IoU: {iou:.6f}\n"
    )

    f.write(
        f"Slice recall: {slice_recall:.6f}\n"
    )

    f.write(
        f"FPR: {fpr:.6f}\n"
    )

    f.write(
        f"Prediction voxels: "
        f"{prediction_voxels}\n"
    )

    f.write(
        f"GT voxels: "
        f"{int(np.count_nonzero(gt))}\n"
    )

    f.write(
        f"GT overlap: {overlap}\n"
    )

    f.write(
        f"Candidate voxels: "
        f"{candidate_voxels}\n"
    )

    f.write(
        f"3D components: "
        f"{len(components)}\n"
    )

    f.write(
        f"Eligible components: "
        f"{len(eligible)}\n"
    )

    f.write(
        f"Selected labels: "
        f"{selected_labels}\n"
    )

    f.write(
        f"Predicted slices: "
        f"{pred_slices}\n"
    )

    f.write(
        f"GT-positive slices: "
        f"{gt_slices}\n"
    )

    f.write(
        f"Detected GT slices: "
        f"{detected_gt_slices}\n"
    )


print(
    "Saved:",
    report_path,
)


# ============================================================================
# FINAL
# ============================================================================

print()
print("=" * 100)
print("DEV53 COMPLETE")
print("=" * 100)

print()
print("FINAL VALIDATION RESULT")
print("-" * 80)

print(
    f"Dice         : {dice:.6f}"
)

print(
    f"IoU          : {iou:.6f}"
)

print(
    f"Slice Recall : {slice_recall:.6f}"
)

print(
    f"FPR          : {fpr:.6f}"
)

print()
print(
    "Selected labels:",
    selected_labels,
)

print()
print("Outputs:")
print(mask_path)
print(selected_path)
print(core_path)
print(all_components_path)
print(metrics_path)
print(visualization_path)
print(report_path)