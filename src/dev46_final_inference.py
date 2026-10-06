from pathlib import Path
import sys
import runpy

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt


# ============================================================
# DEV46 — FINAL TUMOR DETECTION / INFERENCE
#
# Frozen proposed method: DEV39
#
# Pipeline:
#   PET-only SUV >= 2.25
#       ->
#   Exact Dev10 3D component construction
#       ->
#   PET-quality ranking
#       ->
#   hot_fraction >= 0.40
#       ->
#   Top-K = 4
#       ->
#   P70 core
#       ->
#   FINAL TUMOR MASK
#
# IMPORTANT:
#   Ground truth is NOT used for inference.
#   Dev10 source file is NOT modified.
# ============================================================


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ============================================================
# IMPORT FROZEN DEV10 IMPLEMENTATION
# ============================================================

import dev10_lite_hot_core as dev10


# ============================================================
# VALIDATION CASE LOADER
#
# We use the already-established Case-7 loader because
# dev10.load_case() only searches development_cases/.
#
# This gives us the aligned CT and SUV arrays without using GT.
# ============================================================

CASE7_SCRIPT = SRC / "v4_validate_case7.py"

case7_ns = runpy.run_path(
    str(CASE7_SCRIPT),
    run_name="__dev46_case7_loader__"
)

prepare_case7 = case7_ns["prepare_case7"]
normalize_ct = case7_ns["normalize_ct"]


# ============================================================
# CONFIGURATION — FROZEN DEV39
# ============================================================

CASE_NAME = "PETCT_0011f3deaf"

PET_THRESHOLD = 2.25
HOT_THRESHOLD = 3.0

MIN_3D_VOXELS = 75
MIN_3D_SLICES = 2

HOT_FRACTION_GATE = 0.40

TOP_K = 4
CORE_POLICY = "P70"


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "results"
    / "inference"
    / CASE_NAME
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# UTILITY
# ============================================================

def minmax_normalize(values):
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return np.zeros_like(values)

    vmin = np.nanmin(values)
    vmax = np.nanmax(values)

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return np.zeros_like(values)

    if vmax <= vmin:
        return np.zeros_like(values)

    return (values - vmin) / (vmax - vmin)


def safe_core(
    cc,
    suv,
    label,
    policy
):
    """
    Handles the exact Dev10 build_core return type safely.

    Dev10 build_core() may return a tuple containing the
    binary core mask.
    """

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

        raise RuntimeError(
            "build_core() returned a tuple "
            "but no ndarray mask was found."
        )

    return result.astype(bool)


# ============================================================
# PET-QUALITY FEATURE CALCULATION
# ============================================================

def add_pet_quality_features(
    components,
    cc,
    suv
):
    """
    Reconstructs the frozen Dev38 PET-quality ranking.

    Features:
      - mean SUV
      - max SUV
      - hot fraction
      - persistence
      - longest consecutive z-run
      - compactness

    Final quality score:

      0.25 * mean SUV
    + 0.15 * max SUV
    + 0.15 * hot fraction
    + 0.20 * persistence
    + 0.15 * longest run
    + 0.10 * compactness
    """

    df = components.copy()

    feature_rows = []

    max_z = max(
        1,
        suv.shape[0] - 1
    )

    for _, row in df.iterrows():

        label = int(row["label"])

        component = (
            cc == label
        )

        voxels = int(
            component.sum()
        )

        if voxels == 0:
            continue

        z_indices = np.where(
            np.any(component, axis=(1, 2))
        )[0]

        if len(z_indices) == 0:
            continue

        z_start = int(
            z_indices.min()
        )

        z_end = int(
            z_indices.max()
        )

        z_span = (
            z_end
            - z_start
            + 1
        )

        slice_count = len(
            z_indices
        )

        # ----------------------------------------------------
        # Longest consecutive z-run
        # ----------------------------------------------------

        longest_run = 1
        current_run = 1

        for i in range(
            1,
            len(z_indices)
        ):

            if (
                z_indices[i]
                == z_indices[i - 1] + 1
            ):
                current_run += 1
            else:
                current_run = 1

            longest_run = max(
                longest_run,
                current_run
            )

        # ----------------------------------------------------
        # Persistence
        # ----------------------------------------------------

        persistence = (
            slice_count
            / float(z_span)
            if z_span > 0
            else 0.0
        )

        # ----------------------------------------------------
        # SUV statistics
        # ----------------------------------------------------

        component_suv = suv[
            component
        ].astype(np.float64)

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

        # ----------------------------------------------------
        # Compactness
        #
        # voxel count / bounding-box volume
        # ----------------------------------------------------

        coords = np.where(
            component
        )

        z_min, z_max = (
            coords[0].min(),
            coords[0].max()
        )

        y_min, y_max = (
            coords[1].min(),
            coords[1].max()
        )

        x_min, x_max = (
            coords[2].min(),
            coords[2].max()
        )

        bbox_volume = (
            (z_max - z_min + 1)
            * (y_max - y_min + 1)
            * (x_max - x_min + 1)
        )

        compactness = (
            voxels / float(bbox_volume)
            if bbox_volume > 0
            else 0.0
        )

        feature_rows.append(
            {
                "label": label,
                "voxels": voxels,
                "z_start": z_start,
                "z_end": z_end,
                "z_span": z_span,
                "slice_count": slice_count,
                "longest_run": longest_run,
                "persistence": persistence,
                "mean_suv_dev46": mean_suv,
                "max_suv_dev46": max_suv,
                "hot_fraction_dev46": hot_fraction,
                "compactness_dev46": compactness,
            }
        )

    feature_df = pd.DataFrame(
        feature_rows
    )

    if feature_df.empty:
        return feature_df

    # --------------------------------------------------------
    # Normalize features exactly for ranking
    # --------------------------------------------------------

    feature_df["n_mean_suv"] = minmax_normalize(
        feature_df["mean_suv_dev46"]
    )

    feature_df["n_max_suv"] = minmax_normalize(
        feature_df["max_suv_dev46"]
    )

    feature_df["n_hot_fraction"] = minmax_normalize(
        feature_df["hot_fraction_dev46"]
    )

    feature_df["n_persistence"] = minmax_normalize(
        feature_df["persistence"]
    )

    feature_df["n_longest_run"] = minmax_normalize(
        feature_df["longest_run"]
    )

    feature_df["n_compactness"] = minmax_normalize(
        feature_df["compactness_dev46"]
    )

    # --------------------------------------------------------
    # Frozen Dev38 / Dev39 PET-quality score
    # --------------------------------------------------------

    feature_df["pet_quality_score"] = (
        0.25 * feature_df["n_mean_suv"]
        + 0.15 * feature_df["n_max_suv"]
        + 0.15 * feature_df["n_hot_fraction"]
        + 0.20 * feature_df["n_persistence"]
        + 0.15 * feature_df["n_longest_run"]
        + 0.10 * feature_df["n_compactness"]
    )

    return feature_df


# ============================================================
# LOAD CASE
# ============================================================

print()
print("=" * 100)
print("DEV46 — FINAL TUMOR DETECTION / INFERENCE")
print("=" * 100)

print()
print("Case:", CASE_NAME)

print()
print("Frozen Dev39 configuration:")
print("  PET threshold       :", PET_THRESHOLD)
print("  Hot threshold       :", HOT_THRESHOLD)
print("  3D minimum voxels   :", MIN_3D_VOXELS)
print("  3D minimum slices   :", MIN_3D_SLICES)
print("  Hot fraction gate   :", HOT_FRACTION_GATE)
print("  Top-K               :", TOP_K)
print("  Core policy         :", CORE_POLICY)


# ============================================================
# PREPARE CASE
# ============================================================

print()
print("-" * 100)
print("Loading PET/CT...")
print("-" * 100)

prepared, ct_img, ct, body = prepare_case7()

suv = prepared["suv"]

ct = np.asarray(
    ct,
    dtype=np.float32
)

suv = np.asarray(
    suv,
    dtype=np.float32
)

print()
print("CT shape :", ct.shape)
print("SUV shape:", suv.shape)
print("SUV max  :", float(np.max(suv)))


# ============================================================
# DEV10 COMPATIBILITY OBJECT
#
# Dev10's component-table function expects a "gt" key because
# it also supports evaluation. For inference, GT must NOT
# influence the prediction, so we provide an all-zero dummy GT.
# ============================================================

dummy_gt = np.zeros_like(
    suv,
    dtype=bool
)

case = {
    "name": CASE_NAME,
    "ct": ct,
    "suv": suv,
    "gt": dummy_gt,
}


# ============================================================
# STEP 1 — PET-ONLY CANDIDATE
# ============================================================

print()
print("-" * 100)
print("STEP 1 — PET-ONLY CANDIDATE GENERATION")
print("-" * 100)

candidate_volume = (
    suv >= PET_THRESHOLD
).astype(np.uint8)

candidate_voxels = int(
    candidate_volume.sum()
)

print(
    "Candidate voxels:",
    candidate_voxels
)


# ============================================================
# STEP 2 — EXACT DEV10 3D COMPONENT CONSTRUCTION
# ============================================================

print()
print("-" * 100)
print("STEP 2 — 3D CONNECTED COMPONENTS")
print("-" * 100)

components, cc = (
    dev10.build_3d_component_table(
        case,
        candidate_volume
    )
)

print(
    "Initial 3D components:",
    len(components)
)

if len(components) == 0:

    print()
    print("NO TUMOR DETECTED.")
    print(
        "No valid 3D PET components were generated."
    )

    # Save empty mask.
    empty_mask = np.zeros_like(
        suv,
        dtype=np.uint8
    )

    mask_img = sitk.GetImageFromArray(
        empty_mask
    )

    mask_img.CopyInformation(
        ct_img
    )

    sitk.WriteImage(
        mask_img,
        str(
            OUTPUT_DIR
            / "tumor_mask.nii.gz"
        )
    )

    raise SystemExit(0)


# ============================================================
# STEP 3 — PET-QUALITY RANKING
# ============================================================

print()
print("-" * 100)
print("STEP 3 — PET-QUALITY COMPONENT RANKING")
print("-" * 100)

quality_df = add_pet_quality_features(
    components,
    cc,
    suv
)

if quality_df.empty:

    raise RuntimeError(
        "No component features could be calculated."
    )


# ============================================================
# STEP 4 — HOT FRACTION GATE
# ============================================================

eligible = quality_df[
    quality_df[
        "hot_fraction_dev46"
    ]
    >= HOT_FRACTION_GATE
].copy()

print()
print(
    "Components before hot-fraction gate:",
    len(quality_df)
)

print(
    "Eligible components:",
    len(eligible)
)

if eligible.empty:

    print()
    print("NO TUMOR DETECTED.")
    print(
        "No component passed the frozen "
        "Dev39 hot-fraction gate."
    )

    empty_mask = np.zeros_like(
        suv,
        dtype=np.uint8
    )

    mask_img = sitk.GetImageFromArray(
        empty_mask
    )

    mask_img.CopyInformation(
        ct_img
    )

    sitk.WriteImage(
        mask_img,
        str(
            OUTPUT_DIR
            / "tumor_mask.nii.gz"
        )
    )

    raise SystemExit(0)


# ============================================================
# STEP 5 — TOP-K SELECTION
# ============================================================

eligible = eligible.sort_values(
    [
        "pet_quality_score",
        "mean_suv_dev46"
    ],
    ascending=False
).reset_index(
    drop=True
)

selected = eligible.head(
    TOP_K
).copy()

selected_labels = [
    int(x)
    for x in selected["label"].tolist()
]

print()
print(
    "Selected component labels:",
    selected_labels
)

print()
print("Selected components:")

print(
    selected[
        [
            "label",
            "voxels",
            "z_start",
            "z_end",
            "mean_suv_dev46",
            "max_suv_dev46",
            "hot_fraction_dev46",
            "persistence",
            "longest_run",
            "compactness_dev46",
            "pet_quality_score",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# STEP 6 — P70 CORE EXTRACTION
# ============================================================

print()
print("-" * 100)
print("STEP 6 — P70 TUMOR CORE EXTRACTION")
print("-" * 100)

prediction = np.zeros_like(
    suv,
    dtype=bool
)

selected_core_rows = []

for label in selected_labels:

    core = safe_core(
        cc,
        suv,
        label,
        CORE_POLICY
    )

    prediction |= core

    component_mask = (
        cc == label
    )

    selected_core_rows.append(
        {
            "label": label,
            "component_voxels": int(
                component_mask.sum()
            ),
            "core_voxels": int(
                core.sum()
            ),
        }
    )


# ============================================================
# FINAL DETECTION SUMMARY
# ============================================================

prediction_voxels = int(
    prediction.sum()
)

predicted_slices = np.where(
    np.any(
        prediction,
        axis=(1, 2)
    )
)[0]

num_predicted_slices = len(
    predicted_slices
)

tumor_detected = (
    prediction_voxels > 0
)


print()
print("=" * 100)
print("DEV46 — FINAL DETECTION RESULT")
print("=" * 100)

print()
print(
    "Tumor detected:",
    "YES" if tumor_detected else "NO"
)

print(
    "Predicted tumor voxels:",
    prediction_voxels
)

print(
    "Predicted tumor slices:",
    num_predicted_slices
)

print(
    "Selected components:",
    len(selected_labels)
)


# ============================================================
# SAVE TUMOR MASK
# ============================================================

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
    str(mask_path)
)


# ============================================================
# SAVE COMPONENT AUDIT
# ============================================================

component_audit_path = (
    OUTPUT_DIR
    / "selected_components.csv"
)

selected.to_csv(
    component_audit_path,
    index=False
)


# ============================================================
# SAVE FULL COMPONENT TABLE
# ============================================================

all_components_path = (
    OUTPUT_DIR
    / "all_component_scores.csv"
)

quality_df.sort_values(
    "pet_quality_score",
    ascending=False
).to_csv(
    all_components_path,
    index=False
)


# ============================================================
# SAVE CORE SUMMARY
# ============================================================

core_df = pd.DataFrame(
    selected_core_rows
)

core_df.to_csv(
    OUTPUT_DIR
    / "selected_core_summary.csv",
    index=False
)


# ============================================================
# VISUALIZATION
# ============================================================

print()
print("-" * 100)
print("Generating tumor detection images...")
print("-" * 100)


# ------------------------------------------------------------
# PET/CT visualization
# ------------------------------------------------------------

ct_display = normalize_ct(
    ct
).astype(np.float32)

# Find central predicted slice.
if num_predicted_slices > 0:

    z_center = int(
        predicted_slices[
            len(predicted_slices) // 2
        ]
    )

else:

    z_center = (
        suv.shape[0] // 2
    )


def save_overlay(
    z,
    output_path,
    title
):

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    # CT grayscale.
    ax.imshow(
        ct_display[z],
        cmap="gray"
    )

    # PET heatmap with transparency.
    pet_slice = suv[z]

    vmax = np.percentile(
        pet_slice[
            pet_slice > 0
        ],
        99
    ) if np.any(
        pet_slice > 0
    ) else 1.0

    ax.imshow(
        pet_slice,
        cmap="hot",
        alpha=0.35,
        vmin=0,
        vmax=vmax
    )

    # Tumor contour.
    mask_slice = (
        prediction[z]
        .astype(np.uint8)
    )

    contours, _ = cv2.findContours(
        mask_slice,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        cv2.drawContours(
            mask_slice,
            contours,
            -1,
            255,
            1
        )

        ys, xs = np.where(
            mask_slice > 0
        )

        if len(xs) > 0:

            ax.scatter(
                xs,
                ys,
                s=0.15,
                c="red",
                alpha=0.7
            )

    ax.set_title(
        title,
        fontsize=13
    )

    ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


save_overlay(
    z_center,
    OUTPUT_DIR
    / "tumor_detection_axial.png",
    f"Dev39 Tumor Detection — Axial Slice {z_center}"
)


# ============================================================
# CREATE MAXIMUM-PROJECTION OVERVIEW
# ============================================================

projection = np.max(
    prediction.astype(np.uint8),
    axis=0
)

fig, ax = plt.subplots(
    figsize=(8, 8)
)

ax.imshow(
    normalize_ct(
        np.max(
            ct,
            axis=0
        )
    ),
    cmap="gray"
)

ys, xs = np.where(
    projection > 0
)

if len(xs) > 0:

    ax.scatter(
        xs,
        ys,
        s=0.5,
        c="red",
        alpha=0.7
    )

ax.set_title(
    "Dev39 — Tumor Detection Maximum Projection",
    fontsize=13
)

ax.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "tumor_detection_mip.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE DETECTION REPORT
# ============================================================

report_lines = [

    "DEV46 — FINAL TUMOR DETECTION REPORT",
    "=" * 70,
    "",
    f"Case: {CASE_NAME}",
    "",
    "METHOD: Dev39",
    "",
    "Frozen parameters:",
    f"PET threshold       = {PET_THRESHOLD}",
    f"Hot threshold       = {HOT_THRESHOLD}",
    f"3D minimum voxels   = {MIN_3D_VOXELS}",
    f"3D minimum slices   = {MIN_3D_SLICES}",
    f"Ranking             = PET-quality",
    f"Hot fraction gate   = {HOT_FRACTION_GATE}",
    f"Top-K               = {TOP_K}",
    f"Core policy         = {CORE_POLICY}",
    "",
    "RESULT",
    "",
    f"Tumor detected      = {'YES' if tumor_detected else 'NO'}",
    f"Candidate voxels    = {candidate_voxels}",
    f"Initial components  = {len(components)}",
    f"Eligible components = {len(eligible)}",
    f"Selected components = {len(selected_labels)}",
    f"Predicted voxels    = {prediction_voxels}",
    f"Predicted slices    = {num_predicted_slices}",
    "",
    "Selected labels:",
    str(selected_labels),
    "",
    "IMPORTANT:",
    "Ground-truth annotations were NOT used during inference.",
    "The prediction was generated using PET/CT data only.",
    "",
    "Output files:",
    "tumor_mask.nii.gz",
    "tumor_detection_axial.png",
    "tumor_detection_mip.png",
    "selected_components.csv",
    "selected_core_summary.csv",
    "all_component_scores.csv",
]

report_path = (
    OUTPUT_DIR
    / "detection_report.txt"
)

report_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 100)
print("DEV46 COMPLETE")
print("=" * 100)

print()
print("Tumor detected:")
print(
    "YES" if tumor_detected else "NO"
)

print(
    "Predicted tumor voxels:",
    prediction_voxels
)

print(
    "Predicted slices:",
    num_predicted_slices
)

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("Generated files:")

for path in sorted(
    OUTPUT_DIR.iterdir()
):

    print(
        " ",
        path.name
    )

print()
print("=" * 100)