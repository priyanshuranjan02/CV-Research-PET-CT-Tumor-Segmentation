"""
Dev23 — 3D PET/GT Longitudinal Audit

Purpose:
    Analyze the complete GT-positive z-range of
    PETCT_185da4c8b6.

    Main question:
        Does PET uptake disappear while the GT remains present?

    Dev10 is NOT modified.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

RESULT_DIR = (
    ROOT
    / "results"
    / "development_cases"
    / "dev23_3d_pet_gt_longitudinal_results"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

sys.path.insert(0, str(SRC))

import dev10_lite_hot_core as dev10


CASE_NAME = "PETCT_185da4c8b6"

PET_THRESHOLD = 2.25


def classify_pet(gt_max_suv):
    """
    Classification based on maximum SUV inside GT.

    PET_STRONG:
        At least one GT pixel reaches PET candidate threshold.

    PET_WEAK:
        PET exists inside GT but remains below threshold.

    PET_ABSENT:
        No measurable positive PET value inside GT.
    """

    if gt_max_suv >= PET_THRESHOLD:
        return "PET_STRONG"

    if gt_max_suv > 0:
        return "PET_WEAK"

    return "PET_ABSENT"


def calculate_metrics(
    suv_slice,
    gt_slice
):
    """
    Calculate PET/GT spatial and intensity metrics.
    """

    gt_area = int(gt_slice.sum())

    if gt_area == 0:
        return None

    gt_suv = suv_slice[gt_slice]

    gt_mean_suv = float(
        np.mean(gt_suv)
    )

    gt_median_suv = float(
        np.median(gt_suv)
    )

    gt_max_suv = float(
        np.max(gt_suv)
    )

    # --------------------------------------------
    # PET maximum on entire slice
    # --------------------------------------------

    max_y, max_x = np.unravel_index(
        np.argmax(suv_slice),
        suv_slice.shape
    )

    global_pet_max = float(
        suv_slice[max_y, max_x]
    )

    # --------------------------------------------
    # GT centroid
    # --------------------------------------------

    gt_y, gt_x = np.where(gt_slice)

    gt_cx = float(
        gt_x.mean()
    )

    gt_cy = float(
        gt_y.mean()
    )

    # --------------------------------------------
    # Distance PET maximum -> GT
    # --------------------------------------------

    distances = np.sqrt(
        (gt_x - max_x) ** 2 +
        (gt_y - max_y) ** 2
    )

    max_to_gt_nearest = float(
        np.min(distances)
    )

    max_to_gt_centroid = float(
        np.sqrt(
            (max_x - gt_cx) ** 2 +
            (max_y - gt_cy) ** 2
        )
    )

    # --------------------------------------------
    # PET threshold mask
    # --------------------------------------------

    pet_mask = (
        suv_slice >= PET_THRESHOLD
    )

    overlap = np.logical_and(
        pet_mask,
        gt_slice
    ).sum()

    pet_area = int(
        pet_mask.sum()
    )

    union = np.logical_or(
        pet_mask,
        gt_slice
    ).sum()

    coverage = (
        overlap / gt_area
        if gt_area > 0
        else 0.0
    )

    dice = (
        2.0 * overlap /
        (gt_area + pet_area)
        if gt_area + pet_area > 0
        else 0.0
    )

    iou = (
        overlap / union
        if union > 0
        else 0.0
    )

    # --------------------------------------------
    # PET centroid
    # --------------------------------------------

    pet_y, pet_x = np.where(
        pet_mask
    )

    if len(pet_x) > 0:

        pet_cx = float(
            pet_x.mean()
        )

        pet_cy = float(
            pet_y.mean()
        )

        pet_centroid_distance = float(
            np.sqrt(
                (pet_cx - gt_cx) ** 2 +
                (pet_cy - gt_cy) ** 2
            )
        )

    else:

        pet_cx = np.nan
        pet_cy = np.nan
        pet_centroid_distance = np.nan

    classification = classify_pet(
        gt_max_suv
    )

    return {
        "gt_area": gt_area,

        "gt_mean_suv":
            gt_mean_suv,

        "gt_median_suv":
            gt_median_suv,

        "gt_max_suv":
            gt_max_suv,

        "global_pet_max_suv":
            global_pet_max,

        "pet_max_x":
            int(max_x),

        "pet_max_y":
            int(max_y),

        "gt_centroid_x":
            gt_cx,

        "gt_centroid_y":
            gt_cy,

        "max_to_gt_nearest_px":
            max_to_gt_nearest,

        "max_to_gt_centroid_px":
            max_to_gt_centroid,

        "pet_area_ge_2p25":
            pet_area,

        "pet_gt_overlap":
            int(overlap),

        "pet_gt_coverage":
            float(coverage),

        "pet_gt_dice":
            float(dice),

        "pet_gt_iou":
            float(iou),

        "pet_centroid_x":
            pet_cx,

        "pet_centroid_y":
            pet_cy,

        "pet_centroid_distance_px":
            pet_centroid_distance,

        "pet_class":
            classification,
    }


def create_plot(df, output_path):
    """
    Plot GT area and PET uptake over the z-axis.
    """

    fig, ax1 = plt.subplots(
        figsize=(14, 7)
    )

    ax1.plot(
        df["slice"],
        df["gt_area"],
        marker="o",
        linewidth=1.8,
        label="GT area"
    )

    ax1.set_xlabel(
        "Slice (z)"
    )

    ax1.set_ylabel(
        "GT area (pixels)"
    )

    ax1.grid(
        alpha=0.25
    )

    ax2 = ax1.twinx()

    ax2.plot(
        df["slice"],
        df["gt_mean_suv"],
        marker="s",
        linewidth=1.8,
        label="GT mean SUV"
    )

    ax2.plot(
        df["slice"],
        df["gt_max_suv"],
        marker="^",
        linewidth=1.8,
        label="GT max SUV"
    )

    ax2.plot(
        df["slice"],
        df["global_pet_max_suv"],
        marker="x",
        linewidth=1.5,
        label="Global PET max SUV"
    )

    ax2.axhline(
        PET_THRESHOLD,
        linestyle="--",
        linewidth=1.2,
        label="PET threshold 2.25"
    )

    ax2.set_ylabel(
        "SUV"
    )

    ax1.set_title(
        "Dev23 — PET/GT Longitudinal Audit\n"
        "PETCT_185da4c8b6"
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right"
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


def main():

    print("=" * 100)
    print("DEV23 — 3D PET/GT LONGITUDINAL AUDIT")
    print("=" * 100)

    print(
        f"\nCase: {CASE_NAME}"
    )

    case = dev10.load_case(
        CASE_NAME
    )

    ct = case["ct"]
    suv = case["suv"]
    gt = case["gt"]

    print(
        f"CT shape : {ct.shape}"
    )

    print(
        f"PET shape: {suv.shape}"
    )

    print(
        f"GT shape : {gt.shape}"
    )

    gt_bool = gt.astype(bool)

    positive_slices = np.where(
        np.any(
            gt_bool,
            axis=(1, 2)
        )
    )[0]

    print(
        f"\nGT-positive slices: "
        f"{len(positive_slices)}"
    )

    print(
        f"Z range: "
        f"{positive_slices[0]} "
        f"to "
        f"{positive_slices[-1]}"
    )

    rows = []

    for z in positive_slices:

        metrics = calculate_metrics(
            suv[z],
            gt_bool[z]
        )

        metrics["case_id"] = CASE_NAME
        metrics["slice"] = int(z)

        rows.append(
            metrics
        )

    df = pd.DataFrame(
        rows
    )

    # Put important columns first
    ordered_columns = [
        "case_id",
        "slice",
        "gt_area",
        "gt_mean_suv",
        "gt_median_suv",
        "gt_max_suv",
        "global_pet_max_suv",
        "pet_gt_coverage",
        "pet_gt_dice",
        "pet_gt_iou",
        "max_to_gt_nearest_px",
        "max_to_gt_centroid_px",
        "pet_centroid_distance_px",
        "pet_class",
    ]

    remaining_columns = [
        c for c in df.columns
        if c not in ordered_columns
    ]

    df = df[
        ordered_columns +
        remaining_columns
    ]

    # --------------------------------------------
    # Print complete slice table
    # --------------------------------------------

    print("\n" + "=" * 100)
    print("SLICE-LEVEL PET/GT AUDIT")
    print("=" * 100)

    print(
        df[
            [
                "slice",
                "gt_area",
                "gt_mean_suv",
                "gt_max_suv",
                "global_pet_max_suv",
                "pet_gt_coverage",
                "pet_gt_dice",
                "max_to_gt_nearest_px",
                "pet_class",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------
    # Classification summary
    # --------------------------------------------

    print("\n" + "=" * 100)
    print("PET CLASSIFICATION SUMMARY")
    print("=" * 100)

    class_summary = (
        df.groupby(
            "pet_class"
        )
        .agg(
            slices=(
                "slice",
                "count"
            ),

            first_slice=(
                "slice",
                "min"
            ),

            last_slice=(
                "slice",
                "max"
            ),

            mean_gt_area=(
                "gt_area",
                "mean"
            ),

            mean_gt_suv=(
                "gt_mean_suv",
                "mean"
            ),

            mean_gt_max_suv=(
                "gt_max_suv",
                "mean"
            ),

            mean_coverage=(
                "pet_gt_coverage",
                "mean"
            ),

            mean_dice=(
                "pet_gt_dice",
                "mean"
            ),
        )
        .reset_index()
    )

    print(
        class_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------
    # Save files
    # --------------------------------------------

    csv_path = (
        RESULT_DIR /
        "dev23_slice_audit.csv"
    )

    summary_path = (
        RESULT_DIR /
        "dev23_class_summary.csv"
    )

    plot_path = (
        RESULT_DIR /
        "dev23_pet_gt_longitudinal_plot.png"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    class_summary.to_csv(
        summary_path,
        index=False
    )

    create_plot(
        df,
        plot_path
    )

    # --------------------------------------------
    # Transition analysis
    # --------------------------------------------

    print("\n" + "=" * 100)
    print("PET-ABSENCE TRANSITION")
    print("=" * 100)

    absent = df[
        df["pet_class"] == "PET_ABSENT"
    ]

    weak = df[
        df["pet_class"] == "PET_WEAK"
    ]

    strong = df[
        df["pet_class"] == "PET_STRONG"
    ]

    print(
        f"PET_STRONG slices : {len(strong)}"
    )

    print(
        f"PET_WEAK slices   : {len(weak)}"
    )

    print(
        f"PET_ABSENT slices : {len(absent)}"
    )

    if len(absent) > 0:

        print(
            "\nPET_ABSENT range:"
        )

        print(
            f"{absent['slice'].min()} "
            f"to "
            f"{absent['slice'].max()}"
        )

    print("\nOutput files:")

    print(
        f"Slice audit : {csv_path}"
    )

    print(
        f"Class summary: {summary_path}"
    )

    print(
        f"Plot        : {plot_path}"
    )

    print("\n" + "=" * 100)
    print("DEV23 COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()