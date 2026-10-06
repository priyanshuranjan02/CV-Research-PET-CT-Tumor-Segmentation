"""
DEV55 — Validation Component Visual Audit

Purpose
-------
Visual diagnostic for the frozen Dev39 validation pipeline.

This script does NOT:
- modify Dev10
- modify Dev39
- tune parameters
- use GT for component selection
- change the frozen model

It visualizes selected / important validation components:
    84  -> Dev39 rank #1, false positive
    310 -> Dev39 rank #2, strong GT-overlapping component
    309 -> Dev39 rank #6, strong GT-overlapping component
    302 -> Dev39 rank #4, weak GT-overlapping component

For each component, representative axial slices are generated showing:
    - PET/SUV background
    - component mask
    - GT contour
    - component bounding box

The GT is displayed ONLY as a diagnostic reference.
It is never used for selection.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib

import sys

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent)
)

import dev10_lite_hot_core as dev10
import dev39_pet_quality_fpr_control as dev39


# ============================================================================
# CONFIGURATION
# ============================================================================

CASE_NAME = "PETCT_0011f3deaf"

PET_THRESHOLD = 2.25
HOT_THRESHOLD = 3.0

MIN_3D_VOXELS = 75
MIN_3D_SLICES = 2

HOT_FRACTION_GATE = 0.40

TOP_K = 4
CORE_POLICY = "P70"

# Components identified by Dev54
AUDIT_LABELS = [
    84,
    310,
    309,
    302,
]

# Number of representative slices per component
N_SLICES = 5

OUTPUT_DIR = Path(
    "results/validation/dev55_component_visual_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================================
# HELPERS
# ============================================================================

def normalize_display(volume):
    """
    Normalize PET/SUV volume for visualization only.
    Does not modify the data used by the pipeline.
    """

    volume = np.asarray(volume, dtype=np.float32)

    positive = volume[volume > 0]

    if positive.size == 0:
        return np.zeros_like(volume)

    low = np.percentile(positive, 1)
    high = np.percentile(positive, 99)

    if high <= low:
        return np.zeros_like(volume)

    result = (volume - low) / (high - low)

    return np.clip(result, 0.0, 1.0)


def get_component_z_range(cc, label):
    """
    Return z_start, z_end for one 3D connected component.
    """

    coords = np.where(cc == label)

    if len(coords[0]) == 0:
        return None

    z = coords[0]

    return (
        int(z.min()),
        int(z.max())
    )


def choose_representative_slices(
    cc,
    label,
    n_slices=5
):
    """
    Select representative axial slices across the component's
    actual z extent.
    """

    z_range = get_component_z_range(
        cc,
        label
    )

    if z_range is None:
        return []

    z_start, z_end = z_range

    if z_start == z_end:
        return [z_start]

    count = min(
        n_slices,
        z_end - z_start + 1
    )

    slices = np.linspace(
        z_start,
        z_end,
        count
    ).round().astype(int)

    return sorted(
        set(int(z) for z in slices)
    )


def component_bbox(cc, label):
    """
    Return x/y bounding box of a component.
    """

    coords = np.where(cc == label)

    if len(coords[0]) == 0:
        return None

    z, y, x = coords

    return {
        "x_min": int(x.min()),
        "x_max": int(x.max()),
        "y_min": int(y.min()),
        "y_max": int(y.max()),
        "z_min": int(z.min()),
        "z_max": int(z.max()),
    }


def expand_bbox(
    bbox,
    shape,
    padding=15
):
    """
    Expand x/y bounding box for visualization.
    """

    x_min = max(
        0,
        bbox["x_min"] - padding
    )

    x_max = min(
        shape[2] - 1,
        bbox["x_max"] + padding
    )

    y_min = max(
        0,
        bbox["y_min"] - padding
    )

    y_max = min(
        shape[1] - 1,
        bbox["y_max"] + padding
    )

    return (
        x_min,
        x_max,
        y_min,
        y_max
    )


def safe_build_core(
    cc,
    suv,
    label,
    policy
):
    """
    Dev10 build_core returns a tuple.
    Extract the mask safely.
    """

    result = dev10.build_core(
        cc,
        suv,
        label,
        policy
    )

    if isinstance(result, tuple):

        for item in result:

            if isinstance(item, np.ndarray):
                if item.shape == cc.shape:
                    return item.astype(bool)

        raise RuntimeError(
            "Could not identify core mask from build_core() tuple."
        )

    return np.asarray(
        result
    ).astype(bool)


def component_stats(
    cc,
    suv,
    gt,
    label
):
    """
    Post-hoc diagnostic statistics.
    """

    mask = cc == label

    voxels = int(
        mask.sum()
    )

    if voxels == 0:
        return None

    values = suv[mask]

    z = np.where(mask)[0]

    overlap = int(
        np.logical_and(
            mask,
            gt
        ).sum()
    )

    dice = (
        2.0 * overlap
        / (voxels + int(gt.sum()))
        if voxels + int(gt.sum()) > 0
        else 0.0
    )

    return {
        "label": label,
        "voxels": voxels,
        "z_start": int(z.min()),
        "z_end": int(z.max()),
        "z_span": int(z.max() - z.min() + 1),
        "mean_suv": float(np.mean(values)),
        "max_suv": float(np.max(values)),
        "hot_fraction": float(
            np.mean(
                values >= HOT_THRESHOLD
            )
        ),
        "gt_overlap": overlap,
        "dice": dice,
    }


# ============================================================================
# LOAD VALIDATION CASE
# ============================================================================

def load_validation_case():

    print("\n[1] Loading validation case...")

    from v4_validate_case7 import prepare_case7

    loaded = prepare_case7()

    if not isinstance(
        loaded,
        tuple
    ):
        raise RuntimeError(
            "prepare_case7() did not return a tuple."
        )

    validation_case = loaded[0]

    suv = np.asarray(
        validation_case["suv"]
    )

    gt = np.asarray(
        validation_case["gt"]
    ).astype(bool)

    print(
        f"Case                 : "
        f"{validation_case['name']}"
    )

    print(
        f"SUV shape            : "
        f"{suv.shape}"
    )

    print(
        f"GT shape             : "
        f"{gt.shape}"
    )

    print(
        f"SUV max              : "
        f"{float(suv.max()):.6f}"
    )

    print(
        f"GT voxels            : "
        f"{int(gt.sum())}"
    )

    return (
        validation_case,
        suv,
        gt
    )


# ============================================================================
# BUILD EXACT DEV39 PIPELINE
# ============================================================================

def build_pipeline(
    validation_case,
    suv
):

    print(
        "\n[2] Generating exact "
        "Dev39 PET-only candidate..."
    )

    candidate = (
        suv >= PET_THRESHOLD
    ).astype(np.uint8)

    print(
        f"Candidate voxels     : "
        f"{int(candidate.sum())}"
    )

    print(
        "\n[3] Building exact "
        "Dev10 3D component table..."
    )

    components, cc = (
        dev10.build_3d_component_table(
            validation_case,
            candidate
        )
    )

    print(
        f"Total 3D components  : "
        f"{len(components)}"
    )

    print(
        "\n[4] Computing exact "
        "Dev39 component features..."
    )

    components = (
        dev39.add_component_features(
            components,
            cc,
            suv
        )
    )

    components = (
        dev39.add_pet_quality_score(
            components
        )
    )

    # Exact frozen Dev39 eligibility
    eligible = components[
        components["dev39_hot_fraction"]
        >= HOT_FRACTION_GATE
    ].copy()

    # Exact frozen Dev39 ranking
    eligible = (
        eligible.sort_values(
            [
                "dev39_pet_quality",
                "dev39_mean_suv"
            ],
            ascending=[
                False,
                False
            ]
        )
        .reset_index(drop=True)
    )

    eligible["rank"] = (
        np.arange(
            1,
            len(eligible) + 1
        )
    )

    eligible["selected"] = (
        eligible["rank"] <= TOP_K
    )

    print(
        f"Eligible components : "
        f"{len(eligible)}"
    )

    return (
        components,
        eligible,
        cc
    )


# ============================================================================
# SAVE COMPONENT TABLE
# ============================================================================

def save_component_summary(
    eligible,
    cc,
    suv,
    gt
):

    rows = []

    for _, row in eligible.iterrows():

        label = int(
            row["label"]
        )

        stats = component_stats(
            cc,
            suv,
            gt,
            label
        )

        if stats is None:
            continue

        bbox = component_bbox(
            cc,
            label
        )

        rows.append({

            "rank": int(
                row["rank"]
            ),

            "label": label,

            "selected": bool(
                row["selected"]
            ),

            "voxels": stats["voxels"],

            "z_start": stats["z_start"],
            "z_end": stats["z_end"],
            "z_span": stats["z_span"],

            "mean_suv": stats["mean_suv"],
            "max_suv": stats["max_suv"],
            "hot_fraction": stats["hot_fraction"],

            "persistence": float(
                row["dev39_persistence"]
            ),

            "longest_run": int(
                row["dev39_longest_run"]
            ),

            "compactness": float(
                row["dev39_compactness"]
            ),

            "dev39_pet_quality": float(
                row["dev39_pet_quality"]
            ),

            # Diagnostic only
            "gt_overlap": stats["gt_overlap"],
            "gt_dice": stats["dice"],

            "x_min": bbox["x_min"],
            "x_max": bbox["x_max"],
            "y_min": bbox["y_min"],
            "y_max": bbox["y_max"],
        })

    df = pd.DataFrame(
        rows
    )

    output = (
        OUTPUT_DIR
        / "dev55_component_summary.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    return df


# ============================================================================
# VISUALIZE ONE COMPONENT
# ============================================================================

def visualize_component(
    label,
    rank,
    selected,
    cc,
    suv,
    gt,
    summary_row
):

    component_mask = (
        cc == label
    )

    z_slices = (
        choose_representative_slices(
            cc,
            label,
            N_SLICES
        )
    )

    if not z_slices:
        print(
            f"WARNING: component "
            f"{label} has no voxels."
        )
        return

    suv_display = normalize_display(
        suv
    )

    bbox = component_bbox(
        cc,
        label
    )

    (
        x_min,
        x_max,
        y_min,
        y_max
    ) = expand_bbox(
        bbox,
        suv.shape,
        padding=20
    )

    fig, axes = plt.subplots(
        1,
        len(z_slices),
        figsize=(
            4.5 * len(z_slices),
            5
        )
    )

    if len(z_slices) == 1:
        axes = [axes]

    for ax, z in zip(
        axes,
        z_slices
    ):

        image = (
            suv_display[z]
        )

        ax.imshow(
            image,
            cmap="gray",
            origin="lower"
        )

        # Component overlay
        component_slice = (
            component_mask[z]
        )

        ax.imshow(
            np.ma.masked_where(
                ~component_slice,
                component_slice
            ),
            cmap="autumn",
            alpha=0.45,
            origin="lower"
        )

        # GT contour — diagnostic only
        gt_slice = gt[z]

        if np.any(gt_slice):

            ax.contour(
                gt_slice,
                levels=[0.5],
                linewidths=1.5,
                origin="lower"
            )

        # Component bounding box
        rect_x = [
            bbox["x_min"],
            bbox["x_max"],
            bbox["x_max"],
            bbox["x_min"],
            bbox["x_min"],
        ]

        rect_y = [
            bbox["y_min"],
            bbox["y_min"],
            bbox["y_max"],
            bbox["y_max"],
            bbox["y_min"],
        ]

        ax.plot(
            rect_x,
            rect_y,
            linewidth=1.0
        )

        ax.set_xlim(
            x_min,
            x_max
        )

        ax.set_ylim(
            y_min,
            y_max
        )

        ax.set_title(
            f"z={z}"
        )

        ax.set_xlabel(
            "x"
        )

        ax.set_ylabel(
            "y"
        )

    selection_text = (
        "SELECTED"
        if selected
        else "NOT SELECTED"
    )

    fig.suptitle(
        (
            f"Component {label} | "
            f"Dev39 Rank {rank} | "
            f"{selection_text}\n"
            f"Voxels={summary_row['voxels']:,} | "
            f"Mean SUV={summary_row['mean_suv']:.2f} | "
            f"Max SUV={summary_row['max_suv']:.2f} | "
            f"Hot fraction={summary_row['hot_fraction']:.3f} | "
            f"Quality={summary_row['dev39_pet_quality']:.4f}"
        ),
        fontsize=12
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.88
        ]
    )

    output = (
        OUTPUT_DIR
        / f"component_{label}_rank_{rank}.png"
    )

    fig.savefig(
        output,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"Saved visualization: "
        f"{output}"
    )


# ============================================================================
# OVERVIEW FIGURE
# ============================================================================

def create_overview(
    eligible,
    cc,
    suv,
    gt
):

    labels = [
        int(x)
        for x in AUDIT_LABELS
        if np.any(
            cc == int(x)
        )
    ]

    if not labels:
        return

    n = len(labels)

    fig, axes = plt.subplots(
        n,
        3,
        figsize=(
            12,
            4 * n
        )
    )

    if n == 1:
        axes = np.expand_dims(
            axes,
            axis=0
        )

    suv_display = normalize_display(
        suv
    )

    for row_index, label in enumerate(labels):

        component_mask = (
            cc == label
        )

        component_row = (
            eligible[
                eligible["label"] == label
            ].iloc[0]
        )

        rank = int(
            component_row["rank"]
        )

        selected = bool(
            component_row["selected"]
        )

        z_range = get_component_z_range(
            cc,
            label
        )

        if z_range is None:
            continue

        z_start, z_end = z_range

        z_mid = int(
            round(
                (z_start + z_end) / 2
            )
        )

        selected_text = (
            "SELECTED"
            if selected
            else "NOT SELECTED"
        )

        for col, z in enumerate([
            z_start,
            z_mid,
            z_end
        ]):

            ax = axes[
                row_index,
                col
            ]

            ax.imshow(
                suv_display[z],
                cmap="gray",
                origin="lower"
            )

            mask_slice = (
                component_mask[z]
            )

            ax.imshow(
                np.ma.masked_where(
                    ~mask_slice,
                    mask_slice
                ),
                cmap="autumn",
                alpha=0.45,
                origin="lower"
            )

            gt_slice = gt[z]

            if np.any(gt_slice):

                ax.contour(
                    gt_slice,
                    levels=[0.5],
                    linewidths=1.5,
                    origin="lower"
                )

            ax.set_title(
                f"Component {label} | "
                f"Rank {rank} | "
                f"z={z}\n"
                f"{selected_text}"
            )

            ax.axis(
                "off"
            )

    fig.suptitle(
        (
            "DEV55 — Validation Component "
            "Visual Overview\n"
            "Orange = candidate component | "
            "Contour = GT (diagnostic only)"
        ),
        fontsize=14
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.95
        ]
    )

    output = (
        OUTPUT_DIR
        / "dev55_component_overview.png"
    )

    fig.savefig(
        output,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"Saved overview: {output}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print(
        "DEV55 — VALIDATION COMPONENT "
        "VISUAL AUDIT"
    )
    print("=" * 80)

    print(
        f"Case                 : {CASE_NAME}"
    )

    print(
        f"PET threshold        : {PET_THRESHOLD}"
    )

    print(
        f"Hot threshold        : {HOT_THRESHOLD}"
    )

    print(
        f"3D minimum voxels    : {MIN_3D_VOXELS}"
    )

    print(
        f"3D minimum slices    : {MIN_3D_SLICES}"
    )

    print(
        f"Hot fraction gate    : "
        f"{HOT_FRACTION_GATE}"
    )

    print(
        f"Top-K                : {TOP_K}"
    )

    print(
        f"Core policy          : {CORE_POLICY}"
    )

    print(
        f"Audit labels         : "
        f"{AUDIT_LABELS}"
    )

    (
        validation_case,
        suv,
        gt
    ) = load_validation_case()

    (
        components,
        eligible,
        cc
    ) = build_pipeline(
        validation_case,
        suv
    )

    print(
        "\n[5] Saving component summary..."
    )

    summary = save_component_summary(
        eligible,
        cc,
        suv,
        gt
    )

    print(
        "\n[6] Creating component "
        "visualizations..."
    )

    for label in AUDIT_LABELS:

        matching = summary[
            summary["label"] == label
        ]

        if matching.empty:

            print(
                f"Component {label}: "
                f"not present in eligible set."
            )

            continue

        row = matching.iloc[0]

        visualize_component(
            label=label,
            rank=int(row["rank"]),
            selected=bool(
                row["selected"]
            ),
            cc=cc,
            suv=suv,
            gt=gt,
            summary_row=row
        )

    print(
        "\n[7] Creating overview..."
    )

    create_overview(
        eligible,
        cc,
        suv,
        gt
    )

    # -----------------------------------------------------------------
    # Print concise diagnostic summary
    # -----------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "DEV55 — COMPONENT SUMMARY"
    )

    print(
        "=" * 80
    )

    display_columns = [
        "rank",
        "label",
        "selected",
        "voxels",
        "z_start",
        "z_end",
        "mean_suv",
        "max_suv",
        "hot_fraction",
        "persistence",
        "compactness",
        "dev39_pet_quality",
        "gt_overlap",
        "gt_dice",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "DEV55 COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"Output directory:\n"
        f"{OUTPUT_DIR.resolve()}"
    )

    print(
        "\nVisualizations:"
    )

    for label in AUDIT_LABELS:

        path = (
            OUTPUT_DIR
            / f"component_{label}_rank_"
            f"{int(summary[summary['label'] == label]['rank'].iloc[0])}.png"
            if not summary[
                summary["label"] == label
            ].empty
            else None
        )

        if path:
            print(
                f"  {path}"
            )

    print(
        f"\nOverview:"
    )

    print(
        OUTPUT_DIR
        / "dev55_component_overview.png"
    )

    print(
        f"\nCSV:"
    )

    print(
        OUTPUT_DIR
        / "dev55_component_summary.csv"
    )


if __name__ == "__main__":
    main()