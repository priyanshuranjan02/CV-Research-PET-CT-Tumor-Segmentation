from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import dev10_lite_hot_core as dev10


# ============================================================
# DEV48 CONFIGURATION
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

# Number of representative slices per component.
N_SLICES = 5

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "development_cases"
    / "dev48_anatomical_localization_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAFE CORE
# ============================================================

def safe_build_core(cc, suv, label, policy):

    result = dev10.build_core(
        cc,
        suv,
        int(label),
        policy
    )

    if isinstance(result, tuple):

        for item in result:

            if isinstance(item, np.ndarray):
                return item.astype(bool)

    return result.astype(bool)


# ============================================================
# DEV39 PET QUALITY
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
        (values - vmin)
        / (vmax - vmin)
    )


def longest_consecutive_run(z_values):

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

        if (
            z_values[i]
            == z_values[i - 1] + 1
        ):
            current += 1
            best = max(
                best,
                current
            )
        else:
            current = 1

    return best


def calculate_features(
    components,
    cc,
    suv
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

        suv_values = suv[mask]

        voxels = len(coords)

        z_start = int(
            z.min()
        )

        z_end = int(
            z.max()
        )

        z_span = (
            z_end
            - z_start
            + 1
        )

        unique_z = np.unique(z)

        slice_count = len(
            unique_z
        )

        persistence = (
            slice_count
            / suv.shape[0]
        )

        longest_run = (
            longest_consecutive_run(
                unique_z
            )
        )

        bbox_volume = (
            (z.max() - z.min() + 1)
            * (y.max() - y.min() + 1)
            * (x.max() - x.min() + 1)
        )

        compactness = (
            voxels
            / max(
                bbox_volume,
                1
            )
        )

        mean_suv = float(
            np.mean(suv_values)
        )

        max_suv = float(
            np.max(suv_values)
        )

        hot_fraction = float(
            np.mean(
                suv_values
                >= HOT_THRESHOLD
            )
        )

        centroid_z = float(
            np.mean(z)
        )

        centroid_y = float(
            np.mean(y)
        )

        centroid_x = float(
            np.mean(x)
        )

        rows.append({

            "label": label,

            "voxels": voxels,

            "z_start": z_start,
            "z_end": z_end,
            "z_span": z_span,

            "slice_count": slice_count,

            "persistence": persistence,

            "longest_run": longest_run,

            "centroid_z": centroid_z,
            "centroid_y": centroid_y,
            "centroid_x": centroid_x,

            "x_min": int(x.min()),
            "x_max": int(x.max()),
            "y_min": int(y.min()),
            "y_max": int(y.max()),

            "mean_suv": mean_suv,
            "max_suv": max_suv,

            "hot_fraction": hot_fraction,

            "compactness": compactness,
        })

    return pd.DataFrame(rows)


def add_pet_quality_score(df):

    if df.empty:
        return df

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

    # EXACT Dev39 PET-quality score.
    df["pet_quality_score"] = (
        0.25 * df["n_mean_suv"]
        + 0.15 * df["n_max_suv"]
        + 0.15 * df["n_hot_fraction"]
        + 0.20 * df["n_persistence"]
        + 0.15 * df["n_longest_run"]
        + 0.10 * df["n_compactness"]
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

    rows = []

    gt_voxels = int(
        gt.sum()
    )

    for _, row in df.iterrows():

        label = int(
            row["label"]
        )

        mask = (
            cc == label
        )

        overlap = int(
            np.logical_and(
                mask,
                gt
            ).sum()
        )

        component_voxels = int(
            mask.sum()
        )

        dice = (
            2.0 * overlap
            / max(
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
            / max(
                union,
                1
            )
        )

        rows.append({

            "label": label,

            "overlap_gt": overlap,

            "dice_gt": dice,

            "iou_gt": iou,

        })

    gt_df = pd.DataFrame(
        rows
    )

    return df.merge(
        gt_df,
        on="label",
        how="left"
    )


# ============================================================
# REPRESENTATIVE SLICE SELECTION
# ============================================================

def representative_slices(
    z_start,
    z_end,
    n=N_SLICES
):

    if z_end <= z_start:

        return [
            int(z_start)
        ]

    values = np.linspace(
        z_start,
        z_end,
        n
    )

    return sorted(
        set(
            int(round(v))
            for v in values
        )
    )


# ============================================================
# OVERLAY VISUALIZATION
# ============================================================

def make_component_slice(
    ct_slice,
    suv_slice,
    component_slice,
    gt_slice,
    title,
    output_path
):

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    # CT background.
    ax.imshow(
        ct_slice,
        cmap="gray"
    )

    # PET uptake.
    pet_mask = np.ma.masked_where(
        suv_slice < PET_THRESHOLD,
        suv_slice
    )

    ax.imshow(
        pet_mask,
        cmap="hot",
        alpha=0.35
    )

    # Component in red.
    component_overlay = np.ma.masked_where(
        ~component_slice,
        component_slice
    )

    ax.imshow(
        component_overlay,
        cmap="Reds",
        alpha=0.65
    )

    # GT in cyan/green-like visualization.
    if gt_slice is not None:

        gt_overlay = np.ma.masked_where(
            ~gt_slice,
            gt_slice
        )

        ax.imshow(
            gt_overlay,
            cmap="winter",
            alpha=0.40
        )

    ax.set_title(
        title,
        fontsize=12
    )

    ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# CONTACT SHEET
# ============================================================

def make_contact_sheet(
    images,
    titles,
    output_path
):

    if len(images) == 0:
        return

    n = len(images)

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(5 * n, 5)
    )

    if n == 1:
        axes = [axes]

    for ax, image, title in zip(
        axes,
        images,
        titles
    ):

        ax.imshow(
            image
        )

        ax.set_title(
            title,
            fontsize=10
        )

        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# SINGLE COMPONENT VISUALIZATION
# ============================================================

def visualize_component(
    case_name,
    component_row,
    ct,
    suv,
    gt,
    cc,
    rank
):

    label = int(
        component_row["label"]
    )

    z_start = int(
        component_row["z_start"]
    )

    z_end = int(
        component_row["z_end"]
    )

    z_values = representative_slices(
        z_start,
        z_end,
        N_SLICES
    )

    component_mask = (
        cc == label
    )

    case_dir = (
        OUTPUT_DIR
        / "component_visualizations"
        / case_name
        / f"rank_{rank:02d}_label_{label}"
    )

    case_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    contact_images = []
    contact_titles = []

    for z in z_values:

        ct_slice = ct[z]
        suv_slice = suv[z]

        component_slice = (
            component_mask[z]
        )

        gt_slice = None

        if gt is not None:
            gt_slice = gt[z]

        output_path = (
            case_dir
            / f"slice_{z:03d}.png"
        )

        title = (
            f"{case_name} | "
            f"Rank {rank} | "
            f"Label {label} | "
            f"z={z}"
        )

        make_component_slice(
            ct_slice,
            suv_slice,
            component_slice,
            gt_slice,
            title,
            output_path
        )

        # Generate in-memory image for contact sheet.
        fig, ax = plt.subplots(
            figsize=(4, 4)
        )

        ax.imshow(
            ct_slice,
            cmap="gray"
        )

        pet_mask = np.ma.masked_where(
            suv_slice < PET_THRESHOLD,
            suv_slice
        )

        ax.imshow(
            pet_mask,
            cmap="hot",
            alpha=0.35
        )

        component_overlay = np.ma.masked_where(
            ~component_slice,
            component_slice
        )

        ax.imshow(
            component_overlay,
            cmap="Reds",
            alpha=0.65
        )

        if gt_slice is not None:

            gt_overlay = np.ma.masked_where(
                ~gt_slice,
                gt_slice
            )

            ax.imshow(
                gt_overlay,
                cmap="winter",
                alpha=0.40
            )

        ax.axis("off")
        ax.set_title(
            f"z={z}"
        )

        fig.canvas.draw()

        image = np.asarray(
            fig.canvas.buffer_rgba()
        )[:, :, :3]

        plt.close(fig)

        contact_images.append(
            image
        )

        contact_titles.append(
            f"z={z}"
        )

    contact_path = (
        case_dir
        / "contact_sheet.png"
    )

    make_contact_sheet(
        contact_images,
        contact_titles,
        contact_path
    )


# ============================================================
# CASE AUDIT
# ============================================================

def audit_case(case_name):

    print()
    print("=" * 110)
    print(
        f"DEV48 ANATOMICAL AUDIT — {case_name}"
    )
    print("=" * 110)

    case = dev10.load_case(
        case_name
    )

    ct = case["ct"]
    suv = case["suv"]
    gt = case["gt"]

    print(
        "CT shape:",
        ct.shape
    )

    print(
        "SUV shape:",
        suv.shape
    )

    print(
        "GT shape:",
        gt.shape
    )

    # --------------------------------------------------------
    # PET-only candidate
    # --------------------------------------------------------

    candidate_volume = (
        suv >= PET_THRESHOLD
    ).astype(np.uint8)

    print(
        "PET candidate voxels:",
        int(candidate_volume.sum())
    )

    # --------------------------------------------------------
    # Exact Dev10 3D component table
    # --------------------------------------------------------

    components, cc = (
        dev10.build_3d_component_table(
            case,
            candidate_volume
        )
    )

    print(
        "3D components:",
        len(components)
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    df = calculate_features(
        components,
        cc,
        suv
    )

    df = add_pet_quality_score(
        df
    )

    df = add_gt_metrics(
        df,
        cc,
        gt
    )

    # --------------------------------------------------------
    # Dev39 gate
    # --------------------------------------------------------

    df["eligible_dev39"] = (
        df["hot_fraction"]
        >= HOT_FRACTION_GATE
    )

    eligible = df[
        df["eligible_dev39"]
    ].copy()

    eligible = eligible.sort_values(
        [
            "pet_quality_score",
            "mean_suv"
        ],
        ascending=False
    ).reset_index(
        drop=True
    )

    eligible["dev39_rank"] = (
        np.arange(
            len(eligible)
        ) + 1
    )

    selected = eligible.head(
        TOP_K
    ).copy()

    # --------------------------------------------------------
    # Core audit
    # --------------------------------------------------------

    core_rows = []

    for _, row in selected.iterrows():

        label = int(
            row["label"]
        )

        core_mask = safe_build_core(
            cc,
            suv,
            label,
            CORE_POLICY
        )

        core_voxels = int(
            core_mask.sum()
        )

        core_overlap = int(
            np.logical_and(
                core_mask,
                gt
            ).sum()
        )

        gt_voxels = int(
            gt.sum()
        )

        core_dice = (
            2.0 * core_overlap
            / max(
                core_voxels
                + gt_voxels,
                1
            )
        )

        core_rows.append({

            "label": label,

            "core_voxels":
                core_voxels,

            "core_overlap_gt":
                core_overlap,

            "core_dice_gt":
                core_dice,

        })

    core_df = pd.DataFrame(
        core_rows
    )

    selected = selected.merge(
        core_df,
        on="label",
        how="left"
    )

    # --------------------------------------------------------
    # Anatomical descriptors
    # --------------------------------------------------------

    selected["z_center_fraction"] = (
        selected["centroid_z"]
        / max(
            suv.shape[0] - 1,
            1
        )
    )

    selected["y_center_fraction"] = (
        selected["centroid_y"]
        / max(
            suv.shape[1] - 1,
            1
        )
    )

    selected["x_center_fraction"] = (
        selected["centroid_x"]
        / max(
            suv.shape[2] - 1,
            1
        )
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print(
        "SELECTED COMPONENTS"
    )

    print("-" * 110)

    display_columns = [

        "dev39_rank",
        "label",

        "voxels",

        "z_start",
        "z_end",
        "z_span",

        "centroid_z",
        "centroid_y",
        "centroid_x",

        "mean_suv",
        "max_suv",
        "hot_fraction",

        "persistence",
        "longest_run",
        "compactness",

        "pet_quality_score",

        "overlap_gt",
        "dice_gt",

        "core_dice_gt",

    ]

    print(
        selected[
            display_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Visualizations
    # --------------------------------------------------------

    for _, row in selected.iterrows():

        visualize_component(
            case_name,
            row,
            ct,
            suv,
            gt,
            cc,
            int(
                row["dev39_rank"]
            )
        )

    # --------------------------------------------------------
    # Save all component audit
    # --------------------------------------------------------

    df["case"] = case_name

    selected["case"] = case_name

    return df, selected


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 110)
    print(
        "DEV48 — ANATOMICAL LOCALIZATION AUDIT"
    )
    print("=" * 110)

    print(
        "PET threshold:",
        PET_THRESHOLD
    )

    print(
        "Hot threshold:",
        HOT_THRESHOLD
    )

    print(
        "Hot-fraction gate:",
        HOT_FRACTION_GATE
    )

    print(
        "Top-K:",
        TOP_K
    )

    print(
        "Core policy:",
        CORE_POLICY
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "Dev39 parameters are frozen."
    )
    print(
        "This experiment performs localization only."
    )
    print(
        "No anatomical suppression is applied."
    )

    all_components = []
    all_selected = []

    for case_name in CASES:

        components, selected = (
            audit_case(
                case_name
            )
        )

        all_components.append(
            components
        )

        all_selected.append(
            selected
        )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    all_components_df = pd.concat(
        all_components,
        ignore_index=True
    )

    all_selected_df = pd.concat(
        all_selected,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    all_components_path = (
        OUTPUT_DIR
        / "dev48_all_component_localization.csv"
    )

    selected_path = (
        OUTPUT_DIR
        / "dev48_selected_component_localization.csv"
    )

    all_components_df.to_csv(
        all_components_path,
        index=False
    )

    all_selected_df.to_csv(
        selected_path,
        index=False
    )

    # --------------------------------------------------------
    # Case summary
    # --------------------------------------------------------

    summary = (
        all_selected_df
        .groupby("case")
        .agg(

            selected_components=(
                "label",
                "count"
            ),

            selected_volume=(
                "voxels",
                "sum"
            ),

            largest_component=(
                "voxels",
                "max"
            ),

            mean_suv=(
                "mean_suv",
                "mean"
            ),

            mean_hot_fraction=(
                "hot_fraction",
                "mean"
            ),

            mean_persistence=(
                "persistence",
                "mean"
            ),

            gt_overlapping_components=(
                "overlap_gt",
                lambda x: int(
                    (x > 0).sum()
                )
            ),

            mean_component_dice=(
                "dice_gt",
                "mean"
            ),

            mean_core_dice=(
                "core_dice_gt",
                "mean"
            ),

            mean_centroid_z=(
                "centroid_z",
                "mean"
            ),

            mean_centroid_y=(
                "centroid_y",
                "mean"
            ),

            mean_centroid_x=(
                "centroid_x",
                "mean"
            ),

        )
        .reset_index()
    )

    summary_path = (
        OUTPUT_DIR
        / "dev48_case_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------------
    # Print final summary
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print(
        "DEV48 CASE SUMMARY"
    )
    print("=" * 110)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Generated:"
    )

    print(
        "  dev48_all_component_localization.csv"
    )

    print(
        "  dev48_selected_component_localization.csv"
    )

    print(
        "  dev48_case_summary.csv"
    )

    print(
        "  component_visualizations/"
    )

    print()
    print("=" * 110)
    print(
        "DEV48 COMPLETE"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()