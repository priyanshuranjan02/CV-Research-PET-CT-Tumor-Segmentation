from pathlib import Path
import sys
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import dev10_lite_hot_core as dev10


# ============================================================
# DEV47 — PHYSIOLOGICAL FALSE-POSITIVE AUDIT
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
    / "dev47_physiological_false_positive_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SAFE CORE HELPER
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
# PET QUALITY FEATURES
# EXACT FROZEN DEV39 FORMULATION
# ============================================================

def longest_consecutive_run(z_indices):
    if len(z_indices) == 0:
        return 0

    z_indices = sorted(set(int(z) for z in z_indices))

    best = 1
    current = 1

    for i in range(1, len(z_indices)):
        if z_indices[i] == z_indices[i - 1] + 1:
            current += 1
            best = max(best, current)
        else:
            current = 1

    return best


def minmax_normalize(values):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    vmin = np.min(values)
    vmax = np.max(values)

    if vmax - vmin < 1e-12:
        return np.ones_like(values)

    return (values - vmin) / (vmax - vmin)


def calculate_component_features(
    components,
    cc,
    suv
):
    rows = []

    for _, row in components.iterrows():

        label = int(row["label"])
        component_mask = cc == label

        coords = np.argwhere(component_mask)

        if len(coords) == 0:
            continue

        z = coords[:, 0]
        y = coords[:, 1]
        x = coords[:, 2]

        suv_values = suv[component_mask]

        voxels = len(coords)

        z_start = int(z.min())
        z_end = int(z.max())
        z_span = z_end - z_start + 1

        unique_z = np.unique(z)
        slice_count = len(unique_z)

        persistence = slice_count / suv.shape[0]

        longest_run = longest_consecutive_run(unique_z)

        bbox_volume = (
            (z.max() - z.min() + 1)
            * (y.max() - y.min() + 1)
            * (x.max() - x.min() + 1)
        )

        compactness = voxels / max(bbox_volume, 1)

        mean_suv = float(np.mean(suv_values))
        max_suv = float(np.max(suv_values))

        hot_fraction = float(
            np.mean(suv_values >= HOT_THRESHOLD)
        )

        centroid_z = float(np.mean(z))
        centroid_y = float(np.mean(y))
        centroid_x = float(np.mean(x))

        # Normalized distance from image center.
        cz = (suv.shape[0] - 1) / 2.0
        cy = (suv.shape[1] - 1) / 2.0
        cx = (suv.shape[2] - 1) / 2.0

        dz = (centroid_z - cz) / max(cz, 1)
        dy = (centroid_y - cy) / max(cy, 1)
        dx = (centroid_x - cx) / max(cx, 1)

        normalized_center_distance = float(
            np.sqrt(
                dz * dz +
                dy * dy +
                dx * dx
            )
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
            "normalized_center_distance":
                normalized_center_distance,
            "bbox_volume": bbox_volume,
            "compactness": compactness,
            "mean_suv": mean_suv,
            "max_suv": max_suv,
            "hot_fraction": hot_fraction,
        })

    return pd.DataFrame(rows)


# ============================================================
# DEV39 PET-QUALITY SCORE
# ============================================================

def add_pet_quality_score(df):

    if df.empty:
        df["pet_quality_score"] = []
        return df

    df["n_mean_suv"] = minmax_normalize(df["mean_suv"])
    df["n_max_suv"] = minmax_normalize(df["max_suv"])
    df["n_hot_fraction"] = minmax_normalize(df["hot_fraction"])
    df["n_persistence"] = minmax_normalize(df["persistence"])
    df["n_longest_run"] = minmax_normalize(df["longest_run"])
    df["n_compactness"] = minmax_normalize(df["compactness"])

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
# VISUALIZATION
# ============================================================

def save_component_visualization(
    case_name,
    suv,
    ct,
    cc,
    component_row,
    rank
):

    label = int(component_row["label"])

    coords = np.argwhere(cc == label)

    if len(coords) == 0:
        return

    z_values = coords[:, 0]

    # Use middle slice of component.
    z = int(np.median(z_values))

    ct_slice = ct[z]
    suv_slice = suv[z]

    mask = cc[z] == label

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(
        ct_slice,
        cmap="gray"
    )

    # PET uptake shown transparently.
    pet_display = np.ma.masked_where(
        suv_slice < PET_THRESHOLD,
        suv_slice
    )

    ax.imshow(
        pet_display,
        cmap="hot",
        alpha=0.35
    )

    # Selected component in red.
    component_display = np.ma.masked_where(
        ~mask,
        mask
    )

    ax.imshow(
        component_display,
        cmap="Reds",
        alpha=0.70
    )

    ax.set_title(
        f"{case_name}\n"
        f"Component {label} | Rank {rank} | Slice {z}"
    )

    ax.axis("off")

    case_dir = OUTPUT_DIR / "visualizations" / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        case_dir
        / f"component_{label}_rank_{rank}_z_{z}.png"
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# MAIN AUDIT
# ============================================================

def audit_case(case_name):

    print()
    print("=" * 100)
    print(f"DEV47 AUDIT — {case_name}")
    print("=" * 100)

    case = dev10.load_case(case_name)

    ct = case["ct"]
    suv = case["suv"]
    gt = case["gt"]

    print("CT:", ct.shape)
    print("SUV:", suv.shape)
    print("GT:", gt.shape)

    # --------------------------------------------------------
    # 1. PET-only candidate
    # --------------------------------------------------------

    candidate_volume = (
        suv >= PET_THRESHOLD
    ).astype(np.uint8)

    print(
        "PET candidate voxels:",
        int(candidate_volume.sum())
    )

    # --------------------------------------------------------
    # 2. Exact Dev10 3D component construction
    # --------------------------------------------------------

    components, cc = dev10.build_3d_component_table(
        case,
        candidate_volume
    )

    print(
        "Initial 3D components:",
        len(components)
    )

    # --------------------------------------------------------
    # 3. Our own component features
    # --------------------------------------------------------

    features = calculate_component_features(
        components,
        cc,
        suv
    )

    features = add_pet_quality_score(
        features
    )

    # --------------------------------------------------------
    # 4. GT overlap audit
    # --------------------------------------------------------

    gt_rows = []

    for _, row in features.iterrows():

        label = int(row["label"])
        mask = cc == label

        overlap = int(
            np.logical_and(mask, gt).sum()
        )

        component_voxels = int(mask.sum())
        gt_voxels = int(gt.sum())

        dice = (
            2.0 * overlap
            / max(component_voxels + gt_voxels, 1)
        )

        iou = (
            overlap
            / max(
                component_voxels + gt_voxels - overlap,
                1
            )
        )

        gt_rows.append({
            "label": label,
            "overlap_gt": overlap,
            "dice_gt": dice,
            "iou_gt": iou,
        })

    gt_df = pd.DataFrame(gt_rows)

    features = features.merge(
        gt_df,
        on="label",
        how="left"
    )

    features["gt_overlap"] = (
        features["overlap_gt"] > 0
    )

    # --------------------------------------------------------
    # 5. Frozen Dev39 eligibility
    # --------------------------------------------------------

    features["eligible_dev39"] = (
        features["hot_fraction"]
        >= HOT_FRACTION_GATE
    )

    eligible = features[
        features["eligible_dev39"]
    ].copy()

    eligible = eligible.sort_values(
        ["pet_quality_score", "mean_suv"],
        ascending=False
    ).reset_index(drop=True)

    eligible["dev39_rank"] = (
        np.arange(len(eligible)) + 1
    )

    selected = eligible.head(TOP_K).copy()

    selected["selected_dev39"] = True

    # --------------------------------------------------------
    # 6. Core audit for selected components
    # --------------------------------------------------------

    core_rows = []

    for _, row in selected.iterrows():

        label = int(row["label"])

        core_mask = safe_build_core(
            cc,
            suv,
            label,
            CORE_POLICY
        )

        core_voxels = int(core_mask.sum())

        core_overlap = int(
            np.logical_and(
                core_mask,
                gt
            ).sum()
        )

        core_dice = (
            2.0 * core_overlap
            / max(
                core_voxels + int(gt.sum()),
                1
            )
        )

        core_iou = (
            core_overlap
            / max(
                core_voxels + int(gt.sum())
                - core_overlap,
                1
            )
        )

        core_rows.append({
            "label": label,
            "core_voxels": core_voxels,
            "core_overlap_gt": core_overlap,
            "core_dice_gt": core_dice,
            "core_iou_gt": core_iou,
        })

        save_component_visualization(
            case_name,
            suv,
            ct,
            cc,
            row,
            int(row["dev39_rank"])
        )

    core_df = pd.DataFrame(core_rows)

    selected = selected.merge(
        core_df,
        on="label",
        how="left"
    )

    # --------------------------------------------------------
    # 7. Heuristic physiological-risk indicators
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # These are AUDIT FLAGS only.
    # They do NOT claim anatomical diagnosis.
    #

    features["large_component"] = (
        features["voxels"] >=
        features["voxels"].quantile(0.75)
    )

    features["high_persistence"] = (
        features["persistence"] >= 0.50
    )

    features["high_hot_fraction"] = (
        features["hot_fraction"] >= 0.75
    )

    features["central_location"] = (
        features["normalized_center_distance"] <= 0.35
    )

    features["physiological_risk_flags"] = (
        features["large_component"].astype(int)
        + features["high_persistence"].astype(int)
        + features["high_hot_fraction"].astype(int)
        + features["central_location"].astype(int)
    )

    # Add flags to selected table.
    flag_cols = [
        "label",
        "large_component",
        "high_persistence",
        "high_hot_fraction",
        "central_location",
        "physiological_risk_flags",
    ]

    selected = selected.merge(
        features[flag_cols],
        on="label",
        how="left"
    )

    # --------------------------------------------------------
    # Print selected components
    # --------------------------------------------------------

    print()
    print("SELECTED DEV39 COMPONENTS")
    print("-" * 100)

    display_cols = [
        "dev39_rank",
        "label",
        "voxels",
        "z_start",
        "z_end",
        "mean_suv",
        "max_suv",
        "hot_fraction",
        "persistence",
        "longest_run",
        "compactness",
        "normalized_center_distance",
        "pet_quality_score",
        "overlap_gt",
        "dice_gt",
        "core_dice_gt",
        "physiological_risk_flags",
    ]

    print(
        selected[display_cols].to_string(
            index=False
        )
    )

    return features, selected


# ============================================================
# RUN ALL DEVELOPMENT CASES
# ============================================================

def main():

    print()
    print("=" * 110)
    print("DEV47 — PHYSIOLOGICAL FALSE-POSITIVE AUDIT")
    print("=" * 110)

    print("Frozen PET threshold:", PET_THRESHOLD)
    print("Frozen hot threshold:", HOT_THRESHOLD)
    print("Frozen hot fraction gate:", HOT_FRACTION_GATE)
    print("Frozen Top-K:", TOP_K)
    print("Frozen core policy:", CORE_POLICY)

    all_components = []
    all_selected = []

    for case_name in CASES:

        features, selected = audit_case(
            case_name
        )

        features["case"] = case_name
        selected["case"] = case_name

        all_components.append(features)
        all_selected.append(selected)

    all_components_df = pd.concat(
        all_components,
        ignore_index=True
    )

    all_selected_df = pd.concat(
        all_selected,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    all_components_path = (
        OUTPUT_DIR
        / "dev47_all_component_audit.csv"
    )

    selected_path = (
        OUTPUT_DIR
        / "dev47_selected_component_audit.csv"
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
    # Summary
    # --------------------------------------------------------

    summary = (
        all_selected_df
        .groupby("case")
        .agg(
            selected_components=("label", "count"),
            mean_selected_voxels=("voxels", "mean"),
            max_selected_voxels=("voxels", "max"),
            mean_hot_fraction=("hot_fraction", "mean"),
            mean_persistence=("persistence", "mean"),
            mean_quality=("pet_quality_score", "mean"),
            gt_overlapping_selected=(
                "overlap_gt",
                "sum"
            ),
            mean_core_dice=(
                "core_dice_gt",
                "mean"
            ),
            mean_risk_flags=(
                "physiological_risk_flags",
                "mean"
            ),
        )
        .reset_index()
    )

    summary_path = (
        OUTPUT_DIR
        / "dev47_case_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------------
    # Global summary
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("DEV47 SUMMARY")
    print("=" * 110)

    print(
        summary.to_string(index=False)
    )

    print()
    print("Saved:")
    print(all_components_path)
    print(selected_path)
    print(summary_path)

    print()
    print("=" * 110)
    print("DEV47 COMPLETE")
    print("=" * 110)

if __name__ == "__main__":
    main()