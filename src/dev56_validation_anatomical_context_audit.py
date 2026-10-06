from pathlib import Path
import sys
import importlib.util

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# DEV56 — VALIDATION ANATOMICAL CONTEXT AUDIT
#
# PURPOSE:
#   Diagnostic-only audit of anatomical / CT context for
#   Dev39 PET components on validation case 7.
#
# IMPORTANT:
#   - Does NOT modify Dev10.
#   - Does NOT modify Dev39.
#   - Does NOT change ranking.
#   - Does NOT change Top-K.
#   - Does NOT use GT for component selection.
#   - GT is used only AFTER feature extraction for diagnosis.
#
# MAIN QUESTION:
#   Can CT/anatomical context distinguish the large false
#   positive component 84 from true-positive components
#   310 and 309?
# ============================================================


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ============================================================
# CONFIGURATION
# ============================================================

CASE_NAME = "PETCT_0011f3deaf"

PET_THRESHOLD = 2.25
HOT_THRESHOLD = 3.0

MIN_3D_VOXELS = 75
MIN_3D_SLICES = 2

HOT_FRACTION_GATE = 0.40

TOP_K = 4
CORE_POLICY = "P70"

AUDIT_LABELS = [
    84,
    310,
    309,
    302,
]


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "results"
    / "validation"
    / "dev56_anatomical_context_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# IMPORT DEV10
# ============================================================

DEV10_PATH = SRC / "dev10_lite_hot_core.py"

spec10 = importlib.util.spec_from_file_location(
    "dev10_lite_hot_core",
    DEV10_PATH
)

dev10 = importlib.util.module_from_spec(spec10)
spec10.loader.exec_module(dev10)


# ============================================================
# IMPORT DEV39
# ============================================================

DEV39_PATH = SRC / "dev39_pet_quality_fpr_control.py"

spec39 = importlib.util.spec_from_file_location(
    "dev39_pet_quality_fpr_control",
    DEV39_PATH
)

dev39 = importlib.util.module_from_spec(spec39)
spec39.loader.exec_module(dev39)


# ============================================================
# IMPORT VALIDATION CASE
# ============================================================

V4_PATH = SRC / "v4_validate_case7.py"

spec4 = importlib.util.spec_from_file_location(
    "v4_validate_case7",
    V4_PATH
)

v4 = importlib.util.module_from_spec(spec4)
spec4.loader.exec_module(v4)

prepare_case7 = v4.prepare_case7


# ============================================================
# UTILITY
# ============================================================

def safe_float(value):
    if value is None:
        return np.nan

    try:
        value = float(value)

        if not np.isfinite(value):
            return np.nan

        return value

    except Exception:
        return np.nan


# ============================================================
# BODY CENTROID
# ============================================================

def compute_body_centroid(body):

    coords = np.argwhere(
        body.astype(bool)
    )

    if len(coords) == 0:
        return np.array([
            np.nan,
            np.nan,
            np.nan
        ])

    return coords.mean(axis=0)


# ============================================================
# COMPONENT GEOMETRY
# ============================================================

def component_geometry(
    cc,
    label,
    body_centroid
):

    coords = np.argwhere(
        cc == int(label)
    )

    if len(coords) == 0:
        return {
            "centroid_z": np.nan,
            "centroid_y": np.nan,
            "centroid_x": np.nan,
            "bbox_z": 0,
            "bbox_y": 0,
            "bbox_x": 0,
            "body_centroid_distance": np.nan,
            "body_radius_fraction": np.nan,
        }

    centroid = coords.mean(axis=0)

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    bbox_z = z_max - z_min + 1
    bbox_y = y_max - y_min + 1
    bbox_x = x_max - x_min + 1

    distance = np.linalg.norm(
        centroid - body_centroid
    )

    # Approximate normalized radial distance.
    body_coords = np.argwhere(
        cc > 0
    )

    if len(body_coords) > 0:

        body_distances = np.linalg.norm(
            body_coords - body_centroid,
            axis=1
        )

        max_body_radius = np.max(
            body_distances
        )

        if max_body_radius > 0:
            radius_fraction = (
                distance / max_body_radius
            )
        else:
            radius_fraction = np.nan

    else:
        radius_fraction = np.nan

    return {
        "centroid_z": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_x": float(centroid[2]),
        "bbox_z": int(bbox_z),
        "bbox_y": int(bbox_y),
        "bbox_x": int(bbox_x),
        "body_centroid_distance": float(distance),
        "body_radius_fraction": safe_float(
            radius_fraction
        ),
    }


# ============================================================
# CT STATISTICS
# ============================================================

def compute_ct_features(
    ct,
    component_mask
):

    values = ct[
        component_mask
    ].astype(np.float32)

    if len(values) == 0:
        return {
            "ct_mean": np.nan,
            "ct_median": np.nan,
            "ct_std": np.nan,
            "ct_p10": np.nan,
            "ct_p25": np.nan,
            "ct_p75": np.nan,
            "ct_p90": np.nan,
            "ct_min": np.nan,
            "ct_max": np.nan,
            "ct_frac_below_m500": np.nan,
            "ct_frac_m500_to_100": np.nan,
            "ct_frac_above_100": np.nan,
        }

    return {
        "ct_mean": float(np.mean(values)),
        "ct_median": float(np.median(values)),
        "ct_std": float(np.std(values)),
        "ct_p10": float(np.percentile(values, 10)),
        "ct_p25": float(np.percentile(values, 25)),
        "ct_p75": float(np.percentile(values, 75)),
        "ct_p90": float(np.percentile(values, 90)),
        "ct_min": float(np.min(values)),
        "ct_max": float(np.max(values)),

        "ct_frac_below_m500": float(
            np.mean(values < -500)
        ),

        "ct_frac_m500_to_100": float(
            np.mean(
                (values >= -500)
                & (values <= 100)
            )
        ),

        "ct_frac_above_100": float(
            np.mean(values > 100)
        ),
    }


# ============================================================
# CT GRADIENT FEATURES
# ============================================================

def compute_ct_gradient_features(
    ct,
    component_mask
):

    coords = np.argwhere(
        component_mask
    )

    if len(coords) == 0:
        return {
            "ct_gradient_mean": np.nan,
            "ct_gradient_p90": np.nan,
        }

    # Compute gradients slice-wise.
    gz, gy, gx = np.gradient(
        ct.astype(np.float32)
    )

    gradient_mag = np.sqrt(
        gz ** 2
        + gy ** 2
        + gx ** 2
    )

    values = gradient_mag[
        component_mask
    ]

    return {
        "ct_gradient_mean": float(
            np.mean(values)
        ),
        "ct_gradient_p90": float(
            np.percentile(values, 90)
        ),
    }


# ============================================================
# BODY / COMPONENT VOLUME
# ============================================================

def compute_volume_features(
    cc,
    label,
    body
):

    component_mask = (
        cc == int(label)
    )

    voxels = int(
        np.count_nonzero(
            component_mask
        )
    )

    body_voxels = int(
        np.count_nonzero(
            body
        )
    )

    if body_voxels > 0:
        body_fraction = (
            voxels / body_voxels
        )
    else:
        body_fraction = np.nan

    return {
        "component_voxels": voxels,
        "body_voxels": body_voxels,
        "component_body_fraction": safe_float(
            body_fraction
        ),
    }


# ============================================================
# GT POST-HOC AUDIT
# ============================================================

def compute_gt_features(
    cc,
    label,
    gt
):

    component = (
        cc == int(label)
    )

    gt = gt.astype(bool)

    overlap = int(
        np.count_nonzero(
            component & gt
        )
    )

    component_voxels = int(
        np.count_nonzero(
            component
        )
    )

    gt_voxels = int(
        np.count_nonzero(
            gt
        )
    )

    union = int(
        np.count_nonzero(
            component | gt
        )
    )

    if (
        component_voxels
        + gt_voxels
    ) > 0:

        dice = (
            2.0 * overlap
            / (
                component_voxels
                + gt_voxels
            )
        )

    else:
        dice = 0.0

    if union > 0:
        iou = overlap / union
    else:
        iou = 0.0

    return {
        "gt_overlap": overlap,
        "gt_dice": float(dice),
        "gt_iou": float(iou),
    }


# ============================================================
# VISUALIZATION
# ============================================================

def save_component_visualization(
    ct,
    suv,
    gt,
    cc,
    label,
    row
):

    component = (
        cc == int(label)
    )

    coords = np.argwhere(
        component
    )

    if len(coords) == 0:
        return

    centroid = coords.mean(axis=0)

    z = int(round(
        centroid[0]
    ))

    z = max(
        0,
        min(
            z,
            ct.shape[0] - 1
        )
    )

    ct_slice = ct[z]
    suv_slice = suv[z]

    component_slice = component[z]
    gt_slice = gt[z].astype(bool)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    # --------------------------------------------------------
    # CT
    # --------------------------------------------------------

    axes[0].imshow(
        ct_slice,
        cmap="gray"
    )

    axes[0].contour(
        component_slice,
        levels=[0.5],
        linewidths=1.5
    )

    axes[0].set_title(
        f"CT | Component {label}\n"
        f"z={z}"
    )

    axes[0].axis("off")

    # --------------------------------------------------------
    # PET
    # --------------------------------------------------------

    axes[1].imshow(
        suv_slice,
        cmap="gray"
    )

    axes[1].contour(
        component_slice,
        levels=[0.5],
        linewidths=1.5
    )

    axes[1].set_title(
        f"PET | Rank {int(row['rank'])}\n"
        f"SUV mean={row['mean_suv']:.2f}"
    )

    axes[1].axis("off")

    # --------------------------------------------------------
    # CT + GT
    # --------------------------------------------------------

    axes[2].imshow(
        ct_slice,
        cmap="gray"
    )

    axes[2].contour(
        component_slice,
        levels=[0.5],
        linewidths=1.5
    )

    if np.any(gt_slice):

        axes[2].contour(
            gt_slice,
            levels=[0.5],
            linewidths=1.5
        )

    axes[2].set_title(
        f"CT + Component + GT\n"
        f"GT Dice={row['gt_dice']:.3f}"
    )

    axes[2].axis("off")

    fig.suptitle(
        f"DEV56 — Component {label} | "
        f"Rank {int(row['rank'])} | "
        f"{'AUDIT TARGET' if label in AUDIT_LABELS else 'OTHER'}"
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"component_{label}_anatomical_context.png"
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved visualization: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 90)
    print("DEV56 — VALIDATION ANATOMICAL CONTEXT AUDIT")
    print("=" * 90)

    print()
    print("Case                 :", CASE_NAME)
    print("PET threshold        :", PET_THRESHOLD)
    print("Hot threshold        :", HOT_THRESHOLD)
    print("3D minimum voxels    :", MIN_3D_VOXELS)
    print("3D minimum slices    :", MIN_3D_SLICES)
    print("Hot fraction gate    :", HOT_FRACTION_GATE)
    print("Audit labels         :", AUDIT_LABELS)

    # --------------------------------------------------------
    # LOAD VALIDATION CASE
    # --------------------------------------------------------

    print()
    print("[1] Loading validation case...")

    loaded = prepare_case7()

    validation_case = loaded[0]

    ct_img = loaded[1]

    ct = loaded[2].astype(
        np.float32
    )

    gt = validation_case[
        "gt"
    ].astype(bool)

    suv = validation_case[
        "suv"
    ].astype(np.float32)

    body = loaded[3].astype(bool)

    print(
        "CT shape             :",
        ct.shape
    )

    print(
        "SUV shape            :",
        suv.shape
    )

    print(
        "GT voxels            :",
        int(np.count_nonzero(gt))
    )

    print(
        "Body voxels          :",
        int(np.count_nonzero(body))
    )

    print(
        "CT range             :",
        float(np.min(ct)),
        "to",
        float(np.max(ct))
    )

    # --------------------------------------------------------
    # BUILD EXACT DEV39 PET-ONLY CANDIDATE
    # --------------------------------------------------------

    print()
    print(
        "[2] Generating exact Dev39 PET-only candidate..."
    )

    candidate = (
        suv >= PET_THRESHOLD
    ).astype(np.uint8)

    print(
        "Candidate voxels     :",
        int(np.count_nonzero(candidate))
    )

    # --------------------------------------------------------
    # EXACT DEV10 3D COMPONENT TABLE
    # --------------------------------------------------------

    print()
    print(
        "[3] Building exact Dev10 3D component table..."
    )

    components, cc = (
        dev10.build_3d_component_table(
            validation_case,
            candidate
        )
    )

    print(
        "Total 3D components  :",
        len(components)
    )

    # --------------------------------------------------------
    # EXACT DEV39 FEATURES
    # --------------------------------------------------------

    print()
    print(
        "[4] Computing exact Dev39 PET features..."
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

    # --------------------------------------------------------
    # EXACT DEV39 GATE
    # --------------------------------------------------------

    eligible = components[
        components[
            "dev39_hot_fraction"
        ] >= HOT_FRACTION_GATE
    ].copy()

    eligible = eligible.sort_values(
        [
            "dev39_pet_quality",
            "dev39_mean_suv",
        ],
        ascending=False
    ).reset_index(
        drop=True
    )

    eligible["rank"] = (
        np.arange(
            len(eligible)
        ) + 1
    )

    eligible["selected"] = (
        eligible["rank"] <= TOP_K
    )

    print(
        "Eligible components  :",
        len(eligible)
    )

    # --------------------------------------------------------
    # BODY CENTROID
    # --------------------------------------------------------

    body_centroid = (
        compute_body_centroid(
            body
        )
    )

    print(
        "Body centroid        :",
        body_centroid
    )

    # --------------------------------------------------------
    # BUILD AUDIT TABLE
    # --------------------------------------------------------

    print()
    print(
        "[5] Computing anatomical / CT context..."
    )

    rows = []

    for _, component_row in eligible.iterrows():

        label = int(
            component_row["label"]
        )

        component_mask = (
            cc == label
        )

        row = {
            "case": CASE_NAME,
            "rank": int(
                component_row["rank"]
            ),
            "label": label,
            "selected": bool(
                component_row["selected"]
            ),
        }

        # ----------------------------------------------------
        # Existing Dev39 PET features
        # ----------------------------------------------------

        row.update({
            "voxels": int(
                component_row[
                    "dev39_voxels"
                ]
            ),

            "z_start": int(
                component_row[
                    "z_start"
                ]
            ),

            "z_end": int(
                component_row[
                    "z_end"
                ]
            ),

            "z_span": int(
                component_row[
                    "dev39_z_span"
                ]
            ),

            "slice_count": int(
                component_row[
                    "dev39_slice_count"
                ]
            ),

            "longest_run": int(
                component_row[
                    "dev39_longest_run"
                ]
            ),

            "persistence": float(
                component_row[
                    "dev39_persistence"
                ]
            ),

            "mean_suv": float(
                component_row[
                    "dev39_mean_suv"
                ]
            ),

            "max_suv": float(
                component_row[
                    "dev39_max_suv"
                ]
            ),

            "hot_fraction": float(
                component_row[
                    "dev39_hot_fraction"
                ]
            ),

            "compactness": float(
                component_row[
                    "dev39_compactness"
                ]
            ),

            "pet_quality": float(
                component_row[
                    "dev39_pet_quality"
                ]
            ),
        })

        # ----------------------------------------------------
        # Geometry
        # ----------------------------------------------------

        row.update(
            component_geometry(
                cc,
                label,
                body_centroid
            )
        )

        # ----------------------------------------------------
        # Volume
        # ----------------------------------------------------

        row.update(
            compute_volume_features(
                cc,
                label,
                body
            )
        )

        # ----------------------------------------------------
        # CT statistics
        # ----------------------------------------------------

        row.update(
            compute_ct_features(
                ct,
                component_mask
            )
        )

        # ----------------------------------------------------
        # CT gradients
        # ----------------------------------------------------

        row.update(
            compute_ct_gradient_features(
                ct,
                component_mask
            )
        )

        # ----------------------------------------------------
        # GT — POST-HOC ONLY
        # ----------------------------------------------------

        row.update(
            compute_gt_features(
                cc,
                label,
                gt
            )
        )

        rows.append(row)

    audit_df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        / "dev56_anatomical_context_audit.csv"
    )

    audit_df.to_csv(
        csv_path,
        index=False
    )

    print(
        "Saved:",
        csv_path
    )

    # --------------------------------------------------------
    # SAVE FOCUSED TABLE
    # --------------------------------------------------------

    focused_columns = [
        "rank",
        "label",
        "selected",
        "voxels",
        "z_start",
        "z_end",
        "z_span",

        "centroid_z",
        "centroid_y",
        "centroid_x",

        "bbox_z",
        "bbox_y",
        "bbox_x",

        "component_body_fraction",
        "body_centroid_distance",
        "body_radius_fraction",

        "mean_suv",
        "max_suv",
        "hot_fraction",
        "persistence",
        "compactness",
        "pet_quality",

        "ct_mean",
        "ct_median",
        "ct_std",
        "ct_p10",
        "ct_p25",
        "ct_p75",
        "ct_p90",
        "ct_min",
        "ct_max",

        "ct_frac_below_m500",
        "ct_frac_m500_to_100",
        "ct_frac_above_100",

        "ct_gradient_mean",
        "ct_gradient_p90",

        "gt_overlap",
        "gt_dice",
        "gt_iou",
    ]

    focused_df = audit_df[
        focused_columns
    ]

    focused_path = (
        OUTPUT_DIR
        / "dev56_focused_component_table.csv"
    )

    focused_df.to_csv(
        focused_path,
        index=False
    )

    # --------------------------------------------------------
    # PRINT TABLE
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("DEV56 — COMPONENT ANATOMICAL CONTEXT SUMMARY")
    print("=" * 90)

    display_columns = [
        "rank",
        "label",
        "selected",
        "voxels",
        "centroid_z",
        "centroid_y",
        "centroid_x",
        "bbox_z",
        "bbox_y",
        "bbox_x",
        "body_centroid_distance",
        "body_radius_fraction",
        "mean_suv",
        "hot_fraction",
        "pet_quality",
        "ct_mean",
        "ct_median",
        "ct_std",
        "ct_frac_below_m500",
        "ct_frac_m500_to_100",
        "ct_frac_above_100",
        "gt_overlap",
        "gt_dice",
    ]

    print(
        audit_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # AUDIT TARGETS
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("DEV56 — FOUR TARGET COMPONENTS")
    print("=" * 90)

    targets = audit_df[
        audit_df["label"].isin(
            AUDIT_LABELS
        )
    ].copy()

    targets = targets.sort_values(
        "rank"
    )

    print(
        targets[
            display_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # VISUALIZATIONS
    # --------------------------------------------------------

    print()
    print(
        "[6] Creating CT anatomical-context visualizations..."
    )

    for label in AUDIT_LABELS:

        matching = audit_df[
            audit_df["label"] == label
        ]

        if len(matching) == 0:

            print(
                f"Component {label} not found."
            )

            continue

        row = matching.iloc[0]

        save_component_visualization(
            ct,
            suv,
            gt,
            cc,
            label,
            row
        )

    # --------------------------------------------------------
    # SIMPLE DIAGNOSTIC SCATTER
    # --------------------------------------------------------

    print()
    print(
        "[7] Creating diagnostic feature plots..."
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    # Volume vs mean SUV

    axes[0].scatter(
        audit_df["voxels"],
        audit_df["mean_suv"]
    )

    for _, row in audit_df.iterrows():

        if int(row["label"]) in AUDIT_LABELS:

            axes[0].annotate(
                str(int(row["label"])),
                (
                    row["voxels"],
                    row["mean_suv"]
                )
            )

    axes[0].set_xscale(
        "log"
    )

    axes[0].set_xlabel(
        "Component voxels"
    )

    axes[0].set_ylabel(
        "Mean SUV"
    )

    axes[0].set_title(
        "Component volume vs PET intensity"
    )

    # CT mean vs CT soft-tissue fraction

    axes[1].scatter(
        audit_df["ct_mean"],
        audit_df[
            "ct_frac_m500_to_100"
        ]
    )

    for _, row in audit_df.iterrows():

        if int(row["label"]) in AUDIT_LABELS:

            axes[1].annotate(
                str(int(row["label"])),
                (
                    row["ct_mean"],
                    row["ct_frac_m500_to_100"]
                )
            )

    axes[1].set_xlabel(
        "Mean CT intensity"
    )

    axes[1].set_ylabel(
        "Fraction in -500 to 100"
    )

    axes[1].set_title(
        "CT intensity context"
    )

    fig.suptitle(
        "DEV56 — Anatomical Context Diagnostic Plots"
    )

    fig.tight_layout()

    scatter_path = (
        OUTPUT_DIR
        / "dev56_anatomical_feature_plots.png"
    )

    fig.savefig(
        scatter_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "Saved:",
        scatter_path
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("DEV56 COMPLETE")
    print("=" * 90)

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Files:"
    )

    print(
        " - dev56_anatomical_context_audit.csv"
    )

    print(
        " - dev56_focused_component_table.csv"
    )

    print(
        " - dev56_anatomical_feature_plots.png"
    )

    for label in AUDIT_LABELS:

        print(
            f" - component_{label}_anatomical_context.png"
        )


if __name__ == "__main__":
    main()