from pathlib import Path
import sys
import importlib.util

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# DEV57 — SPATIAL / ANATOMICAL POSITION AUDIT
#
# PURPOSE:
#   Diagnostic-only audit of spatial position and component
#   geometry for Dev39 PET components on validation case 7.
#
# IMPORTANT:
#   - Does NOT modify Dev10.
#   - Does NOT modify Dev39.
#   - Does NOT change ranking.
#   - Does NOT change Top-K.
#   - Does NOT create a new prediction.
#   - GT is used ONLY after all spatial features are computed.
#
# MAIN QUESTION:
#
#   Can normalized anatomical position and component size
#   distinguish false-positive Component 84 from true-positive
#   Components 310 and 309?
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

AUDIT_LABELS = [
    84,
    310,
    309,
    302,
]


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "results"
    / "validation"
    / "dev57_spatial_anatomical_position_audit"
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
# BODY GEOMETRY
# ============================================================

def compute_body_geometry(body):

    body = body.astype(bool)

    coords = np.argwhere(body)

    if len(coords) == 0:

        return {
            "body_centroid_z": np.nan,
            "body_centroid_y": np.nan,
            "body_centroid_x": np.nan,
            "body_z_min": np.nan,
            "body_z_max": np.nan,
            "body_y_min": np.nan,
            "body_y_max": np.nan,
            "body_x_min": np.nan,
            "body_x_max": np.nan,
            "body_z_span": np.nan,
            "body_y_span": np.nan,
            "body_x_span": np.nan,
            "body_radius_max": np.nan,
        }

    centroid = coords.mean(axis=0)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    z_min, y_min, x_min = mins
    z_max, y_max, x_max = maxs

    distances = np.linalg.norm(
        coords - centroid,
        axis=1
    )

    return {
        "body_centroid_z": float(centroid[0]),
        "body_centroid_y": float(centroid[1]),
        "body_centroid_x": float(centroid[2]),

        "body_z_min": int(z_min),
        "body_z_max": int(z_max),

        "body_y_min": int(y_min),
        "body_y_max": int(y_max),

        "body_x_min": int(x_min),
        "body_x_max": int(x_max),

        "body_z_span": int(
            z_max - z_min + 1
        ),

        "body_y_span": int(
            y_max - y_min + 1
        ),

        "body_x_span": int(
            x_max - x_min + 1
        ),

        "body_radius_max": float(
            np.max(distances)
        ),
    }


# ============================================================
# COMPONENT SPATIAL FEATURES
# ============================================================

def compute_spatial_features(
    cc,
    label,
    body_geometry,
    body
):

    component = (
        cc == int(label)
    )

    coords = np.argwhere(
        component
    )

    if len(coords) == 0:

        return {}

    centroid = coords.mean(axis=0)

    z = centroid[0]
    y = centroid[1]
    x = centroid[2]

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    bbox_z = int(
        z_max - z_min + 1
    )

    bbox_y = int(
        y_max - y_min + 1
    )

    bbox_x = int(
        x_max - x_min + 1
    )

    bbox_volume = (
        bbox_z
        * bbox_y
        * bbox_x
    )

    component_voxels = len(coords)

    if bbox_volume > 0:

        bbox_fill_ratio = (
            component_voxels
            / bbox_volume
        )

    else:

        bbox_fill_ratio = np.nan

    # --------------------------------------------------------
    # Body centroid distance
    # --------------------------------------------------------

    body_centroid = np.array([
        body_geometry[
            "body_centroid_z"
        ],
        body_geometry[
            "body_centroid_y"
        ],
        body_geometry[
            "body_centroid_x"
        ],
    ])

    component_centroid = np.array([
        z,
        y,
        x,
    ])

    centroid_distance = np.linalg.norm(
        component_centroid
        - body_centroid
    )

    body_radius = (
        body_geometry[
            "body_radius_max"
        ]
    )

    if body_radius > 0:

        normalized_radius = (
            centroid_distance
            / body_radius
        )

    else:

        normalized_radius = np.nan

    # --------------------------------------------------------
    # Normalized anatomical coordinates
    # --------------------------------------------------------

    body_z_min = body_geometry[
        "body_z_min"
    ]

    body_y_min = body_geometry[
        "body_y_min"
    ]

    body_x_min = body_geometry[
        "body_x_min"
    ]

    body_z_span = body_geometry[
        "body_z_span"
    ]

    body_y_span = body_geometry[
        "body_y_span"
    ]

    body_x_span = body_geometry[
        "body_x_span"
    ]

    if body_z_span > 0:

        normalized_z = (
            z - body_z_min
        ) / body_z_span

    else:

        normalized_z = np.nan

    if body_y_span > 0:

        normalized_y = (
            y - body_y_min
        ) / body_y_span

    else:

        normalized_y = np.nan

    if body_x_span > 0:

        normalized_x = (
            x - body_x_min
        ) / body_x_span

    else:

        normalized_x = np.nan

    # --------------------------------------------------------
    # Distance to body boundaries
    # --------------------------------------------------------

    distance_z_low = (
        z - body_z_min
    )

    distance_z_high = (
        body_geometry[
            "body_z_max"
        ] - z
    )

    distance_y_low = (
        y - body_y_min
    )

    distance_y_high = (
        body_geometry[
            "body_y_max"
        ] - y
    )

    distance_x_low = (
        x - body_x_min
    )

    distance_x_high = (
        body_geometry[
            "body_x_max"
        ] - x
    )

    nearest_body_box_boundary = min(
        distance_z_low,
        distance_z_high,
        distance_y_low,
        distance_y_high,
        distance_x_low,
        distance_x_high,
    )

    # --------------------------------------------------------
    # Component volume
    # --------------------------------------------------------

    body_voxels = int(
        np.count_nonzero(body)
    )

    if body_voxels > 0:

        body_volume_fraction = (
            component_voxels
            / body_voxels
        )

    else:

        body_volume_fraction = np.nan

    # --------------------------------------------------------
    # Aspect ratios
    # --------------------------------------------------------

    sorted_dims = sorted(
        [
            bbox_z,
            bbox_y,
            bbox_x,
        ],
        reverse=True
    )

    longest_dim = sorted_dims[0]
    middle_dim = sorted_dims[1]
    shortest_dim = sorted_dims[2]

    if middle_dim > 0:

        longest_middle_ratio = (
            longest_dim
            / middle_dim
        )

    else:

        longest_middle_ratio = np.nan

    if shortest_dim > 0:

        longest_shortest_ratio = (
            longest_dim
            / shortest_dim
        )

    else:

        longest_shortest_ratio = np.nan

    return {

        # Component centroid
        "centroid_z": float(z),
        "centroid_y": float(y),
        "centroid_x": float(x),

        # Normalized body coordinates
        "normalized_z": safe_float(
            normalized_z
        ),
        "normalized_y": safe_float(
            normalized_y
        ),
        "normalized_x": safe_float(
            normalized_x
        ),

        # Body-relative radial position
        "body_centroid_distance": float(
            centroid_distance
        ),

        "normalized_radial_distance": safe_float(
            normalized_radius
        ),

        # Distance to body bounding box
        "nearest_body_box_boundary": float(
            nearest_body_box_boundary
        ),

        # Component bounds
        "component_z_min": int(z_min),
        "component_z_max": int(z_max),

        "component_y_min": int(y_min),
        "component_y_max": int(y_max),

        "component_x_min": int(x_min),
        "component_x_max": int(x_max),

        # Component dimensions
        "bbox_z": bbox_z,
        "bbox_y": bbox_y,
        "bbox_x": bbox_x,

        "bbox_volume": int(
            bbox_volume
        ),

        "bbox_fill_ratio": safe_float(
            bbox_fill_ratio
        ),

        # Size
        "component_voxels": int(
            component_voxels
        ),

        "body_voxels": int(
            body_voxels
        ),

        "component_body_fraction": safe_float(
            body_volume_fraction
        ),

        # Shape
        "longest_middle_ratio": safe_float(
            longest_middle_ratio
        ),

        "longest_shortest_ratio": safe_float(
            longest_shortest_ratio
        ),
    }


# ============================================================
# GT POST-HOC FEATURES
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

    component_voxels = int(
        np.count_nonzero(component)
    )

    gt_voxels = int(
        np.count_nonzero(gt)
    )

    overlap = int(
        np.count_nonzero(
            component & gt
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

        iou = (
            overlap
            / union
        )

    else:

        iou = 0.0

    if gt_voxels > 0:

        gt_recall = (
            overlap
            / gt_voxels
        )

    else:

        gt_recall = 0.0

    if component_voxels > 0:

        component_precision = (
            overlap
            / component_voxels
        )

    else:

        component_precision = 0.0

    return {

        "gt_overlap": overlap,

        "gt_dice": float(
            dice
        ),

        "gt_iou": float(
            iou
        ),

        "gt_recall": float(
            gt_recall
        ),

        "component_precision": float(
            component_precision
        ),
    }


# ============================================================
# VISUALIZATION
# ============================================================

def save_spatial_visualization(
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

    centroid = coords.mean(
        axis=0
    )

    z = int(
        round(
            centroid[0]
        )
    )

    z = max(
        0,
        min(
            z,
            ct.shape[0] - 1
        )
    )

    ct_slice = ct[z]
    suv_slice = suv[z]

    component_slice = (
        component[z]
    )

    gt_slice = (
        gt[z].astype(bool)
    )

    # --------------------------------------------------------
    # CT figure
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        ct_slice,
        cmap="gray"
    )

    plt.contour(
        component_slice,
        levels=[0.5],
        linewidths=1.5
    )

    if np.any(gt_slice):

        plt.contour(
            gt_slice,
            levels=[0.5],
            linewidths=1.5
        )

    plt.title(
        f"DEV57 Component {label}\n"
        f"Rank {int(row['rank'])} | z={z}\n"
        f"Radial={row['normalized_radial_distance']:.3f} | "
        f"Body fraction={row['component_body_fraction']:.5f}"
    )

    plt.axis("off")

    ct_path = (
        OUTPUT_DIR
        / f"component_{label}_spatial_ct.png"
    )

    plt.savefig(
        ct_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # PET figure
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        suv_slice,
        cmap="gray"
    )

    plt.contour(
        component_slice,
        levels=[0.5],
        linewidths=1.5
    )

    if np.any(gt_slice):

        plt.contour(
            gt_slice,
            levels=[0.5],
            linewidths=1.5
        )

    plt.title(
        f"DEV57 Component {label} — PET\n"
        f"Mean SUV={row['mean_suv']:.2f} | "
        f"Hot fraction={row['hot_fraction']:.3f}"
    )

    plt.axis("off")

    pet_path = (
        OUTPUT_DIR
        / f"component_{label}_spatial_pet.png"
    )

    plt.savefig(
        pet_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved component {label} visualizations"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 90)
    print("DEV57 — SPATIAL / ANATOMICAL POSITION AUDIT")
    print("=" * 90)

    print()
    print("Case                 :", CASE_NAME)
    print("PET threshold        :", PET_THRESHOLD)
    print("Hot threshold        :", HOT_THRESHOLD)
    print("3D minimum voxels    :", MIN_3D_VOXELS)
    print("3D minimum slices    :", MIN_3D_SLICES)
    print("Hot fraction gate    :", HOT_FRACTION_GATE)
    print("Top-K                 :", TOP_K)
    print("Audit labels         :", AUDIT_LABELS)

    # --------------------------------------------------------
    # LOAD CASE
    # --------------------------------------------------------

    print()
    print("[1] Loading validation case...")

    loaded = prepare_case7()

    validation_case = loaded[0]

    ct = loaded[2].astype(
        np.float32
    )

    body = loaded[3].astype(
        bool
    )

    suv = validation_case[
        "suv"
    ].astype(
        np.float32
    )

    gt = validation_case[
        "gt"
    ].astype(
        bool
    )

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

    # --------------------------------------------------------
    # BODY GEOMETRY
    # --------------------------------------------------------

    print()
    print(
        "[2] Computing body geometry..."
    )

    body_geometry = (
        compute_body_geometry(
            body
        )
    )

    print(
        "Body centroid        :",
        (
            body_geometry[
                "body_centroid_z"
            ],
            body_geometry[
                "body_centroid_y"
            ],
            body_geometry[
                "body_centroid_x"
            ],
        )
    )

    print(
        "Body spans           :",
        (
            body_geometry[
                "body_z_span"
            ],
            body_geometry[
                "body_y_span"
            ],
            body_geometry[
                "body_x_span"
            ],
        )
    )

    print(
        "Body radius          :",
        body_geometry[
            "body_radius_max"
        ]
    )

    # --------------------------------------------------------
    # EXACT DEV39 CANDIDATE
    # --------------------------------------------------------

    print()
    print(
        "[3] Generating exact Dev39 PET-only candidate..."
    )

    candidate = (
        suv >= PET_THRESHOLD
    ).astype(
        np.uint8
    )

    print(
        "Candidate voxels     :",
        int(
            np.count_nonzero(
                candidate
            )
        )
    )

    # --------------------------------------------------------
    # EXACT DEV10 COMPONENTS
    # --------------------------------------------------------

    print()
    print(
        "[4] Building exact Dev10 3D component table..."
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
        "[5] Computing exact Dev39 PET features..."
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
    # EXACT DEV39 GATE / RANK
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
    # SPATIAL AUDIT
    # --------------------------------------------------------

    print()
    print(
        "[6] Computing spatial/anatomical features..."
    )

    rows = []

    for _, component_row in eligible.iterrows():

        label = int(
            component_row["label"]
        )

        row = {

            "case": CASE_NAME,

            "rank": int(
                component_row[
                    "rank"
                ]
            ),

            "label": label,

            "selected": bool(
                component_row[
                    "selected"
                ]
            ),

            # ----------------------------------------------
            # Dev39 PET features
            # ----------------------------------------------

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
        }

        # ----------------------------------------------
        # Spatial features
        # ----------------------------------------------

        row.update(
            compute_spatial_features(
                cc,
                label,
                body_geometry,
                body
            )
        )

        # ----------------------------------------------
        # GT post-hoc only
        # ----------------------------------------------

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
    # SAVE FULL CSV
    # --------------------------------------------------------

    full_path = (
        OUTPUT_DIR
        / "dev57_spatial_anatomical_position_audit.csv"
    )

    audit_df.to_csv(
        full_path,
        index=False
    )

    print()
    print(
        "Saved:",
        full_path
    )

    # --------------------------------------------------------
    # FOCUSED CSV
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

        "normalized_z",
        "normalized_y",
        "normalized_x",

        "body_centroid_distance",
        "normalized_radial_distance",
        "nearest_body_box_boundary",

        "bbox_z",
        "bbox_y",
        "bbox_x",
        "bbox_volume",
        "bbox_fill_ratio",

        "component_body_fraction",

        "longest_middle_ratio",
        "longest_shortest_ratio",

        "mean_suv",
        "max_suv",
        "hot_fraction",
        "persistence",
        "compactness",
        "pet_quality",

        "gt_overlap",
        "gt_dice",
        "gt_iou",
        "gt_recall",
        "component_precision",
    ]

    focused_df = audit_df[
        focused_columns
    ]

    focused_path = (
        OUTPUT_DIR
        / "dev57_focused_spatial_table.csv"
    )

    focused_df.to_csv(
        focused_path,
        index=False
    )

    # --------------------------------------------------------
    # PRINT ALL COMPONENTS
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("DEV57 — ALL ELIGIBLE COMPONENTS")
    print("=" * 90)

    display_columns = [

        "rank",
        "label",
        "selected",

        "voxels",

        "centroid_z",
        "centroid_y",
        "centroid_x",

        "normalized_z",
        "normalized_y",
        "normalized_x",

        "normalized_radial_distance",
        "nearest_body_box_boundary",

        "bbox_z",
        "bbox_y",
        "bbox_x",

        "component_body_fraction",
        "bbox_fill_ratio",

        "mean_suv",
        "hot_fraction",
        "pet_quality",

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
    # TARGET COMPONENTS
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("DEV57 — FOUR TARGET COMPONENTS")
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
        "[7] Creating spatial visualizations..."
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

        save_spatial_visualization(
            ct,
            suv,
            gt,
            cc,
            label,
            row
        )

    # --------------------------------------------------------
    # SCATTER 1:
    # COMPONENT SIZE VS RADIAL POSITION
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        audit_df[
            "normalized_radial_distance"
        ],
        audit_df[
            "voxels"
        ]
    )

    for _, row in audit_df.iterrows():

        label = int(
            row["label"]
        )

        if label in AUDIT_LABELS:

            plt.annotate(
                str(label),
                (
                    row[
                        "normalized_radial_distance"
                    ],
                    row["voxels"]
                )
            )

    plt.yscale(
        "log"
    )

    plt.xlabel(
        "Normalized radial distance from body centroid"
    )

    plt.ylabel(
        "Component voxels"
    )

    plt.title(
        "DEV57 — Component Size vs Anatomical Position"
    )

    plt.grid(
        alpha=0.25
    )

    size_position_path = (
        OUTPUT_DIR
        / "dev57_size_vs_radial_position.png"
    )

    plt.savefig(
        size_position_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "Saved:",
        size_position_path
    )

    # --------------------------------------------------------
    # SCATTER 2:
    # NORMALIZED Z VS COMPONENT SIZE
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        audit_df[
            "normalized_z"
        ],
        audit_df[
            "voxels"
        ]
    )

    for _, row in audit_df.iterrows():

        label = int(
            row["label"]
        )

        if label in AUDIT_LABELS:

            plt.annotate(
                str(label),
                (
                    row[
                        "normalized_z"
                    ],
                    row["voxels"]
                )
            )

    plt.yscale(
        "log"
    )

    plt.xlabel(
        "Normalized z position"
    )

    plt.ylabel(
        "Component voxels"
    )

    plt.title(
        "DEV57 — Component Size vs Normalized Z Position"
    )

    plt.grid(
        alpha=0.25
    )

    z_position_path = (
        OUTPUT_DIR
        / "dev57_size_vs_normalized_z.png"
    )

    plt.savefig(
        z_position_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "Saved:",
        z_position_path
    )

    # --------------------------------------------------------
    # SCATTER 3:
    # PET QUALITY VS RADIAL POSITION
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        audit_df[
            "normalized_radial_distance"
        ],
        audit_df[
            "pet_quality"
        ]
    )

    for _, row in audit_df.iterrows():

        label = int(
            row["label"]
        )

        if label in AUDIT_LABELS:

            plt.annotate(
                str(label),
                (
                    row[
                        "normalized_radial_distance"
                    ],
                    row[
                        "pet_quality"
                    ]
                )
            )

    plt.xlabel(
        "Normalized radial distance from body centroid"
    )

    plt.ylabel(
        "Dev39 PET quality"
    )

    plt.title(
        "DEV57 — PET Quality vs Anatomical Position"
    )

    plt.grid(
        alpha=0.25
    )

    quality_position_path = (
        OUTPUT_DIR
        / "dev57_pet_quality_vs_radial_position.png"
    )

    plt.savefig(
        quality_position_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "Saved:",
        quality_position_path
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("DEV57 COMPLETE")
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
        " - dev57_spatial_anatomical_position_audit.csv"
    )

    print(
        " - dev57_focused_spatial_table.csv"
    )

    print(
        " - dev57_size_vs_radial_position.png"
    )

    print(
        " - dev57_size_vs_normalized_z.png"
    )

    print(
        " - dev57_pet_quality_vs_radial_position.png"
    )

    for label in AUDIT_LABELS:

        print(
            f" - component_{label}_spatial_ct.png"
        )

        print(
            f" - component_{label}_spatial_pet.png"
        )


if __name__ == "__main__":
    main()