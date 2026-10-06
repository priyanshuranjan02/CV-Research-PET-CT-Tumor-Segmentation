from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import SimpleITK as sitk


# =========================================================
# LOAD ALL DEFINITIONS FROM EXISTING DUAL-CORE EXPERIMENT
# WITHOUT RUNNING ITS MAIN SECTION
# =========================================================

source = Path(
    "src/v4_neighbor_dual_core_experiment.py"
).read_text()

prefix = source.split(
    "# MAIN",
    1
)[0]

exec(
    compile(
        prefix,
        "v4_neighbor_dual_core_experiment.py",
        "exec"
    ),
    globals()
)


# =========================================================
# OVERSIZED-CONTOUR RESCUE SETTINGS
# =========================================================

RESCUE_THRESHOLDS = [
    1.25,
    1.50,
]

RESCUE_MIN_COMPONENT_AREA = 20
RESCUE_MAX_NEIGHBOR_DISTANCE = 30


# =========================================================
# LOAD CT / BODY FOR RESCUE
# =========================================================

def load_ct_and_body(case_dir):

    ct_dir = find_series(
        case_dir,
        "GK p.v.3"
    )

    ct_img = read_series(
        ct_dir
    )

    ct = sitk.GetArrayFromImage(
        ct_img
    ).astype(np.float32)

    body = build_body_mask(
        ct
    )

    return ct, body


# =========================================================
# BUILD OVERSIZED PET RESCUE
# =========================================================

def build_oversized_rescue(
    prepared,
    ct,
    body,
    threshold
):

    suv = prepared["suv"]

    candidate_masks = prepared[
        "candidate_masks"
    ]

    accepted = prepared[
        "accepted"
    ]

    # -----------------------------------------------------
    # Real V4 accepted candidate centroids.
    # These are the spatial anchors.
    # -----------------------------------------------------

    accepted_centroids = {}

    for i, (
        z,
        candidate_mask,
        _
    ) in enumerate(candidate_masks):

        if not accepted[i]:
            continue

        ys, xs = np.where(
            candidate_mask > 0
        )

        if len(xs) == 0:
            continue

        accepted_centroids.setdefault(
            int(z),
            []
        ).append(
            (
                float(xs.mean()),
                float(ys.mean())
            )
        )

    ct_display = normalize_ct(
        ct
    )

    opening_kernel = np.ones(
        (5, 5),
        np.uint8
    )

    rescue_volume = np.zeros(
        suv.shape,
        dtype=np.uint8
    )

    rescue_count = 0

    selected_rows = []

    # -----------------------------------------------------
    # Examine every CT slice.
    # -----------------------------------------------------

    for z in range(
        ct.shape[0]
    ):

        anchors = []

        for nz in [
            z - 2,
            z - 1,
            z + 1,
            z + 2
        ]:

            for cx, cy in accepted_centroids.get(
                nz,
                []
            ):

                anchors.append(
                    (
                        cx,
                        cy
                    )
                )

        if not anchors:
            continue

        body_slice = body[z]

        if not np.any(
            body_slice
        ):
            continue

        # -------------------------------------------------
        # Recreate the exact V4 CT candidate morphology.
        # -------------------------------------------------

        adaptive = cv2.adaptiveThreshold(
            ct_display[z],
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15,
            3
        )

        opened = cv2.morphologyEx(
            adaptive,
            cv2.MORPH_OPEN,
            opening_kernel
        )

        opened = np.where(
            body_slice,
            opened,
            0
        ).astype(np.uint8)

        opened[:5, :] = 0
        opened[-5:, :] = 0
        opened[:, :5] = 0
        opened[:, -5:] = 0

        contours, _ = cv2.findContours(
            opened,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        eligible_components = []

        # -------------------------------------------------
        # Inspect oversized contours only.
        # -------------------------------------------------

        for contour_idx, contour in enumerate(
            contours
        ):

            contour_area = float(
                cv2.contourArea(
                    contour
                )
            )

            if contour_area <= MAX_CANDIDATE_AREA:
                continue

            contour_mask = np.zeros_like(
                opened,
                dtype=np.uint8
            )

            cv2.drawContours(
                contour_mask,
                [contour],
                -1,
                1,
                -1
            )

            contour_pixels = (
                contour_mask > 0
            )

            # -------------------------------------------------
            # PET-connected components inside the oversized
            # contour.
            # -------------------------------------------------

            hot = (
                contour_pixels
                & np.isfinite(suv[z])
                & (suv[z] >= threshold)
            ).astype(np.uint8)

            n, labels, stats, centroids = (
                cv2.connectedComponentsWithStats(
                    hot,
                    connectivity=8
                )
            )

            for label in range(
                1,
                n
            ):

                area = int(
                    stats[
                        label,
                        cv2.CC_STAT_AREA
                    ]
                )

                if area < RESCUE_MIN_COMPONENT_AREA:
                    continue

                component = (
                    labels == label
                )

                cx, cy = centroids[
                    label
                ]

                nearest_distance = min(
                    np.hypot(
                        cx - ax,
                        cy - ay
                    )
                    for ax, ay in anchors
                )

                if (
                    nearest_distance
                    > RESCUE_MAX_NEIGHBOR_DISTANCE
                ):
                    continue

                values = suv[z][
                    component
                ]

                values = values[
                    np.isfinite(values)
                ]

                if values.size == 0:
                    continue

                mean_suv = float(
                    np.mean(values)
                )

                eligible_components.append(
                    {
                        "slice": z,
                        "contour_idx": contour_idx,
                        "contour_area": contour_area,
                        "label": label,
                        "area": area,
                        "mean_suv": mean_suv,
                        "distance": float(
                            nearest_distance
                        ),
                        "component": component,
                        "cx": float(cx),
                        "cy": float(cy),
                    }
                )

        # -------------------------------------------------
        # Actual automatic selection rule:
        # highest mean SUV.
        # -------------------------------------------------

        if not eligible_components:
            continue

        selected = max(
            eligible_components,
            key=lambda x: x[
                "mean_suv"
            ]
        )

        rescue_volume[z][
            selected["component"]
        ] = 1

        rescue_count += 1

        selected_rows.append(
            {
                "slice": z,
                "threshold": threshold,
                "contour_area":
                    selected["contour_area"],
                "component_area":
                    selected["area"],
                "mean_suv":
                    selected["mean_suv"],
                "neighbor_distance":
                    selected["distance"],
            }
        )

    return (
        rescue_volume,
        rescue_count,
        selected_rows
    )


# =========================================================
# BUILD COMPLETE PREDICTION
# =========================================================

def build_prediction_with_rescue(
    prepared,
    ct,
    body,
    rescue_threshold
):

    reference = prepared[
        "reference"
    ]

    candidate_masks = prepared[
        "candidate_masks"
    ]

    accepted_reference = prepared[
        "accepted"
    ]

    accepted_slices = {
        int(reference.iloc[i]["slice"])
        for i, flag in enumerate(
            accepted_reference
        )
        if flag
    }

    background = prepared[
        "background"
    ]

    suv = prepared[
        "suv"
    ]

    candidate_volume = np.zeros(
        suv.shape,
        dtype=np.uint8
    )

    # -----------------------------------------------------
    # EXISTING DUAL-CORE V4 BRANCH
    # -----------------------------------------------------

    for i, (
        z,
        candidate_mask,
        _
    ) in enumerate(candidate_masks):

        candidate_suv = float(
            reference.iloc[i][
                "candidate_suv"
            ]
        )

        local_difference = float(
            reference.iloc[i][
                "local_difference"
            ]
        )

        local_ratio = float(
            reference.iloc[i][
                "local_ratio"
            ]
        )

        gate_count = (
            int(
                candidate_suv
                >= MIN_CANDIDATE_SUV
            )
            + int(
                local_difference
                >= MIN_LOCAL_DIFFERENCE
            )
            + int(
                local_ratio
                >= MIN_LOCAL_RATIO
            )
        )

        original_v4_accepted = bool(
            accepted_reference[i]
        )

        accepted_neighbor = any(
            0 < abs(
                z - accepted_z
            ) <= 2
            for accepted_z
            in accepted_slices
        )

        if not original_v4_accepted and not (
            gate_count >= 2
            and accepted_neighbor
        ):
            continue

        candidate_pixels = (
            candidate_mask > 0
        )

        # Original V4 3/3 candidate.
        # Recovered 2/3 candidate.
        if original_v4_accepted:

            threshold = 2.00

        else:

            threshold = 1.50

        core = (
            candidate_pixels
            & (
                suv[z] >= threshold
            )
        )

        candidate_volume[z][
            core
        ] = 1

    # -----------------------------------------------------
    # OVERSIZED-CONTOUR PET RESCUE
    # -----------------------------------------------------

    rescue_volume, rescue_count, rescue_rows = (
        build_oversized_rescue(
            prepared,
            ct,
            body,
            rescue_threshold
        )
    )

    candidate_volume = np.maximum(
        candidate_volume,
        rescue_volume
    )

    # -----------------------------------------------------
    # SAME V4 3D FILTERING
    # -----------------------------------------------------

    candidate_img = sitk.GetImageFromArray(
        candidate_volume
    )

    cc_img = sitk.ConnectedComponent(
        candidate_img
    )

    cc = sitk.GetArrayFromImage(
        cc_img
    )

    stats = (
        sitk.LabelShapeStatisticsImageFilter()
    )

    stats.Execute(
        cc_img
    )

    final_mask = np.zeros_like(
        candidate_volume,
        dtype=np.uint8
    )

    for label in stats.GetLabels():

        component = (
            cc == label
        )

        voxel_count = int(
            component.sum()
        )

        if voxel_count < MIN_3D_VOXELS:
            continue

        z_indices = np.where(
            component
        )[0]

        if len(z_indices) == 0:
            continue

        z_span = (
            int(z_indices.max())
            - int(z_indices.min())
            + 1
        )

        if z_span < MIN_3D_SLICES:
            continue

        final_mask[
            component
        ] = 1

    return (
        final_mask,
        rescue_count,
        rescue_rows
    )


# =========================================================
# MAIN
# =========================================================

print()
print("=" * 72)
print("===== V4 OVERSIZED PET-RESCUE EXPERIMENT =====")
print("=" * 72)

prepared_cases = []

case_dirs = dict(
    CASES
)

for name, case_dir in CASES:

    prepared_cases.append(
        prepare_case(
            name,
            case_dir
        )
    )

all_results = []
all_rescue_rows = []

for rescue_threshold in RESCUE_THRESHOLDS:

    rule = (
        "DUAL_CORE_2.00_1.50"
        f"_RESCUE_{rescue_threshold:.2f}"
    )

    print()
    print("-" * 72)
    print("RULE:", rule)
    print("-" * 72)

    case_metrics = []

    for prepared in prepared_cases:

        case_name = prepared[
            "name"
        ]

        ct, body = load_ct_and_body(
            case_dirs[
                case_name
            ]
        )

        prediction, rescue_count, rescue_rows = (
            build_prediction_with_rescue(
                prepared,
                ct,
                body,
                rescue_threshold
            )
        )

        metrics = evaluate(
            prediction,
            prepared["gt"]
        )

        row = {
            "rule": rule,
            "case": case_name,
            "rescue_threshold":
                rescue_threshold,
            "rescue_selected_slices":
                rescue_count,
            **metrics,
        }

        all_results.append(
            row
        )

        case_metrics.append(
            metrics
        )

        for rescue_row in rescue_rows:

            all_rescue_rows.append(
                {
                    "case": case_name,
                    "rule": rule,
                    **rescue_row,
                }
            )

        print(
            f"{case_name}: "
            f"Dice={metrics['dice']:.4f}, "
            f"IoU={metrics['iou']:.4f}, "
            f"Recall={metrics['slice_recall']:.4f}, "
            f"FPR={metrics['fpr']:.4f}, "
            f"Voxels="
            f"{metrics['prediction_voxels']}, "
            f"RescueSlices="
            f"{rescue_count}"
        )

    print(
        "MACRO:",
        f"Dice="
        f"{np.mean([m['dice'] for m in case_metrics]):.4f}, ",
        f"IoU="
        f"{np.mean([m['iou'] for m in case_metrics]):.4f}, ",
        f"Recall="
        f"{np.mean([m['slice_recall'] for m in case_metrics]):.4f}, ",
        f"FPR="
        f"{np.mean([m['fpr'] for m in case_metrics]):.4f}"
    )


# =========================================================
# SAVE
# =========================================================

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    "development_cases/"
    "v4_oversized_rescue_experiment_cases.csv",
    index=False
)

macro_df = (
    results_df
    .groupby("rule")
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
    )
    .reset_index()
)

macro_df.to_csv(
    "development_cases/"
    "v4_oversized_rescue_experiment_macro.csv",
    index=False
)

pd.DataFrame(
    all_rescue_rows
).to_csv(
    "development_cases/"
    "v4_oversized_rescue_selected_components.csv",
    index=False
)

print()
print("=" * 72)
print("===== MACRO RESULTS =====")
print("=" * 72)

print(
    macro_df.to_string(
        index=False
    )
)

print()
print("Saved:")
print(
    "development_cases/"
    "v4_oversized_rescue_experiment_cases.csv"
)

print(
    "development_cases/"
    "v4_oversized_rescue_experiment_macro.csv"
)

print(
    "development_cases/"
    "v4_oversized_rescue_selected_components.csv"
)

print()
print("===== EXPERIMENT COMPLETE =====")
