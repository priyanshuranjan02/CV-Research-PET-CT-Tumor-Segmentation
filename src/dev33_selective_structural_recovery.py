from pathlib import Path
import sys
import numpy as np
import pandas as pd
import cv2

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

# Structural recovery parameters
MIN_FALLBACK_SLICES = 2
MIN_FALLBACK_VOXELS = 100

MAX_DISTANCE_TO_BASELINE = 8
MAX_DISTANCE_TO_FALLBACK = 8

MAX_EXPANSION_RADIUS = 2


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
# OUTPUT
# ============================================================

OUT_DIR = (
    ROOT
    / "results"
    / "development_cases"
    / "dev33_selective_structural_recovery"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXACT DEV27 FALLBACK
# ============================================================

def build_pet_fallback(suv_slice):
    """
    Exact Dev27 fallback:

    PET >= 2.25
    connected components
    area >= 100
    max SUV >= 4.0
    """

    hot = (
        suv_slice >= PET_THRESHOLD
    ).astype(np.uint8)

    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            hot,
            connectivity=8
        )
    )

    fallback = np.zeros_like(
        hot,
        dtype=np.uint8
    )

    accepted = []

    for label in range(1, n_labels):

        area = int(
            stats[label, cv2.CC_STAT_AREA]
        )

        if area < MIN_COMPONENT_AREA:
            continue

        component = (
            labels == label
        )

        max_suv = float(
            np.max(
                suv_slice[component]
            )
        )

        if max_suv < MIN_MAX_SUV:
            continue

        fallback[component] = 1

        ys, xs = np.where(component)

        accepted.append({
            "label": int(label),
            "area": area,
            "max_suv": max_suv,
            "y_min": int(ys.min()),
            "y_max": int(ys.max()),
            "x_min": int(xs.min()),
            "x_max": int(xs.max()),
        })

    return fallback, accepted


# ============================================================
# 2D DISTANCE / NEIGHBORHOOD HELPERS
# ============================================================

def dilate_small(mask, radius=1):

    size = 2 * radius + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (size, size)
    )

    return cv2.dilate(
        mask.astype(np.uint8),
        kernel,
        iterations=1
    )


def minimum_distance(mask_a, mask_b):
    """
    Approximate minimum 2D distance between
    two binary masks.
    """

    if (
        np.count_nonzero(mask_a) == 0
        or np.count_nonzero(mask_b) == 0
    ):
        return np.inf

    expanded = dilate_small(
        mask_a,
        MAX_EXPANSION_RADIUS
    )

    if np.any(
        (expanded > 0) &
        (mask_b > 0)
    ):
        return 0.0

    # Use contours for a conservative pixel distance.
    pts_a = np.column_stack(
        np.where(mask_a > 0)
    )

    pts_b = np.column_stack(
        np.where(mask_b > 0)
    )

    if len(pts_a) == 0 or len(pts_b) == 0:
        return np.inf

    # Avoid enormous pairwise matrices.
    # Sample if necessary.
    if len(pts_a) > 2000:
        pts_a = pts_a[::max(1, len(pts_a) // 2000)]

    if len(pts_b) > 2000:
        pts_b = pts_b[::max(1, len(pts_b) // 2000)]

    diff = (
        pts_a[:, None, :]
        - pts_b[None, :, :]
    )

    dist2 = np.sum(
        diff.astype(np.float32) ** 2,
        axis=2
    )

    return float(
        np.sqrt(np.min(dist2))
    )


# ============================================================
# FALLBACK 3D COMPONENTS
# ============================================================

def build_fallback_3d_components(
    fallback_volume,
    suv
):
    """
    Build 3D connected components from
    fallback voxels only.
    """

    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            fallback_volume[0].astype(np.uint8),
            connectivity=8
        )
    )

    # Use scipy-free 3D flood fill.
    # 6-connectivity is deliberately conservative.

    volume = (
        fallback_volume > 0
    )

    visited = np.zeros(
        volume.shape,
        dtype=bool
    )

    components = []

    Z, H, W = volume.shape

    for z in range(Z):

        ys, xs = np.where(
            volume[z] & ~visited[z]
        )

        for y, x in zip(ys, xs):

            if visited[z, y, x]:
                continue

            stack = [(z, y, x)]
            visited[z, y, x] = True

            voxels = []

            while stack:

                cz, cy, cx = stack.pop()

                voxels.append(
                    (cz, cy, cx)
                )

                neighbors = (
                    (cz - 1, cy, cx),
                    (cz + 1, cy, cx),
                    (cz, cy - 1, cx),
                    (cz, cy + 1, cx),
                    (cz, cy, cx - 1),
                    (cz, cy, cx + 1),
                )

                for nz, ny, nx in neighbors:

                    if (
                        nz < 0 or nz >= Z
                        or ny < 0 or ny >= H
                        or nx < 0 or nx >= W
                    ):
                        continue

                    if visited[nz, ny, nx]:
                        continue

                    if not volume[nz, ny, nx]:
                        continue

                    visited[nz, ny, nx] = True
                    stack.append(
                        (nz, ny, nx)
                    )

            if len(voxels) < MIN_FALLBACK_VOXELS:
                continue

            arr = np.asarray(
                voxels,
                dtype=np.int32
            )

            zs = arr[:, 0]
            ys = arr[:, 1]
            xs = arr[:, 2]

            components.append({
                "voxels": arr,
                "voxel_count": len(arr),
                "z_start": int(zs.min()),
                "z_end": int(zs.max()),
                "z_span": int(
                    zs.max() - zs.min() + 1
                ),
                "mean_suv": float(
                    np.mean(
                        suv[
                            zs,
                            ys,
                            xs
                        ]
                    )
                ),
                "max_suv": float(
                    np.max(
                        suv[
                            zs,
                            ys,
                            xs
                        ]
                    )
                ),
            })

    return components


# ============================================================
# SELECTIVE STRUCTURAL RECOVERY
# ============================================================

def selective_recovery(
    baseline,
    fallback,
    suv
):
    """
    Add only structurally supported fallback
    voxels to the baseline.

    No GT information is used.
    """

    result = baseline.copy()

    fallback_components = (
        build_fallback_3d_components(
            fallback,
            suv
        )
    )

    accepted = []

    for idx, comp in enumerate(
        fallback_components
    ):

        # ----------------------------------------------------
        # Basic persistence requirement
        # ----------------------------------------------------

        if (
            comp["z_span"]
            < MIN_FALLBACK_SLICES
        ):
            continue

        if (
            comp["voxel_count"]
            < MIN_FALLBACK_VOXELS
        ):
            continue

        # ----------------------------------------------------
        # Require strong PET signal
        # ----------------------------------------------------

        if comp["max_suv"] < MIN_MAX_SUV:
            continue

        # ----------------------------------------------------
        # Check distance to baseline candidate
        # ----------------------------------------------------

        arr = comp["voxels"]

        z_min = max(
            0,
            comp["z_start"] - 1
        )

        z_max = min(
            baseline.shape[0] - 1,
            comp["z_end"] + 1
        )

        near_baseline = False
        best_distance = np.inf

        for z in range(
            z_min,
            z_max + 1
        ):

            component_slice = np.zeros(
                baseline.shape[1:],
                dtype=np.uint8
            )

            current = (
                arr[:, 0] == z
            )

            if not np.any(current):
                continue

            ys = arr[current, 1]
            xs = arr[current, 2]

            component_slice[
                ys,
                xs
            ] = 1

            distance = minimum_distance(
                component_slice,
                baseline[z]
            )

            best_distance = min(
                best_distance,
                distance
            )

            if (
                distance
                <= MAX_DISTANCE_TO_BASELINE
            ):
                near_baseline = True
                break

        # ----------------------------------------------------
        # If not close to baseline, reject.
        # ----------------------------------------------------

        if not near_baseline:
            continue

        # ----------------------------------------------------
        # Accepted component
        # ----------------------------------------------------

        result[
            arr[:, 0],
            arr[:, 1],
            arr[:, 2]
        ] = 1

        accepted.append({
            "component": idx,
            "voxel_count": comp["voxel_count"],
            "z_span": comp["z_span"],
            "mean_suv": comp["mean_suv"],
            "max_suv": comp["max_suv"],
            "baseline_distance": best_distance,
        })

    return result, accepted


# ============================================================
# EVALUATION
# ============================================================

def evaluate_variant(
    case,
    volume
):

    components, cc = (
        dev10.build_3d_component_table(
            case,
            volume
        )
    )

    metrics = dev10.evaluate_selection(
        case,
        components,
        cc,
        RANKING_FEATURE,
        CORE_POLICY,
        TOP_K
    )

    return (
        metrics,
        components,
        cc
    )


# ============================================================
# CASE ANALYSIS
# ============================================================

def analyze_case(case_name):

    print("\n" + "=" * 80)
    print(case_name)
    print("=" * 80)

    case = dev10.load_case(
        case_name
    )

    baseline = (
        dev10.generate_candidate_volume(
            case
        )
    )

    suv = case["suv"]

    # --------------------------------------------------------
    # Exact Dev27 fallback
    # --------------------------------------------------------

    fallback = np.zeros_like(
        baseline,
        dtype=np.uint8
    )

    fallback_2d_components = 0
    fallback_slices = 0

    for z in range(
        suv.shape[0]
    ):

        # Only recover when Dev10
        # candidate is completely empty.
        if np.count_nonzero(
            baseline[z]
        ) != 0:
            continue

        mask, accepted = (
            build_pet_fallback(
                suv[z]
            )
        )

        if np.count_nonzero(
            mask
        ) == 0:
            continue

        fallback[z] = mask

        fallback_slices += 1
        fallback_2d_components += len(
            accepted
        )

    # --------------------------------------------------------
    # Selective structural recovery
    # --------------------------------------------------------

    selective, accepted = (
        selective_recovery(
            baseline,
            fallback,
            suv
        )
    )

    print(
        f"Fallback slices: "
        f"{fallback_slices}"
    )

    print(
        f"Fallback 2D components: "
        f"{fallback_2d_components}"
    )

    print(
        f"Structurally accepted "
        f"components: {len(accepted)}"
    )

    for item in accepted:

        print(
            "  "
            f"component={item['component']} "
            f"voxels={item['voxel_count']} "
            f"z_span={item['z_span']} "
            f"meanSUV={item['mean_suv']:.3f} "
            f"maxSUV={item['max_suv']:.3f} "
            f"distance={item['baseline_distance']:.2f}"
        )

    variants = {
        "baseline": baseline,
        "fallback": np.maximum(
            baseline,
            fallback
        ),
        "selective": selective,
    }

    rows = []

    for name, volume in variants.items():

        metrics, components, cc = (
            evaluate_variant(
                case,
                volume
            )
        )

        row = {
            "case": case_name,
            "variant": name,
            "dice": float(
                metrics["dice"]
            ),
            "iou": float(
                metrics["iou"]
            ),
            "slice_recall": float(
                metrics["slice_recall"]
            ),
            "fpr": float(
                metrics["fpr"]
            ),
            "prediction_voxels": int(
                metrics[
                    "prediction_voxels"
                ]
            ),
            "candidate_voxels": int(
                np.count_nonzero(
                    volume
                )
            ),
            "components_3d": int(
                len(components)
            ),
            "accepted_structural": int(
                len(accepted)
            ),
        }

        rows.append(row)

        print(
            f"{name:12s} "
            f"Dice={row['dice']:.6f} "
            f"IoU={row['iou']:.6f} "
            f"Recall={row['slice_recall']:.6f} "
            f"FPR={row['fpr']:.6f} "
            f"Pred={row['prediction_voxels']}"
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
        summary[
            "total_prediction_voxels"
        ]
        - baseline[
            "total_prediction_voxels"
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    case_path = (
        OUT_DIR
        / "dev33_case_results.csv"
    )

    summary_path = (
        OUT_DIR
        / "dev33_policy_summary.csv"
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
    # Print
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("DEV33 MACRO SUMMARY")
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