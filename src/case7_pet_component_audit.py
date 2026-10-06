from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import SimpleITK as sitk


ROOT = Path("verified_case")

CT_DIR = next(
    p for p in ROOT.rglob("4.000000-GK p.v.3-58263")
    if p.is_dir() and "__MACOSX" not in str(p)
)

PET_DIR = next(
    p for p in ROOT.rglob("7.000000-PET corr.-78839")
    if p.is_dir() and "__MACOSX" not in str(p)
)

GT_PATH = ROOT / "tumor_mask_ct_visible.nii.gz"


def read_series(directory):
    reader = sitk.ImageSeriesReader()

    ids = reader.GetGDCMSeriesIDs(str(directory))

    if not ids:
        raise RuntimeError(
            f"No DICOM series found: {directory}"
        )

    files = reader.GetGDCMSeriesFileNames(
        str(directory),
        ids[0]
    )

    reader.SetFileNames(files)

    return reader.Execute()


def build_body_mask(ct):

    body = np.zeros_like(
        ct,
        dtype=np.uint8
    )

    kernel = np.ones(
        (11, 11),
        np.uint8
    )

    for z in range(ct.shape[0]):

        binary = (
            ct[z] > -900
        ).astype(np.uint8) * 255

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        contour = max(
            contours,
            key=cv2.contourArea
        )

        cv2.drawContours(
            body[z],
            [contour],
            -1,
            1,
            -1
        )

    return body.astype(bool)


def calculate_suv(
    activity,
    weight_kg,
    dose_bq
):
    return (
        activity
        * weight_kg
        * 1000.0
        / dose_bq
    )


print("\n" + "=" * 80)
print("CASE 7 — PET-ONLY CONNECTED COMPONENT AUDIT")
print("=" * 80)

ct_img = read_series(
    CT_DIR
)

pet_img = read_series(
    PET_DIR
)

ct = sitk.GetArrayFromImage(
    ct_img
).astype(np.float32)

pet_stored = sitk.GetArrayFromImage(
    pet_img
).astype(np.float64)

print("\nCT:", ct.shape)
print("PET:", pet_stored.shape)


# Same calibration used by Case 7.
weight = 68.0
dose = 327000000.0

suv_native = calculate_suv(
    pet_stored,
    weight,
    dose
)

suv_native_img = sitk.GetImageFromArray(
    suv_native.astype(np.float32)
)

suv_native_img.CopyInformation(
    pet_img
)

suv_ct_img = sitk.Resample(
    suv_native_img,
    ct_img,
    sitk.Transform(),
    sitk.sitkLinear,
    0.0,
    sitk.sitkFloat32
)

suv = sitk.GetArrayFromImage(
    suv_ct_img
).astype(np.float64)


gt = (
    sitk.GetArrayFromImage(
        sitk.ReadImage(
            str(GT_PATH)
        )
    ) > 0
)

body = build_body_mask(
    ct
)

print(
    "\nSUV max:",
    float(np.nanmax(suv))
)

print(
    "GT voxels:",
    int(gt.sum())
)


# ---------------------------------------------------------
# Analyze PET connected components.
# ---------------------------------------------------------

thresholds = [
    1.00,
    1.25,
    1.50,
    2.00,
    2.25
]

all_rows = []

for threshold in thresholds:

    print("\n" + "-" * 80)
    print(
        f"PET SUV >= {threshold:.2f}"
    )
    print("-" * 80)

    binary = (
        body
        & np.isfinite(suv)
        & (suv >= threshold)
    ).astype(np.uint8)

    cc_img = sitk.ConnectedComponent(
        sitk.GetImageFromArray(binary)
    )

    cc = sitk.GetArrayFromImage(
        cc_img
    )

    stats = sitk.LabelShapeStatisticsImageFilter()

    stats.Execute(
        cc_img
    )

    components = []

    for label in stats.GetLabels():

        component = (
            cc == label
        )

        voxels = int(
            component.sum()
        )

        if voxels < 20:
            continue

        z_indices = np.where(
            component
        )[0]

        if len(z_indices) == 0:
            continue

        overlap = int(
            np.logical_and(
                component,
                gt
            ).sum()
        )

        gt_voxels = int(
            gt.sum()
        )

        dice = (
            2.0 * overlap
            / (
                voxels
                + gt_voxels
                + 1e-9
            )
        )

        iou = (
            overlap
            / (
                voxels
                + gt_voxels
                - overlap
                + 1e-9
            )
        )

        gt_z = np.where(gt)[0]

        component_z_min = int(
            z_indices.min()
        )

        component_z_max = int(
            z_indices.max()
        )

        gt_z_min = int(
            gt_z.min()
        )

        gt_z_max = int(
            gt_z.max()
        )

        ys, xs = np.where(
            np.any(
                component,
                axis=0
            )
        )

        # True 3D centroid.
        coords = np.argwhere(
            component
        )

        cz = float(
            coords[:, 0].mean()
        )

        cy = float(
            coords[:, 1].mean()
        )

        cx = float(
            coords[:, 2].mean()
        )

        gt_coords = np.argwhere(
            gt
        )

        gcz = float(
            gt_coords[:, 0].mean()
        )

        gcy = float(
            gt_coords[:, 1].mean()
        )

        gcx = float(
            gt_coords[:, 2].mean()
        )

        centroid_distance = float(
            np.sqrt(
                (cx - gcx) ** 2
                + (cy - gcy) ** 2
                + (cz - gcz) ** 2
            )
        )

        values = suv[
            component
        ]

        values = values[
            np.isfinite(values)
        ]

        components.append(
            {
                "threshold": threshold,
                "label": int(label),
                "voxels": voxels,
                "z_min": component_z_min,
                "z_max": component_z_max,
                "z_span": (
                    component_z_max
                    - component_z_min
                    + 1
                ),
                "overlap_gt": overlap,
                "dice_gt": dice,
                "iou_gt": iou,
                "mean_suv": float(
                    np.mean(values)
                ),
                "median_suv": float(
                    np.median(values)
                ),
                "max_suv": float(
                    np.max(values)
                ),
                "p95_suv": float(
                    np.percentile(
                        values,
                        95
                    )
                ),
                "centroid_distance_3d": (
                    centroid_distance
                ),
            }
        )

    components_df = pd.DataFrame(
        components
    )

    print(
        "Components >=20 voxels:",
        len(components_df)
    )

    if components_df.empty:
        continue

    top_overlap = (
        components_df
        .sort_values(
            [
                "overlap_gt",
                "dice_gt",
                "mean_suv"
            ],
            ascending=[
                False,
                False,
                False
            ]
        )
        .head(10)
    )

    print(
        "\nTop components by GT overlap:"
    )

    print(
        top_overlap.to_string(
            index=False
        )
    )

    best = top_overlap.iloc[0]

    print(
        "\nBEST:"
    )

    print(
        "  overlap =",
        int(best["overlap_gt"])
    )

    print(
        "  Dice    =",
        f"{best['dice_gt']:.4f}"
    )

    print(
        "  IoU     =",
        f"{best['iou_gt']:.4f}"
    )

    print(
        "  voxels  =",
        int(best["voxels"])
    )

    print(
        "  z-range =",
        int(best["z_min"]),
        "-",
        int(best["z_max"])
    )

    print(
        "  mean SUV =",
        f"{best['mean_suv']:.4f}"
    )

    print(
        "  max SUV  =",
        f"{best['max_suv']:.4f}"
    )

    print(
        "  3D centroid distance =",
        f"{best['centroid_distance_3d']:.2f}"
    )

    all_rows.extend(
        components
    )


# ---------------------------------------------------------
# Save complete audit.
# ---------------------------------------------------------

out = (
    ROOT.parent
    / "validation_results"
    / "PETCT_0011f3deaf"
    / "case7_pet_connected_component_audit.csv"
)

out.parent.mkdir(
    parents=True,
    exist_ok=True
)

pd.DataFrame(
    all_rows
).to_csv(
    out,
    index=False
)

print(
    "\nSaved:",
    out
)

print(
    "\n===== PET COMPONENT AUDIT COMPLETE ====="
)
