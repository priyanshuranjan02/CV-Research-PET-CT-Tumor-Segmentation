from pathlib import Path
import runpy
import numpy as np
import cv2
import pandas as pd


# ---------------------------------------------------------
# Load exact Case 7 preparation utilities.
# ---------------------------------------------------------

ns = runpy.run_path(
    "src/v4_validate_case7.py",
    run_name="__case7_body_component_audit__"
)

prepare_case7 = ns["prepare_case7"]
normalize_ct = ns["normalize_ct"]

prepared, ct_img, ct, body = prepare_case7()

gt = prepared["gt"]

ct_display = normalize_ct(ct)

GT_SLICES = np.where(
    np.any(
        gt,
        axis=(1, 2)
    )
)[0]


rows = []


for z in GT_SLICES:

    gt_slice = gt[z]

    # -----------------------------------------------------
    # Recreate the EXACT raw body threshold.
    # -----------------------------------------------------

    binary = (
        ct[z] > -900
    ).astype(np.uint8) * 255

    kernel = np.ones(
        (11, 11),
        np.uint8
    )

    closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    components = []

    for idx, contour in enumerate(contours):

        component_mask = np.zeros(
            gt_slice.shape,
            dtype=np.uint8
        )

        cv2.drawContours(
            component_mask,
            [contour],
            -1,
            1,
            -1
        )

        area = int(
            component_mask.sum()
        )

        overlap = int(
            np.logical_and(
                component_mask > 0,
                gt_slice
            ).sum()
        )

        components.append(
            {
                "component_id": int(idx),
                "area": area,
                "gt_overlap": overlap,
            }
        )

    components.sort(
        key=lambda x: x["area"],
        reverse=True
    )

    for rank, component in enumerate(
        components
    ):
        component["area_rank"] = rank + 1

    largest_area = (
        components[0]["area"]
        if components
        else 0
    )

    gt_component = None

    for component in components:

        if component["gt_overlap"] > 0:

            gt_component = component

            break

    if gt_component is None:

        rows.append(
            {
                "slice": int(z),
                "gt_area": int(gt_slice.sum()),
                "num_components": len(components),
                "largest_area": largest_area,
                "largest_gt_overlap": 0,
                "gt_component_area": 0,
                "gt_component_rank": -1,
                "gt_component_area_fraction_of_largest": 0.0,
                "current_body_gt_overlap": int(
                    np.logical_and(
                        body[z],
                        gt_slice
                    ).sum()
                )
            }
        )

    else:

        current_overlap = int(
            np.logical_and(
                body[z],
                gt_slice
            ).sum()
        )

        rows.append(
            {
                "slice": int(z),
                "gt_area": int(gt_slice.sum()),
                "num_components": len(components),
                "largest_area": largest_area,
                "largest_gt_overlap": (
                    components[0]["gt_overlap"]
                    if components
                    else 0
                ),
                "gt_component_area":
                    gt_component["area"],
                "gt_component_rank":
                    gt_component["area_rank"],
                "gt_component_area_fraction_of_largest":
                    (
                        gt_component["area"]
                        / largest_area
                        if largest_area > 0
                        else 0
                    ),
                "current_body_gt_overlap":
                    current_overlap
            }
        )


df = pd.DataFrame(rows)


# ---------------------------------------------------------
# Print.
# ---------------------------------------------------------

pd.set_option(
    "display.max_rows",
    100
)

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    220
)

print("\n" + "=" * 100)
print("CASE 7 — BODY CONNECTED-COMPONENT AUDIT")
print("=" * 100)

print(
    df.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

print(
    "GT-positive slices:",
    len(df)
)

print(
    "Slices where GT is inside largest component:",
    int(
        (df["gt_component_rank"] == 1).sum()
    )
)

print(
    "Slices where GT is in non-largest component:",
    int(
        (df["gt_component_rank"] > 1).sum()
    )
)

print(
    "Slices with no GT-containing component:",
    int(
        (df["gt_component_rank"] == -1).sum()
    )
)

print(
    "\nSlices where GT is in a non-largest component:"
)

print(
    df[
        df["gt_component_rank"] > 1
    ][
        [
            "slice",
            "gt_area",
            "num_components",
            "largest_area",
            "gt_component_area",
            "gt_component_rank",
            "gt_component_area_fraction_of_largest",
            "current_body_gt_overlap"
        ]
    ].to_string(
        index=False
    )
)


# ---------------------------------------------------------
# Save.
# ---------------------------------------------------------

out = (
    Path("validation_results")
    / "PETCT_0011f3deaf"
    / "case7_body_component_audit.csv"
)

out.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    out,
    index=False
)

print(
    "\nSaved:",
    out
)

print(
    "\n===== BODY COMPONENT AUDIT COMPLETE ====="
)
