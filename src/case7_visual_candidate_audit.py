from pathlib import Path
import runpy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import SimpleITK as sitk


ROOT = Path(".")

# ---------------------------------------------------------
# Load existing Case 7 pipeline without changing it.
# ---------------------------------------------------------

ns = runpy.run_path(
    "src/v4_validate_case7.py",
    run_name="__case7_visual_audit__"
)

prepare_case7 = ns["prepare_case7"]

prepared, ct_img, ct, body = prepare_case7()

candidate_masks = prepared["candidate_masks"]
suv = prepared["suv"]
gt = prepared["gt"]
reference = prepared["reference"]

normalize_ct = ns["normalize_ct"]


# ---------------------------------------------------------
# Representative slices:
#
# 278, 283 = early lesion where candidates are near but miss
# 290      = transition region
# 355,356  = strongest candidate overlap
# 360      = dramatic candidate failure
# ---------------------------------------------------------

SLICES = [
    278,
    283,
    290,
    355,
    356,
    360
]


ct_display = normalize_ct(ct)

out_dir = (
    ROOT
    / "validation_results"
    / "PETCT_0011f3deaf"
)

out_dir.mkdir(
    parents=True,
    exist_ok=True
)


for z in SLICES:

    # -----------------------------------------------------
    # All candidates for this slice.
    # -----------------------------------------------------

    indices = [
        i
        for i, (cz, _, _) in enumerate(candidate_masks)
        if int(cz) == z
    ]

    # Candidate with maximum PET median.
    best_i = None

    if indices:

        best_i = max(
            indices,
            key=lambda i: float(
                reference.iloc[i]["candidate_suv"]
            )
        )

    # Candidate with maximum GT overlap.
    best_overlap_i = None
    best_overlap = -1

    if indices and gt[z].any():

        for i in indices:

            mask = (
                candidate_masks[i][1] > 0
            )

            overlap = int(
                np.logical_and(
                    mask,
                    gt[z]
                ).sum()
            )

            if overlap > best_overlap:
                best_overlap = overlap
                best_overlap_i = i

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 10)
    )

    # -----------------------------------------------------
    # 1. CT + GT
    # -----------------------------------------------------

    ax = axes[0, 0]

    ax.imshow(
        ct_display[z],
        cmap="gray"
    )

    gt_contours, _ = cv2.findContours(
        gt[z].astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if gt_contours:
        ax.contour(
            gt[z],
            levels=[0.5],
            linewidths=2
        )

    ax.set_title(
        f"Slice {z}: CT + GT"
    )

    ax.axis("off")


    # -----------------------------------------------------
    # 2. PET SUV
    # -----------------------------------------------------

    ax = axes[0, 1]

    im = ax.imshow(
        suv[z],
        cmap="hot",
        vmin=0,
        vmax=max(
            3.0,
            float(
                np.percentile(
                    suv[z],
                    99.5
                )
            )
        )
    )

    ax.contour(
        gt[z],
        levels=[0.5],
        linewidths=2
    )

    ax.set_title(
        f"Slice {z}: PET SUV + GT"
    )

    ax.axis("off")

    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046
    )


    # -----------------------------------------------------
    # 3. CT candidate with maximum median SUV
    # -----------------------------------------------------

    ax = axes[0, 2]

    ax.imshow(
        ct_display[z],
        cmap="gray"
    )

    if best_i is not None:

        mask = (
            candidate_masks[best_i][1] > 0
        )

        ax.contour(
            mask,
            levels=[0.5],
            linewidths=2
        )

        ax.contour(
            gt[z],
            levels=[0.5],
            linewidths=2
        )

        r = reference.iloc[best_i]

        ax.set_title(
            "Max candidate SUV\n"
            f"SUV={r['candidate_suv']:.4f}, "
            f"area={int(mask.sum())}"
        )

    else:
        ax.set_title(
            "No candidate"
        )

    ax.axis("off")


    # -----------------------------------------------------
    # 4. Candidate masks + GT
    # -----------------------------------------------------

    ax = axes[1, 0]

    ax.imshow(
        ct_display[z],
        cmap="gray"
    )

    for i in indices:

        mask = (
            candidate_masks[i][1] > 0
        )

        ax.contour(
            mask,
            levels=[0.5],
            linewidths=0.4,
            alpha=0.25
        )

    ax.contour(
        gt[z],
        levels=[0.5],
        linewidths=2
    )

    ax.set_title(
        f"All {len(indices)} CT candidates + GT"
    )

    ax.axis("off")


    # -----------------------------------------------------
    # 5. Best-overlap candidate
    # -----------------------------------------------------

    ax = axes[1, 1]

    ax.imshow(
        ct_display[z],
        cmap="gray"
    )

    if best_overlap_i is not None:

        mask = (
            candidate_masks[
                best_overlap_i
            ][1] > 0
        )

        ax.contour(
            mask,
            levels=[0.5],
            linewidths=2
        )

        ax.contour(
            gt[z],
            levels=[0.5],
            linewidths=2
        )

        r = reference.iloc[
            best_overlap_i
        ]

        ax.set_title(
            "Best GT-overlap candidate\n"
            f"overlap={best_overlap}, "
            f"SUV={r['candidate_suv']:.4f}"
        )

    else:
        ax.set_title(
            "No GT overlap"
        )

    ax.axis("off")


    # -----------------------------------------------------
    # 6. PET + candidate
    # -----------------------------------------------------

    ax = axes[1, 2]

    ax.imshow(
        suv[z],
        cmap="hot",
        vmin=0,
        vmax=max(
            3.0,
            float(
                np.percentile(
                    suv[z],
                    99.5
                )
            )
        )
    )

    if best_i is not None:

        mask = (
            candidate_masks[best_i][1] > 0
        )

        ax.contour(
            mask,
            levels=[0.5],
            linewidths=2
        )

    ax.contour(
        gt[z],
        levels=[0.5],
        linewidths=2
    )

    ax.set_title(
        "PET + candidate + GT"
    )

    ax.axis("off")


    fig.suptitle(
        f"Case 7 Candidate Geometry Audit — Slice {z}",
        fontsize=16
    )

    fig.tight_layout()

    out_path = (
        out_dir
        / f"case7_geometry_slice_{z}.png"
    )

    fig.savefig(
        out_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "Saved:",
        out_path
    )


print("\n===== VISUAL AUDIT COMPLETE =====")
