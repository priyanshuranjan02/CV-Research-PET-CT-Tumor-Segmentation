"""
Dev22 — Visual PET/GT/CT Alignment Diagnostic

Purpose:
    Visually inspect the exact Dev10 CT-contour failure slices.

    For each selected slice, generate a 4-panel figure:

        1. CT image + GT contour
        2. PET/SUV image + GT contour
        3. PET/SUV image + PET threshold mask + GT contour
        4. CT image + GT contour + PET maximum

    Dev10 is NOT modified.
"""

from pathlib import Path
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

RESULT_DIR = (
    ROOT
    / "results"
    / "development_cases"
    / "dev22_visual_alignment_results"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC))

import dev10_lite_hot_core as dev10


CASES = [
    "PETCT_0b57b247b6",
    "PETCT_185da4c8b6",
]

PET_THRESHOLD = 2.25


def get_dev10_ct_contours(ct_slice, body_slice):
    """
    Exact Dev10 CT contour generation.
    """
    ct_display = dev10.normalize_ct(ct_slice)

    kernel = np.ones((5, 5), np.uint8)

    opened = cv2.morphologyEx(
        ct_display,
        cv2.MORPH_OPEN,
        kernel
    )

    binary = cv2.adaptiveThreshold(
        opened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        3
    )

    binary[body_slice == 0] = 0

    binary[:5, :] = 0
    binary[-5:, :] = 0
    binary[:, :5] = 0
    binary[:, -5:] = 0

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return [
        c for c in contours
        if cv2.contourArea(c) >= 100
    ]


def find_baseline_failures(case):
    """
    Find exact Dev10 CT-contour failure slices.
    """
    ct = case["ct"]
    gt = case["gt"]

    body = dev10.build_union_body_mask(ct)

    gt_bool = gt.astype(bool)

    positive_slices = np.where(
        np.any(gt_bool, axis=(1, 2))
    )[0]

    failures = []

    for z in positive_slices:

        gt_slice = gt_bool[z]

        contours = get_dev10_ct_contours(
            ct[z],
            body[z]
        )

        mask = np.zeros(
            gt_slice.shape,
            dtype=np.uint8
        )

        if contours:
            cv2.drawContours(
                mask,
                contours,
                -1,
                1,
                -1
            )

        overlap = np.logical_and(
            mask > 0,
            gt_slice
        ).sum()

        if overlap == 0:
            failures.append(z)

    return failures


def normalize_for_display(image):
    """
    Robust 0-1 normalization for visualization only.
    """
    image = image.astype(np.float32)

    lo = np.percentile(image, 1)
    hi = np.percentile(image, 99)

    if hi <= lo:
        return np.zeros_like(image)

    image = np.clip(
        (image - lo) / (hi - lo),
        0,
        1
    )

    return image


def draw_gt_contour(ax, gt):
    """
    Draw GT boundary.
    """
    contours, _ = cv2.findContours(
        gt.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        points = contour[:, 0, :]

        if len(points) >= 2:
            ax.plot(
                points[:, 0],
                points[:, 1],
                linewidth=1.5,
                label="GT"
            )


def draw_ct_contours(ax, contours):
    """
    Draw Dev10 CT contours.
    """
    for contour in contours:
        points = contour[:, 0, :]

        if len(points) >= 2:
            ax.plot(
                points[:, 0],
                points[:, 1],
                linewidth=0.8,
                alpha=0.7
            )


def save_slice_visualization(
    case_name,
    z,
    ct_slice,
    suv_slice,
    gt_slice,
    body_slice,
    output_path
):

    ct_display = normalize_for_display(
        ct_slice
    )

    suv_display = normalize_for_display(
        suv_slice
    )

    contours = get_dev10_ct_contours(
        ct_slice,
        body_slice
    )

    pet_mask = suv_slice >= PET_THRESHOLD

    max_y, max_x = np.unravel_index(
        np.argmax(suv_slice),
        suv_slice.shape
    )

    gt_y, gt_x = np.where(gt_slice)

    gt_cx = float(gt_x.mean())
    gt_cy = float(gt_y.mean())

    max_suv = float(
        suv_slice[max_y, max_x]
    )

    gt_mean_suv = float(
        np.mean(suv_slice[gt_slice])
    )

    gt_max_suv = float(
        np.max(suv_slice[gt_slice])
    )

    pet_overlap = np.logical_and(
        pet_mask,
        gt_slice
    ).sum()

    gt_area = gt_slice.sum()

    pet_coverage = (
        pet_overlap / gt_area
        if gt_area > 0
        else 0.0
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 12)
    )

    ax1, ax2, ax3, ax4 = axes.ravel()

    # --------------------------------------------------
    # 1. CT + GT
    # --------------------------------------------------

    ax1.imshow(
        ct_display,
        cmap="gray"
    )

    draw_gt_contour(
        ax1,
        gt_slice
    )

    draw_ct_contours(
        ax1,
        contours
    )

    ax1.set_title(
        "CT + GT + Dev10 CT contours"
    )

    ax1.axis("off")

    # --------------------------------------------------
    # 2. PET + GT
    # --------------------------------------------------

    ax2.imshow(
        suv_display,
        cmap="hot"
    )

    draw_gt_contour(
        ax2,
        gt_slice
    )

    ax2.scatter(
        [max_x],
        [max_y],
        marker="x",
        s=100,
        linewidths=2
    )

    ax2.set_title(
        "PET/SUV + GT + PET maximum"
    )

    ax2.axis("off")

    # --------------------------------------------------
    # 3. PET threshold + GT
    # --------------------------------------------------

    ax3.imshow(
        pet_mask,
        cmap="gray"
    )

    draw_gt_contour(
        ax3,
        gt_slice
    )

    ax3.set_title(
        f"PET >= {PET_THRESHOLD} + GT"
    )

    ax3.axis("off")

    # --------------------------------------------------
    # 4. CT + GT + PET maximum
    # --------------------------------------------------

    ax4.imshow(
        ct_display,
        cmap="gray"
    )

    draw_gt_contour(
        ax4,
        gt_slice
    )

    ax4.scatter(
        [max_x],
        [max_y],
        marker="x",
        s=100,
        linewidths=2
    )

    ax4.scatter(
        [gt_cx],
        [gt_cy],
        marker="+",
        s=100,
        linewidths=2
    )

    ax4.set_title(
        "CT + GT centroid + PET maximum"
    )

    ax4.axis("off")

    fig.suptitle(
        (
            f"{case_name} | slice {z}\n"
            f"GT mean SUV={gt_mean_suv:.3f} | "
            f"GT max SUV={gt_max_suv:.3f} | "
            f"PET max={max_suv:.3f} | "
            f"PET>=2.25 GT coverage={pet_coverage:.3f}"
        ),
        fontsize=13
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


def main():

    print("=" * 100)
    print("DEV22 — VISUAL PET/GT/CT ALIGNMENT DIAGNOSTIC")
    print("=" * 100)

    for case_name in CASES:

        print("\n" + "=" * 100)
        print(f"===== CASE: {case_name} =====")
        print("=" * 100)

        case = dev10.load_case(case_name)

        ct = case["ct"]
        suv = case["suv"]
        gt = case["gt"]

        failures = find_baseline_failures(
            case
        )

        print(
            f"Exact Dev10 CT failures: "
            f"{len(failures)}"
        )

        if not failures:
            print("No failure slices.")
            continue

        print(
            "Failure slices:",
            failures
        )

        body = dev10.build_union_body_mask(
            ct
        )

        case_dir = (
            RESULT_DIR / case_name
        )

        case_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for z in failures:

            gt_slice = gt[z].astype(bool)

            output_path = (
                case_dir /
                f"slice_{z:04d}.png"
            )

            save_slice_visualization(
                case_name=case_name,
                z=z,
                ct_slice=ct[z],
                suv_slice=suv[z],
                gt_slice=gt_slice,
                body_slice=body[z],
                output_path=output_path
            )

            print(
                f"Saved: {output_path}"
            )

    print("\n" + "=" * 100)
    print("DEV22 COMPLETE")
    print("=" * 100)

    print(
        f"Visualizations: {RESULT_DIR}"
    )


if __name__ == "__main__":
    main()