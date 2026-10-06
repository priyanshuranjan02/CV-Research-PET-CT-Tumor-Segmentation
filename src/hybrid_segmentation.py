from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom


def find_series_directory(case_root: Path, description_text: str) -> Path:
    matches = [
        p for p in case_root.rglob("*")
        if p.is_dir() and description_text in p.name
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one directory containing "
            f"'{description_text}', found {len(matches)}"
        )

    return matches[0]


def read_dicom_series(series_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(series_dir))

    if not files:
        raise RuntimeError(f"No DICOM files found in {series_dir}")

    reader.SetFileNames(files)
    return reader.Execute()


def normalize_ct_hu(ct_slice: np.ndarray) -> np.ndarray:
    """
    Fixed CT window used for candidate generation.
    HU range: -1000 to +1000.
    """
    img = np.clip(ct_slice, -1000.0, 1000.0)
    img = ((img + 1000.0) / 2000.0 * 255.0)
    return img.astype(np.uint8)


def build_body_mask(ct_array: np.ndarray) -> np.ndarray:
    """
    Conservative whole-body ROI using HU > -900 followed by
    morphological closing and largest connected component.
    """
    body = np.zeros_like(ct_array, dtype=np.uint8)

    kernel = np.ones((11, 11), np.uint8)

    for z in range(ct_array.shape[0]):
        binary = (ct_array[z] > -900).astype(np.uint8) * 255

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)

        cv2.drawContours(
            body[z],
            [contour],
            -1,
            1,
            -1,
        )

    return body.astype(bool)


def robust_z(value: float, median: float, mad: float) -> float:
    return float((value - median) / (1.4826 * mad + 1e-6))


def percentile_rank(values):
    values = np.asarray(values, dtype=np.float64)

    if len(values) <= 1:
        return np.ones(len(values), dtype=np.float64)

    order = np.argsort(np.argsort(values))
    return order / (len(values) - 1)


def candidate_features(
    contour,
    ct_img,
    pet_slice,
    body_slice,
):
    mask = np.zeros_like(ct_img, dtype=np.uint8)

    cv2.drawContours(
        mask,
        [contour],
        -1,
        1,
        -1,
    )

    pixels = mask > 0

    if not np.any(pixels):
        return None

    area = float(cv2.contourArea(contour))

    perimeter = float(cv2.arcLength(contour, True))

    circularity = (
        4.0 * np.pi * area / (perimeter * perimeter + 1e-6)
    )

    ct_values = ct_img[pixels].astype(np.float32)
    pet_values = pet_slice[pixels].astype(np.float32)

    body_pet = pet_slice[body_slice]

    if body_pet.size == 0:
        return None

    pet_median = float(np.median(body_pet))
    pet_mad = float(
        np.median(np.abs(body_pet - pet_median))
    )

    pet_mean = float(np.mean(pet_values))
    pet_p90 = float(np.percentile(pet_values, 90))

    high_threshold = float(
        np.percentile(body_pet, 90)
    )

    pet_high_fraction = float(
        np.mean(pet_values >= high_threshold)
    )

    pet_robust_z = robust_z(
        pet_mean,
        pet_median,
        pet_mad,
    )

    # Simple CT texture proxy:
    # intensity variation inside the candidate.
    ct_std = float(np.std(ct_values))

    # Edge-density / texture proxy using Laplacian variance.
    lap = cv2.Laplacian(
        ct_img,
        cv2.CV_32F,
    )
    texture = float(
        np.var(lap[pixels])
    )

    x, y, w, h = cv2.boundingRect(contour)

    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": float(
            np.clip(circularity, 0.0, 1.0)
        ),
        "ct_mean": float(np.mean(ct_values)),
        "ct_std": ct_std,
        "texture": texture,
        "pet_mean": pet_mean,
        "pet_p90": pet_p90,
        "pet_high_fraction": pet_high_fraction,
        "pet_robust_z": pet_robust_z,
        "bbox_w": w,
        "bbox_h": h,
        "mask": mask,
    }


def generate_candidates(
    ct_array,
    pet_array,
    body,
):
    """
    Generate candidate contours slice-by-slice.

    Candidate generation uses:
      CT adaptive threshold
      -> morphological opening
      -> body ROI
      -> contour extraction

    Selection is then performed using hybrid CT + PET features.
    """
    final_mask = np.zeros_like(ct_array, dtype=np.uint8)
    records = []

    kernel = np.ones((5, 5), np.uint8)

    for z in range(ct_array.shape[0]):

        ct_display = normalize_ct_hu(ct_array[z])

        body_slice = body[z]

        if not np.any(body_slice):
            continue

        adaptive = cv2.adaptiveThreshold(
            ct_display,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15,
            3,
        )

        # Opening only.
        # Closing was shown to collapse the image into a giant component.
        clean = cv2.morphologyEx(
            adaptive,
            cv2.MORPH_OPEN,
            kernel,
        )

        clean = np.where(
            body_slice,
            clean,
            0,
        ).astype(np.uint8)

        # Remove image border.
        clean[:5, :] = 0
        clean[-5:, :] = 0
        clean[:, :5] = 0
        clean[:, -5:] = 0

        contours, _ = cv2.findContours(
            clean,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates = []

        for contour in contours:

            area = cv2.contourArea(contour)

            # Candidate size range.
            if area < 100 or area > 15000:
                continue

            features = candidate_features(
                contour,
                ct_display,
                pet_array[z],
                body_slice,
            )

            if features is None:
                continue

            features["z"] = z
            features["contour"] = contour

            candidates.append(features)

        if not candidates:
            continue

        # -------------------------------------------------
        # Hybrid feature ranking
        # -------------------------------------------------

        pet_mean_rank = percentile_rank(
            [c["pet_mean"] for c in candidates]
        )

        pet_high_rank = percentile_rank(
            [c["pet_high_fraction"] for c in candidates]
        )

        texture_rank = percentile_rank(
            [c["texture"] for c in candidates]
        )

        area_rank = percentile_rank(
            [np.log1p(c["area"]) for c in candidates]
        )

        # Shape preference:
        # round/compact regions score higher.
        shape_score = np.array(
            [c["circularity"] for c in candidates],
            dtype=np.float64,
        )

        scores = (
            0.45 * pet_mean_rank
            + 0.20 * pet_high_rank
            + 0.15 * texture_rank
            + 0.10 * shape_score
            + 0.10 * area_rank
        )

        for i, candidate in enumerate(candidates):
            candidate["score"] = float(scores[i])

        # Require evidence of elevated PET uptake.
        # This prevents selection of purely CT-driven structures.
        eligible = [
            c
            for c in candidates
            if c["pet_high_fraction"] >= 0.20
            and c["pet_robust_z"] >= 1.0
        ]

        if eligible:
            best = max(
                eligible,
                key=lambda c: c["score"],
            )

            # Conservative score threshold.
            if best["score"] >= 0.55:
                final_mask[
                    z
                ][best["mask"] > 0] = 1

        # Save candidate information for analysis.
        for candidate in candidates:
            records.append({
                "slice": z,
                "area": candidate["area"],
                "circularity": candidate["circularity"],
                "ct_mean": candidate["ct_mean"],
                "ct_std": candidate["ct_std"],
                "texture": candidate["texture"],
                "pet_mean": candidate["pet_mean"],
                "pet_p90": candidate["pet_p90"],
                "pet_high_fraction": candidate["pet_high_fraction"],
                "pet_robust_z": candidate["pet_robust_z"],
                "score": candidate["score"],
            })

    return final_mask, pd.DataFrame(records)


def dice_score(pred, gt):
    pred = pred > 0
    gt = gt > 0

    intersection = np.logical_and(
        pred,
        gt,
    ).sum()

    denominator = (
        pred.sum() + gt.sum()
    )

    if denominator == 0:
        return 1.0

    return float(
        2.0 * intersection / denominator
    )


def iou_score(pred, gt):
    pred = pred > 0
    gt = gt > 0

    intersection = np.logical_and(
        pred,
        gt,
    ).sum()

    union = np.logical_or(
        pred,
        gt,
    ).sum()

    if union == 0:
        return 1.0

    return float(
        intersection / union
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--case-dir",
        required=True,
        help="Development case directory",
    )

    args = parser.parse_args()

    case_dir = Path(args.case_dir)

    print("Case:", case_dir)

    # Locate series
    ct_dir = find_series_directory(
        case_dir,
        "GK p.v.3",
    )

    pet_dir = find_series_directory(
        case_dir,
        "PET corr.",
    )

    print("CT:", ct_dir)
    print("PET:", pet_dir)

    # Load volumes
    ct = read_dicom_series(ct_dir)
    pet = read_dicom_series(pet_dir)

    print("\nOriginal geometries:")
    print("CT :", ct.GetSize(), ct.GetSpacing())
    print("PET:", pet.GetSize(), pet.GetSpacing())

    # Resample PET to CT geometry.
    pet_on_ct = sitk.Resample(
        pet,
        ct,
        sitk.Transform(),
        sitk.sitkLinear,
        0.0,
        pet.GetPixelID(),
    )

    ct_array = sitk.GetArrayFromImage(
        ct
    ).astype(np.float32)

    pet_array = sitk.GetArrayFromImage(
        pet_on_ct
    ).astype(np.float32)

    print(
        "\nPET resampled to CT:",
        pet_on_ct.GetSize(),
    )

    # Build body ROI
    print("\nBuilding body ROI...")
    body = build_body_mask(ct_array)

    print(
        "Body ROI voxels:",
        int(body.sum()),
    )

    # Generate hybrid segmentation
    print("\nRunning hybrid candidate scoring...")

    prediction, candidates = generate_candidates(
        ct_array,
        pet_array,
        body,
    )

    # Save prediction
    prediction_img = sitk.GetImageFromArray(
        prediction.astype(np.uint8)
    )
    prediction_img.CopyInformation(ct)

    prediction_path = (
        case_dir / "hybrid_segmentation_v1.nii.gz"
    )

    sitk.WriteImage(
        prediction_img,
        str(prediction_path),
    )

    # Save candidate table
    candidate_csv = (
        case_dir / "hybrid_candidate_scores.csv"
    )

    candidates.to_csv(
        candidate_csv,
        index=False,
    )

    print("\nPrediction saved:")
    print(prediction_path)

    print("Candidate table saved:")
    print(candidate_csv)

    # -------------------------------------------------
    # Development evaluation
    # -------------------------------------------------

    gt_path = (
        case_dir /
        "tumor_mask_ct_visible.nii.gz"
    )

    if gt_path.exists():

        gt_img = sitk.ReadImage(
            str(gt_path)
        )

        gt_array = sitk.GetArrayFromImage(
            gt_img
        ) > 0

        pred_array = prediction > 0

        positive_gt = np.any(
            gt_array,
            axis=(1, 2)
        )

        positive_pred = np.any(
            pred_array,
            axis=(1, 2)
        )

        tp_slices = np.logical_and(
            positive_gt,
            positive_pred,
        ).sum()

        fp_slices = np.logical_and(
            ~positive_gt,
            positive_pred,
        ).sum()

        total_gt_positive = positive_gt.sum()
        total_gt_negative = (~positive_gt).sum()

        print("\n===== HYBRID V1 DEVELOPMENT RESULT =====")

        print(
            "Total CT slices:",
            len(positive_gt),
        )

        print(
            "GT positive slices:",
            int(total_gt_positive),
        )

        print(
            "Predicted positive slices:",
            int(positive_pred.sum()),
        )

        print(
            "Slice recall:",
            round(
                float(
                    tp_slices /
                    max(total_gt_positive, 1)
                ),
                4,
            ),
        )

        print(
            "Mean Dice:",
            round(
                dice_score(
                    prediction,
                    gt_array,
                ),
                4,
            ),
        )

        print(
            "IoU:",
            round(
                iou_score(
                    prediction,
                    gt_array,
                ),
                4,
            ),
        )

        print(
            "False-positive slices:",
            int(fp_slices),
        )

        print(
            "False-positive rate:",
            round(
                float(
                    fp_slices /
                    max(total_gt_negative, 1)
                ),
                4,
            ),
        )

        print(
            "\nPrediction voxels:",
            int(pred_array.sum()),
        )

        print(
            "GT voxels:",
            int(gt_array.sum()),
        )


if __name__ == "__main__":
    main()
