"""
Dev43 — Frozen validation on PETCT_0011f3deaf

Compares:
1. Original Dev10-Lite
2. Frozen Dev39

Dev39:
    PET-only SUV >= 2.25
    -> exact Dev10 3D components
    -> Dev38 PET-quality ranking
    -> hot_fraction >= 0.40
    -> Top-K 4
    -> P70 core

DO NOT MODIFY dev10_lite_hot_core.py.
"""

from pathlib import Path
import sys
import runpy
import numpy as np
import pandas as pd
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import dev10_lite_hot_core as dev10


CASE_NAME = "PETCT_0011f3deaf"

PET_THRESHOLD = 2.25
HOT_THRESHOLD = 3.0
HOT_FRACTION_GATE = 0.40
TOP_K = 4
CORE_POLICY = "P70"

GT_PATH = ROOT / "verified_case" / "tumor_mask_ct.nii.gz"

OUTPUT_DIR = ROOT / "results" / "development_cases" / "dev43_validation_case7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_case7():
    """
    Reuse the existing Case-7 DICOM/SUV loader.
    """
    loader_path = SRC / "v4_validate_case7.py"

    ns = runpy.run_path(
        str(loader_path),
        run_name="__dev43_loader__"
    )

    if "prepare_case7" not in ns:
        raise RuntimeError(
            "prepare_case7() was not found in v4_validate_case7.py"
        )

    prepared, ct_img, ct, body = ns["prepare_case7"]()

    # prepare_case7() already performs the project's PET/SUV preparation.
    # Find the aligned SUV array from the returned prepared dictionary.
    if isinstance(prepared, dict):
        suv = None

        for key in ["suv", "SUV", "pet_suv", "aligned_suv"]:
            if key in prepared:
                suv = prepared[key]
                break

        if suv is None:
            raise RuntimeError(
                "Could not find aligned SUV array in prepare_case7() output. "
                f"Available keys: {list(prepared.keys())}"
            )
    else:
        raise RuntimeError(
            "Unexpected prepare_case7() return type: "
            f"{type(prepared)}"
        )

    ct = np.asarray(ct, dtype=np.float32)
    suv = np.asarray(suv, dtype=np.float32)

    return ct, suv, ct_img


def load_gt():
    gt_img = sitk.ReadImage(str(GT_PATH))
    gt = sitk.GetArrayFromImage(gt_img) > 0
    return gt, gt_img


def check_geometry(ct, suv, gt, gt_img):
    print("\n===== GEOMETRY CHECK =====")
    print("CT :", ct.shape)
    print("SUV:", suv.shape)
    print("GT :", gt.shape)

    if ct.shape != gt.shape:
        raise ValueError(
            f"CT/GT shape mismatch: {ct.shape} vs {gt.shape}"
        )

    if suv.shape != ct.shape:
        raise ValueError(
            f"SUV/CT shape mismatch: {suv.shape} vs {ct.shape}"
        )

    print("GT spacing :", gt_img.GetSpacing())
    print("GT origin  :", gt_img.GetOrigin())
    print("GT direction:", gt_img.GetDirection())

    print("GT voxels:", int(gt.sum()))

    print("Geometry: PASS")


def build_pet_only_candidate(suv):
    return (suv >= PET_THRESHOLD).astype(np.uint8)


def add_dev39_features(components, cc, suv):
    """
    Add the frozen Dev38 PET-quality features.

    These are computed only from the PET component itself.
    """
    df = components.copy()

    records = []

    for _, row in df.iterrows():
        label = int(row["label"])
        mask = cc == label

        coords = np.where(mask)

        if len(coords[0]) == 0:
            records.append({
                "label": label,
                "voxels_dev39": 0,
                "z_span_dev39": 0,
                "slice_count_dev39": 0,
                "longest_run_dev39": 0,
                "persistence_dev39": 0.0,
                "compactness_dev39": 0.0,
                "mean_suv_dev39": 0.0,
                "max_suv_dev39": 0.0,
                "hot_fraction_dev39": 0.0,
            })
            continue

        z = coords[0]

        voxels = int(mask.sum())
        z_unique = np.unique(z)

        z_span = int(z_unique[-1] - z_unique[0] + 1)
        slice_count = int(len(z_unique))

        # Longest consecutive z run.
        longest_run = 1
        current_run = 1

        for i in range(1, len(z_unique)):
            if z_unique[i] == z_unique[i - 1] + 1:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 1

        persistence = (
            float(longest_run) / float(z_span)
            if z_span > 0 else 0.0
        )

        z0, z1 = z.min(), z.max()
        y0, y1 = coords[1].min(), coords[1].max()
        x0, x1 = coords[2].min(), coords[2].max()

        bbox_volume = (
            (z1 - z0 + 1)
            * (y1 - y0 + 1)
            * (x1 - x0 + 1)
        )

        compactness = (
            float(voxels) / float(bbox_volume)
            if bbox_volume > 0 else 0.0
        )

        vals = suv[mask]

        mean_suv = float(vals.mean())
        max_suv = float(vals.max())
        hot_fraction = float(np.mean(vals >= HOT_THRESHOLD))

        records.append({
            "label": label,
            "voxels_dev39": voxels,
            "z_span_dev39": z_span,
            "slice_count_dev39": slice_count,
            "longest_run_dev39": longest_run,
            "persistence_dev39": persistence,
            "compactness_dev39": compactness,
            "mean_suv_dev39": mean_suv,
            "max_suv_dev39": max_suv,
            "hot_fraction_dev39": hot_fraction,
        })

    features = pd.DataFrame(records)

    df = df.merge(features, on="label", how="left")

    # Normalize each feature across the current case.
    def norm(col):
        x = df[col].astype(float).to_numpy()
        lo = np.nanmin(x)
        hi = np.nanmax(x)

        if hi <= lo:
            return np.zeros_like(x)

        return (x - lo) / (hi - lo)

    n_mean = norm("mean_suv_dev39")
    n_max = norm("max_suv_dev39")
    n_hot = norm("hot_fraction_dev39")
    n_persist = norm("persistence_dev39")
    n_run = norm("longest_run_dev39")
    n_compact = norm("compactness_dev39")

    # Exact Dev38 PET-quality score.
    df["dev39_pet_quality"] = (
        0.25 * n_mean
        + 0.15 * n_max
        + 0.15 * n_hot
        + 0.20 * n_persist
        + 0.15 * n_run
        + 0.10 * n_compact
    )

    return df


def safe_core_mask(cc, suv, label, policy):
    result = dev10.build_core(
        cc,
        suv,
        int(label),
        policy
    )

    if isinstance(result, tuple):
        arrays = [
            x for x in result
            if isinstance(x, np.ndarray)
        ]

        if not arrays:
            raise RuntimeError(
                "build_core returned tuple but no ndarray was found."
            )

        return arrays[0].astype(bool)

    return result.astype(bool)


def evaluate_case(case, ranking_df, cc, ranking_col):
    """
    Exact Dev10 evaluator on the supplied selected/ranked components.
    """
    metrics = dev10.evaluate_selection(
        case,
        ranking_df,
        cc,
        ranking_col,
        CORE_POLICY,
        TOP_K
    )

    return metrics


def run_original_dev10(case):
    """
    Original Dev10 pipeline.
    """
    candidate = dev10.generate_candidate_volume(case)

    components, cc = dev10.build_3d_component_table(
        case,
        candidate
    )

    metrics = dev10.evaluate_selection(
        case,
        components,
        cc,
        "mean_suv_hot3_a0p5",
        CORE_POLICY,
        TOP_K
    )

    return candidate, components, cc, metrics


def run_dev39(case):
    """
    Frozen Dev39 pipeline.
    """
    suv = case["suv"]

    candidate = build_pet_only_candidate(suv)

    components, cc = dev10.build_3d_component_table(
        case,
        candidate
    )

    components = add_dev39_features(
        components,
        cc,
        suv
    )

    # Frozen Dev39 gate.
    eligible = components[
        components["hot_fraction_dev39"] >= HOT_FRACTION_GATE
    ].copy()

    eligible = eligible.sort_values(
        ["dev39_pet_quality", "mean_suv"],
        ascending=[False, False]
    ).reset_index(drop=True)

    selected = eligible.head(TOP_K).copy()

    # evaluate_selection expects the ranking column to exist.
    metrics = evaluate_case(
        case,
        selected,
        cc,
        "dev39_pet_quality"
    )

    return candidate, components, eligible, selected, cc, metrics


def main():
    print("=" * 100)
    print("DEV43 — FROZEN VALIDATION")
    print("=" * 100)

    print("\nConfiguration:")
    print("Case:", CASE_NAME)
    print("PET threshold:", PET_THRESHOLD)
    print("Hot threshold:", HOT_THRESHOLD)
    print("Hot fraction gate:", HOT_FRACTION_GATE)
    print("Top-K:", TOP_K)
    print("Core:", CORE_POLICY)
    print("GT:", GT_PATH)

    print("\n===== LOADING CASE 7 =====")

    ct, suv, ct_img = load_case7()
    gt, gt_img = load_gt()

    check_geometry(
        ct,
        suv,
        gt,
        gt_img
    )

    case = {
        "name": CASE_NAME,
        "ct": ct,
        "suv": suv,
        "gt": gt,
    }

    print("\n===== ORIGINAL DEV10 =====")

    (
        dev10_candidate,
        dev10_components,
        dev10_cc,
        dev10_metrics,
    ) = run_original_dev10(case)

    print("Candidate voxels:",
          int(dev10_candidate.sum()))
    print("3D components:",
          len(dev10_components))

    print("\nDev10 metrics:")
    for k, v in dev10_metrics.items():
        print(f"{k}: {v}")

    print("\n===== FROZEN DEV39 =====")

    (
        dev39_candidate,
        dev39_components,
        dev39_eligible,
        dev39_selected,
        dev39_cc,
        dev39_metrics,
    ) = run_dev39(case)

    print("Candidate voxels:",
          int(dev39_candidate.sum()))
    print("3D components:",
          len(dev39_components))
    print("Eligible components:",
          len(dev39_eligible))
    print("Selected components:",
          list(dev39_selected["label"].astype(int)))

    print("\nDev39 metrics:")
    for k, v in dev39_metrics.items():
        print(f"{k}: {v}")

    # ------------------------------------------------------------------
    # Save component audit.
    # ------------------------------------------------------------------

    audit = dev39_components.copy()

    audit["eligible_dev39"] = audit["label"].isin(
        dev39_eligible["label"]
    )

    audit["selected_dev39"] = audit["label"].isin(
        dev39_selected["label"]
    )

    audit_path = OUTPUT_DIR / "dev43_dev39_component_audit.csv"
    audit.to_csv(audit_path, index=False)

    # ------------------------------------------------------------------
    # Save summary.
    # ------------------------------------------------------------------

    def get_metric(metrics, key):
        value = metrics.get(key, np.nan)
        try:
            return float(value)
        except Exception:
            return np.nan

    summary = pd.DataFrame([
        {
            "case": CASE_NAME,
            "method": "Dev10",
            "dice": get_metric(dev10_metrics, "dice"),
            "iou": get_metric(dev10_metrics, "iou"),
            "slice_recall": get_metric(
                dev10_metrics,
                "slice_recall"
            ),
            "fpr": get_metric(dev10_metrics, "fpr"),
            "prediction_voxels": get_metric(
                dev10_metrics,
                "prediction_voxels"
            ),
            "candidate_voxels": int(
                dev10_candidate.sum()
            ),
            "components": len(dev10_components),
        },
        {
            "case": CASE_NAME,
            "method": "Dev39",
            "dice": get_metric(dev39_metrics, "dice"),
            "iou": get_metric(dev39_metrics, "iou"),
            "slice_recall": get_metric(
                dev39_metrics,
                "slice_recall"
            ),
            "fpr": get_metric(dev39_metrics, "fpr"),
            "prediction_voxels": get_metric(
                dev39_metrics,
                "prediction_voxels"
            ),
            "candidate_voxels": int(
                dev39_candidate.sum()
            ),
            "components": len(dev39_components),
        },
    ])

    summary["dice_delta_vs_dev10"] = (
        summary["dice"]
        - summary.loc[
            summary["method"] == "Dev10",
            "dice"
        ].iloc[0]
    )

    summary["fpr_delta_vs_dev10"] = (
        summary["fpr"]
        - summary.loc[
            summary["method"] == "Dev10",
            "fpr"
        ].iloc[0]
    )

    summary_path = OUTPUT_DIR / "dev43_case_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 100)
    print("DEV43 COMPARISON")
    print("=" * 100)

    print(summary.to_string(index=False))

    print("\nSaved:")
    print(summary_path)
    print(audit_path)

    print("\n===== DEV43 COMPLETE =====")


if __name__ == "__main__":
    main()