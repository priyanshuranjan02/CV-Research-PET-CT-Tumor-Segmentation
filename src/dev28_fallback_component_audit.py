from pathlib import Path
import sys
import gc
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
import dev10_lite_hot_core as dev10

CASES = [
    "PETCT_0168f65af8",
    "PETCT_04606080a0",
    "PETCT_04ab5c61c9",
    "PETCT_0b57b247b6",
    "PETCT_11afab3485",
    "PETCT_185da4c8b6",
]

PET_THRESHOLD = 2.25
MIN_COMPONENT_AREA = 100
MIN_MAX_SUV = 4.0
RANKING = "mean_suv_hot3_a0p5"
CORE_POLICY = "P70"
TOP_K = 4

RESULT_DIR = ROOT / "results" / "development_cases" / "dev28_fallback_component_audit"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def build_pet_fallback(suv_slice):
    hot = (suv_slice >= PET_THRESHOLD).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hot, connectivity=8)
    fallback = np.zeros_like(hot, dtype=np.uint8)
    accepted = []
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < MIN_COMPONENT_AREA:
            continue
        mask = labels == label
        values = suv_slice[mask]
        if values.size == 0:
            continue
        max_suv = float(np.max(values))
        if max_suv < MIN_MAX_SUV:
            continue
        fallback[mask] = 1
        accepted.append({
            "label": label,
            "area": area,
            "mean_suv": float(np.mean(values)),
            "max_suv": max_suv,
            "mask": mask,
        })
    return fallback, accepted


def trace_case(case, fallback_records, components, cc):
    gt = case["gt"]
    suv = case["suv"]

    if components.empty:
        ranked = components.copy()
    else:
        ranked = components.sort_values(
            [RANKING, "mean_suv"], ascending=[False, False]
        ).reset_index(drop=True)

    rank_map = {int(r["label"]): i + 1 for i, (_, r) in enumerate(ranked.iterrows())}
    selected = {int(x) for x in ranked.head(TOP_K)["label"].tolist()}
    eligible = {int(r["label"]): r for _, r in components.iterrows()}

    max_label = int(cc.max())
    voxel_counts = np.bincount(cc.ravel(), minlength=max_label + 1)
    z_start = np.full(max_label + 1, cc.shape[0], dtype=np.int32)
    z_end = np.full(max_label + 1, -1, dtype=np.int32)
    for z in range(cc.shape[0]):
        labs = np.unique(cc[z])
        labs = labs[labs > 0]
        if labs.size:
            z_start[labs] = np.minimum(z_start[labs], z)
            z_end[labs] = np.maximum(z_end[labs], z)

    rows = []
    for rec in fallback_records:
        z = rec["slice"]
        mask = rec["mask"]
        gt_slice = gt[z]
        gt_overlap_2d = int(np.logical_and(mask, gt_slice).sum())
        gt_voxels_2d = int(gt_slice.sum())
        dice_2d = (2.0 * gt_overlap_2d / (rec["area"] + gt_voxels_2d)
                   if rec["area"] + gt_voxels_2d else 0.0)

        labels = [int(x) for x in np.unique(cc[z][mask]) if int(x) > 0]
        if not labels:
            labels = [0]

        for label in labels:
            is_eligible = label in eligible
            if label > 0:
                fallback_in_3d = int(np.logical_and(mask, cc[z] == label).sum())
                voxels = int(voxel_counts[label])
                z0 = int(z_start[label]) if z_end[label] >= 0 else -1
                z1 = int(z_end[label]) if z_end[label] >= 0 else -1
            else:
                fallback_in_3d = 0
                voxels = 0
                z0 = z1 = -1
            span = z1 - z0 + 1 if z0 >= 0 else 0

            if is_eligible:
                r3 = eligible[label]
                comp_dice = float(r3["dice_gt"])
                comp_iou = float(r3["iou_gt"])
                comp_overlap = int(r3["overlap_gt"])
                mean_suv_3d = float(r3["mean_suv"])
                rank_value = float(r3[RANKING])
                rank = rank_map[label]
                selected_topk = label in selected
            else:
                comp_dice = comp_iou = 0.0
                comp_overlap = 0
                mean_suv_3d = rank_value = np.nan
                rank = -1
                selected_topk = False

            core_threshold = np.nan
            core_voxels = 0
            fallback_in_core = 0
            core_gt_overlap = 0
            core_dice = 0.0

            if is_eligible and selected_topk:
                core, core_threshold, core_voxels = dev10.build_core(
                    cc, suv, label, CORE_POLICY
                )
                if core is not None:
                    fallback_in_core = int(np.logical_and(mask, core[z]).sum())
                    core_gt_overlap = int(np.logical_and(core, gt).sum())
                    gt_voxels = int(gt.sum())
                    core_dice = (2.0 * core_gt_overlap / (core_voxels + gt_voxels)
                                 if core_voxels + gt_voxels else 0.0)
                    del core

            rows.append({
                "case_id": rec["case_id"],
                "slice": z,
                "fallback_label_2d": rec["fallback_label"],
                "fallback_area": rec["area"],
                "fallback_mean_suv": rec["mean_suv"],
                "fallback_max_suv": rec["max_suv"],
                "fallback_gt_overlap_2d": gt_overlap_2d,
                "fallback_dice_2d": dice_2d,
                "3d_label": label,
                "3d_component_eligible": is_eligible,
                "3d_component_voxels": voxels,
                "3d_z_start": z0,
                "3d_z_end": z1,
                "3d_z_span": span,
                "fallback_voxels_in_3d_component": fallback_in_3d,
                "3d_mean_suv": mean_suv_3d,
                "3d_ranking_value": rank_value,
                "3d_component_gt_overlap": comp_overlap,
                "3d_component_dice": comp_dice,
                "3d_component_iou": comp_iou,
                "ranking_rank": rank,
                "selected_top4": selected_topk,
                "core_policy": CORE_POLICY,
                "core_threshold": core_threshold,
                "core_voxels": core_voxels,
                "fallback_voxels_in_core": fallback_in_core,
                "core_gt_overlap": core_gt_overlap,
                "core_dice": core_dice,
            })
    return rows


def main():
    print("=" * 115)
    print("DEV28 — FALLBACK COMPONENT-LEVEL AUDIT")
    print("=" * 115)
    print(f"Fallback: PET >= {PET_THRESHOLD}, area >= {MIN_COMPONENT_AREA}, max SUV >= {MIN_MAX_SUV}")
    print(f"Downstream: {RANKING}, {CORE_POLICY}, Top-K={TOP_K}")

    all_rows = []
    summaries = []

    for case_name in CASES:
        print("\n" + "=" * 115)
        print(f"===== CASE: {case_name} =====")
        print("=" * 115)
        case = dev10.load_case(case_name)

        baseline = dev10.generate_candidate_volume(case)
        experimental = baseline.copy()
        records = []

        for z in range(case["ct"].shape[0]):
            if np.count_nonzero(baseline[z]) != 0:
                continue
            fallback, accepted = build_pet_fallback(case["suv"][z])
            for item in accepted:
                records.append({
                    "case_id": case_name,
                    "slice": z,
                    "fallback_label": int(item["label"]),
                    "area": int(item["area"]),
                    "mean_suv": float(item["mean_suv"]),
                    "max_suv": float(item["max_suv"]),
                    "mask": item["mask"],
                })
            if accepted:
                experimental[z] = np.maximum(experimental[z], fallback)

        baseline_components, baseline_cc = dev10.build_3d_component_table(case, baseline)
        experimental_components, experimental_cc = dev10.build_3d_component_table(case, experimental)

        base_m = dev10.evaluate_selection(case, baseline_components, baseline_cc, RANKING, CORE_POLICY, TOP_K)
        exp_m = dev10.evaluate_selection(case, experimental_components, experimental_cc, RANKING, CORE_POLICY, TOP_K)

        rows = trace_case(case, records, experimental_components, experimental_cc)
        all_rows.extend(rows)

        if rows:
            adf = pd.DataFrame(rows)
            grouped = adf.groupby(["slice", "fallback_label_2d"])
            with_gt = int((grouped["fallback_gt_overlap_2d"].max() > 0).sum())
            survived = int(grouped["3d_component_eligible"].max().sum())
            selected = int(grouped["selected_top4"].max().sum())
            reached_core = int((grouped["fallback_voxels_in_core"].sum() > 0).sum())
        else:
            with_gt = survived = selected = reached_core = 0

        summaries.append({
            "case_id": case_name,
            "fallback_components": len(records),
            "with_gt_overlap": with_gt,
            "survived_3d_filter": survived,
            "selected_top4": selected,
            "reached_p70_core": reached_core,
            "baseline_dice": base_m["dice"],
            "experimental_dice": exp_m["dice"],
            "delta_dice": exp_m["dice"] - base_m["dice"],
            "baseline_slice_recall": base_m["slice_recall"],
            "experimental_slice_recall": exp_m["slice_recall"],
            "delta_slice_recall": exp_m["slice_recall"] - base_m["slice_recall"],
            "baseline_fpr": base_m["fpr"],
            "experimental_fpr": exp_m["fpr"],
            "delta_fpr": exp_m["fpr"] - base_m["fpr"],
        })

        print(f"Accepted fallback components: {len(records)}")
        print(f"  With GT overlap: {with_gt}/{len(records)}")
        print(f"  Survived 3D filter: {survived}/{len(records)}")
        print(f"  Selected Top-4: {selected}/{len(records)}")
        print(f"  Reached P70 core: {reached_core}/{len(records)}")
        print(f"  Dice: {base_m['dice']:.6f} -> {exp_m['dice']:.6f}")
        print(f"  Slice recall: {base_m['slice_recall']:.6f} -> {exp_m['slice_recall']:.6f}")
        print(f"  FPR: {base_m['fpr']:.8f} -> {exp_m['fpr']:.8f}")

        del case, baseline, experimental, baseline_components, baseline_cc, experimental_components, experimental_cc
        gc.collect()

    audit_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summaries)
    audit_path = RESULT_DIR / "dev28_fallback_component_audit.csv"
    summary_path = RESULT_DIR / "dev28_case_summary.csv"
    audit_df.to_csv(audit_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 115)
    print("DEV28 COMPLETE")
    print("=" * 115)
    print(f"Detailed audit: {audit_path}")
    print(f"Case summary:   {summary_path}")
    print("Dev10 was NOT modified.")


if __name__ == "__main__":
    main()