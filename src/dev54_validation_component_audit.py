"""
Dev54 — Validation Component Audit

Purpose:
    Audit ALL eligible PET-only components on the validation case
    using the exact frozen Dev39 ranking pipeline.

IMPORTANT:
    - Dev10 is NOT modified.
    - Dev39 is NOT modified.
    - GT is NOT used for component selection/ranking.
    - GT is used ONLY after ranking for post-hoc audit.
    - Ranking is exactly the frozen Dev39 ranking.
    - Top-K = 4, Core = P70.

Outputs:
    results/validation/dev54_component_audit/
        dev54_all_eligible_components.csv
        dev54_selected_components.csv
        dev54_core_audit.csv
        dev54_validation_audit.txt
"""

from pathlib import Path
import sys
import io
import contextlib

import numpy as np
import pandas as pd
import nibabel as nib

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import dev10_lite_hot_core as dev10
import dev39_pet_quality_fpr_control as dev39


# ---------------------------------------------------------------------
# Frozen Dev39 configuration
# ---------------------------------------------------------------------

CASE_NAME = "PETCT_0011f3deaf"

PET_THRESHOLD = 2.25
HOT_THRESHOLD = 3.0

MIN_3D_VOXELS = 75
MIN_3D_SLICES = 2

HOT_FRACTION_GATE = 0.40

TOP_K = 4
CORE_POLICY = "P70"


# ---------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------

OUT_DIR = (
    ROOT
    / "results"
    / "validation"
    / "dev54_component_audit"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def dice_score(pred, gt):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)

    p = int(pred.sum())
    g = int(gt.sum())

    if p + g == 0:
        return 1.0

    overlap = int(np.logical_and(pred, gt).sum())

    return (2.0 * overlap) / (p + g)


def iou_score(pred, gt):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)

    intersection = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())

    if union == 0:
        return 1.0

    return intersection / union


def safe_build_core(cc, suv, label, policy):
    """
    Dev10 build_core returns a tuple.

    This wrapper extracts the actual core mask robustly without
    modifying Dev10.
    """

    result = dev10.build_core(
        cc,
        suv,
        int(label),
        policy
    )

    if isinstance(result, tuple):
        # Dev10's current build_core returns a tuple whose mask is
        # the first ndarray-like boolean volume.
        for item in result:
            if isinstance(item, np.ndarray):
                if item.shape == cc.shape:
                    return item.astype(bool)

        raise RuntimeError(
            f"Could not identify core mask for label {label} "
            f"from build_core() tuple."
        )

    if isinstance(result, np.ndarray):
        return result.astype(bool)

    raise RuntimeError(
        f"Unexpected build_core() return type: {type(result)}"
    )


def component_metrics(mask, gt):
    """
    Post-hoc GT audit of a component.
    """

    mask = np.asarray(mask, dtype=bool)
    gt = np.asarray(gt, dtype=bool)

    component_voxels = int(mask.sum())
    gt_voxels = int(gt.sum())

    overlap = int(np.logical_and(mask, gt).sum())

    precision = (
        overlap / component_voxels
        if component_voxels > 0
        else 0.0
    )

    recall = (
        overlap / gt_voxels
        if gt_voxels > 0
        else 0.0
    )

    dice = dice_score(mask, gt)
    iou = iou_score(mask, gt)

    return {
        "component_voxels": component_voxels,
        "gt_overlap_voxels": overlap,
        "gt_precision": precision,
        "gt_recall": recall,
        "component_dice": dice,
        "component_iou": iou,
    }


def z_extent(mask):
    """
    Return z_start, z_end, z_span.
    """

    coords = np.where(np.any(mask, axis=(1, 2)))[0]

    if len(coords) == 0:
        return -1, -1, 0

    z_start = int(coords.min())
    z_end = int(coords.max())
    z_span = z_end - z_start + 1

    return z_start, z_end, z_span


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 80)
    print("DEV54 — VALIDATION COMPONENT AUDIT")
    print("=" * 80)

    print(f"Case                 : {CASE_NAME}")
    print(f"PET threshold        : {PET_THRESHOLD}")
    print(f"Hot threshold        : {HOT_THRESHOLD}")
    print(f"3D minimum voxels    : {MIN_3D_VOXELS}")
    print(f"3D minimum slices    : {MIN_3D_SLICES}")
    print(f"Hot fraction gate    : {HOT_FRACTION_GATE}")
    print(f"Top-K                : {TOP_K}")
    print(f"Core policy          : {CORE_POLICY}")

    
    # -----------------------------------------------------------------
    # Load validation case
    # -----------------------------------------------------------------

    print("\n[1] Loading validation case...")

    validation_script = SRC / "v4_validate_case7.py"

    if not validation_script.exists():
        raise FileNotFoundError(
            f"Validation loader not found:\n{validation_script}"
        )

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "v4_validate_case7",
        validation_script
    )

    validation_module = importlib.util.module_from_spec(spec)

    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(validation_module)

    if not hasattr(validation_module, "prepare_case7"):
        raise AttributeError(
            "v4_validate_case7.py does not expose prepare_case7()."
        )

    with contextlib.redirect_stdout(io.StringIO()):
        loaded = validation_module.prepare_case7()

    if not isinstance(loaded, tuple):
        raise TypeError(
            f"Unexpected prepare_case7() return type: {type(loaded)}"
        )

    if len(loaded) != 4:
        raise RuntimeError(
            f"Expected 4 values from prepare_case7(), got {len(loaded)}"
        )

    validation_case = loaded[0]

    if not isinstance(validation_case, dict):
        raise TypeError(
            "prepare_case7()[0] is not a dictionary."
        )

    print("Validation case loaded successfully.")

    print(
        "Available validation keys:",
        list(validation_case.keys())
    )

    # Exact validation arrays from prepare_case7()
    suv = np.asarray(
        validation_case["suv"],
        dtype=np.float32
    )

    gt = np.asarray(
        validation_case["gt"],
        dtype=bool
    )

    # CT is not required by Dev54 because this is a PET-only
    # Dev39 component audit.

    print(f"SUV shape             : {suv.shape}")
    print(f"GT shape              : {gt.shape}")
    print(
        f"SUV max               : "
        f"{float(np.nanmax(suv)):.6f}"
    )
    print(
        f"GT voxels             : "
        f"{int(gt.sum())}"
    )

    if suv.shape != gt.shape:
        raise ValueError(
            f"SUV/GT shape mismatch: "
            f"{suv.shape} vs {gt.shape}"
        )

    # -----------------------------------------------------------------
    # Exact Dev39 PET-only candidate
    # -----------------------------------------------------------------

    print("\n[2] Generating exact Dev39 PET-only candidate...")

    candidate = (
        np.asarray(suv, dtype=np.float32) >= PET_THRESHOLD
    ).astype(np.uint8)

    candidate_voxels = int(candidate.sum())

    print(f"Candidate voxels     : {candidate_voxels}")

    # -----------------------------------------------------------------
    # Exact Dev10 3D component table
    # -----------------------------------------------------------------

    print("\n[3] Building exact Dev10 3D component table...")

    components, cc = dev10.build_3d_component_table(
        validation_case,
        candidate
    )

    print(f"Total 3D components  : {len(components)}")

    # -----------------------------------------------------------------
    # Exact Dev39 component features
    # -----------------------------------------------------------------

    print("\n[4] Computing exact Dev39 component features...")

    components = dev39.add_component_features(
        components,
        cc,
        suv
    )

    # -----------------------------------------------------------------
    # Exact Dev39 PET quality score
    # -----------------------------------------------------------------

    components = dev39.add_pet_quality_score(
        components
    )

    # -----------------------------------------------------------------
    # Apply EXACT Dev39 eligibility gate
    #
    # Winner was:
    #     hot_fraction >= 0.40
    #
    # Selection remains completely GT-free.
    # -----------------------------------------------------------------

    eligible = components[
        components["dev39_hot_fraction"] >= HOT_FRACTION_GATE
    ].copy()

    print(f"Eligible components : {len(eligible)}")

    # -----------------------------------------------------------------
    # Exact Dev39 ranking
    # -----------------------------------------------------------------

    eligible = eligible.sort_values(
        ["dev39_pet_quality", "dev39_mean_suv"],
        ascending=[False, False]
    ).reset_index(drop=True)

    eligible["rank"] = np.arange(
        1,
        len(eligible) + 1
    )

    eligible["selected"] = (
        eligible["rank"] <= TOP_K
    )

    # -----------------------------------------------------------------
    # Audit every eligible component
    # -----------------------------------------------------------------

    print("\n[5] Auditing all eligible components...")

    audit_rows = []
    core_rows = []

    for _, row in eligible.iterrows():

        label = int(row["label"])

        component_mask = (
            cc == label
        )

        # -------------------------------------------------------------
        # Component-level GT audit
        # -------------------------------------------------------------

        comp_gt = component_metrics(
            component_mask,
            gt
        )

        z_start, z_end, z_span = z_extent(
            component_mask
        )

        # -------------------------------------------------------------
        # Core generated from this component alone.
        #
        # This is post-hoc audit only.
        # GT does not influence core construction.
        # -------------------------------------------------------------

        core_mask = safe_build_core(
            cc,
            suv,
            label,
            CORE_POLICY
        )

        core_gt = component_metrics(
            core_mask,
            gt
        )

        core_z_start, core_z_end, core_z_span = z_extent(
            core_mask
        )

        # -------------------------------------------------------------
        # Component-level audit row
        # -------------------------------------------------------------

        audit_rows.append({

            # Identification / ranking
            "rank": int(row["rank"]),
            "label": label,
            "selected": bool(row["selected"]),

            "voxels": int(row["dev39_voxels"]),
            "slice_count": int(row["dev39_slice_count"]),
            "z_start": z_start,
            "z_end": z_end,
            "z_span": z_span,
            "longest_run": int(row["dev39_longest_run"]),
            "persistence": float(row["dev39_persistence"]),
            "compactness": float(row["dev39_compactness"]),

            # PET characteristics
            "mean_suv": float(row["dev39_mean_suv"]),
            "max_suv": float(row["dev39_max_suv"]),
            "hot_fraction": float(row["dev39_hot_fraction"]),

            # Exact Dev39 quality score
            "dev39_pet_quality": float(
                row["dev39_pet_quality"]
            ),

            # Post-hoc GT audit
            **comp_gt,

            # Core audit
            "core_voxels": int(core_gt["component_voxels"]),
            "core_gt_overlap_voxels": int(
                core_gt["gt_overlap_voxels"]
            ),
            "core_gt_precision": float(
                core_gt["gt_precision"]
            ),
            "core_gt_recall": float(
                core_gt["gt_recall"]
            ),
            "core_dice": float(
                core_gt["component_dice"]
            ),
            "core_iou": float(
                core_gt["component_iou"]
            ),

            # Core geometry
            "core_z_start": core_z_start,
            "core_z_end": core_z_end,
            "core_z_span": core_z_span,
        })

        # -------------------------------------------------------------
        # Dedicated core audit row
        # -------------------------------------------------------------

        core_rows.append({

            "rank": int(row["rank"]),
            "label": label,
            "selected": bool(row["selected"]),

            "component_voxels": int(row["dev39_voxels"]),
            "core_voxels": int(core_mask.sum()),

            "gt_voxels": int(gt.sum()),

            "component_gt_overlap": int(
                comp_gt["gt_overlap_voxels"]
            ),

            "core_gt_overlap": int(
                core_gt["gt_overlap_voxels"]
            ),

            "component_dice": float(
                comp_gt["component_dice"]
            ),

            "core_dice": float(
                core_gt["component_dice"]
            ),

            "component_iou": float(
                comp_gt["component_iou"]
            ),

            "core_iou": float(
                core_gt["component_iou"]
            ),

            "component_precision": float(
                comp_gt["gt_precision"]
            ),

            "core_precision": float(
                core_gt["gt_precision"]
            ),

            "component_recall": float(
                comp_gt["gt_recall"]
            ),

            "core_recall": float(
                core_gt["gt_recall"]
            ),
        })

    audit_df = pd.DataFrame(audit_rows)

    core_df = pd.DataFrame(core_rows)

    # -----------------------------------------------------------------
    # Selected component table
    # -----------------------------------------------------------------

    selected_df = audit_df[
        audit_df["selected"] == True
    ].copy()

    # -----------------------------------------------------------------
    # Save CSVs
    # -----------------------------------------------------------------

    all_path = (
        OUT_DIR
        / "dev54_all_eligible_components.csv"
    )

    selected_path = (
        OUT_DIR
        / "dev54_selected_components.csv"
    )

    core_path = (
        OUT_DIR
        / "dev54_core_audit.csv"
    )

    audit_df.to_csv(
        all_path,
        index=False
    )

    selected_df.to_csv(
        selected_path,
        index=False
    )

    core_df.to_csv(
        core_path,
        index=False
    )

    # -----------------------------------------------------------------
    # Print complete audit table
    # -----------------------------------------------------------------

    print("\n" + "=" * 120)
    print("ALL ELIGIBLE COMPONENTS")
    print("=" * 120)

    display_columns = [
        "rank",
        "label",
        "selected",
        "voxels",
        "mean_suv",
        "max_suv",
        "hot_fraction",
        "persistence",
        "longest_run",
        "compactness",
        "dev39_pet_quality",
        "gt_overlap_voxels",
        "gt_precision",
        "gt_recall",
        "component_dice",
        "component_iou",
        "core_voxels",
        "core_gt_overlap_voxels",
        "core_dice",
        "core_iou",
    ]

    print(
        audit_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # -----------------------------------------------------------------
    # Selected summary
    # -----------------------------------------------------------------

    print("\n" + "=" * 100)
    print("SELECTED TOP-K COMPONENTS")
    print("=" * 100)

    print(
        selected_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # -----------------------------------------------------------------
    # Best GT-overlap components
    #
    # This is diagnostic only.
    # It must NOT be interpreted as part of the selection pipeline.
    # -----------------------------------------------------------------

    best_gt = audit_df.sort_values(
        "component_dice",
        ascending=False
    ).head(5)

    print("\n" + "=" * 100)
    print("TOP 5 COMPONENTS BY POST-HOC GT DICE")
    print("(Diagnostic only — NOT used by Dev39 selection)")
    print("=" * 100)

    print(
        best_gt[
            [
                "rank",
                "label",
                "selected",
                "voxels",
                "mean_suv",
                "max_suv",
                "hot_fraction",
                "dev39_pet_quality",
                "gt_overlap_voxels",
                "component_dice",
                "component_iou",
                "core_dice",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # -----------------------------------------------------------------
    # Rank of best GT component
    # -----------------------------------------------------------------

    best_component = audit_df.iloc[
        audit_df["component_dice"].argmax()
    ]

    best_core = audit_df.iloc[
        audit_df["core_dice"].argmax()
    ]

    # -----------------------------------------------------------------
    # Validation audit report
    # -----------------------------------------------------------------

    report_path = (
        OUT_DIR
        / "dev54_validation_audit.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "DEV54 — VALIDATION COMPONENT AUDIT\n"
        )
        f.write("=" * 80 + "\n\n")

        f.write(
            "This is a post-hoc audit of the frozen Dev39 "
            "validation pipeline.\n"
        )
        f.write(
            "Ground truth was NOT used for component "
            "selection or ranking.\n\n"
        )

        f.write(
            f"Case: {CASE_NAME}\n"
        )
        f.write(
            f"PET threshold: {PET_THRESHOLD}\n"
        )
        f.write(
            f"Hot threshold: {HOT_THRESHOLD}\n"
        )
        f.write(
            f"3D minimum voxels: {MIN_3D_VOXELS}\n"
        )
        f.write(
            f"3D minimum slices: {MIN_3D_SLICES}\n"
        )
        f.write(
            f"Hot fraction gate: {HOT_FRACTION_GATE}\n"
        )
        f.write(
            f"Top-K: {TOP_K}\n"
        )
        f.write(
            f"Core policy: {CORE_POLICY}\n\n"
        )

        f.write(
            f"Candidate voxels: {candidate_voxels}\n"
        )
        f.write(
            f"Total 3D components: {len(components)}\n"
        )
        f.write(
            f"Eligible components: {len(eligible)}\n"
        )
        f.write(
            f"Selected components: {len(selected_df)}\n"
        )
        f.write(
            f"GT voxels: {int(gt.sum())}\n\n"
        )

        f.write(
            "Selected labels:\n"
        )

        for _, r in selected_df.iterrows():
            f.write(
                f"  Rank {int(r['rank'])}: "
                f"label {int(r['label'])}\n"
            )

        f.write("\n")

        f.write(
            "Best component by post-hoc GT Dice:\n"
        )
        f.write(
            f"  Rank: {int(best_component['rank'])}\n"
        )
        f.write(
            f"  Label: {int(best_component['label'])}\n"
        )
        f.write(
            f"  Selected by Dev39: "
            f"{bool(best_component['selected'])}\n"
        )
        f.write(
            f"  Component Dice: "
            f"{best_component['component_dice']:.6f}\n"
        )
        f.write(
            f"  Component IoU: "
            f"{best_component['component_iou']:.6f}\n"
        )
        f.write(
            f"  GT overlap: "
            f"{int(best_component['gt_overlap_voxels'])}\n"
        )

        f.write("\n")

        f.write(
            "Best component by post-hoc P70 core Dice:\n"
        )
        f.write(
            f"  Rank: {int(best_core['rank'])}\n"
        )
        f.write(
            f"  Label: {int(best_core['label'])}\n"
        )
        f.write(
            f"  Selected by Dev39: "
            f"{bool(best_core['selected'])}\n"
        )
        f.write(
            f"  Core Dice: "
            f"{best_core['core_dice']:.6f}\n"
        )
        f.write(
            f"  Core IoU: "
            f"{best_core['core_iou']:.6f}\n"
        )

        f.write("\n\n")

        f.write(
            "FULL COMPONENT AUDIT\n"
        )
        f.write("=" * 120 + "\n\n"
        )

        f.write(
            audit_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}"
            )
        )

        f.write("\n")

    # -----------------------------------------------------------------
    # Final console summary
    # -----------------------------------------------------------------

    print("\n" + "=" * 80)
    print("DEV54 COMPLETE")
    print("=" * 80)

    print(f"All eligible CSV     : {all_path}")
    print(f"Selected CSV         : {selected_path}")
    print(f"Core audit CSV       : {core_path}")
    print(f"Audit report         : {report_path}")

    print("\nSelected labels:")
    print(
        selected_df[
            ["rank", "label"]
        ].to_string(index=False)
    )

    print("\nBest post-hoc component:")
    print(
        f"  Rank                : "
        f"{int(best_component['rank'])}"
    )
    print(
        f"  Label               : "
        f"{int(best_component['label'])}"
    )
    print(
        f"  Selected by Dev39   : "
        f"{bool(best_component['selected'])}"
    )
    print(
        f"  Component Dice      : "
        f"{best_component['component_dice']:.6f}"
    )

    print("\nBest post-hoc P70 core:")
    print(
        f"  Rank                : "
        f"{int(best_core['rank'])}"
    )
    print(
        f"  Label               : "
        f"{int(best_core['label'])}"
    )
    print(
        f"  Selected by Dev39   : "
        f"{bool(best_core['selected'])}"
    )
    print(
        f"  Core Dice           : "
        f"{best_core['core_dice']:.6f}"
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()