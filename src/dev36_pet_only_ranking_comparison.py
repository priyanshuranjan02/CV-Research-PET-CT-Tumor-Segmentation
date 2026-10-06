"""
Dev36 - PET-only Ranking Comparison

Purpose
-------
Compare alternative ranking strategies on the SAME PET-only candidate
volume and SAME Dev10 3D components.

Pipeline:

    PET >= 2.25
          |
          v
    Dev10 3D components
          |
          v
    Alternative ranking
          |
          v
       Top-4
          |
          v
        P70
          |
          v
    Final evaluation

IMPORTANT
---------
- dev10_lite_hot_core.py is NOT modified.
- GT is NEVER used for ranking.
- GT is used only for diagnostic auditing.
- Candidate generation is identical for every policy.
- 3D component construction is identical for every policy.
- Core construction is identical for every policy.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import dev10_lite_hot_core as dev10


# ============================================================
# CONFIG
# ============================================================

CASES = [
    "PETCT_0168f65af8",
    "PETCT_04606080a0",
    "PETCT_04ab5c61c9",
    "PETCT_0b57b247b6",
    "PETCT_11afab3485",
    "PETCT_185da4c8b6",
]

PET_THRESHOLD = 2.25
TOP_K = 4
CORE_POLICY = "P70"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "development_cases"
    / "dev36_pet_only_ranking_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SAFE NUMERIC HELPERS
# ============================================================

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0


def safe_int(x):
    try:
        return int(x)
    except Exception:
        return 0


# ============================================================
# PET-ONLY CANDIDATE
# ============================================================

def build_pet_only(case):
    """
    Exact PET-only candidate used in Dev34/Dev35.
    """

    suv = case["suv"]

    return (suv >= PET_THRESHOLD).astype(np.uint8)


# ============================================================
# RANKING FEATURES
# ============================================================

def add_ranking_features(components):
    """
    Add ranking features using ONLY component/PET information.

    No GT is used.
    """

    df = components.copy()

    # --------------------------------------------------------
    # Basic numerical safety
    # --------------------------------------------------------

    for col in [
        "mean_suv",
        "max_suv",
        "frac_ge_3",
        "mean_suv_hot3_a0p5",
        "voxels",
    ]:
        if col not in df.columns:
            df[col] = 0.0

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).fillna(0.0)

    # --------------------------------------------------------
    # Locate the component-size column
    # --------------------------------------------------------

    if "voxels" not in df.columns:

        if "voxel_count" in df.columns:
            df["voxels"] = df["voxel_count"]

        elif "component_voxels" in df.columns:
            df["voxels"] = df["component_voxels"]

        else:
            raise KeyError(
                "Could not find component voxel-count column."
            )

    # --------------------------------------------------------
    # Additional PET-only scores
    # --------------------------------------------------------

    df["rank_mean_suv"] = df["mean_suv"]

    df["rank_frac_ge_3"] = df["frac_ge_3"]

    df["rank_max_suv"] = df["max_suv"]

    df["rank_volume"] = np.log1p(
        df["voxels"]
    )

    # Mean SUV weighted by component size.
    #
    # log(volume) prevents very large components from
    # completely dominating.
    df["rank_mean_suv_logvol"] = (
        df["mean_suv"]
        * np.log1p(df["voxels"])
    )

    # Hot fraction × mean SUV.
    df["rank_hot_intensity"] = (
        df["frac_ge_3"]
        * df["mean_suv"]
    )

    # Original Dev10 feature.
    df["rank_dev10"] = df["mean_suv_hot3_a0p5"]

    # Combined PET score.
    #
    # Normalize each feature across components first.
    def normalize(series):

        s_min = series.min()
        s_max = series.max()

        if s_max - s_min < 1e-12:
            return pd.Series(
                np.zeros(len(series)),
                index=series.index,
            )

        return (
            (series - s_min)
            / (s_max - s_min)
        )

    n_mean = normalize(df["mean_suv"])
    n_hot = normalize(df["frac_ge_3"])
    n_max = normalize(df["max_suv"])
    n_vol = normalize(np.log1p(df["voxels"]))

    df["rank_combined"] = (
        0.40 * n_mean
        + 0.30 * n_hot
        + 0.20 * n_max
        + 0.10 * n_vol
    )

    return df


# ============================================================
# CORE AUDIT
# ============================================================

def get_core_mask(cc, suv, label, policy):
    """
    Correctly handle the Dev10 build_core return value.

    Dev10 build_core returns a tuple, not a bare ndarray.
    """

    result = dev10.build_core(
        cc,
        suv,
        int(label),
        policy,
    )

    if isinstance(result, tuple):

        # The first ndarray in the tuple is the core mask.
        for item in result:

            if isinstance(item, np.ndarray):
                return item.astype(bool)

        raise RuntimeError(
            "build_core returned a tuple but no ndarray "
            "core mask was found."
        )

    return result.astype(bool)


# ============================================================
# GT AUDIT
# ============================================================

def audit_ranking(
    case,
    ranked_df,
    cc,
):
    """
    Diagnostic audit.

    GT is NOT involved in ranking.

    It is only used to determine:
    - whether a component overlaps GT
    - component Dice
    - whether Top-4 contains GT
    - P70 core Dice
    """

    suv = case["suv"]
    gt = case["gt"] > 0

    rows = []

    for rank_idx, row in ranked_df.iterrows():

        label = safe_int(row["label"])

        component_mask = (
            cc == label
        )

        component_voxels = int(
            component_mask.sum()
        )

        if component_voxels == 0:
            continue

        gt_overlap = int(
            np.logical_and(
                component_mask,
                gt,
            ).sum()
        )

        gt_voxels = int(gt.sum())

        denom = (
            component_voxels
            + gt_voxels
        )

        if denom > 0:
            component_dice = (
                2.0 * gt_overlap / denom
            )
        else:
            component_dice = 0.0

        # ----------------------------------------------------
        # Exact P70 core
        # ----------------------------------------------------

        core_mask = get_core_mask(
            cc,
            suv,
            label,
            CORE_POLICY,
        )

        core_voxels = int(
            core_mask.sum()
        )

        core_gt_overlap = int(
            np.logical_and(
                core_mask,
                gt,
            ).sum()
        )

        core_denom = (
            core_voxels
            + gt_voxels
        )

        if core_denom > 0:
            core_dice = (
                2.0
                * core_gt_overlap
                / core_denom
            )
        else:
            core_dice = 0.0

        rows.append({
            "case": case["name"],
            "label": label,
            "rank": rank_idx + 1,

            "component_voxels":
                component_voxels,

            "mean_suv":
                safe_float(row["mean_suv"]),

            "max_suv":
                safe_float(row["max_suv"]),

            "frac_ge_3":
                safe_float(row["frac_ge_3"]),

            "mean_suv_hot3_a0p5":
                safe_float(
                    row["mean_suv_hot3_a0p5"]
                ),

            "gt_overlap_voxels":
                gt_overlap,

            "component_gt_dice":
                component_dice,

            "selected_top4":
                rank_idx < TOP_K,

            "core_voxels":
                core_voxels,

            "core_gt_overlap_voxels":
                core_gt_overlap,

            "core_gt_dice":
                core_dice,

            "gt_overlapping_component":
                gt_overlap > 0,
        })

    return pd.DataFrame(rows)


# ============================================================
# ONE RANKING POLICY
# ============================================================

def evaluate_policy(
    case,
    components,
    cc,
    policy_name,
    ranking_column,
):
    """
    Evaluate one ranking policy with exact Dev10 downstream logic.
    """

    ranked = components.copy()

    # --------------------------------------------------------
    # Sort exactly as Dev10 does:
    #
    # ranking feature descending
    # mean_suv descending
    # --------------------------------------------------------

    ranked = ranked.sort_values(
        [
            ranking_column,
            "mean_suv",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Exact Dev10 evaluation
    # --------------------------------------------------------

    metrics = dev10.evaluate_selection(
        case,
        ranked,
        cc,
        ranking_column,
        CORE_POLICY,
        TOP_K,
    )

    # --------------------------------------------------------
    # Ranking audit
    # --------------------------------------------------------

    audit = audit_ranking(
        case,
        ranked,
        cc,
    )

    audit["policy"] = policy_name

    # --------------------------------------------------------
    # Diagnostic values
    # --------------------------------------------------------

    gt_components = audit[
        audit["gt_overlapping_component"]
    ]

    top4_gt = audit[
        audit["selected_top4"]
        & audit["gt_overlapping_component"]
    ]

    if len(gt_components) > 0:
        best_rank = int(
            gt_components["rank"].min()
        )

        best_component_dice = float(
            gt_components[
                "component_gt_dice"
            ].max()
        )

        best_core_dice = float(
            gt_components[
                "core_gt_dice"
            ].max()
        )

    else:
        best_rank = np.nan
        best_component_dice = 0.0
        best_core_dice = 0.0

    case_summary = {
        "case": case["name"],
        "policy": policy_name,

        "num_components":
            len(components),

        "num_gt_components":
            len(gt_components),

        "gt_components_in_top4":
            len(top4_gt),

        "best_gt_component_rank":
            best_rank,

        "best_component_gt_dice":
            best_component_dice,

        "best_core_gt_dice":
            best_core_dice,

        "final_dice":
            safe_float(
                metrics.get("dice")
            ),

        "final_iou":
            safe_float(
                metrics.get("iou")
            ),

        "final_slice_recall":
            safe_float(
                metrics.get("slice_recall")
            ),

        "final_fpr":
            safe_float(
                metrics.get("fpr")
            ),

        "prediction_voxels":
            safe_int(
                metrics.get(
                    "prediction_voxels"
                )
            ),
    }

    return audit, case_summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("DEV36 - PET-ONLY RANKING COMPARISON")
    print("=" * 90)

    print("\nConfiguration")
    print("----------------")
    print(f"PET threshold : {PET_THRESHOLD}")
    print(f"Top-K         : {TOP_K}")
    print(f"Core policy   : {CORE_POLICY}")

    all_audits = []
    all_summaries = []

    # --------------------------------------------------------
    # Ranking policies
    # --------------------------------------------------------

    POLICIES = {
        "dev10":
            "rank_dev10",

        "mean_suv":
            "rank_mean_suv",

        "frac_ge_3":
            "rank_frac_ge_3",

        "max_suv":
            "rank_max_suv",

        "volume":
            "rank_volume",

        "mean_suv_logvol":
            "rank_mean_suv_logvol",

        "hot_intensity":
            "rank_hot_intensity",

        "combined":
            "rank_combined",
    }

    # --------------------------------------------------------
    # Cases
    # --------------------------------------------------------

    for case_name in CASES:

        print("\n" + "=" * 90)
        print(f"CASE: {case_name}")
        print("=" * 90)

        case = dev10.load_case(
            case_name
        )

        # ----------------------------------------------------
        # SAME PET-only candidate for every policy
        # ----------------------------------------------------

        candidate = build_pet_only(
            case
        )

        print(
            "PET candidate voxels:",
            int(candidate.sum())
        )

        # ----------------------------------------------------
        # SAME Dev10 3D components
        # ----------------------------------------------------

        components, cc = (
            dev10.build_3d_component_table(
                case,
                candidate,
            )
        )

        print(
            "Dev10 3D components:",
            len(components)
        )

        # ----------------------------------------------------
        # Add ranking features
        # ----------------------------------------------------

        components = add_ranking_features(
            components
        )

        # ----------------------------------------------------
        # Evaluate every policy
        # ----------------------------------------------------

        for policy_name, ranking_column in POLICIES.items():

            print(
                f"\n  Running policy: {policy_name}"
            )

            audit, summary = evaluate_policy(
                case,
                components,
                cc,
                policy_name,
                ranking_column,
            )

            all_audits.append(
                audit
            )

            all_summaries.append(
                summary
            )

            # ------------------------------------------------
            # Print compact diagnostic
            # ------------------------------------------------

            print(
                "    best GT rank:",
                summary[
                    "best_gt_component_rank"
                ]
            )

            print(
                "    GT components in Top-4:",
                summary[
                    "gt_components_in_top4"
                ]
            )

            print(
                "    final Dice:",
                f"{summary['final_dice']:.6f}"
            )

    # ========================================================
    # SAVE COMPONENT AUDIT
    # ========================================================

    audit_df = pd.concat(
        all_audits,
        ignore_index=True,
    )

    summary_df = pd.DataFrame(
        all_summaries
    )

    audit_path = (
        OUTPUT_DIR
        / "dev36_component_audit.csv"
    )

    case_path = (
        OUTPUT_DIR
        / "dev36_case_policy_summary.csv"
    )

    audit_df.to_csv(
        audit_path,
        index=False,
    )

    summary_df.to_csv(
        case_path,
        index=False,
    )

    # ========================================================
    # MACRO SUMMARY
    # ========================================================

    macro = (
        summary_df
        .groupby("policy")
        .agg(
            macro_dice=(
                "final_dice",
                "mean",
            ),

            macro_iou=(
                "final_iou",
                "mean",
            ),

            macro_slice_recall=(
                "final_slice_recall",
                "mean",
            ),

            macro_fpr=(
                "final_fpr",
                "mean",
            ),

            total_prediction_voxels=(
                "prediction_voxels",
                "sum",
            ),

            mean_gt_components_in_top4=(
                "gt_components_in_top4",
                "mean",
            ),

            mean_best_gt_rank=(
                "best_gt_component_rank",
                "mean",
            ),

            mean_best_component_dice=(
                "best_component_gt_dice",
                "mean",
            ),

            mean_best_core_dice=(
                "best_core_gt_dice",
                "mean",
            ),
        )
        .reset_index()
    )

    macro = macro.sort_values(
        "macro_dice",
        ascending=False,
    )

    macro_path = (
        OUTPUT_DIR
        / "dev36_macro_summary.csv"
    )

    macro.to_csv(
        macro_path,
        index=False,
    )

    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print("\n")
    print("=" * 90)
    print("DEV36 COMPLETE")
    print("=" * 90)

    print("\nMACRO RESULTS")
    print("-" * 90)

    print(
        macro.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(audit_path)
    print(case_path)
    print(macro_path)


if __name__ == "__main__":
    main()