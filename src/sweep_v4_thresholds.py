from pathlib import Path

import numpy as np
import pandas as pd


AUDIT = Path(
    "development_cases/"
    "v3_candidate_gt_audit_all_cases.csv"
)


df = pd.read_csv(AUDIT)

print("===== V4 THRESHOLD SWEEP =====")
print("Candidates:", len(df))


# ---------------------------------------------------------
# Define candidate-level positives
# ---------------------------------------------------------

df["positive_01"] = (
    df["candidate_dice"] >= 0.10
)

df["positive_03"] = (
    df["candidate_dice"] >= 0.30
)


# ---------------------------------------------------------
# Threshold grids
# ---------------------------------------------------------

SUV_THRESHOLDS = [
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
]

DIFF_THRESHOLDS = [
    0.00,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.75,
]

RATIO_THRESHOLDS = [
    1.00,
    1.10,
    1.20,
    1.30,
    1.40,
    1.50,
    1.60,
    1.75,
    2.00,
    2.25,
    2.50,
]


# ---------------------------------------------------------
# Evaluate one threshold combination
# ---------------------------------------------------------

def evaluate(
    selected,
    truth
):

    tp = int(
        np.logical_and(
            selected,
            truth
        ).sum()
    )

    fp = int(
        np.logical_and(
            selected,
            ~truth
        ).sum()
    )

    fn = int(
        np.logical_and(
            ~selected,
            truth
        ).sum()
    )

    selected_count = int(
        selected.sum()
    )

    truth_count = int(
        truth.sum()
    )

    precision = (
        tp / selected_count
        if selected_count > 0
        else 0.0
    )

    recall = (
        tp / truth_count
        if truth_count > 0
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return (
        tp,
        fp,
        fn,
        selected_count,
        precision,
        recall,
        f1
    )


# ---------------------------------------------------------
# Sweep
# ---------------------------------------------------------

results_01 = []
results_03 = []

for suv in SUV_THRESHOLDS:

    for diff in DIFF_THRESHOLDS:

        for ratio in RATIO_THRESHOLDS:

            selected = (
                (df["candidate_suv"] >= suv)
                &
                (df["local_difference"] >= diff)
                &
                (df["local_ratio"] >= ratio)
            ).to_numpy()

            # Dice >= 0.10 target
            (
                tp,
                fp,
                fn,
                count,
                precision,
                recall,
                f1
            ) = evaluate(
                selected,
                df["positive_01"].to_numpy()
            )

            results_01.append(
                {
                    "suv": suv,
                    "difference": diff,
                    "ratio": ratio,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "selected": count,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )

            # Dice >= 0.30 target
            (
                tp,
                fp,
                fn,
                count,
                precision,
                recall,
                f1
            ) = evaluate(
                selected,
                df["positive_03"].to_numpy()
            )

            results_03.append(
                {
                    "suv": suv,
                    "difference": diff,
                    "ratio": ratio,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "selected": count,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )


r01 = pd.DataFrame(
    results_01
)

r03 = pd.DataFrame(
    results_03
)


# ---------------------------------------------------------
# Remove useless configurations
# ---------------------------------------------------------

r01 = r01[
    r01["selected"] > 0
]

r03 = r03[
    r03["selected"] > 0
]


# ---------------------------------------------------------
# Top configurations
# ---------------------------------------------------------

print()
print("=" * 70)
print("TOP THRESHOLDS — CANDIDATE DICE >= 0.10")
print("=" * 70)

top01 = r01.sort_values(
    ["f1", "precision", "recall"],
    ascending=False
).head(20)

print(
    top01.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print()
print("=" * 70)
print("TOP THRESHOLDS — CANDIDATE DICE >= 0.30")
print("=" * 70)

top03 = r03.sort_values(
    ["f1", "precision", "recall"],
    ascending=False
).head(20)

print(
    top03.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ---------------------------------------------------------
# Best precision configurations with recall constraint
# ---------------------------------------------------------

print()
print("=" * 70)
print("HIGH-PRECISION OPTIONS — Dice >= 0.10")
print("=" * 70)

for minimum_recall in [
    0.25,
    0.50,
    0.75
]:

    subset = r01[
        r01["recall"] >= minimum_recall
    ]

    if len(subset) == 0:
        print(
            f"Recall >= {minimum_recall:.2f}: none"
        )
        continue

    best = subset.sort_values(
        ["precision", "f1"],
        ascending=False
    ).iloc[0]

    print()
    print(
        f"Recall >= {minimum_recall:.2f}"
    )
    print(
        best.to_string()
    )


# ---------------------------------------------------------
# Save sweep
# ---------------------------------------------------------

out01 = Path(
    "development_cases/"
    "v4_threshold_sweep_dice01.csv"
)

out03 = Path(
    "development_cases/"
    "v4_threshold_sweep_dice03.csv"
)

r01.to_csv(
    out01,
    index=False
)

r03.to_csv(
    out03,
    index=False
)


print()
print("=" * 70)
print("SWEEP COMPLETE")
print("=" * 70)

print(
    "Saved:",
    out01
)

print(
    "Saved:",
    out03
)