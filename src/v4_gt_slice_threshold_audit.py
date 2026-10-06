from pathlib import Path
import pandas as pd
import SimpleITK as sitk
import numpy as np

CASES = [
    (
        "PETCT_0168f65af8",
        "development_cases/PETCT_0168f65af8/"
        "manifest-1789661648760/FDG-PET-CT-Lesions/"
        "PETCT_0168f65af8/hybrid_v4_balanced_candidate_scores.csv",
        "development_cases/PETCT_0168f65af8/"
        "tumor_mask_ct_visible.nii.gz",
    ),
    (
        "PETCT_04606080a0",
        "development_cases/PETCT_04606080a0/"
        "FDG-PET-CT-Lesions/PETCT_04606080a0/"
        "hybrid_v4_balanced_candidate_scores.csv",
        "development_cases/PETCT_04606080a0/"
        "tumor_mask_ct_visible.nii.gz",
    ),
    (
        "PETCT_04ab5c61c9",
        "development_cases/PETCT_04ab5c61c9/"
        "manifest-1789918980481/FDG-PET-CT-Lesions/"
        "PETCT_04ab5c61c9/hybrid_v4_balanced_candidate_scores.csv",
        "development_cases/PETCT_04ab5c61c9/"
        "tumor_mask_ct_visible.nii.gz",
    ),
    (
        "PETCT_0b57b247b6",
        "development_cases/PETCT_0b57b247b6/"
        "manifest-1789920519449/FDG-PET-CT-Lesions/"
        "PETCT_0b57b247b6/hybrid_v4_balanced_candidate_scores.csv",
        "development_cases/PETCT_0b57b247b6/"
        "tumor_mask_ct_visible.nii.gz",
    ),
    (
        "PETCT_11afab3485",
        "development_cases/PETCT_11afab3485/"
        "manifest-1789921915902/FDG-PET-CT-Lesions/"
        "PETCT_11afab3485/hybrid_v4_balanced_candidate_scores.csv",
        "development_cases/PETCT_11afab3485/"
        "tumor_mask_ct_visible.nii.gz",
    ),
    (
        "PETCT_185da4c8b6",
        "development_cases/PETCT_185da4c8b6/"
        "manifest-1789923055836/FDG-PET-CT-Lesions/"
        "PETCT_185da4c8b6/hybrid_v4_balanced_candidate_scores.csv",
        "development_cases/PETCT_185da4c8b6/"
        "tumor_mask_ct_visible.nii.gz",
    ),
]

MIN_LOCAL_RATIO = 1.6
MIN_LOCAL_DIFFERENCE = 0.5
MIN_CANDIDATE_SUV = 1.0

print("\n===== V4 GT-SLICE THRESHOLD AUDIT =====\n")

summary = []

for case, csv_path, gt_path in CASES:

    print("=" * 72)
    print(case)
    print("=" * 72)

    csv_path = Path(csv_path)
    gt_path = Path(gt_path)

    if not csv_path.exists():
        print("CSV NOT FOUND:", csv_path)
        print()
        continue

    if not gt_path.exists():
        print("GT NOT FOUND:", gt_path)
        print()
        continue

    df = pd.read_csv(csv_path)

    gt = (
        sitk.GetArrayFromImage(
            sitk.ReadImage(str(gt_path))
        ) > 0
    )

    gt_slices = np.where(
        np.any(gt, axis=(1, 2))
    )[0]

    gt_df = df[
        df["slice"].isin(gt_slices)
    ].copy()

    ratio_pass = (
        gt_df["local_ratio"]
        >= MIN_LOCAL_RATIO
    )

    difference_pass = (
        gt_df["local_difference"]
        >= MIN_LOCAL_DIFFERENCE
    )

    suv_pass = (
        gt_df["candidate_suv"]
        >= MIN_CANDIDATE_SUV
    )

    all_pass = (
        ratio_pass
        & difference_pass
        & suv_pass
    )

    print("GT positive slices:", len(gt_slices))
    print(
        "Candidate rows on GT slices:",
        len(gt_df)
    )
    print(
        "Ratio >= 1.6:",
        int(ratio_pass.sum())
    )
    print(
        "Difference >= 0.5:",
        int(difference_pass.sum())
    )
    print(
        "SUV >= 1.0:",
        int(suv_pass.sum())
    )
    print(
        "ALL THREE PASS:",
        int(all_pass.sum())
    )

    accepted_slices = set(
        df.loc[
            df["accepted"] == 1,
            "slice"
        ].astype(int)
    )

    accepted_gt_slices = [
        int(z)
        for z in gt_slices
        if int(z) in accepted_slices
    ]

    print(
        "GT slices with accepted candidate:",
        len(accepted_gt_slices),
        "/",
        len(gt_slices),
        accepted_gt_slices
    )

    print(
        "\nStrongest candidate on each GT slice "
        "(ranked by difference):"
    )

    for z in gt_slices:

        s = gt_df[
            gt_df["slice"] == z
        ]

        if len(s) == 0:
            continue

        row = s.sort_values(
            [
                "local_difference",
                "local_ratio",
                "candidate_suv"
            ],
            ascending=False
        ).iloc[0]

        print(
            f"  Slice {int(z)}: "
            f"SUV={row['candidate_suv']:.3f}, "
            f"ratio={row['local_ratio']:.3f}, "
            f"diff={row['local_difference']:.3f}, "
            f"area={row['area']:.1f}, "
            f"accepted={int(row['accepted'])}"
        )

    summary.append({
        "case": case,
        "gt_slices": len(gt_slices),
        "candidate_rows": len(gt_df),
        "ratio_pass": int(ratio_pass.sum()),
        "difference_pass": int(difference_pass.sum()),
        "suv_pass": int(suv_pass.sum()),
        "all_three_pass": int(all_pass.sum()),
        "accepted_gt_slices": len(
            accepted_gt_slices
        ),
    })

    print()

print("=" * 72)
print("SUMMARY")
print("=" * 72)

if summary:
    print(
        pd.DataFrame(summary).to_string(
            index=False
        )
    )

print("\nAudit complete.")
