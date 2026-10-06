from pathlib import Path
import pandas as pd
import SimpleITK as sitk
import numpy as np

CASES = [
    ("PETCT_0168f65af8","1789661648760"),
    ("PETCT_04606080a0","1789711980684"),
    ("PETCT_04ab5c61c9","1789918980481"),
    ("PETCT_0b57b247b6","1789920519449"),
    ("PETCT_11afab3485","1789921915902"),
    ("PETCT_185da4c8b6","1789923055836"),
]
BASE = Path("development_cases")

print("\n===== V4 CANDIDATE SLICE AUDIT =====\n")

for case, manifest in CASES:
    csv_path = BASE/case/f"manifest-{manifest}"/"FDG-PET-CT-Lesions"/case/"hybrid_v4_balanced_candidate_scores.csv"
    gt_path = BASE/case/"tumor_mask_ct_visible.nii.gz"

    print("="*70)
    print(case)
    print("="*70)

    if not csv_path.exists():
        print("CSV NOT FOUND:", csv_path, "\n")
        continue
    if not gt_path.exists():
        print("GT NOT FOUND:", gt_path, "\n")
        continue

    df = pd.read_csv(csv_path)
    gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path))) > 0
    gt_slices = np.where(np.any(gt, axis=(1,2)))[0]

    candidate_slices = set(df["slice"].astype(int))
    accepted_df = df[df["accepted"] == 1]
    accepted_slices = set(accepted_df["slice"].astype(int))

    any_candidate = [int(z) for z in gt_slices if int(z) in candidate_slices]
    accepted = [int(z) for z in gt_slices if int(z) in accepted_slices]
    no_candidate = [int(z) for z in gt_slices if int(z) not in candidate_slices]
    rejected = [int(z) for z in gt_slices if int(z) in candidate_slices and int(z) not in accepted_slices]

    print("GT positive slices:", len(gt_slices), gt_slices.tolist())
    print("Total candidate rows:", len(df))
    print("Accepted candidates:", len(accepted_df))
    print("GT slices with ANY candidate:", len(any_candidate), "/", len(gt_slices), any_candidate)
    print("GT slices with ACCEPTED candidate:", len(accepted), "/", len(gt_slices), accepted)
    print("GT slices with NO candidate:", no_candidate)
    print("GT slices with candidate but NONE accepted:", rejected)
    print()

print("\nAudit complete.")