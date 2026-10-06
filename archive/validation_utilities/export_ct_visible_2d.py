from pathlib import Path
import numpy as np
import SimpleITK as sitk

CT_DIR = Path("verified_case/ct_2d")
MASK_3D = Path("verified_case/tumor_mask_ct_visible.nii.gz")
OUT_DIR = Path("verified_case/masks_ct_visible_2d")

OUT_DIR.mkdir(parents=True, exist_ok=True)

mask = sitk.ReadImage(str(MASK_3D))
mask_np = sitk.GetArrayFromImage(mask)

positive = 0

for z in range(mask_np.shape[0]):

    binary = (mask_np[z] > 0).astype(np.uint8) * 255

    if binary.any():
        positive += 1

    output = OUT_DIR / f"slice_{z:03d}.png"

    sitk.WriteImage(
        sitk.GetImageFromArray(binary),
        str(output)
    )

print("=== CT-VISIBLE MASK EXPORT ===")
print("Total slices:", mask_np.shape[0])
print("Positive slices:", positive)
print("Output:", OUT_DIR)
