from pathlib import Path
import numpy as np
import SimpleITK as sitk

ct_root = Path("verified_case/ct")
mask_path = Path("verified_case/tumor_mask_ct.nii.gz")

ct_out = Path("verified_case/ct_2d")
mask_out = Path("verified_case/masks_2d")

ct_out.mkdir(parents=True, exist_ok=True)
mask_out.mkdir(parents=True, exist_ok=True)

# Find and sort CT DICOM files physically
import pydicom

ct_files = []

for p in ct_root.rglob("*.dcm"):
    if "__MACOSX" in str(p):
        continue

    ds = pydicom.dcmread(p, stop_before_pixels=True)

    if getattr(ds, "Modality", None) == "CT":
        ct_files.append(p)

ct_files.sort(
    key=lambda p: float(
        pydicom.dcmread(p, stop_before_pixels=True).ImagePositionPatient[2]
    )
)

# Read CT volume
reader = sitk.ImageSeriesReader()
reader.SetFileNames([str(p) for p in ct_files])
ct = reader.Execute()

# Read expert mask
mask = sitk.ReadImage(str(mask_path))
mask = sitk.Cast(mask > 0, sitk.sitkUInt8)

ct_np = sitk.GetArrayFromImage(ct)
mask_np = sitk.GetArrayFromImage(mask)

print("CT shape:", ct_np.shape)
print("Mask shape:", mask_np.shape)

positive_count = 0

for z in range(ct_np.shape[0]):

    # CT in Hounsfield Units -> display/input representation
    ct_slice = ct_np[z]

    ct_display = np.clip(ct_slice, -200, 300)
    ct_display = ((ct_display + 200) / 500 * 255).astype(np.uint8)

    # Binary expert mask
    expert_mask = (mask_np[z] > 0).astype(np.uint8) * 255

    # Use identical index for CT and mask
    ct_path = ct_out / f"slice_{z:03d}.tif"
    mask_path_out = mask_out / f"slice_{z:03d}.png"

    sitk.WriteImage(
        sitk.GetImageFromArray(ct_display),
        str(ct_path)
    )

    sitk.WriteImage(
        sitk.GetImageFromArray(expert_mask),
        str(mask_path_out)
    )

    if expert_mask.any():
        positive_count += 1

print("\nExport complete.")
print("Total CT slices:", ct_np.shape[0])
print("Positive slices:", positive_count)
print("CT directory:", ct_out)
print("Mask directory:", mask_out)
