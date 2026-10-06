from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk


ROOT = Path("development_cases/PETCT_0b57b247b6")

PT_UID = "1.3.6.1.4.1.14519.5.2.1.269602498229315711806794469221706441962"
SEG_UID = "1.3.6.1.4.1.14519.5.2.1.181267673274669528392473579178665773480"

OUT_PATH = ROOT / "tumor_mask_pet_aligned.nii.gz"


# ---------------------------------------------------------
# Find PET and SEG files
# ---------------------------------------------------------
pet_files = []
seg_file = None

for path in ROOT.rglob("*.dcm"):
    try:
        ds = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True
        )

        uid = getattr(ds, "SeriesInstanceUID", None)

        if uid == PT_UID:
            pet_files.append(path)

        elif uid == SEG_UID:
            seg_file = path

    except Exception:
        pass


if len(pet_files) != 326:
    raise RuntimeError(
        f"Expected 326 PET files, found {len(pet_files)}"
    )

if seg_file is None:
    raise RuntimeError("SEG file not found.")


# ---------------------------------------------------------
# Load PET geometry
# ---------------------------------------------------------
reader = sitk.ImageSeriesReader()

pet_names = reader.GetGDCMSeriesFileNames(
    str(pet_files[0].parent),
    PT_UID
)

reader.SetFileNames(pet_names)
pet_img = reader.Execute()


# ---------------------------------------------------------
# Read SEG pixel data
# ---------------------------------------------------------
seg = pydicom.dcmread(
    str(seg_file),
    force=True
)

seg_array = seg.pixel_array

print("===== CASE 4 SEG ALIGNMENT =====")
print("Native SEG shape:", seg_array.shape)
print("Native SEG foreground voxels:",
      int(np.count_nonzero(seg_array)))


if seg_array.shape != (326, 400, 400):
    raise RuntimeError(
        f"Unexpected SEG shape: {seg_array.shape}"
    )


# ---------------------------------------------------------
# Flip Y axis
#
# SEG orientation:
# [1, 0, 0, 0, -1, 0]
#
# PET orientation:
# identity
#
# Therefore reverse the second array dimension.
# ---------------------------------------------------------
aligned = seg_array[:, ::-1, :]

aligned = (aligned > 0).astype(np.uint8)


# ---------------------------------------------------------
# Convert to SimpleITK image
# ---------------------------------------------------------
mask_img = sitk.GetImageFromArray(aligned)

mask_img.SetSpacing(pet_img.GetSpacing())
mask_img.SetOrigin(pet_img.GetOrigin())
mask_img.SetDirection(pet_img.GetDirection())


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
sitk.WriteImage(mask_img, str(OUT_PATH))


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------
foreground = int(np.count_nonzero(aligned))

positive_slices = np.where(
    np.any(aligned > 0, axis=(1, 2))
)[0]

voxel_volume_cm3 = np.prod(pet_img.GetSpacing()) / 1000.0
volume_cm3 = foreground * voxel_volume_cm3


print()
print("===== ALIGNED PET MASK =====")
print("Saved:", OUT_PATH)
print("PET size:", pet_img.GetSize())
print("PET spacing:", pet_img.GetSpacing())
print("Foreground voxels:", foreground)
print("Positive PET slices:", len(positive_slices))

if len(positive_slices):
    print(
        "PET slice range:",
        int(positive_slices.min()),
        "to",
        int(positive_slices.max())
    )

print("Volume (cm³):", round(float(volume_cm3), 3))