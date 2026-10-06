from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk


ROOT = Path("development_cases/PETCT_04ab5c61c9")

CT_UID = "1.3.6.1.4.1.14519.5.2.1.232341950326768109489919195931758650209"
PT_UID = "1.3.6.1.4.1.14519.5.2.1.184146994116738426472805672529258365051"

PET_MASK = ROOT / "tumor_mask_pet_aligned.nii.gz"
CT_MASK = ROOT / "tumor_mask_ct_raw.nii.gz"


# ---------------------------------------------------------
# Find CT and PET DICOM files
# ---------------------------------------------------------
ct_files = []
pt_files = []

for path in ROOT.rglob("*.dcm"):
    try:
        ds = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True
        )

        uid = getattr(ds, "SeriesInstanceUID", None)

        if uid == CT_UID:
            ct_files.append(path)

        elif uid == PT_UID:
            pt_files.append(path)

    except Exception:
        pass


if len(ct_files) != 340:
    raise RuntimeError(
        f"Expected 340 CT files, found {len(ct_files)}"
    )

if len(pt_files) != 284:
    raise RuntimeError(
        f"Expected 284 PET files, found {len(pt_files)}"
    )


# ---------------------------------------------------------
# Read CT
# ---------------------------------------------------------
ct_reader = sitk.ImageSeriesReader()
ct_names = ct_reader.GetGDCMSeriesFileNames(
    str(ct_files[0].parent),
    CT_UID
)
ct_reader.SetFileNames(ct_names)
ct_img = ct_reader.Execute()


# ---------------------------------------------------------
# Read PET
# ---------------------------------------------------------
pt_reader = sitk.ImageSeriesReader()
pt_names = pt_reader.GetGDCMSeriesFileNames(
    str(pt_files[0].parent),
    PT_UID
)
pt_reader.SetFileNames(pt_names)
pet_img = pt_reader.Execute()


# ---------------------------------------------------------
# Load aligned PET mask
# ---------------------------------------------------------
pet_mask = sitk.ReadImage(str(PET_MASK))


print("===== CASE 3 CT / PET =====")
print("CT size:", ct_img.GetSize())
print("CT spacing:", ct_img.GetSpacing())
print("CT origin:", ct_img.GetOrigin())
print("CT direction:", ct_img.GetDirection())

print()
print("PET size:", pet_img.GetSize())
print("PET spacing:", pet_img.GetSpacing())
print("PET origin:", pet_img.GetOrigin())
print("PET direction:", pet_img.GetDirection())


# ---------------------------------------------------------
# Frame of Reference check
# ---------------------------------------------------------
ct_ds = pydicom.dcmread(
    str(ct_files[0]),
    stop_before_pixels=True
)

pt_ds = pydicom.dcmread(
    str(pt_files[0]),
    stop_before_pixels=True
)

ct_for = getattr(ct_ds, "FrameOfReferenceUID", None)
pt_for = getattr(pt_ds, "FrameOfReferenceUID", None)

print()
print("CT FrameOfReferenceUID:", ct_for)
print("PET FrameOfReferenceUID:", pt_for)
print("Frame of Reference match:", ct_for == pt_for)


# ---------------------------------------------------------
# PET -> CT resampling
# ---------------------------------------------------------
ct_mask = sitk.Resample(
    pet_mask,
    ct_img,
    sitk.Transform(),
    sitk.sitkNearestNeighbor,
    0,
    sitk.sitkUInt8
)

sitk.WriteImage(ct_mask, str(CT_MASK))


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------
mask_arr = sitk.GetArrayFromImage(ct_mask)
ct_arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)

foreground = mask_arr > 0

voxels = int(np.count_nonzero(foreground))

positive_slices = np.where(
    np.any(foreground, axis=(1, 2))
)[0]

voxel_volume_cm3 = np.prod(ct_img.GetSpacing()) / 1000.0
volume_cm3 = voxels * voxel_volume_cm3


print()
print("===== RESAMPLED CT MASK =====")
print("CT mask size:", ct_mask.GetSize())
print("Foreground voxels:", voxels)
print("Positive CT slices:", len(positive_slices))

if len(positive_slices):
    print(
        "CT slice range:",
        int(positive_slices.min()),
        "to",
        int(positive_slices.max())
    )

print("Raw CT mask volume (cm³):",
      round(float(volume_cm3), 3))


# ---------------------------------------------------------
# Connected components + CT HU statistics
# ---------------------------------------------------------
cc = sitk.ConnectedComponent(ct_mask)
cc_arr = sitk.GetArrayFromImage(cc)

num_components = int(cc_arr.max())

print()
print("===== CT-MAPPED COMPONENTS =====")
print("Number of components:", num_components)

for label in range(1, num_components + 1):

    component = cc_arr == label

    count = int(np.count_nonzero(component))

    if count == 0:
        continue

    z = np.where(np.any(component, axis=(1, 2)))[0]

    values = ct_arr[component]

    background_fraction = float(
        np.mean(values <= -900)
    )

    volume = count * voxel_volume_cm3

    print(
        f"Component {label}: "
        f"voxels={count} "
        f"volume_cm3={volume:.3f} "
        f"z={int(z.min())}..{int(z.max())} "
        f"HUmin={values.min():.0f} "
        f"HUmax={values.max():.0f} "
        f"HUmean={values.mean():.2f} "
        f"HUmedian={np.median(values):.2f} "
        f"background_fraction={background_fraction:.4f}"
    )


print()
print("Saved:", CT_MASK)