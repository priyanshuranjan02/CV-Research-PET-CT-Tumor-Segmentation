from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk


ROOT = Path("development_cases/PETCT_11afab3485")

CT_UID = "1.3.6.1.4.1.14519.5.2.1.167235741250989372348731643027849097647"

RAW_MASK = ROOT / "tumor_mask_ct_raw.nii.gz"
VISIBLE_MASK = ROOT / "tumor_mask_ct_visible.nii.gz"


# Find CT files
ct_files = []

for path in ROOT.rglob("*.dcm"):
    try:
        ds = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True
        )

        if getattr(ds, "SeriesInstanceUID", None) == CT_UID:
            ct_files.append(path)

    except Exception:
        pass


if len(ct_files) != 340:
    raise RuntimeError(
        f"Expected 340 CT files, found {len(ct_files)}"
    )


# Load CT
reader = sitk.ImageSeriesReader()
ct_names = reader.GetGDCMSeriesFileNames(
    str(ct_files[0].parent),
    CT_UID
)
reader.SetFileNames(ct_names)
ct_img = reader.Execute()

ct = sitk.GetArrayFromImage(ct_img).astype(np.float32)


# Load raw mapped mask
raw_img = sitk.ReadImage(str(RAW_MASK))
raw = sitk.GetArrayFromImage(raw_img)

if raw.shape != ct.shape:
    raise RuntimeError(
        f"Shape mismatch: CT={ct.shape}, mask={raw.shape}"
    )


# Connected components
cc_img = sitk.ConnectedComponent(
    sitk.GetImageFromArray((raw > 0).astype(np.uint8))
)
cc = sitk.GetArrayFromImage(cc_img)

num_components = int(cc.max())

visible = np.zeros_like(raw, dtype=np.uint8)

print("===== CASE 5 CT REFERENCE FILTER =====")
print("Components:", num_components)

for label in range(1, num_components + 1):

    component = cc == label
    count = int(np.count_nonzero(component))

    if count == 0:
        continue

    values = ct[component]
    background_fraction = float(
        np.mean(values <= -900)
    )

    print(
        f"Component {label}: "
        f"voxels={count} "
        f"background_fraction={background_fraction:.4f}"
    )

    # Retain every component that contains at least some
    # CT-visible anatomy. Only completely CT-invisible
    # components are removed.
    if background_fraction < 1.0:
        visible[component] = 1


# Save
visible_img = sitk.GetImageFromArray(visible)
visible_img.CopyInformation(raw_img)

sitk.WriteImage(
    visible_img,
    str(VISIBLE_MASK)
)


# Final statistics
foreground = visible > 0

z = np.where(
    np.any(foreground, axis=(1, 2))
)[0]

voxel_volume_cm3 = np.prod(ct_img.GetSpacing()) / 1000.0
volume_cm3 = int(np.count_nonzero(foreground)) * voxel_volume_cm3

print()
print("===== CASE 5 CT REFERENCE MASK =====")
print("Saved:", VISIBLE_MASK)
print("Foreground voxels:", int(np.count_nonzero(foreground)))
print("Positive CT slices:", len(z))

if len(z):
    print(
        "CT slice range:",
        int(z.min()),
        "to",
        int(z.max())
    )

print("Volume (cm³):", round(float(volume_cm3), 3))