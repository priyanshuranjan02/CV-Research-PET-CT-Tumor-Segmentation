from pathlib import Path
import numpy as np
import pydicom
import pydicom_seg
import SimpleITK as sitk

PET_ROOT = Path("verified_case/pet_reference")
CT_ROOT = Path("verified_case/ct")
SEG_ROOT = Path("verified_case/seg")

OUTPUT = Path("verified_case/tumor_mask_ct_visible.nii.gz")
OVERLAY_DIR = Path("verified_case/ct_visible_overlays")
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Load PET
# ---------------------------------------------------------
pet_files = [
    p for p in PET_ROOT.rglob("*.dcm")
    if "__MACOSX" not in str(p)
]

pet_files.sort(
    key=lambda p: float(
        pydicom.dcmread(p, stop_before_pixels=True).ImagePositionPatient[2]
    )
)

pet_reader = sitk.ImageSeriesReader()
pet_reader.SetFileNames([str(p) for p in pet_files])
pet = pet_reader.Execute()

# ---------------------------------------------------------
# Load CT
# ---------------------------------------------------------
ct_files = [
    p for p in CT_ROOT.rglob("*.dcm")
    if "__MACOSX" not in str(p)
]

ct_files.sort(
    key=lambda p: float(
        pydicom.dcmread(p, stop_before_pixels=True).ImagePositionPatient[2]
    )
)

ct_reader = sitk.ImageSeriesReader()
ct_reader.SetFileNames([str(p) for p in ct_files])
ct = ct_reader.Execute()

# ---------------------------------------------------------
# Decode native PET SEG
# ---------------------------------------------------------
seg_file = next(
    p for p in SEG_ROOT.rglob("*.dcm")
    if "__MACOSX" not in str(p)
)

seg_ds = pydicom.dcmread(seg_file)
reader = pydicom_seg.SegmentReader()
result = reader.read(seg_ds)

tumor = sitk.Cast(
    result.segment_image(1) > 0,
    sitk.sitkUInt8
)

# ---------------------------------------------------------
# Find connected components in PET space
# ---------------------------------------------------------
cc = sitk.ConnectedComponent(tumor, True)

stats = sitk.LabelShapeStatisticsImageFilter()
stats.Execute(cc)

labels = sorted(
    stats.GetLabels(),
    key=lambda x: stats.GetNumberOfPixels(x),
    reverse=True
)

print("=== PET COMPONENTS → CT ===")

visible_ct_mask = sitk.Image(
    ct.GetSize(),
    sitk.sitkUInt8
)
visible_ct_mask.CopyInformation(ct)

for rank, label in enumerate(labels, 1):

    component = sitk.Cast(
        cc == label,
        sitk.sitkUInt8
    )

    mapped = sitk.Resample(
        component,
        ct,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8
    )

    mapped_np = sitk.GetArrayFromImage(mapped)

    # Inspect CT HU at mapped component locations.
    ct_np = sitk.GetArrayFromImage(ct)
    values = ct_np[mapped_np > 0]

    # A component is considered CT-visible when its mapped
    # voxels contain something other than pure CT background.
    visible = np.any(values > -1000)

    print(
        f"Component {rank}: "
        f"PET voxels={stats.GetNumberOfPixels(label)}, "
        f"mapped CT voxels={len(values)}, "
        f"CT-visible={visible}"
    )

    if visible:
        visible_ct_mask = sitk.Maximum(
            visible_ct_mask,
            mapped
        )

# ---------------------------------------------------------
# Save CT-visible reference mask
# ---------------------------------------------------------
sitk.WriteImage(
    visible_ct_mask,
    str(OUTPUT)
)

print("\nSaved:", OUTPUT)

# ---------------------------------------------------------
# Report positive CT slices
# ---------------------------------------------------------
mask_np = sitk.GetArrayFromImage(visible_ct_mask) > 0

positive = np.where(
    mask_np.reshape(mask_np.shape[0], -1).sum(axis=1) > 0
)[0]

print("CT-visible expert voxels:", int(mask_np.sum()))
print("CT-visible positive slices:", len(positive))
print(
    "Positive CT slice range:",
    int(positive[0]),
    "-",
    int(positive[-1])
)

# ---------------------------------------------------------
# Generate overlays around each visible component
# ---------------------------------------------------------
selected = [280, 291, 354, 357, 360]

for z in selected:

    if z >= ct.GetDepth():
        continue

    ct_slice = ct[:, :, z]
    mask_slice = visible_ct_mask[:, :, z]

    display = sitk.IntensityWindowing(
        ct_slice,
        -200,
        300,
        0,
        255
    )

    display = sitk.Cast(
        display,
        sitk.sitkUInt8
    )

    overlay = sitk.LabelOverlay(
        display,
        mask_slice,
        opacity=0.60
    )

    path = OVERLAY_DIR / f"slice_{z:03d}_ct_visible_expert.png"
    sitk.WriteImage(overlay, str(path))
    print("Saved:", path)

print("\nDONE")
