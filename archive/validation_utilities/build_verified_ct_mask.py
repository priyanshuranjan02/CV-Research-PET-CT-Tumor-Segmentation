from pathlib import Path
import csv
import numpy as np
import pydicom
import SimpleITK as sitk
import pydicom_seg

CT_SERIES_UID = "1.3.6.1.4.1.14519.5.2.1.239537544060707922869401739533283658263"

CT_ROOT = Path("verified_case/ct")
SEG_ROOT = Path("verified_case/seg")

# ---------------------------------------------------------
# 1. Find all CT DICOM slices belonging to the verified CT series
# ---------------------------------------------------------
ct_files = []

for path in CT_ROOT.rglob("*.dcm"):
    if "__MACOSX" in str(path):
        continue

    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)

        if (
            getattr(ds, "Modality", None) == "CT"
            and getattr(ds, "SeriesInstanceUID", None) == CT_SERIES_UID
        ):
            ct_files.append(path)

    except Exception:
        pass

if not ct_files:
    raise RuntimeError("No CT DICOM files found for the expected SeriesInstanceUID.")

print("CT slices found:", len(ct_files))

# ---------------------------------------------------------
# 2. Sort CT slices by physical Z position
# ---------------------------------------------------------
def z_position(path):
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    return float(ds.ImagePositionPatient[2])

ct_files.sort(key=z_position)

# ---------------------------------------------------------
# 3. Read CT volume
# ---------------------------------------------------------
reader = sitk.ImageSeriesReader()
reader.SetFileNames([str(p) for p in ct_files])
ct_image = reader.Execute()

print("\n=== CT VOLUME ===")
print("Size:", ct_image.GetSize())
print("Spacing:", ct_image.GetSpacing())
print("Origin:", ct_image.GetOrigin())
print("Direction:", ct_image.GetDirection())

# ---------------------------------------------------------
# 4. Decode DICOM-SEG
# ---------------------------------------------------------
seg_files = [
    p for p in SEG_ROOT.rglob("*.dcm")
    if "__MACOSX" not in str(p)
]

if len(seg_files) != 1:
    raise RuntimeError(f"Expected exactly 1 SEG DICOM, found {len(seg_files)}")

seg_file = seg_files[0]

seg_ds = pydicom.dcmread(seg_file)

segment_reader = pydicom_seg.SegmentReader()
seg_result = segment_reader.read(seg_ds)

print("\n=== SEGMENT ===")
print("Available segments:", seg_result.available_segments)

seg_image = seg_result.segment_image(1)

print("SEG size:", seg_image.GetSize())
print("SEG spacing:", seg_image.GetSpacing())
print("SEG origin:", seg_image.GetOrigin())
print("SEG direction:", seg_image.GetDirection())

# ---------------------------------------------------------
# 5. Resample expert SEG onto CT grid
#    Nearest-neighbor preserves binary labels.
# ---------------------------------------------------------
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(ct_image)
resampler.SetInterpolator(sitk.sitkNearestNeighbor)
resampler.SetTransform(sitk.Transform())
resampler.SetDefaultPixelValue(0)
resampler.SetOutputPixelType(sitk.sitkUInt8)

ct_mask = resampler.Execute(seg_image)

# ---------------------------------------------------------
# 6. Convert to NumPy for analysis
# ---------------------------------------------------------
mask_array = sitk.GetArrayFromImage(ct_mask)
mask_array = (mask_array > 0).astype(np.uint8)

print("\n=== CT-ALIGNED MASK ===")
print("Mask shape (Z,Y,X):", mask_array.shape)
print("Foreground voxels:", int(mask_array.sum()))

positive_slices = np.where(mask_array.reshape(mask_array.shape[0], -1).sum(axis=1) > 0)[0]

print("Positive CT slices:", len(positive_slices))

if len(positive_slices) > 0:
    print("First positive slice:", int(positive_slices[0]))
    print("Last positive slice:", int(positive_slices[-1]))
else:
    raise RuntimeError(
        "No foreground voxels remain after resampling. "
        "Do not continue until the geometry is investigated."
    )

# ---------------------------------------------------------
# 7. Save CT-aligned 3D expert mask
# ---------------------------------------------------------
output_dir = Path("verified_case")
output_dir.mkdir(exist_ok=True)

mask_image = sitk.GetImageFromArray(mask_array)
mask_image.CopyInformation(ct_image)

mask_path = output_dir / "tumor_mask_ct.nii.gz"
sitk.WriteImage(mask_image, str(mask_path))

print("\nSaved:", mask_path)

# ---------------------------------------------------------
# 8. Save positive-slice information
# ---------------------------------------------------------
csv_path = output_dir / "positive_ct_slices.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ct_slice_index", "tumor_pixels"])

    for z in positive_slices:
        pixels = int(mask_array[z].sum())
        writer.writerow([int(z), pixels])

print("Saved:", csv_path)

print("\nDONE")
