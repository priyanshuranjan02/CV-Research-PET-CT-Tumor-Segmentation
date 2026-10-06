from pathlib import Path
import pydicom
import SimpleITK as sitk
import pydicom_seg

PET_ROOT = Path("verified_case/pet_reference")
SEG_ROOT = Path("verified_case/seg")

# ---------------------------------------------------------
# Find the 326 PET DICOM files
# ---------------------------------------------------------
pet_files = []

for p in PET_ROOT.rglob("*.dcm"):
    if "__MACOSX" in str(p):
        continue

    ds = pydicom.dcmread(p, stop_before_pixels=True)

    if getattr(ds, "Modality", None) == "PT":
        pet_files.append(p)

if len(pet_files) != 326:
    raise RuntimeError(
        f"Expected 326 PET files, found {len(pet_files)}"
    )

# Sort by physical Z position
pet_files.sort(
    key=lambda p: float(
        pydicom.dcmread(
            p,
            stop_before_pixels=True
        ).ImagePositionPatient[2]
    )
)

# ---------------------------------------------------------
# Read PET volume
# ---------------------------------------------------------
reader = sitk.ImageSeriesReader()
reader.SetFileNames([str(p) for p in pet_files])
pet_image = reader.Execute()

print("=== PET VOLUME ===")
print("Size:", pet_image.GetSize())
print("Spacing:", pet_image.GetSpacing())
print("Origin:", pet_image.GetOrigin())
print("Direction:", pet_image.GetDirection())

# ---------------------------------------------------------
# Decode SEG
# ---------------------------------------------------------
seg_file = next(
    p for p in SEG_ROOT.rglob("*.dcm")
    if "__MACOSX" not in str(p)
)

seg_ds = pydicom.dcmread(seg_file)

seg_reader = pydicom_seg.SegmentReader()
result = seg_reader.read(seg_ds)

seg_image = result.segment_image(1)

print("\n=== DICOM SEG ===")
print("Size:", seg_image.GetSize())
print("Spacing:", seg_image.GetSpacing())
print("Origin:", seg_image.GetOrigin())
print("Direction:", seg_image.GetDirection())

# ---------------------------------------------------------
# Compare geometry
# ---------------------------------------------------------
print("\n=== GEOMETRY COMPARISON ===")

same_size = pet_image.GetSize() == seg_image.GetSize()
same_spacing = np_allclose = all(
    abs(a - b) < 1e-4
    for a, b in zip(
        pet_image.GetSpacing(),
        seg_image.GetSpacing()
    )
)

same_origin = all(
    abs(a - b) < 1e-3
    for a, b in zip(
        pet_image.GetOrigin(),
        seg_image.GetOrigin()
    )
)

same_direction = all(
    abs(a - b) < 1e-4
    for a, b in zip(
        pet_image.GetDirection(),
        seg_image.GetDirection()
    )
)

print("Same size:", same_size)
print("Same spacing:", same_spacing)
print("Same origin:", same_origin)
print("Same direction:", same_direction)

if all([
    same_size,
    same_spacing,
    same_origin,
    same_direction
]):
    print("\nSUCCESS: SEG geometry matches the actual PET volume.")
else:
    print("\nWARNING: PET and SEG geometry differ.")

# ---------------------------------------------------------
# Foreground statistics
# ---------------------------------------------------------
mask = sitk.GetArrayFromImage(
    sitk.Cast(seg_image > 0, sitk.sitkUInt8)
)

print("\n=== MASK ===")
print("Shape:", mask.shape)
print("Foreground voxels:", int(mask.sum()))

if mask.sum() > 0:
    positive = (mask > 0).sum(axis=(1, 2))
    positive_slices = positive > 0

    print("Positive PET slices:", int(positive_slices.sum()))
    print(
        "First positive PET slice:",
        int(positive_slices.nonzero()[0][0])
    )
    print(
        "Last positive PET slice:",
        int(positive_slices.nonzero()[0][-1])
    )
