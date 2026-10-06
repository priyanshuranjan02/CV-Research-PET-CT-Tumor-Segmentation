from pathlib import Path
import numpy as np
import pydicom
import pydicom_seg
import SimpleITK as sitk

PET_ROOT = Path("verified_case/pet_reference")
CT_ROOT = Path("verified_case/ct")
SEG_ROOT = Path("verified_case/seg")

# ---------------------------------------------------------
# Load PET
# ---------------------------------------------------------
pet_files = [
    p for p in PET_ROOT.rglob("*.dcm")
    if "__MACOSX" not in str(p)
]

pet_files.sort(
    key=lambda p: float(
        pydicom.dcmread(
            p,
            stop_before_pixels=True
        ).ImagePositionPatient[2]
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
        pydicom.dcmread(
            p,
            stop_before_pixels=True
        ).ImagePositionPatient[2]
    )
)

ct_reader = sitk.ImageSeriesReader()
ct_reader.SetFileNames([str(p) for p in ct_files])
ct = ct_reader.Execute()

# ---------------------------------------------------------
# Load native PET SEG
# ---------------------------------------------------------
seg_file = next(
    p for p in SEG_ROOT.rglob("*.dcm")
    if "__MACOSX" not in str(p)
)

seg_ds = pydicom.dcmread(seg_file)

seg_reader = pydicom_seg.SegmentReader()
result = seg_reader.read(seg_ds)

mask = sitk.Cast(
    result.segment_image(1) > 0,
    sitk.sitkUInt8
)

# ---------------------------------------------------------
# Connected components
# ---------------------------------------------------------
cc = sitk.ConnectedComponent(mask, True)

stats = sitk.LabelShapeStatisticsImageFilter()
stats.Execute(cc)

labels = sorted(
    stats.GetLabels(),
    key=lambda x: stats.GetNumberOfPixels(x),
    reverse=True
)

print("=== PET → CT COMPONENT MAPPING ===")
print("PET size:", pet.GetSize())
print("CT size:", ct.GetSize())

for rank, label in enumerate(labels, 1):

    # Bounding box of component in PET voxel space
    x, y, z, sx, sy, sz = stats.GetBoundingBox(label)

    # Centroid in physical PET coordinates
    centroid = stats.GetCentroid(label)

    # Map PET physical point → CT continuous index
    ct_index = ct.TransformPhysicalPointToContinuousIndex(
        centroid
    )

    inside = (
        0 <= ct_index[0] < ct.GetWidth()
        and
        0 <= ct_index[1] < ct.GetHeight()
        and
        0 <= ct_index[2] < ct.GetDepth()
    )

    print(f"\nComponent {rank}")
    print("  Label:", label)
    print("  PET voxels:", stats.GetNumberOfPixels(label))
    print(
        "  PET centroid physical:",
        tuple(round(v, 3) for v in centroid)
    )
    print(
        "  CT continuous index:",
        tuple(round(v, 3) for v in ct_index)
    )
    print("  Inside CT volume:", inside)
    print(
        "  PET bounding box:",
        f"x={x}, y={y}, z={z}, "
        f"size=({sx}, {sy}, {sz})"
    )

print("\n=== DONE ===")
