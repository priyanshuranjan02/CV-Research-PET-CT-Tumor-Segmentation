from pathlib import Path
import numpy as np
import pydicom
import pydicom_seg
import SimpleITK as sitk
import cv2

PET_ROOT = Path("verified_case/pet_reference")
CT_ROOT = Path("verified_case/ct")
SEG_ROOT = Path("verified_case/seg")
BODY_ROOT = Path("verified_case/improved_body_masked_2d")

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

seg = sitk.Cast(
    result.segment_image(1) > 0,
    sitk.sitkUInt8
)

# Connected components in PET space
cc = sitk.ConnectedComponent(seg, True)

stats = sitk.LabelShapeStatisticsImageFilter()
stats.Execute(cc)

labels = sorted(
    stats.GetLabels(),
    key=lambda x: stats.GetNumberOfPixels(x),
    reverse=True
)

print("=== PET TUMOR COMPONENT → CT VISIBILITY ===")
print("CT size:", ct.GetSize())
print()

for rank, label in enumerate(labels, 1):

    # Isolate one PET component
    component = sitk.Cast(
        cc == label,
        sitk.sitkUInt8
    )

    # Map this PET component into CT physical space
    mapped = sitk.Resample(
        component,
        ct,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8
    )

    mapped_np = sitk.GetArrayFromImage(mapped) > 0

    total_voxels = int(mapped_np.sum())

    # Check against our improved body mask.
    body_hits = 0
    body_total = 0
    slice_stats = []

    for z in range(mapped_np.shape[0]):

        if not mapped_np[z].any():
            continue

        body_path = BODY_ROOT / f"slice_{z:03d}.tif"
        body = cv2.imread(
            str(body_path),
            cv2.IMREAD_GRAYSCALE
        )

        if body is None:
            continue

        component_pixels = mapped_np[z]
        hits = int(
            np.logical_and(
                component_pixels,
                body > 0
            ).sum()
        )

        count = int(component_pixels.sum())

        body_hits += hits
        body_total += count

        slice_stats.append(
            (z, count, hits)
        )

    body_fraction = (
        body_hits / body_total
        if body_total > 0 else 0.0
    )

    centroid = stats.GetCentroid(label)

    ct_centroid = ct.TransformPhysicalPointToContinuousIndex(
        centroid
    )

    print(f"Component {rank}")
    print("  PET voxels:", stats.GetNumberOfPixels(label))
    print("  CT-mapped voxels:", total_voxels)
    print(
        "  CT centroid index:",
        tuple(round(v, 2) for v in ct_centroid)
    )
    print("  CT Z range:", end=" ")

    if slice_stats:
        print(
            f"{min(x[0] for x in slice_stats)} - "
            f"{max(x[0] for x in slice_stats)}"
        )
    else:
        print("N/A")

    print(
        "  Fraction inside improved CT body mask:",
        f"{100 * body_fraction:.2f}%"
    )

    print()

print("=== DONE ===")
