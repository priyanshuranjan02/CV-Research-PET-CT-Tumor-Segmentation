from pathlib import Path
import numpy as np
import pydicom
import pydicom_seg
import SimpleITK as sitk

PET_ROOT = Path("verified_case/pet_reference")
CT_ROOT = Path("verified_case/ct")
SEG_ROOT = Path("verified_case/seg")

# -----------------------------
# Load PET
# -----------------------------
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

# -----------------------------
# Load CT
# -----------------------------
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

ct_np = sitk.GetArrayFromImage(ct)

# -----------------------------
# Load SEG
# -----------------------------
seg_file = next(
    p for p in SEG_ROOT.rglob("*.dcm")
    if "__MACOSX" not in str(p)
)

seg_ds = pydicom.dcmread(seg_file)

reader = pydicom_seg.SegmentReader()
result = reader.read(seg_ds)

seg = sitk.Cast(
    result.segment_image(1) > 0,
    sitk.sitkUInt8
)

# Connected components
cc = sitk.ConnectedComponent(seg, True)

stats = sitk.LabelShapeStatisticsImageFilter()
stats.Execute(cc)

labels = sorted(
    stats.GetLabels(),
    key=lambda x: stats.GetNumberOfPixels(x),
    reverse=True
)

print("=== CT INTENSITY AT MAPPED PET TUMOR COMPONENTS ===")

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

    mapped_np = sitk.GetArrayFromImage(mapped) > 0

    values = ct_np[mapped_np]

    centroid = stats.GetCentroid(label)
    ct_index = ct.TransformPhysicalPointToContinuousIndex(
        centroid
    )

    print(f"\nComponent {rank}")
    print("  PET voxels:", stats.GetNumberOfPixels(label))
    print("  Mapped CT voxels:", len(values))
    print(
        "  CT centroid index:",
        tuple(round(v, 2) for v in ct_index)
    )

    if len(values) > 0:
        print("  CT min HU:", float(values.min()))
        print("  CT max HU:", float(values.max()))
        print("  CT mean HU:", float(values.mean()))
        print("  CT median HU:", float(np.median(values)))
        print(
            "  Voxels <= -900 HU:",
            int((values <= -900).sum()),
            "/",
            len(values)
        )

print("\n=== DONE ===")
