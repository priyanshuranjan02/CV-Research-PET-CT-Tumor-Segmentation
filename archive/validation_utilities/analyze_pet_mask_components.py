from pathlib import Path
import SimpleITK as sitk

SEG_ROOT = Path("verified_case/seg")

seg_file = next(
    p for p in SEG_ROOT.rglob("*.dcm")
    if "__MACOSX" not in str(p)
)

import pydicom
import pydicom_seg

ds = pydicom.dcmread(seg_file)

reader = pydicom_seg.SegmentReader()
result = reader.read(ds)

mask = sitk.Cast(
    result.segment_image(1) > 0,
    sitk.sitkUInt8
)

# 3D connected components in native PET space
cc = sitk.ConnectedComponent(mask, True)

stats = sitk.LabelShapeStatisticsImageFilter()
stats.Execute(cc)

spacing = mask.GetSpacing()
voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]

components = []

for label in stats.GetLabels():
    voxels = stats.GetNumberOfPixels(label)
    x, y, z, sx, sy, sz = stats.GetBoundingBox(label)

    components.append({
        "label": label,
        "voxels": voxels,
        "volume_cm3": voxels * voxel_volume_mm3 / 1000.0,
        "x": x,
        "y": y,
        "z": z,
        "size_x": sx,
        "size_y": sy,
        "size_z": sz,
    })

components.sort(
    key=lambda c: c["voxels"],
    reverse=True
)

print("=== NATIVE PET EXPERT MASK ===")
print("Mask size:", mask.GetSize())
print("Spacing:", mask.GetSpacing())
print("Foreground voxels:", int(sitk.GetArrayFromImage(mask).sum()))
print("Connected components:", len(components))

for i, c in enumerate(components, 1):
    print(f"\nComponent {i}")
    print("  Voxels:", c["voxels"])
    print(f"  Volume: {c['volume_cm3']:.4f} cm^3")
    print(
        "  Bounding box:",
        f"x={c['x']}, y={c['y']}, z={c['z']},",
        f"size=({c['size_x']}, {c['size_y']}, {c['size_z']})"
    )

print("\n=== DONE ===")
