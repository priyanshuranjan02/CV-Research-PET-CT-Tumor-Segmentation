from pathlib import Path
import numpy as np
import pydicom
import SimpleITK as sitk
import pydicom_seg

PET_ROOT = Path("verified_case/pet_reference")
SEG_ROOT = Path("verified_case/seg")
OUT_DIR = Path("verified_case/pet_seg_overlays")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Find PET DICOM files
# ---------------------------------------------------------
pet_files = []

for p in PET_ROOT.rglob("*.dcm"):
    if "__MACOSX" in str(p):
        continue

    ds = pydicom.dcmread(p, stop_before_pixels=True)

    if getattr(ds, "Modality", None) == "PT":
        pet_files.append(p)

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
pet = reader.Execute()

# ---------------------------------------------------------
# Decode expert SEG
# ---------------------------------------------------------
seg_file = next(
    p for p in SEG_ROOT.rglob("*.dcm")
    if "__MACOSX" not in str(p)
)

seg_ds = pydicom.dcmread(seg_file)
seg_reader = pydicom_seg.SegmentReader()
result = seg_reader.read(seg_ds)

seg = result.segment_image(1)
seg = sitk.Cast(seg > 0, sitk.sitkUInt8)

pet_np = sitk.GetArrayFromImage(pet)
seg_np = sitk.GetArrayFromImage(seg)

positive = np.where(
    seg_np.reshape(seg_np.shape[0], -1).sum(axis=1) > 0
)[0]

print("Positive PET slices:", len(positive))
print("First:", int(positive[0]))
print("Last:", int(positive[-1]))

# ---------------------------------------------------------
# Select beginning, middle and end
# ---------------------------------------------------------
selected = [
    int(positive[0]),
    int(positive[len(positive) // 2]),
    int(positive[-1]),
]

for z in selected:

    pet_slice = pet_np[z].astype(np.float32)
    mask_slice = seg_np[z].astype(np.uint8)

    # Robust PET display normalization
    lo = np.percentile(pet_slice, 1)
    hi = np.percentile(pet_slice, 99)

    if hi <= lo:
        hi = lo + 1.0

    display = np.clip(
        (pet_slice - lo) / (hi - lo) * 255,
        0,
        255
    ).astype(np.uint8)

    # Convert to SimpleITK image
    pet_display = sitk.GetImageFromArray(display)

    # Overlay expert tumor mask
    overlay = sitk.LabelOverlay(
        pet_display,
        sitk.GetImageFromArray(mask_slice),
        opacity=0.65
    )

    output_path = OUT_DIR / f"pet_seg_overlay_{z:03d}.png"
    sitk.WriteImage(overlay, str(output_path))

    print("Saved:", output_path)

print("\nPET/SEG overlay generation complete.")
