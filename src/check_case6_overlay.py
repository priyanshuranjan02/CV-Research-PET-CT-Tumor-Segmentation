from pathlib import Path
import numpy as np
import cv2
import pydicom
import SimpleITK as sitk


ROOT = Path("development_cases/PETCT_185da4c8b6")
MASK_PATH = ROOT / "tumor_mask_ct_visible.nii.gz"
OUT_DIR = ROOT / "reference_overlay_check"

OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Find CT DICOM series
# ---------------------------------------------------------
series = {}

for path in ROOT.rglob("*"):
    if not path.is_file():
        continue

    try:
        ds = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True
        )

        uid = getattr(ds, "SeriesInstanceUID", None)
        modality = getattr(ds, "Modality", None)

        if uid and modality == "CT":
            series.setdefault(uid, []).append(path)

    except Exception:
        continue


if not series:
    raise RuntimeError("No CT DICOM series found.")

# Select the largest CT series
ct_uid, ct_files = max(series.items(), key=lambda x: len(x[1]))

print("CT Series UID:")
print(ct_uid)
print("CT files:", len(ct_files))


# ---------------------------------------------------------
# Load CT volume
# ---------------------------------------------------------
reader = sitk.ImageSeriesReader()
reader.SetFileNames([str(p) for p in sorted(ct_files)])
ct_img = reader.Execute()

ct = sitk.GetArrayFromImage(ct_img).astype(np.float32)


# ---------------------------------------------------------
# Load reference mask
# ---------------------------------------------------------
mask_img = sitk.ReadImage(str(MASK_PATH))
mask = sitk.GetArrayFromImage(mask_img)


print()
print("CT shape:", ct.shape)
print("Mask shape:", mask.shape)


# ---------------------------------------------------------
# Geometry check
# ---------------------------------------------------------
print()
print("===== GEOMETRY CHECK =====")
print("Size:",
      ct_img.GetSize() == mask_img.GetSize())

print("Spacing:",
      np.allclose(ct_img.GetSpacing(), mask_img.GetSpacing()))

print("Origin:",
      np.allclose(ct_img.GetOrigin(), mask_img.GetOrigin()))

print("Direction:",
      np.allclose(ct_img.GetDirection(), mask_img.GetDirection()))


if ct.shape != mask.shape:
    raise RuntimeError(
        f"Shape mismatch: CT={ct.shape}, MASK={mask.shape}"
    )


# ---------------------------------------------------------
# Positive slices
# ---------------------------------------------------------
positive = np.where(
    np.any(mask > 0, axis=(1, 2))
)[0]

print()
print("Positive CT slices:", len(positive))
print("Slice range:", int(positive.min()), "to", int(positive.max()))


# Select 4 representative slices
selected = [
    int(positive[0]),
    int(positive[len(positive) // 3]),
    int(positive[(2 * len(positive)) // 3]),
    int(positive[-1]),
]

selected = sorted(set(selected))

print("Selected slices:", selected)


# ---------------------------------------------------------
# Create overlays
# ---------------------------------------------------------
for z in selected:

    img = ct[z]

    # Robust display window
    lo, hi = np.percentile(img, [1, 99])

    vis = np.clip(
        (img - lo) / (hi - lo + 1e-8) * 255,
        0,
        255
    ).astype(np.uint8)

    overlay = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    binary = (mask[z] > 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        overlay,
        f"Case 6 - CT slice {z}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    output = OUT_DIR / f"slice_{z:03d}.png"

    cv2.imwrite(str(output), overlay)

    print("Saved:", output)


print()
print("===== DONE =====")