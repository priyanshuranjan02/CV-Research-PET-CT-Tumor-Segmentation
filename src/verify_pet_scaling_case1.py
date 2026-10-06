from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk


CASE = Path(
    "development_cases/PETCT_0168f65af8/"
    "manifest-1789661648760/FDG-PET-CT-Lesions/"
    "PETCT_0168f65af8"
)

PET_DIR = next(
    p for p in CASE.rglob("*")
    if p.is_dir() and "PET corr." in p.name
)


# ---------------------------------------------------------
# PET files in SimpleITK's actual reading order
# ---------------------------------------------------------
reader = sitk.ImageSeriesReader()

files = reader.GetGDCMSeriesFileNames(
    str(PET_DIR)
)

reader.SetFileNames(files)
pet_img = reader.Execute()

pet_sitk = sitk.GetArrayFromImage(
    pet_img
).astype(np.float64)


print("===== CASE 1 PET SCALING CHECK =====")
print("PET files:", len(files))
print("SimpleITK PET shape:", pet_sitk.shape)


# ---------------------------------------------------------
# Compare SimpleITK values with raw DICOM pixels
# ---------------------------------------------------------
ratios_to_raw = []
ratios_to_slope = []

slopes = []
raw_maxima = []
sitk_maxima = []

for i, path in enumerate(files):

    ds = pydicom.dcmread(
        str(path),
        force=True
    )

    raw = ds.pixel_array.astype(
        np.float64
    )

    slope = float(
        getattr(
            ds,
            "RescaleSlope",
            1.0
        )
    )

    raw_max = float(
        np.max(raw)
    )

    sitk_max = float(
        np.max(pet_sitk[i])
    )

    slopes.append(slope)
    raw_maxima.append(raw_max)
    sitk_maxima.append(sitk_max)

    if raw_max > 0:

        ratios_to_raw.append(
            sitk_max / raw_max
        )

        ratios_to_slope.append(
            sitk_max / (raw_max * slope)
        )


ratios_to_raw = np.asarray(
    ratios_to_raw
)

ratios_to_slope = np.asarray(
    ratios_to_slope
)

slopes = np.asarray(
    slopes
)

raw_maxima = np.asarray(
    raw_maxima
)

sitk_maxima = np.asarray(
    sitk_maxima
)


# ---------------------------------------------------------
# Scaling comparison
# ---------------------------------------------------------
print()
print("===== SIMPLEITK vs RAW DICOM =====")

print(
    "Median SimpleITK/raw ratio:",
    round(
        float(np.median(ratios_to_raw)),
        6
    )
)

print(
    "Median SimpleITK/(raw*slope) ratio:",
    round(
        float(np.median(ratios_to_slope)),
        6
    )
)


near_raw = np.mean(
    np.isclose(
        ratios_to_raw,
        1.0,
        rtol=0.02,
        atol=0.02
    )
)

near_rescaled = np.mean(
    np.isclose(
        ratios_to_slope,
        1.0,
        rtol=0.02,
        atol=0.02
    )
)

print(
    "Slices where SimpleITK ≈ raw:",
    f"{near_raw * 100:.1f}%"
)

print(
    "Slices where SimpleITK ≈ raw*slope:",
    f"{near_rescaled * 100:.1f}%"
)


# ---------------------------------------------------------
# PET metadata
# ---------------------------------------------------------
first = pydicom.dcmread(
    str(files[0]),
    stop_before_pixels=True,
    force=True
)

weight = getattr(
    first,
    "PatientWeight",
    None
)

units = getattr(
    first,
    "Units",
    None
)

dose = None

seq = getattr(
    first,
    "RadiopharmaceuticalInformationSequence",
    None
)

if seq:
    dose = getattr(
        seq[0],
        "RadionuclideTotalDose",
        None
    )


print()
print("===== PET METADATA =====")
print("Units:", units)
print("PatientWeightKg:", weight)
print("InjectedDoseBq:", dose)
print(
    "Slope range:",
    float(slopes.min()),
    "to",
    float(slopes.max())
)


# ---------------------------------------------------------
# Dynamic SUV max
# ---------------------------------------------------------
if weight is not None and dose is not None:

    activity = (
        pet_sitk
        * slopes[:, None, None]
    )

    suv = (
        activity
        * float(weight)
        * 1000.0
        / float(dose)
    )

    max_index = np.unravel_index(
        np.argmax(suv),
        suv.shape
    )

    z = max_index[0]

    print()
    print("===== DYNAMIC SUV CHECK =====")
    print(
        "SUV max:",
        round(
            float(np.max(suv)),
            4
        )
    )

    print(
        "Max SUV slice:",
        int(z)
    )

    print(
        "Slope at max SUV slice:",
        float(slopes[z])
    )

    print(
        "Raw pixel max at max SUV slice:",
        float(raw_maxima[z])
    )

    print(
        "SimpleITK pixel max at max SUV slice:",
        float(sitk_maxima[z])
    )


# ---------------------------------------------------------
# Top 10 SUV slices
# ---------------------------------------------------------
if weight is not None and dose is not None:

    suv_slice_max = np.max(
        suv,
        axis=(1, 2)
    )

    top = np.argsort(
        suv_slice_max
    )[-10:][::-1]

    print()
    print("===== TOP 10 SUV SLICES =====")

    for z in top:

        print(
            f"z={int(z):3d} "
            f"SUVmax={suv_slice_max[z]:8.4f} "
            f"slope={slopes[z]:8.6f} "
            f"raw_max={raw_maxima[z]:8.1f}"
        )


print()
print("===== DONE =====")