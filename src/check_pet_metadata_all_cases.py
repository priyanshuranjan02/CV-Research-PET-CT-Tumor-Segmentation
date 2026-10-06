from pathlib import Path
import pydicom


CASES = [
    "PETCT_0168f65af8",
    "PETCT_04606080a0",
    "PETCT_04ab5c61c9",
    "PETCT_0b57b247b6",
    "PETCT_11afab3485",
    "PETCT_185da4c8b6",
]


def get_first_pet_file(case_dir):
    for path in case_dir.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=True
            )

            if getattr(ds, "Modality", None) == "PT":
                return path, ds

        except Exception:
            continue

    raise RuntimeError(f"No PET file found in {case_dir}")


def get_radiopharmaceutical_value(ds, name, default=None):
    seq = getattr(
        ds,
        "RadiopharmaceuticalInformationSequence",
        None
    )

    if not seq:
        return default

    item = seq[0]

    return getattr(
        item,
        name,
        default
    )


ROOT = Path("development_cases")

print("===== PET METADATA AUDIT =====")

for case in CASES:

    case_dir = ROOT / case

    path, ds = get_first_pet_file(case_dir)

    slope = getattr(
        ds,
        "RescaleSlope",
        None
    )

    intercept = getattr(
        ds,
        "RescaleIntercept",
        None
    )

    weight = getattr(
        ds,
        "PatientWeight",
        None
    )

    units = getattr(
        ds,
        "Units",
        None
    )

    decay = getattr(
        ds,
        "DecayCorrection",
        None
    )

    dose = get_radiopharmaceutical_value(
        ds,
        "RadionuclideTotalDose"
    )

    start_time = get_radiopharmaceutical_value(
        ds,
        "RadiopharmaceuticalStartTime"
    )

    series_uid = getattr(
        ds,
        "SeriesInstanceUID",
        None
    )

    print()
    print("CASE:", case)
    print("PET file:", path)
    print("SeriesInstanceUID:", series_uid)
    print("RescaleSlope:", slope)
    print("RescaleIntercept:", intercept)
    print("PatientWeightKg:", weight)
    print("InjectedDoseBq:", dose)
    print("Units:", units)
    print("DecayCorrection:", decay)
    print("RadiopharmaceuticalStartTime:", start_time)

print()
print("===== DONE =====")