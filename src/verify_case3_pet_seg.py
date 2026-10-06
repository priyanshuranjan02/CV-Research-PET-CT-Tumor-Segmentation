from pathlib import Path
import pydicom
import SimpleITK as sitk


ROOT = Path("development_cases/PETCT_04ab5c61c9")

CT_UID = "1.3.6.1.4.1.14519.5.2.1.232341950326768109489919195931758650209"
PT_UID = "1.3.6.1.4.1.14519.5.2.1.184146994116738426472805672529258365051"
SEG_UID = "1.3.6.1.4.1.14519.5.2.1.103712160193489524006965439384048373756"


# ---------------------------------------------------------
# Find DICOM files
# ---------------------------------------------------------
ct_files = []
pt_files = []
seg_file = None

for path in ROOT.rglob("*.dcm"):
    try:
        ds = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True
        )

        uid = getattr(ds, "SeriesInstanceUID", None)

        if uid == CT_UID:
            ct_files.append(path)

        elif uid == PT_UID:
            pt_files.append(path)

        elif uid == SEG_UID:
            seg_file = path

    except Exception:
        pass


print("===== CASE 3 PET / SEG VERIFICATION =====")
print("CT files:", len(ct_files))
print("PET files:", len(pt_files))
print("SEG file:", seg_file)


if len(ct_files) != 340:
    raise RuntimeError(f"Expected 340 CT files, found {len(ct_files)}")

if len(pt_files) != 284:
    raise RuntimeError(f"Expected 284 PET files, found {len(pt_files)}")

if seg_file is None:
    raise RuntimeError("SEG file not found.")


# ---------------------------------------------------------
# Load PET with SimpleITK
# ---------------------------------------------------------
reader = sitk.ImageSeriesReader()

pt_names = reader.GetGDCMSeriesFileNames(
    str(pt_files[0].parent),
    PT_UID
)

reader.SetFileNames(pt_names)
pet_img = reader.Execute()


print()
print("===== PET GEOMETRY =====")
print("Size:", pet_img.GetSize())
print("Spacing:", pet_img.GetSpacing())
print("Origin:", pet_img.GetOrigin())
print("Direction:", pet_img.GetDirection())


# ---------------------------------------------------------
# Read SEG DICOM metadata
# ---------------------------------------------------------
seg = pydicom.dcmread(
    str(seg_file),
    force=True
)

print()
print("===== SEG METADATA =====")
print("Rows:", getattr(seg, "Rows", None))
print("Columns:", getattr(seg, "Columns", None))
print("NumberOfFrames:", getattr(seg, "NumberOfFrames", None))
print("FrameOfReferenceUID:",
      getattr(seg, "FrameOfReferenceUID", None))


# ---------------------------------------------------------
# Segment information
# ---------------------------------------------------------
segments = (
    seg.SegmentSequence
    if hasattr(seg, "SegmentSequence")
    else []
)

for item in segments:
    print()
    print("Segment Number:", getattr(item, "SegmentNumber", None))
    print("Segment Label:", getattr(item, "SegmentLabel", None))
    print("Segment Algorithm Type:",
          getattr(item, "SegmentAlgorithmType", None))


# ---------------------------------------------------------
# Check referenced PET series
# ---------------------------------------------------------
referenced_series = set()

try:
    for item in seg.ReferencedSeriesSequence:
        referenced_series.add(
            item.SeriesInstanceUID
        )
except AttributeError:
    pass


# Some SEG files store references deeper in the structure
if not referenced_series:

    try:
        for study in seg.ReferencedStudySequence:
            for series in study.ReferencedSeriesSequence:
                referenced_series.add(
                    series.SeriesInstanceUID
                )
    except AttributeError:
        pass


print()
print("===== PET REFERENCE =====")
print("Referenced PET Series UID(s):")
for uid in referenced_series:
    print(uid)

print("Expected PET Series UID:")
print(PT_UID)

print(
    "PET reference match:",
    PT_UID in referenced_series
)


# ---------------------------------------------------------
# Shared functional groups
# ---------------------------------------------------------
try:
    shared = seg.SharedFunctionalGroupsSequence[0]

    if hasattr(shared, "PixelMeasuresSequence"):
        pm = shared.PixelMeasuresSequence[0]

        print()
        print("===== SEG PIXEL MEASURES =====")
        print("Pixel Spacing:", pm.PixelSpacing)
        print("Slice Thickness:", getattr(pm, "SliceThickness", None))

    if hasattr(shared, "PlaneOrientationSequence"):
        po = shared.PlaneOrientationSequence[0]

        print()
        print("===== SEG ORIENTATION =====")
        print(
            "ImageOrientationPatient:",
            po.ImageOrientationPatient
        )

except Exception as e:
    print("Shared geometry unavailable:", e)


# ---------------------------------------------------------
# First-frame position
# ---------------------------------------------------------
try:
    frame0 = seg.PerFrameFunctionalGroupsSequence[0]

    if hasattr(frame0, "PlanePositionSequence"):
        pp = frame0.PlanePositionSequence[0]

        print()
        print("===== SEG FIRST FRAME POSITION =====")
        print(
            "ImagePositionPatient:",
            pp.ImagePositionPatient
        )

except Exception as e:
    print("Frame position unavailable:", e)


# ---------------------------------------------------------
# Compare PET FrameOfReferenceUID
# ---------------------------------------------------------
first_pet = pydicom.dcmread(
    str(pt_files[0]),
    stop_before_pixels=True
)

pet_for_uid = getattr(
    first_pet,
    "FrameOfReferenceUID",
    None
)

seg_for_uid = getattr(
    seg,
    "FrameOfReferenceUID",
    None
)

print()
print("===== FRAME OF REFERENCE =====")
print("PET FrameOfReferenceUID:", pet_for_uid)
print("SEG FrameOfReferenceUID:", seg_for_uid)
print("Frame of Reference match:", pet_for_uid == seg_for_uid)


print()
print("===== COMPLETE =====")