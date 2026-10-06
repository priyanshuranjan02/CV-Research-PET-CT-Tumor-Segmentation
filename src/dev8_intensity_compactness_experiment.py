from pathlib import Path
import sys
import io
from contextlib import redirect_stdout

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ============================================================
# LOAD COMMON DICOM / PET HELPERS
# ============================================================

with redirect_stdout(io.StringIO()):

    source = (
        SRC
        / "v4_neighbor_dual_core_experiment.py"
    ).read_text()

    prefix = source.split(
        "# MAIN",
        1
    )[0]

    exec(
        compile(
            prefix,
            "v4_neighbor_dual_core_experiment.py",
            "exec"
        ),
        globals()
    )


# ============================================================
# SIX VERIFIED DEVELOPMENT CASES
# ============================================================

CASES = [
    "PETCT_0168f65af8",
    "PETCT_04606080a0",
    "PETCT_04ab5c61c9",
    "PETCT_0b57b247b6",
    "PETCT_11afab3485",
    "PETCT_185da4c8b6",
]


# ============================================================
# EXPERIMENT PARAMETERS
# ============================================================

PET_THRESHOLD = 2.25

MIN_CONTOUR_AREA = 100

MIN_PET_COMPONENT_AREA = 20

MIN_3D_VOXELS = 75
MIN_3D_SLICES = 2


# ============================================================
# NO HARD UPPER VOLUME CAP
# ============================================================

VOLUME_POLICY = "NO_CAP"


# ============================================================
# TOP-K VALUES
# ============================================================

TOP_K_VALUES = [
    1,
    2,
    3,
    4,
    5,
    6,
]


# ============================================================
# COMPACTNESS EXPONENTS
#
# score =
#     SUV * compactness^alpha
#
# alpha=0:
#     equivalent to raw SUV ranking
#
# Higher alpha:
#     stronger preference for compact components
# ============================================================

COMPACTNESS_EXPONENTS = [
    0.10,
    0.20,
    0.30,
    0.40,
]


# ============================================================
# SCORE FEATURES
# ============================================================

SCORE_FEATURES = [
    "mean_suv",
    "median_suv",
]


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "development_cases"
    / "dev8_intensity_compactness_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SUMMARY_CSV = (
    OUTPUT_DIR
    / "dev8_intensity_compactness_summary.csv"
)

COMPONENT_CSV = (
    OUTPUT_DIR
    / "dev8_intensity_compactness_components.csv"
)


# ============================================================
# UNION BODY MASK
#
# Same body ROI used in the previous diagnostic work.
# ============================================================

def build_union_body_mask(ct):

    body = np.zeros_like(
        ct,
        dtype=np.uint8
    )

    kernel = np.ones(
        (11, 11),
        np.uint8
    )

    for z in range(
        ct.shape[0]
    ):

        binary = (
            ct[z] > -900
        ).astype(
            np.uint8
        ) * 255

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            if cv2.contourArea(
                contour
            ) <= 0:
                continue

            cv2.drawContours(
                body[z],
                [contour],
                -1,
                1,
                -1
            )

    return body.astype(bool)


# ============================================================
# LOCAL PET PARAMETER READER
# ============================================================

def read_pet_parameters_local(pet_dir):

    reader = sitk.ImageSeriesReader()

    files = reader.GetGDCMSeriesFileNames(
        str(pet_dir)
    )

    if not files:

        raise RuntimeError(
            f"No PET DICOM files found in {pet_dir}"
        )

    slopes = []
    intercepts = []

    weight = None
    dose = None
    units = None

    for file_path in files:

        ds = pydicom.dcmread(
            str(file_path),
            stop_before_pixels=True,
            force=True
        )

        slopes.append(
            float(
                getattr(
                    ds,
                    "RescaleSlope",
                    1.0
                )
            )
        )

        intercepts.append(
            float(
                getattr(
                    ds,
                    "RescaleIntercept",
                    0.0
                )
            )
        )

        # ----------------------------------------------------
        # Patient weight
        # ----------------------------------------------------

        if weight is None:

            value = getattr(
                ds,
                "PatientWeight",
                None
            )

            if value is not None:
                weight = float(value)

        # ----------------------------------------------------
        # Injected dose
        # ----------------------------------------------------

        if dose is None:

            seq = getattr(
                ds,
                "RadiopharmaceuticalInformationSequence",
                None
            )

            if seq:

                dose_value = getattr(
                    seq[0],
                    "RadionuclideTotalDose",
                    None
                )

                if dose_value is not None:
                    dose = float(
                        dose_value
                    )

        # ----------------------------------------------------
        # PET Units
        # ----------------------------------------------------

        if units is None:

            units = getattr(
                ds,
                "Units",
                None
            )

    if weight is None:

        raise RuntimeError(
            f"PatientWeight not found in {pet_dir}"
        )

    if dose is None:

        raise RuntimeError(
            f"Injected dose not found in {pet_dir}"
        )

    return (
        np.asarray(
            slopes,
            dtype=np.float64
        ),
        np.asarray(
            intercepts,
            dtype=np.float64
        ),
        weight,
        dose,
        units,
        files
    )


# ============================================================
# LOAD ONE DEVELOPMENT CASE
# ============================================================

def load_case(case_name):

    case_dir = (
        ROOT
        / "development_cases"
        / case_name
    )

    if not case_dir.exists():

        raise FileNotFoundError(
            f"Missing case directory: {case_dir}"
        )

    print()
    print("=" * 110)
    print(
        f"===== LOADING {case_name} ====="
    )
    print("=" * 110)

    # ========================================================
    # CT
    # ========================================================

    ct_matches = [

        p

        for p in case_dir.rglob("*")

        if (
            p.is_dir()
            and "GK p.v.3" in p.name
            and "verified_download" not in p.parts
            and "__MACOSX" not in p.parts
        )
    ]

    if len(ct_matches) != 1:

        raise RuntimeError(
            f"{case_name}: expected exactly one clean CT "
            f"directory, found {len(ct_matches)}"
        )

    ct_dir = ct_matches[0]

    ct_img = read_series(
        ct_dir
    )

    ct = (
        sitk.GetArrayFromImage(
            ct_img
        )
        .astype(
            np.float32
        )
    )

    # ========================================================
    # PET
    # ========================================================

    pet_matches = [

        p

        for p in case_dir.rglob("*")

        if (
            p.is_dir()
            and "PET corr." in p.name
            and "verified_download" not in p.parts
            and "__MACOSX" not in p.parts
        )
    ]

    if len(pet_matches) != 1:

        raise RuntimeError(
            f"{case_name}: expected exactly one clean PET "
            f"directory, found {len(pet_matches)}"
        )

    pet_dir = pet_matches[0]

    pet_img = read_series(
        pet_dir
    )

    (
        pet_slopes,
        pet_intercepts,
        patient_weight,
        injected_dose,
        pet_units,
        pet_files
    ) = read_pet_parameters_local(
        pet_dir
    )

    pet_stored = (
        sitk.GetArrayFromImage(
            pet_img
        )
        .astype(
            np.float64
        )
    )

    if (
        pet_stored.shape[0]
        != len(pet_slopes)
    ):

        raise RuntimeError(
            f"{case_name}: PET slice count "
            f"{pet_stored.shape[0]} != "
            f"calibration count "
            f"{len(pet_slopes)}"
        )

    # ========================================================
    # NATIVE PET -> SUV
    #
    # IMPORTANT:
    # SimpleITK already applies
    # RescaleSlope / RescaleIntercept.
    #
    # DO NOT apply them again.
    # ========================================================

    if str(
        pet_units
    ).upper() != "BQML":

        raise RuntimeError(
            f"{case_name}: expected PET units BQML, "
            f"found {pet_units}"
        )

    suv_native = calculate_suv(
        pet_stored,
        patient_weight,
        injected_dose
    )

    # ========================================================
    # PET SUV -> CT SPACE
    # ========================================================

    suv_native_img = (
        sitk.GetImageFromArray(
            suv_native.astype(
                np.float32
            )
        )
    )

    suv_native_img.CopyInformation(
        pet_img
    )

    suv_ct_img = sitk.Resample(
        suv_native_img,
        ct_img,
        sitk.Transform(),
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32
    )

    suv = (
        sitk.GetArrayFromImage(
            suv_ct_img
        )
        .astype(
            np.float64
        )
    )

    # ========================================================
    # GROUND TRUTH
    #
    # Evaluation only.
    # ========================================================

    gt_path = (
        case_dir
        / "tumor_mask_ct_visible.nii.gz"
    )

    if not gt_path.exists():

        raise FileNotFoundError(
            f"{case_name}: GT not found: "
            f"{gt_path}"
        )

    gt_img = sitk.ReadImage(
        str(gt_path)
    )

    gt = (
        sitk.GetArrayFromImage(
            gt_img
        ) > 0
    )

    # ========================================================
    # BODY
    # ========================================================

    body = build_union_body_mask(
        ct
    )

    # ========================================================
    # LOG
    # ========================================================

    print(
        "CT:",
        ct.shape
    )

    print(
        "PET:",
        pet_stored.shape
    )

    print(
        "SUV max:",
        round(
            float(
                np.nanmax(suv)
            ),
            4
        )
    )

    print(
        "GT voxels:",
        int(
            gt.sum()
        )
    )

    print(
        "GT-positive slices:",
        int(
            np.any(
                gt,
                axis=(1, 2)
            ).sum()
        )
    )

    return {
        "name":
            case_name,

        "case_dir":
            case_dir,

        "ct_img":
            ct_img,

        "ct":
            ct,

        "suv":
            suv,

        "gt":
            gt,

        "body":
            body,
    }


# ============================================================
# GENERATE 3D PET COMPONENTS
#
# Pipeline:
#
#   CT
#     ↓
#   Adaptive threshold
#     ↓
#   Morphological opening
#     ↓
#   Union body ROI
#     ↓
#   CT contours
#     ↓
#   PET >= 2.25
#     ↓
#   2D PET connected components
#     ↓
#   3D connected components
#     ↓
#   >=75 voxels and >=2 slices
#
# GT is NOT used during generation.
# ============================================================

def generate_3d_components(case):

    ct = case["ct"]
    suv = case["suv"]
    body = case["body"]

    ct_display = normalize_ct(
        ct
    )

    opening_kernel = np.ones(
        (5, 5),
        np.uint8
    )

    raw_volume = np.zeros(
        suv.shape,
        dtype=np.uint8
    )

    contour_count = 0
    pet_component_count = 0

    # ========================================================
    # SLICE-WISE CANDIDATE GENERATION
    # ========================================================

    for z in range(
        ct.shape[0]
    ):

        body_slice = body[z]

        if not np.any(
            body_slice
        ):
            continue

        # ----------------------------------------------------
        # Adaptive CT threshold
        # ----------------------------------------------------

        adaptive = cv2.adaptiveThreshold(
            ct_display[z],
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15,
            3
        )

        # ----------------------------------------------------
        # Morphological opening
        # ----------------------------------------------------

        opened = cv2.morphologyEx(
            adaptive,
            cv2.MORPH_OPEN,
            opening_kernel
        )

        # ----------------------------------------------------
        # Body restriction
        # ----------------------------------------------------

        opened = np.where(
            body_slice,
            opened,
            0
        ).astype(
            np.uint8
        )

        # ----------------------------------------------------
        # Border exclusion
        # ----------------------------------------------------

        opened[:5, :] = 0
        opened[-5:, :] = 0
        opened[:, :5] = 0
        opened[:, -5:] = 0

        # ----------------------------------------------------
        # Find CT contours
        # ----------------------------------------------------

        contours, _ = cv2.findContours(
            opened,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            contour_area = float(
                cv2.contourArea(
                    contour
                )
            )

            if (
                contour_area
                < MIN_CONTOUR_AREA
            ):
                continue

            contour_count += 1

            contour_mask = np.zeros(
                opened.shape,
                dtype=np.uint8
            )

            cv2.drawContours(
                contour_mask,
                [contour],
                -1,
                1,
                -1
            )

            contour_pixels = (
                contour_mask > 0
            )

            # ------------------------------------------------
            # PET threshold inside CT contour
            # ------------------------------------------------

            hot = (
                contour_pixels
                & np.isfinite(
                    suv[z]
                )
                & (
                    suv[z]
                    >= PET_THRESHOLD
                )
            ).astype(
                np.uint8
            )

            if not np.any(
                hot
            ):
                continue

            (
                n_components,
                labels,
                stats,
                centroids
            ) = cv2.connectedComponentsWithStats(
                hot,
                connectivity=8
            )

            # ------------------------------------------------
            # Keep ALL valid 2D PET components
            # ------------------------------------------------

            for component_id in range(
                1,
                n_components
            ):

                area = int(
                    stats[
                        component_id,
                        cv2.CC_STAT_AREA
                    ]
                )

                if (
                    area
                    < MIN_PET_COMPONENT_AREA
                ):
                    continue

                component = (
                    labels
                    == component_id
                )

                raw_volume[z][
                    component
                ] = 1

                pet_component_count += 1

    # ========================================================
    # 3D CONNECTED COMPONENTS
    # ========================================================

    raw_img = sitk.GetImageFromArray(
        raw_volume
    )

    cc_img = sitk.ConnectedComponent(
        raw_img
    )

    cc = sitk.GetArrayFromImage(
        cc_img
    )

    stats = sitk.LabelShapeStatisticsImageFilter()

    stats.Execute(
        cc_img
    )

    components = []
    label_masks = {}

    # ========================================================
    # EXTRACT COMPONENT FEATURES
    # ========================================================

    for label in stats.GetLabels():

        component = (
            cc == label
        )

        voxels = int(
            component.sum()
        )

        z_indices = np.where(
            component
        )[0]

        if len(
            z_indices
        ) == 0:
            continue

        z_start = int(
            z_indices.min()
        )

        z_end = int(
            z_indices.max()
        )

        z_span = (
            z_end
            - z_start
            + 1
        )

        # ----------------------------------------------------
        # Minimum 3D coherence
        # ----------------------------------------------------

        if (
            voxels
            < MIN_3D_VOXELS
        ):
            continue

        if (
            z_span
            < MIN_3D_SLICES
        ):
            continue

        # ====================================================
        # 3D BOUNDING BOX
        # ====================================================

        coords = np.where(
            component
        )

        z_min = int(
            coords[0].min()
        )

        z_max = int(
            coords[0].max()
        )

        y_min = int(
            coords[1].min()
        )

        y_max = int(
            coords[1].max()
        )

        x_min = int(
            coords[2].min()
        )

        x_max = int(
            coords[2].max()
        )

        bbox_z = (
            z_max
            - z_min
            + 1
        )

        bbox_y = (
            y_max
            - y_min
            + 1
        )

        bbox_x = (
            x_max
            - x_min
            + 1
        )

        bbox_volume = (
            bbox_z
            * bbox_y
            * bbox_x
        )

        # ----------------------------------------------------
        # Bounding-box compactness / extent
        #
        # range approximately:
        #   0 < compactness <= 1
        # ----------------------------------------------------

        compactness = (
            voxels
            / max(
                bbox_volume,
                1
            )
        )

        # ====================================================
        # PET FEATURES
        # ====================================================

        values = suv[
            component
        ]

        if values.size == 0:
            continue

        mean_suv = float(
            np.mean(
                values
            )
        )

        median_suv = float(
            np.median(
                values
            )
        )

        max_suv = float(
            np.max(
                values
            )
        )

        # ====================================================
        # GT AUDIT
        #
        # Evaluation only.
        # ====================================================

        overlap = int(
            np.logical_and(
                component,
                case["gt"]
            ).sum()
        )

        gt_voxels = int(
            case["gt"].sum()
        )

        dice = (
            2.0
            * overlap
            / (
                voxels
                + gt_voxels
            )
            if (
                voxels
                + gt_voxels
            ) > 0
            else 0.0
        )

        union = (
            voxels
            + gt_voxels
            - overlap
        )

        iou = (
            overlap
            / union
            if union > 0
            else 0.0
        )

        label_masks[
            int(label)
        ] = component

        components.append(
            {
                "label":
                    int(label),

                "voxels":
                    voxels,

                "z_start":
                    z_start,

                "z_end":
                    z_end,

                "z_span":
                    z_span,

                "bbox_x":
                    bbox_x,

                "bbox_y":
                    bbox_y,

                "bbox_z":
                    bbox_z,

                "bbox_volume":
                    int(bbox_volume),

                "compactness":
                    float(compactness),

                "mean_suv":
                    mean_suv,

                "median_suv":
                    median_suv,

                "max_suv":
                    max_suv,

                "overlap_gt":
                    overlap,

                "dice_gt":
                    float(dice),

                "iou_gt":
                    float(iou),
            }
        )

    components_df = pd.DataFrame(
        components
    )

    # ========================================================
    # ADD COMPACTNESS SCORES
    # ========================================================

    if not components_df.empty:

        for alpha in COMPACTNESS_EXPONENTS:

            alpha_name = (
                f"{alpha:.2f}"
                .replace(".", "p")
            )

            components_df[
                f"mean_suv_compact_a{alpha_name}"
            ] = (
                components_df[
                    "mean_suv"
                ]
                * np.power(
                    np.maximum(
                        components_df[
                            "compactness"
                        ].astype(float),
                        1e-12
                    ),
                    alpha
                )
            )

            components_df[
                f"median_suv_compact_a{alpha_name}"
            ] = (
                components_df[
                    "median_suv"
                ]
                * np.power(
                    np.maximum(
                        components_df[
                            "compactness"
                        ].astype(float),
                        1e-12
                    ),
                    alpha
                )
            )

    return (
        components_df,
        label_masks,
        {
            "contours":
                contour_count,

            "pet_components_2d":
                pet_component_count,

            "raw_volume":
                raw_volume,
        }
    )


# ============================================================
# EVALUATE COMPONENT SELECTION
# ============================================================

def evaluate_selection(
    case,
    components,
    label_masks,
    score_column,
    top_k
):

    if components.empty:

        return {
            "eligible_components":
                0,

            "selected_components":
                0,

            "prediction_voxels":
                0,

            "gt_overlap":
                0,

            "dice":
                0.0,

            "iou":
                0.0,

            "slice_recall":
                0.0,

            "fpr":
                0.0,

            "predicted_slices":
                0,

            "tp_slices":
                0,

            "fp_slices":
                0,

            "fn_slices":
                int(
                    np.any(
                        case["gt"],
                        axis=(1, 2)
                    ).sum()
                ),

            "labels":
                ""
        }

    # ========================================================
    # NO HARD UPPER CAP
    # ========================================================

    eligible = components.copy()

    # ========================================================
    # RANK
    #
    # Primary:
    #   selected score
    #
    # Secondary:
    #   median SUV
    #
    # Tertiary:
    #   mean SUV
    #
    # Final:
    #   smaller volume
    # ========================================================

    ranked = eligible.sort_values(
        [
            score_column,
            "median_suv",
            "mean_suv",
            "voxels"
        ],
        ascending=[
            False,
            False,
            False,
            True
        ]
    )

    selected = ranked.head(
        top_k
    )

    # ========================================================
    # BUILD PREDICTION
    # ========================================================

    prediction = np.zeros_like(
        case["gt"],
        dtype=np.uint8
    )

    for label in selected[
        "label"
    ]:

        prediction[
            label_masks[
                int(label)
            ]
        ] = 1

    pred = (
        prediction > 0
    )

    gt = case["gt"]

    # ========================================================
    # VOXEL METRICS
    # ========================================================

    overlap = int(
        np.logical_and(
            pred,
            gt
        ).sum()
    )

    pred_voxels = int(
        pred.sum()
    )

    gt_voxels = int(
        gt.sum()
    )

    dice = (
        2.0
        * overlap
        / (
            pred_voxels
            + gt_voxels
        )
        if (
            pred_voxels
            + gt_voxels
        ) > 0
        else 0.0
    )

    union = (
        pred_voxels
        + gt_voxels
        - overlap
    )

    iou = (
        overlap
        / union
        if union > 0
        else 0.0
    )

    # ========================================================
    # SLICE METRICS
    # ========================================================

    gt_positive = np.any(
        gt,
        axis=(1, 2)
    )

    pred_positive = np.any(
        pred,
        axis=(1, 2)
    )

    tp = np.logical_and(
        gt_positive,
        pred_positive
    ).sum()

    fp = np.logical_and(
        ~gt_positive,
        pred_positive
    ).sum()

    fn = np.logical_and(
        gt_positive,
        ~pred_positive
    ).sum()

    slice_recall = (
        tp
        / max(
            gt_positive.sum(),
            1
        )
    )

    fpr = (
        fp
        / max(
            (~gt_positive).sum(),
            1
        )
    )

    return {

        "eligible_components":
            int(
                len(eligible)
            ),

        "selected_components":
            int(
                len(selected)
            ),

        "prediction_voxels":
            pred_voxels,

        "gt_overlap":
            overlap,

        "dice":
            float(dice),

        "iou":
            float(iou),

        "slice_recall":
            float(slice_recall),

        "fpr":
            float(fpr),

        "predicted_slices":
            int(
                pred_positive.sum()
            ),

        "tp_slices":
            int(tp),

        "fp_slices":
            int(fp),

        "fn_slices":
            int(fn),

        "labels":
            ",".join(
                str(
                    int(x)
                )
                for x in selected[
                    "label"
                ]
            )
    }


# ============================================================
# BUILD EXPERIMENT CONFIGURATIONS
# ============================================================

def build_configurations():

    configurations = []

    # --------------------------------------------------------
    # Raw baseline rankings
    # --------------------------------------------------------

    configurations.append(
        {
            "score_name":
                "mean_suv",

            "score_column":
                "mean_suv",

            "alpha":
                0.0
        }
    )

    configurations.append(
        {
            "score_name":
                "median_suv",

            "score_column":
                "median_suv",

            "alpha":
                0.0
        }
    )

    # --------------------------------------------------------
    # Compactness-weighted rankings
    # --------------------------------------------------------

    for alpha in COMPACTNESS_EXPONENTS:

        alpha_name = (
            f"{alpha:.2f}"
            .replace(".", "p")
        )

        configurations.append(
            {
                "score_name":
                    f"mean_suv_compact_a{alpha_name}",

                "score_column":
                    f"mean_suv_compact_a{alpha_name}",

                "alpha":
                    alpha
            }
        )

        configurations.append(
            {
                "score_name":
                    f"median_suv_compact_a{alpha_name}",

                "score_column":
                    f"median_suv_compact_a{alpha_name}",

                "alpha":
                    alpha
            }
        )

    return configurations


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 115)
print(
    "===== DEV8 PET INTENSITY + 3D COMPACTNESS EXPERIMENT ====="
)
print("=" * 115)

print()

print(
    "PET threshold:",
    PET_THRESHOLD
)

print(
    "3D filter:",
    f">={MIN_3D_VOXELS} voxels, "
    f">={MIN_3D_SLICES} slices"
)

print(
    "Volume policy:",
    VOLUME_POLICY
)

print(
    "Compactness exponents:",
    COMPACTNESS_EXPONENTS
)

print(
    "K values:",
    TOP_K_VALUES
)


configurations = build_configurations()

print()

print(
    "Number of score configurations:",
    len(configurations)
)


# ============================================================
# STORAGE
# ============================================================

all_results = []
all_components = []


# ============================================================
# PROCESS SIX CASES
# ============================================================

for case_name in CASES:

    case = load_case(
        case_name
    )

    (
        components,
        label_masks,
        generation_stats
    ) = generate_3d_components(
        case
    )

    if components.empty:

        print(
            f"{case_name}: "
            "no 3D components survived."
        )

        continue

    print()

    print(
        f"{case_name}: "
        f"3D components="
        f"{len(components)}, "
        f"2D PET components="
        f"{generation_stats['pet_components_2d']}"
    )

    # --------------------------------------------------------
    # Component audit
    # --------------------------------------------------------

    for _, row in components.iterrows():

        row_dict = row.to_dict()

        row_dict[
            "case"
        ] = case_name

        row_dict[
            "threshold"
        ] = PET_THRESHOLD

        all_components.append(
            row_dict
        )

    # --------------------------------------------------------
    # Evaluate every scoring configuration
    # --------------------------------------------------------

    for configuration in configurations:

        score_name = configuration[
            "score_name"
        ]

        score_column = configuration[
            "score_column"
        ]

        alpha = configuration[
            "alpha"
        ]

        for top_k in TOP_K_VALUES:

            metrics = evaluate_selection(
                case,
                components,
                label_masks,
                score_column,
                top_k
            )

            all_results.append(
                {
                    "case":
                        case_name,

                    "score_name":
                        score_name,

                    "score_column":
                        score_column,

                    "compactness_alpha":
                        alpha,

                    "volume_policy":
                        VOLUME_POLICY,

                    "top_k":
                        top_k,

                    **metrics
                }
            )


# ============================================================
# DATAFRAMES
# ============================================================

results_df = pd.DataFrame(
    all_results
)

components_df = pd.DataFrame(
    all_components
)

if results_df.empty:

    raise RuntimeError(
        "No experiment results were generated."
    )


# ============================================================
# CASE-LEVEL RESULTS
# ============================================================

print()
print("=" * 115)
print(
    "===== CASE-LEVEL RESULTS ====="
)
print("=" * 115)

print()

case_display = (
    results_df[
        [
            "case",
            "score_name",
            "compactness_alpha",
            "top_k",
            "prediction_voxels",
            "dice",
            "iou",
            "slice_recall",
            "fpr",
            "eligible_components",
            "selected_components",
            "labels"
        ]
    ]
    .sort_values(
        [
            "score_name",
            "top_k",
            "case"
        ]
    )
)

print(
    case_display.to_string(
        index=False
    )
)


# ============================================================
# MACRO RESULTS
# ============================================================

macro = (
    results_df
    .groupby(
        [
            "score_name",
            "compactness_alpha",
            "volume_policy",
            "top_k"
        ],
        as_index=False
    )
    .agg(
        macro_dice=(
            "dice",
            "mean"
        ),

        macro_iou=(
            "iou",
            "mean"
        ),

        macro_slice_recall=(
            "slice_recall",
            "mean"
        ),

        macro_fpr=(
            "fpr",
            "mean"
        ),

        total_prediction_voxels=(
            "prediction_voxels",
            "sum"
        ),

        mean_prediction_voxels=(
            "prediction_voxels",
            "mean"
        )
    )
)


print()
print("=" * 115)
print(
    "===== MACRO DEVELOPMENT RESULTS ====="
)
print("=" * 115)

print()

print(
    macro.sort_values(
        "macro_dice",
        ascending=False
    ).to_string(
        index=False
    )
)


# ============================================================
# TOP 10 CONFIGURATIONS
# ============================================================

top10 = (
    macro
    .sort_values(
        [
            "macro_dice",
            "macro_slice_recall"
        ],
        ascending=[
            False,
            False
        ]
    )
    .head(10)
)


print()
print("=" * 115)
print(
    "===== TOP 10 CONFIGURATIONS BY MACRO DICE ====="
)
print("=" * 115)

print()

print(
    top10.to_string(
        index=False
    )
)


# ============================================================
# TOP CONFIGURATION BY EACH ALPHA
# ============================================================

print()
print("=" * 115)
print(
    "===== BEST CONFIGURATION FOR EACH COMPACTNESS ALPHA ====="
)
print("=" * 115)

print()

alpha_summary = []

for alpha in [
    0.0,
    *COMPACTNESS_EXPONENTS
]:

    subset = macro[
        macro[
            "compactness_alpha"
        ]
        == alpha
    ].copy()

    if subset.empty:
        continue

    best = (
        subset
        .sort_values(
            "macro_dice",
            ascending=False
        )
        .iloc[0]
    )

    alpha_summary.append(
        best.to_dict()
    )

alpha_summary_df = pd.DataFrame(
    alpha_summary
)

print(
    alpha_summary_df.to_string(
        index=False
    )
)


# ============================================================
# GT-OVERLAPPING COMPONENT AUDIT
#
# Evaluation / diagnostic only.
# ============================================================

print()
print("=" * 115)
print(
    "===== GT-OVERLAPPING 3D COMPONENT AUDIT ====="
)
print("=" * 115)

overlap_components = (
    components_df[
        components_df[
            "overlap_gt"
        ] > 0
    ]
    .copy()
)

if overlap_components.empty:

    print(
        "No overlapping components found."
    )

else:

    print()

    audit_columns = [

        "case",

        "label",

        "voxels",

        "z_start",

        "z_end",

        "z_span",

        "bbox_x",

        "bbox_y",

        "bbox_z",

        "bbox_volume",

        "compactness",

        "mean_suv",

        "median_suv",

        "max_suv",

        "overlap_gt",

        "dice_gt",

        "iou_gt"
    ]

    print(
        overlap_components
        .sort_values(
            [
                "case",
                "dice_gt"
            ],
            ascending=[
                True,
                False
            ]
        )[
            audit_columns
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    SUMMARY_CSV,
    index=False
)

components_df.to_csv(
    COMPONENT_CSV,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 115)
print(
    "===== DEV8 EXPERIMENT COMPLETE ====="
)
print("=" * 115)

print()

print(
    "Summary CSV:"
)

print(
    SUMMARY_CSV
)

print()

print(
    "Component CSV:"
)

print(
    COMPONENT_CSV
)