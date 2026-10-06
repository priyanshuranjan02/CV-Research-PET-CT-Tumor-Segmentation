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
# LOAD WORKING V4 HELPERS
#
# IMPORTANT:
# Do NOT import Dev7.
#
# Dev7 currently contains a syntax error. We therefore load
# only the reusable functions from the already-working V4
# implementation.
# ============================================================

V4_PATH = (
    SRC
    / "v4_neighbor_dual_core_experiment.py"
)

if not V4_PATH.exists():
    raise FileNotFoundError(
        f"Could not find V4 helper file:\n{V4_PATH}"
    )

with redirect_stdout(io.StringIO()):

    source = V4_PATH.read_text(
        encoding="utf-8"
    )

    if "# MAIN" not in source:
        raise RuntimeError(
            "Could not find '# MAIN' marker in V4 script."
        )

    prefix = source.split(
        "# MAIN",
        1
    )[0]

    exec(
        compile(
            prefix,
            str(V4_PATH),
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
# DEV9 PARAMETERS
# ============================================================

PET_THRESHOLD = 2.25

MIN_CONTOUR_AREA = 100
MIN_PET_COMPONENT_AREA = 20

MIN_3D_VOXELS = 75
MIN_3D_SLICES = 2

TOP_K_VALUES = [
    1,
    2,
    3,
    4,
    5,
    6,
]

SCORE_FEATURES = [
    "mean_suv",
    "median_suv",
    "p90_suv",
    "p95_suv",
    "max_suv",
    "mean_excess_suv",

    "mean_suv_hot3_a0p5",
    "mean_suv_hot4_a0p5",
    "mean_suv_hot5_a0p5",

    "p95_hot4_a0p5",
    "p95_hot5_a0p5",
]


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "development_cases"
    / "dev9_pet_intensity_distribution_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "dev9_pet_intensity_distribution_summary.csv"
)

COMPONENT_CSV = (
    OUTPUT_DIR
    / "dev9_pet_intensity_distribution_components.csv"
)


# ============================================================
# UNION BODY MASK
#
# Same body ROI used in previous diagnostic experiments.
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
#
# SimpleITK already applies the DICOM slope/intercept.
# Therefore we do NOT apply them again.
# ============================================================

def read_pet_parameters_local(
    pet_dir
):

    reader = sitk.ImageSeriesReader()

    files = reader.GetGDCMSeriesFileNames(
        str(pet_dir)
    )

    if not files:

        raise RuntimeError(
            f"No PET DICOM files found in {pet_dir}"
        )

    weight = None
    dose = None
    units = None

    slopes = []
    intercepts = []

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

        if weight is None:

            value = getattr(
                ds,
                "PatientWeight",
                None
            )

            if value is not None:
                weight = float(value)

        if dose is None:

            seq = getattr(
                ds,
                "RadiopharmaceuticalInformationSequence",
                None
            )

            if seq:

                value = getattr(
                    seq[0],
                    "RadionuclideTotalDose",
                    None
                )

                if value is not None:
                    dose = float(value)

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
# LOAD CASE
# ============================================================

def load_case(
    case_name
):

    case_dir = (
        ROOT
        / "development_cases"
        / case_name
    )

    if not case_dir.exists():

        raise FileNotFoundError(
            f"Missing case directory:\n{case_dir}"
        )

    print()
    print("=" * 100)
    print(
        f"===== LOADING {case_name} ====="
    )
    print("=" * 100)

    # --------------------------------------------------------
    # CT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PET
    # --------------------------------------------------------

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
            f"{pet_stored.shape[0]} != calibration count "
            f"{len(pet_slopes)}"
        )

    # --------------------------------------------------------
    # PET -> SUV
    #
    # IMPORTANT:
    # SimpleITK already applied DICOM rescaling.
    # --------------------------------------------------------

    if (
        str(pet_units).upper()
        != "BQML"
    ):

        raise RuntimeError(
            f"{case_name}: expected PET units BQML, "
            f"found {pet_units}"
        )

    suv_native = calculate_suv(
        pet_stored,
        patient_weight,
        injected_dose
    )

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

    # --------------------------------------------------------
    # PET SUV -> CT space
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    gt_path = (
        case_dir
        / "tumor_mask_ct_visible.nii.gz"
    )

    if not gt_path.exists():

        raise FileNotFoundError(
            f"{case_name}: GT not found:\n{gt_path}"
        )

    gt_img = sitk.ReadImage(
        str(gt_path)
    )

    gt = (
        sitk.GetArrayFromImage(
            gt_img
        ) > 0
    )

    # --------------------------------------------------------
    # Body
    # --------------------------------------------------------

    body = build_union_body_mask(
        ct
    )

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
        "name": case_name,
        "case_dir": case_dir,
        "ct_img": ct_img,
        "ct": ct,
        "suv": suv,
        "gt": gt,
        "body": body,
    }


# ============================================================
# GENERATE 3D PET COMPONENTS
#
# Candidate generation is unchanged from the previous
# component-selection experiments.
#
# PET threshold = 2.25
#
# GT is NOT used in candidate generation.
# ============================================================

def generate_3d_components(
    case
):

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

    # --------------------------------------------------------
    # Slice-wise candidate generation
    # --------------------------------------------------------

    for z in range(
        ct.shape[0]
    ):

        body_slice = body[z]

        if not np.any(
            body_slice
        ):
            continue

        adaptive = cv2.adaptiveThreshold(
            ct_display[z],
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15,
            3
        )

        opened = cv2.morphologyEx(
            adaptive,
            cv2.MORPH_OPEN,
            opening_kernel
        )

        opened = np.where(
            body_slice,
            opened,
            0
        ).astype(
            np.uint8
        )

        # ----------------------------------------------------
        # Remove border candidates
        # ----------------------------------------------------

        opened[:5, :] = 0
        opened[-5:, :] = 0
        opened[:, :5] = 0
        opened[:, -5:] = 0

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

            if contour_area < MIN_CONTOUR_AREA:
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

    # --------------------------------------------------------
    # 3D connected components
    # --------------------------------------------------------

    raw_img = sitk.GetImageFromArray(
        raw_volume
    )

    cc_img = sitk.ConnectedComponent(
        raw_img
    )

    cc = sitk.GetArrayFromImage(
        cc_img
    )

    stats = (
        sitk.LabelShapeStatisticsImageFilter()
    )

    stats.Execute(
        cc_img
    )

    components = []
    label_masks = {}

    # --------------------------------------------------------
    # Analyze every 3D component
    # --------------------------------------------------------

    for label in stats.GetLabels():

        label = int(label)

        component = (
            cc == label
        )

        voxels = int(
            component.sum()
        )

        if voxels < MIN_3D_VOXELS:
            continue

        coords = np.where(
            component
        )

        if len(coords[0]) == 0:
            continue

        z_indices = coords[0]

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

        if z_span < MIN_3D_SLICES:
            continue

        values = suv[
            component
        ]

        values = values[
            np.isfinite(values)
        ]

        if values.size == 0:
            continue

        # ----------------------------------------------------
        # Basic PET features
        # ----------------------------------------------------

        mean_suv = float(
            np.mean(values)
        )

        median_suv = float(
            np.median(values)
        )

        max_suv = float(
            np.max(values)
        )

        p90_suv = float(
            np.percentile(
                values,
                90
            )
        )

        p95_suv = float(
            np.percentile(
                values,
                95
            )
        )

        # ----------------------------------------------------
        # Hot voxel fractions
        # ----------------------------------------------------

        frac_ge_3 = float(
            np.mean(
                values >= 3.0
            )
        )

        frac_ge_4 = float(
            np.mean(
                values >= 4.0
            )
        )

        frac_ge_5 = float(
            np.mean(
                values >= 5.0
            )
        )

        # ----------------------------------------------------
        # Mean excess SUV
        # ----------------------------------------------------

        excess = np.maximum(
            values - PET_THRESHOLD,
            0.0
        )

        mean_excess_suv = float(
            np.mean(excess)
        )

        # ----------------------------------------------------
        # Distribution-aware scores
        # ----------------------------------------------------

        mean_suv_hot3_a0p5 = (
            mean_suv
            * np.sqrt(
                frac_ge_3
            )
        )

        mean_suv_hot4_a0p5 = (
            mean_suv
            * np.sqrt(
                frac_ge_4
            )
        )

        mean_suv_hot5_a0p5 = (
            mean_suv
            * np.sqrt(
                frac_ge_5
            )
        )

        p95_hot4_a0p5 = (
            p95_suv
            * np.sqrt(
                frac_ge_4
            )
        )

        p95_hot5_a0p5 = (
            p95_suv
            * np.sqrt(
                frac_ge_5
            )
        )

        # ----------------------------------------------------
        # GT audit
        #
        # Used ONLY for evaluation.
        # Never for ranking.
        # ----------------------------------------------------

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
            label
        ] = component

        components.append(
            {
                "label": label,
                "voxels": voxels,
                "z_start": z_start,
                "z_end": z_end,
                "z_span": z_span,

                "mean_suv": mean_suv,
                "median_suv": median_suv,
                "p90_suv": p90_suv,
                "p95_suv": p95_suv,
                "max_suv": max_suv,

                "frac_ge_3": frac_ge_3,
                "frac_ge_4": frac_ge_4,
                "frac_ge_5": frac_ge_5,

                "mean_excess_suv":
                    mean_excess_suv,

                "mean_suv_hot3_a0p5":
                    mean_suv_hot3_a0p5,

                "mean_suv_hot4_a0p5":
                    mean_suv_hot4_a0p5,

                "mean_suv_hot5_a0p5":
                    mean_suv_hot5_a0p5,

                "p95_hot4_a0p5":
                    p95_hot4_a0p5,

                "p95_hot5_a0p5":
                    p95_hot5_a0p5,

                "overlap_gt":
                    overlap,

                "dice_gt":
                    float(dice),

                "iou_gt":
                    float(iou),
            }
        )

    return (
        pd.DataFrame(
            components
        ),
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
# EVALUATE ONE CONFIGURATION
#
# Selection does NOT use GT.
# GT is used only after prediction is constructed.
# ============================================================

def evaluate_selection(
    case,
    components,
    label_masks,
    score_feature,
    top_k
):

    ranked = components.sort_values(
        [
            score_feature,
            "median_suv",
            "mean_suv",
        ],
        ascending=[
            False,
            False,
            False
        ]
    )

    selected = ranked.head(
        top_k
    )

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

    # --------------------------------------------------------
    # Voxel metrics
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Slice metrics
    # --------------------------------------------------------

    gt_positive = np.any(
        gt,
        axis=(1, 2)
    )

    pred_positive = np.any(
        pred,
        axis=(1, 2)
    )

    tp = int(
        np.logical_and(
            gt_positive,
            pred_positive
        ).sum()
    )

    fp = int(
        np.logical_and(
            ~gt_positive,
            pred_positive
        ).sum()
    )

    fn = int(
        np.logical_and(
            gt_positive,
            ~pred_positive
        ).sum()
    )

    slice_recall = (
        tp
        / max(
            int(gt_positive.sum()),
            1
        )
    )

    fpr = (
        fp
        / max(
            int((~gt_positive).sum()),
            1
        )
    )

    return {
        "selected_components":
            int(len(selected)),

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
            tp,

        "fp_slices":
            fp,

        "fn_slices":
            fn,

        "labels":
            ",".join(
                str(
                    int(x)
                )
                for x in selected[
                    "label"
                ]
            ),
    }


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 120)
print(
    "===== DEV9 PET INTENSITY-DISTRIBUTION COMPONENT SELECTION ====="
)
print("=" * 120)

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
    "NO HARD UPPER CAP"
)

print(
    "K values:",
    TOP_K_VALUES
)

print()
print(
    "Scores:"
)

for score in SCORE_FEATURES:
    print(
        "  -",
        score
    )


# ============================================================
# STORAGE
# ============================================================

all_results = []
all_components = []


# ============================================================
# PROCESS EACH CASE
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

    print()
    print(
        f"{case_name}: "
        f"3D components={len(components)}, "
        f"2D PET components="
        f"{generation_stats['pet_components_2d']}"
    )

    if components.empty:

        continue

    # --------------------------------------------------------
    # Store component audit
    # --------------------------------------------------------

    for _, row in components.iterrows():

        row_dict = row.to_dict()

        row_dict[
            "case"
        ] = case_name

        row_dict[
            "pet_threshold"
        ] = PET_THRESHOLD

        all_components.append(
            row_dict
        )

    # --------------------------------------------------------
    # Evaluate all score × K combinations
    # --------------------------------------------------------

    for score_feature in SCORE_FEATURES:

        for top_k in TOP_K_VALUES:

            metrics = evaluate_selection(
                case,
                components,
                label_masks,
                score_feature,
                top_k
            )

            all_results.append(
                {
                    "case":
                        case_name,

                    "score_feature":
                        score_feature,

                    "volume_policy":
                        "NO_CAP",

                    "top_k":
                        top_k,

                    "eligible_components":
                        int(
                            len(components)
                        ),

                    **metrics,
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


# ============================================================
# CASE-LEVEL RESULTS
# ============================================================

print()
print("=" * 120)
print("===== CASE-LEVEL RESULTS =====")
print("=" * 120)
print()

case_display = (
    results_df[
        [
            "case",
            "score_feature",
            "top_k",
            "prediction_voxels",
            "dice",
            "iou",
            "slice_recall",
            "fpr",
            "selected_components",
            "labels",
        ]
    ]
    .sort_values(
        [
            "score_feature",
            "top_k",
            "case",
        ]
    )
)

print(
    case_display.to_string(
        index=False
    )
)


# ============================================================
# MACRO AGGREGATION
# ============================================================

macro = (
    results_df
    .groupby(
        [
            "score_feature",
            "volume_policy",
            "top_k",
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
        ),
    )
)


# ============================================================
# MACRO RESULTS
# ============================================================

print()
print("=" * 120)
print("===== MACRO DEVELOPMENT RESULTS =====")
print("=" * 120)
print()

macro_sorted = (
    macro
    .sort_values(
        [
            "macro_dice",
            "macro_iou",
        ],
        ascending=[
            False,
            False
        ]
    )
)

print(
    macro_sorted.to_string(
        index=False
    )
)


# ============================================================
# TOP 15 CONFIGURATIONS
# ============================================================

print()
print("=" * 120)
print(
    "===== TOP 15 CONFIGURATIONS BY MACRO DICE ====="
)
print("=" * 120)
print()

print(
    macro_sorted
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# BEST K FOR EACH SCORE
# ============================================================

print()
print("=" * 120)
print("===== BEST K FOR EACH SCORE FEATURE =====")
print("=" * 120)
print()

best_per_score = (
    macro
    .sort_values(
        [
            "score_feature",
            "macro_dice",
            "macro_iou",
        ],
        ascending=[
            True,
            False,
            False,
        ]
    )
    .groupby(
        "score_feature",
        as_index=False
    )
    .head(1)
    .sort_values(
        "macro_dice",
        ascending=False
    )
)

print(
    best_per_score.to_string(
        index=False
    )
)


# ============================================================
# GT-OVERLAPPING COMPONENT AUDIT
# ============================================================

print()
print("=" * 120)
print("===== GT-OVERLAPPING COMPONENT INTENSITY AUDIT =====")
print("=" * 120)
print()

if components_df.empty:

    print(
        "No components available."
    )

else:

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
            "No GT-overlapping components found."
        )

    else:

        audit_columns = [
            "case",
            "label",
            "voxels",
            "z_start",
            "z_end",
            "z_span",

            "mean_suv",
            "median_suv",
            "p90_suv",
            "p95_suv",
            "max_suv",

            "frac_ge_3",
            "frac_ge_4",
            "frac_ge_5",

            "mean_excess_suv",

            "mean_suv_hot3_a0p5",
            "mean_suv_hot4_a0p5",
            "mean_suv_hot5_a0p5",

            "p95_hot4_a0p5",
            "p95_hot5_a0p5",

            "overlap_gt",
            "dice_gt",
            "iou_gt",
        ]

        print(
            overlap_components
            .sort_values(
                [
                    "case",
                    "dice_gt",
                ],
                ascending=[
                    True,
                    False,
                ]
            )[
                audit_columns
            ]
            .to_string(
                index=False
            )
        )


# ============================================================
# GLOBAL FEATURE SUMMARY
# ============================================================

print()
print("=" * 120)
print("===== GLOBAL PET FEATURE SUMMARY =====")
print("=" * 120)
print()

feature_columns = [
    "mean_suv",
    "median_suv",
    "p90_suv",
    "p95_suv",
    "max_suv",
    "frac_ge_3",
    "frac_ge_4",
    "frac_ge_5",
    "mean_excess_suv",
]

if not components_df.empty:

    summary_rows = []

    gt_components = (
        components_df[
            components_df[
                "overlap_gt"
            ] > 0
        ]
    )

    for feature in feature_columns:

        all_values = (
            components_df[
                feature
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )

        gt_values = (
            gt_components[
                feature
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )

        summary_rows.append(
            {
                "feature":
                    feature,

                "all_components_mean":
                    float(
                        all_values.mean()
                    )
                    if not all_values.empty
                    else 0.0,

                "all_components_median":
                    float(
                        all_values.median()
                    )
                    if not all_values.empty
                    else 0.0,

                "gt_overlap_mean":
                    float(
                        gt_values.mean()
                    )
                    if not gt_values.empty
                    else 0.0,

                "gt_overlap_median":
                    float(
                        gt_values.median()
                    )
                    if not gt_values.empty
                    else 0.0,
            }
        )

    feature_summary_df = pd.DataFrame(
        summary_rows
    )

    print(
        feature_summary_df.to_string(
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
print("=" * 120)
print("===== DEV9 COMPLETE =====")
print("=" * 120)

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