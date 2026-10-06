from pathlib import Path
import sys
import io
import gc
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
# LOAD ONLY WORKING V4 HELPERS
# ============================================================

V4_PATH = SRC / "v4_neighbor_dual_core_experiment.py"

if not V4_PATH.exists():
    raise FileNotFoundError(
        f"Missing V4 helper file:\n{V4_PATH}"
    )

with redirect_stdout(io.StringIO()):

    source = V4_PATH.read_text(
        encoding="utf-8"
    )

    if "# MAIN" not in source:
        raise RuntimeError(
            "Could not find '# MAIN' in V4 script."
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
# CASES
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
# PARAMETERS
# ============================================================

PET_THRESHOLD = 2.25

HOT_THRESHOLD = 3.0

MIN_CONTOUR_AREA = 100
MIN_PET_COMPONENT_AREA = 20

MIN_3D_VOXELS = 75
MIN_3D_SLICES = 2

CORE_POLICIES = [
    "P70",
    "P80",
    "P90",
]

TOP_K_VALUES = [
    1,
    2,
    3,
    4,
]


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "development_cases"
    / "dev10_lite_hot_core_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "dev10_lite_summary.csv"
)

COMPONENT_CSV = (
    OUTPUT_DIR
    / "dev10_lite_components.csv"
)


# ============================================================
# BODY MASK
# ============================================================

def build_union_body_mask(ct):

    body = np.zeros(
        ct.shape,
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
# PET PARAMETER READER
#
# SimpleITK has already applied DICOM rescale slope/intercept.
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
            f"PatientWeight not found: {pet_dir}"
        )

    if dose is None:
        raise RuntimeError(
            f"Injected dose not found: {pet_dir}"
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
            f"{case_name}: expected one clean CT directory, "
            f"found {len(ct_matches)}"
        )

    ct_dir = ct_matches[0]

    ct_img = read_series(
        ct_dir
    )

    # Keep CT as float32.
    # It will be deleted after candidate generation.
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
            f"{case_name}: expected one clean PET directory, "
            f"found {len(pet_matches)}"
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
            np.float32
        )
    )

    if (
        pet_stored.shape[0]
        != len(pet_slopes)
    ):

        raise RuntimeError(
            f"{case_name}: PET slices "
            f"{pet_stored.shape[0]} != "
            f"calibration count "
            f"{len(pet_slopes)}"
        )

    if (
        str(pet_units).upper()
        != "BQML"
    ):

        raise RuntimeError(
            f"{case_name}: expected BQML, "
            f"found {pet_units}"
        )

    # --------------------------------------------------------
    # Correct SUV calculation.
    #
    # No second DICOM rescale application.
    # --------------------------------------------------------

    suv_native = calculate_suv(
        pet_stored,
        patient_weight,
        injected_dose
    ).astype(
        np.float32
    )

    suv_native_img = sitk.GetImageFromArray(
        suv_native
    )

    suv_native_img.CopyInformation(
        pet_img
    )

    # --------------------------------------------------------
    # PET -> CT space
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
            np.float32
        )
    )

    # --------------------------------------------------------
    # GT
    # --------------------------------------------------------

    gt_path = (
        case_dir
        / "tumor_mask_ct_visible.nii.gz"
    )

    if not gt_path.exists():

        raise FileNotFoundError(
            f"Missing GT:\n{gt_path}"
        )

    gt_img = sitk.ReadImage(
        str(gt_path)
    )

    gt = (
        sitk.GetArrayFromImage(
            gt_img
        ) > 0
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

    # Free large temporary objects.
    del (
        ct_img,
        pet_img,
        pet_stored,
        suv_native,
        suv_native_img,
        suv_ct_img,
        gt_img,
        pet_slopes,
        pet_intercepts,
        pet_files
    )

    gc.collect()

    return {
        "name": case_name,
        "ct": ct,
        "suv": suv,
        "gt": gt,
    }


# ============================================================
# GENERATE RAW 3D CANDIDATE VOLUME
#
# Same basic candidate generation used previously.
# ============================================================

def generate_candidate_volume(
    case
):

    ct = case["ct"]
    suv = case["suv"]

    body = build_union_body_mask(
        ct
    )

    ct_display = normalize_ct(
        ct
    )

    opening_kernel = np.ones(
        (5, 5),
        np.uint8
    )

    # 1 byte per voxel.
    raw_volume = np.zeros(
        suv.shape,
        dtype=np.uint8
    )

    contour_count = 0
    pet_component_count = 0

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

            area = float(
                cv2.contourArea(
                    contour
                )
            )

            if area < MIN_CONTOUR_AREA:
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

            hot = (
                contour_mask
                & (
                    suv[z]
                    >= PET_THRESHOLD
                ).astype(
                    np.uint8
                )
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

                if area < MIN_PET_COMPONENT_AREA:
                    continue

                component = (
                    labels
                    == component_id
                )

                raw_volume[z][
                    component
                ] = 1

                pet_component_count += 1

    # Free body-related temporary data.
    del (
        body,
        ct_display,
        ct
    )

    gc.collect()

    print(
        "2D PET components:",
        pet_component_count
    )

    print(
        "CT contours:",
        contour_count
    )

    return raw_volume


# ============================================================
# 3D COMPONENT ANALYSIS
#
# IMPORTANT:
# We do NOT store a mask for every component.
# Only the compact connected-component label volume is kept.
# ============================================================

def build_3d_component_table(
    case,
    raw_volume
):

    suv = case["suv"]
    gt = case["gt"]

    raw_img = sitk.GetImageFromArray(
        raw_volume
    )

    cc_img = sitk.ConnectedComponent(
        raw_img
    )

    # Raw binary candidate volume no longer needed.
    del raw_img
    del raw_volume

    gc.collect()

    cc = (
        sitk.GetArrayFromImage(
            cc_img
        )
        .astype(
            np.int32,
            copy=False
        )
    )

    # We no longer need the SimpleITK CC image.
    del cc_img

    gc.collect()

    max_label = int(
        cc.max()
    )

    if max_label == 0:

        return (
            pd.DataFrame(),
            cc
        )

    # --------------------------------------------------------
    # First pass:
    # voxel count and z-range
    # --------------------------------------------------------

    voxel_counts = np.bincount(
        cc.ravel(),
        minlength=max_label + 1
    )

    z_start = np.full(
        max_label + 1,
        cc.shape[0],
        dtype=np.int32
    )

    z_end = np.full(
        max_label + 1,
        -1,
        dtype=np.int32
    )

    # --------------------------------------------------------
    # Second pass:
    # PET intensity sums and hot counts.
    #
    # Slice-wise to avoid creating huge temporary masks.
    # --------------------------------------------------------

    suv_sums = np.zeros(
        max_label + 1,
        dtype=np.float64
    )

    hot_counts = np.zeros(
        max_label + 1,
        dtype=np.int64
    )

    for z in range(
        cc.shape[0]
    ):

        labels = cc[z]

        positive = (
            labels > 0
        )

        if not np.any(
            positive
        ):
            continue

        label_values = labels[
            positive
        ]

        suv_values = suv[z][
            positive
        ]

        # Sum SUV by label.
        slice_sums = np.bincount(
            label_values,
            weights=suv_values,
            minlength=max_label + 1
        )

        suv_sums += slice_sums

        # Count hot voxels by label.
        hot = (
            suv_values
            >= HOT_THRESHOLD
        )

        if np.any(
            hot
        ):

            slice_hot = np.bincount(
                label_values[hot],
                minlength=max_label + 1
            )

            hot_counts += slice_hot

        labels_present = np.unique(
            label_values
        )

        z_start[
            labels_present
        ] = np.minimum(
            z_start[
                labels_present
            ],
            z
        )

        z_end[
            labels_present
        ] = np.maximum(
            z_end[
                labels_present
            ],
            z
        )

    # --------------------------------------------------------
    # Build component table
    # --------------------------------------------------------

    rows = []

    gt_voxels = int(
        gt.sum()
    )

    for label in range(
        1,
        max_label + 1
    ):

        voxels = int(
            voxel_counts[label]
        )

        if voxels < MIN_3D_VOXELS:
            continue

        if z_end[label] < 0:
            continue

        z_span = (
            int(z_end[label])
            - int(z_start[label])
            + 1
        )

        if z_span < MIN_3D_SLICES:
            continue

        mean_suv = (
            float(
                suv_sums[label]
                / voxels
            )
        )

        frac_hot = (
            float(
                hot_counts[label]
                / voxels
            )
        )

        # Dev9's best-style ranking feature.
        mean_suv_hot3_a0p5 = (
            mean_suv
            * np.sqrt(
                frac_hot
            )
        )

        # ----------------------------------------------------
        # GT audit
        #
        # A full component mask is created ONE AT A TIME.
        # ----------------------------------------------------

        component_mask = (
            cc == label
        )

        overlap = int(
            np.logical_and(
                component_mask,
                gt
            ).sum()
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

        rows.append(
            {
                "label":
                    label,

                "voxels":
                    voxels,

                "z_start":
                    int(
                        z_start[label]
                    ),

                "z_end":
                    int(
                        z_end[label]
                    ),

                "z_span":
                    z_span,

                "mean_suv":
                    mean_suv,

                "frac_ge_3":
                    frac_hot,

                "mean_suv_hot3_a0p5":
                    mean_suv_hot3_a0p5,

                "overlap_gt":
                    overlap,

                "dice_gt":
                    float(dice),

                "iou_gt":
                    float(iou),
            }
        )

        del component_mask

    components = pd.DataFrame(
        rows
    )

    return (
        components,
        cc
    )


# ============================================================
# GET CORE THRESHOLD
# ============================================================

def get_core_threshold(
    values,
    policy
):

    if policy == "P70":

        threshold = np.percentile(
            values,
            70
        )

    elif policy == "P80":

        threshold = np.percentile(
            values,
            80
        )

    elif policy == "P90":

        threshold = np.percentile(
            values,
            90
        )

    else:

        raise ValueError(
            f"Unknown policy: {policy}"
        )

    return max(
        float(threshold),
        PET_THRESHOLD
    )


# ============================================================
# BUILD REFINED CORE FOR ONE SELECTED COMPONENT
#
# No 3D connected-component calculation inside the core.
#
# We simply keep the high-SUV portion of the selected
# component.
# ============================================================

def build_core(
    cc,
    suv,
    label,
    policy
):

    component = (
        cc == label
    )

    values = suv[
        component
    ]

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:

        del component

        return (
            None,
            PET_THRESHOLD,
            0
        )

    threshold = get_core_threshold(
        values,
        policy
    )

    core = (
        component
        & np.isfinite(suv)
        & (
            suv
            >= threshold
        )
    )

    core_voxels = int(
        core.sum()
    )

    del (
        component,
        values
    )

    return (
        core,
        threshold,
        core_voxels
    )


# ============================================================
# EVALUATE ONE CONFIGURATION
#
# Only TOP-K selected components are refined.
# ============================================================

def evaluate_selection(
    case,
    components,
    cc,
    ranking_feature,
    core_policy,
    top_k
):

    ranked = (
        components
        .sort_values(
            [
                ranking_feature,
                "mean_suv",
            ],
            ascending=[
                False,
                False,
            ]
        )
    )

    selected = ranked.head(
        top_k
    )

    prediction = np.zeros_like(
        case["gt"],
        dtype=np.uint8
    )

    thresholds = []
    core_voxel_counts = []

    for label in selected[
        "label"
    ]:

        (
            core,
            threshold,
            core_voxels
        ) = build_core(
            cc,
            case["suv"],
            int(label),
            core_policy
        )

        thresholds.append(
            threshold
        )

        core_voxel_counts.append(
            core_voxels
        )

        if core is not None:

            prediction[
                core
            ] = 1

            del core

        gc.collect()

    pred = (
        prediction > 0
    )

    gt = case["gt"]

    # --------------------------------------------------------
    # Dice / IoU
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
            int(
                gt_positive.sum()
            ),
            1
        )
    )

    fpr = (
        fp
        / max(
            int(
                (~gt_positive).sum()
            ),
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

        "mean_core_voxels":
            float(
                np.mean(
                    core_voxel_counts
                )
            )
            if core_voxel_counts
            else 0.0,

        "labels":
            ",".join(
                str(
                    int(label)
                )
                for label in selected[
                    "label"
                ]
            ),

        "core_thresholds":
            ",".join(
                f"{x:.3f}"
                for x in thresholds
            ),
    }


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 115)
print(
    "===== DEV10-LITE HOT-CORE REFINEMENT ====="
)
print("=" * 115)

print()
print(
    "PET candidate threshold:",
    PET_THRESHOLD
)

print(
    "Hot feature threshold:",
    HOT_THRESHOLD
)

print(
    "3D filter:",
    f">={MIN_3D_VOXELS} voxels, "
    f">={MIN_3D_SLICES} slices"
)

print(
    "Core policies:",
    CORE_POLICIES
)

print(
    "Top-K:",
    TOP_K_VALUES
)

print(
    "Ranking:",
    "mean_suv_hot3_a0p5"
)


# ============================================================
# STORAGE
# ============================================================

all_results = []
all_component_rows = []


# ============================================================
# CASE LOOP
# ============================================================

for case_name in CASES:

    case = None
    raw_volume = None
    components = None
    cc = None

    try:

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        case = load_case(
            case_name
        )

        # ----------------------------------------------------
        # Candidate generation
        # ----------------------------------------------------

        raw_volume = (
            generate_candidate_volume(
                case
            )
        )

        # ----------------------------------------------------
        # 3D components
        # ----------------------------------------------------

        (
            components,
            cc
        ) = build_3d_component_table(
            case,
            raw_volume
        )

        raw_volume = None

        print()
        print(
            f"{case_name}: "
            f"eligible 3D components="
            f"{len(components)}"
        )

        if components.empty:

            print(
                "No eligible components."
            )

            continue

        # ----------------------------------------------------
        # Component audit
        # ----------------------------------------------------

        for _, row in components.iterrows():

            row_dict = row.to_dict()

            row_dict[
                "case"
            ] = case_name

            all_component_rows.append(
                row_dict
            )

        # ----------------------------------------------------
        # Evaluate only 3 × 4 = 12 configurations.
        # ----------------------------------------------------

        for core_policy in CORE_POLICIES:

            for top_k in TOP_K_VALUES:

                metrics = evaluate_selection(
                    case,
                    components,
                    cc,
                    "mean_suv_hot3_a0p5",
                    core_policy,
                    top_k
                )

                all_results.append(
                    {
                        "case":
                            case_name,

                        "ranking_feature":
                            "mean_suv_hot3_a0p5",

                        "core_policy":
                            core_policy,

                        "top_k":
                            top_k,

                        **metrics
                    }
                )

        # ----------------------------------------------------
        # Case summary
        # ----------------------------------------------------

        case_results = (
            pd.DataFrame(
                [
                    r
                    for r in all_results
                    if r["case"] == case_name
                ]
            )
        )

        best_case = (
            case_results
            .sort_values(
                [
                    "dice",
                    "iou",
                ],
                ascending=[
                    False,
                    False,
                ]
            )
            .iloc[0]
        )

        print()
        print(
            f"{case_name} best:"
        )

        print(
            "  Core:",
            best_case[
                "core_policy"
            ]
        )

        print(
            "  K:",
            int(
                best_case[
                    "top_k"
                ]
            )
        )

        print(
            "  Dice:",
            round(
                float(
                    best_case[
                        "dice"
                    ]
                ),
                6
            )
        )

        print(
            "  IoU:",
            round(
                float(
                    best_case[
                        "iou"
                    ]
                ),
                6
            )
        )

        print(
            "  Prediction voxels:",
            int(
                best_case[
                    "prediction_voxels"
                ]
            )
        )

    finally:

        # ----------------------------------------------------
        # Aggressive cleanup before next case.
        # ----------------------------------------------------

        del case
        del raw_volume
        del components
        del cc

        gc.collect()


# ============================================================
# DATAFRAMES
# ============================================================

results_df = pd.DataFrame(
    all_results
)

components_df = pd.DataFrame(
    all_component_rows
)


# ============================================================
# MACRO RESULTS
# ============================================================

print()
print("=" * 115)
print(
    "===== DEV10-LITE MACRO DEVELOPMENT RESULTS ====="
)
print("=" * 115)

macro = (
    results_df
    .groupby(
        [
            "ranking_feature",
            "core_policy",
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

        mean_core_voxels=(
            "mean_core_voxels",
            "mean"
        ),
    )
    .sort_values(
        [
            "macro_dice",
            "macro_iou",
        ],
        ascending=[
            False,
            False,
        ]
    )
)

print()

print(
    macro.to_string(
        index=False
    )
)


# ============================================================
# TOP 10
# ============================================================

print()
print("=" * 115)
print(
    "===== TOP 10 DEV10-LITE CONFIGURATIONS ====="
)
print("=" * 115)

print()

print(
    macro
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# BEST CORE POLICY
# ============================================================

print()
print("=" * 115)
print(
    "===== BEST CONFIGURATION FOR EACH CORE POLICY ====="
)
print("=" * 115)

best_policy = (
    macro
    .sort_values(
        [
            "core_policy",
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
        "core_policy",
        as_index=False
    )
    .head(1)
    .sort_values(
        "macro_dice",
        ascending=False
    )
)

print()

print(
    best_policy.to_string(
        index=False
    )
)


# ============================================================
# GT-OVERLAPPING COMPONENT AUDIT
# ============================================================

print()
print("=" * 115)
print(
    "===== GT-OVERLAPPING COMPONENT AUDIT ====="
)
print("=" * 115)

if components_df.empty:

    print(
        "No components."
    )

else:

    overlap = (
        components_df[
            components_df[
                "overlap_gt"
            ] > 0
        ]
        .sort_values(
            [
                "case",
                "dice_gt",
            ],
            ascending=[
                True,
                False,
            ]
        )
    )

    if overlap.empty:

        print(
            "No GT-overlapping components."
        )

    else:

        print()

        print(
            overlap[
                [
                    "case",
                    "label",
                    "voxels",
                    "z_start",
                    "z_end",
                    "z_span",
                    "mean_suv",
                    "frac_ge_3",
                    "mean_suv_hot3_a0p5",
                    "overlap_gt",
                    "dice_gt",
                    "iou_gt",
                ]
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
    "===== DEV10-LITE COMPLETE ====="
)
print("=" * 115)

print()
print(
    "Summary:"
)

print(
    SUMMARY_CSV
)

print()
print(
    "Components:"
)

print(
    COMPONENT_CSV
)