"""
DEV59 — FINAL RESEARCH PACKAGE

Purpose:
    Freeze the completed PET-CT tumor segmentation study into a
    reproducible final research package.

IMPORTANT:
    - No new model tuning
    - No new feature engineering
    - No changes to Dev10
    - No changes to Dev39
    - Dev39 remains the proposed method
    - Dev58 is diagnostic only

Frozen proposed method:
    PET threshold       = 2.25 SUV
    Hot threshold       = 3.0 SUV
    3D filter           = >=75 voxels and >=2 slices
    PET quality ranking = Dev39
    Hot fraction gate   = >=0.40
    Top-K               = 4
    Core policy         = P70
"""

from pathlib import Path
import shutil
import pandas as pd
from datetime import datetime


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FINAL_DIR = PROJECT_ROOT / "results" / "final"

DEV45_DIR = (
    PROJECT_ROOT
    / "results"
    / "development_cases"
    / "dev45_final_package"
)

DEV53_DIR = (
    PROJECT_ROOT
    / "results"
    / "validation"
    / "dev53_final_validation"
)

DEV58_DIR = (
    PROJECT_ROOT
    / "results"
    / "development_cases"
    / "dev58_cohort_spatial_separation_audit"
)

FINAL_FIGURES = FINAL_DIR / "figures"


# ============================================================
# FROZEN METHOD
# ============================================================

METHOD = {
    "PET threshold SUV": 2.25,
    "Hot threshold SUV": 3.0,
    "3D minimum voxels": 75,
    "3D minimum slices": 2,
    "Hot fraction gate": 0.40,
    "Top-K components": 4,
    "Core policy": "P70",
    "Ranking": "Dev39 PET quality score",
}


# ============================================================
# HELPERS
# ============================================================

def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[COPIED] {src.name}")
    else:
        print(f"[SKIP] Missing: {src}")


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ============================================================
# CREATE DIRECTORIES
# ============================================================

FINAL_DIR.mkdir(parents=True, exist_ok=True)
FINAL_FIGURES.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("DEV59 — FINAL RESEARCH PACKAGE")
print("=" * 80)

print(f"\nProject root:")
print(PROJECT_ROOT)

print(f"\nFinal directory:")
print(FINAL_DIR)


# ============================================================
# 1. COPY DEV45 FROZEN DEVELOPMENT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("1. FROZEN DEVELOPMENT RESULTS")
print("=" * 80)

dev45_files = [
    "dev45_final_results.csv",
    "dev45_per_case_results.csv",
    "dev45_method_comparison.csv",
    "dev45_ablation_summary.csv",
    "dev45_parameters.csv",
    "dev45_reproducibility_report.txt",
]

for filename in dev45_files:
    copy_if_exists(
        DEV45_DIR / filename,
        FINAL_DIR / filename,
    )


# ============================================================
# 2. COPY DEV53 VALIDATION RESULTS
# ============================================================

print("\n" + "=" * 80)
print("2. FINAL VALIDATION RESULTS")
print("=" * 80)

dev53_files = [
    "validation_metrics.csv",
    "selected_components.csv",
    "selected_core_summary.csv",
    "all_component_scores.csv",
    "validation_report.txt",
    "axial_validation_overlay.png",
    "tumor_mask.nii.gz",
]

for filename in dev53_files:
    copy_if_exists(
        DEV53_DIR / filename,
        FINAL_DIR / filename,
    )


# ============================================================
# 3. COPY DEV58 DIAGNOSTIC RESULTS
# ============================================================

print("\n" + "=" * 80)
print("3. DEV58 COHORT DIAGNOSTIC RESULTS")
print("=" * 80)

dev58_files = [
    "dev58_all_eligible_components.csv",
    "dev58_group_feature_summary.csv",
    "dev58_case_summary.csv",
]

for filename in dev58_files:
    copy_if_exists(
        DEV58_DIR / filename,
        FINAL_DIR / filename,
    )


dev58_figures = [
    "dev58_size_vs_radial_cohort.png",
    "dev58_size_vs_z_cohort.png",
    "dev58_pet_quality_vs_radial_cohort.png",
    "dev58_pet_quality_vs_size_cohort.png",
]

for filename in dev58_figures:
    copy_if_exists(
        DEV58_DIR / filename,
        FINAL_FIGURES / filename,
    )


# ============================================================
# 4. CREATE FINAL PARAMETER TABLE
# ============================================================

print("\n" + "=" * 80)
print("4. FINAL PARAMETERS")
print("=" * 80)

parameter_rows = [
    ["PET candidate threshold", "2.25", "SUV"],
    ["Hot feature threshold", "3.0", "SUV"],
    ["3D minimum component size", "75", "voxels"],
    ["3D minimum slice span", "2", "slices"],
    ["PET quality ranking", "Dev39 PET quality score", "ranking"],
    ["Hot fraction gate", "0.40", "fraction"],
    ["Top-K", "4", "components"],
    ["Core extraction", "P70", "percentile"],
    ["Candidate strategy", "PET-only", "candidate generation"],
    ["Proposed version", "Dev39", "frozen"],
]

parameters_df = pd.DataFrame(
    parameter_rows,
    columns=["Parameter", "Value", "Unit / Description"],
)

parameters_df.to_csv(
    FINAL_DIR / "final_parameters.csv",
    index=False,
)

print("[CREATED] final_parameters.csv")


# ============================================================
# 5. BUILD FINAL DEVELOPMENT VS VALIDATION TABLE
# ============================================================

print("\n" + "=" * 80)
print("5. DEVELOPMENT VS VALIDATION")
print("=" * 80)

development_row = {
    "Dataset": "Development cohort",
    "Cases": 6,
    "Method": "Dev39",
    "Dice": 0.120468,
    "IoU": 0.070577,
    "Slice Recall": 0.376596,
    "FPR": 0.085242,
    "Prediction Voxels": 376252,
}

validation_row = {
    "Dataset": "Validation case",
    "Cases": 1,
    "Method": "Dev39",
    "Dice": 0.304136,
    "IoU": 0.179340,
    "Slice Recall": 0.521739,
    "FPR": 0.845420,
    "Prediction Voxels": 52000,
}

comparison_df = pd.DataFrame(
    [development_row, validation_row]
)

comparison_df.to_csv(
    FINAL_DIR / "final_development_vs_validation.csv",
    index=False,
)

print("[CREATED] final_development_vs_validation.csv")


# ============================================================
# 6. FINAL ABLATION SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("6. ABLATION SUMMARY")
print("=" * 80)

ablation_rows = [
    {
        "Method": "Dev10 baseline",
        "Dice": 0.100912,
        "IoU": 0.059005,
        "Slice Recall": 0.332675,
        "FPR": 0.081641,
        "Status": "Baseline",
    },
    {
        "Method": "Dev39 proposed",
        "Dice": 0.120468,
        "IoU": 0.070577,
        "Slice Recall": 0.376596,
        "FPR": 0.085242,
        "Status": "Final proposed method",
    },
    {
        "Method": "Dev52 PET + CT structural",
        "Dice": 0.119794,
        "IoU": 0.070125,
        "Slice Recall": 0.362854,
        "FPR": 0.077767,
        "Status": "Rejected; lower Dice/recall",
    },
]

ablation_df = pd.DataFrame(ablation_rows)

ablation_df["Dice Improvement vs Baseline"] = (
    ablation_df["Dice"] - 0.100912
)

ablation_df.to_csv(
    FINAL_DIR / "final_ablation_table.csv",
    index=False,
)

print("[CREATED] final_ablation_table.csv")


# ============================================================
# 7. FINAL RESEARCH SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("7. FINAL RESEARCH SUMMARY")
print("=" * 80)

summary = f"""
PET-CT TUMOR SEGMENTATION — FINAL RESEARCH SUMMARY
==================================================

Project:
Automatic Tumor Segmentation in PET-CT Images Using
Hybrid Intensity and Texture-Based Techniques

FINAL PROPOSED METHOD
---------------------
The final proposed method is Dev39.

Pipeline:
1. Load CT-aligned SUV PET volume.
2. Generate PET candidate voxels using SUV >= 2.25.
3. Extract connected 3D components.
4. Retain components with >=75 voxels and >=2 slices.
5. Compute PET component quality features.
6. Apply hot-fraction gate >=0.40.
7. Rank eligible components using the Dev39 PET quality score.
8. Select Top-K = 4 components.
9. Extract P70 PET hot core.
10. Produce the final 3D segmentation.

FROZEN PARAMETERS
-----------------
PET threshold:       2.25 SUV
Hot threshold:       3.0 SUV
3D minimum voxels:   75
3D minimum slices:   2
Hot fraction gate:   0.40
Top-K:               4
Core policy:         P70
Ranking:             Dev39 PET quality score

DEVELOPMENT RESULTS
-------------------
Cases:               6
Dice:                0.120468
IoU:                 0.070577
Slice recall:        0.376596
FPR:                 0.085242
Prediction voxels:   376252

BASELINE COMPARISON
-------------------
Dev10 Dice:          0.100912
Dev39 Dice:          0.120468

Absolute Dice improvement:
+0.019556

Relative Dice improvement:
approximately +19.38 percent

VALIDATION RESULT
-----------------
Validation case:
PETCT_0011f3deaf

Dice:                0.304136
IoU:                 0.179340
Slice recall:        0.521739
FPR:                 0.845420
Prediction voxels:   52000
GT overlap voxels:   1651

IMPORTANT INTERPRETATION
------------------------
The validation Dice is higher than the development-cohort mean,
but the validation false-positive rate is substantially higher.

Therefore the validation result must not be presented as evidence
of uniformly strong segmentation performance.

The main observed limitation is false-positive PET uptake.
The component audit demonstrated that high PET intensity,
persistence and PET quality do not always distinguish tumor
uptake from physiologic/background uptake.

DEV58 COHORT DIAGNOSTIC FINDING
-------------------------------
The cohort diagnostic analysis found stronger spatial separation
than pure PET intensity for some successful components.

However, the distributions still overlap substantially and the
development cohort is small. Therefore an additional spatial
ranking model was NOT introduced into the final method.

This avoids overfitting the final method to the available
development cases.

FINAL RESEARCH CONCLUSION
-------------------------
The study demonstrates a reproducible PET-based candidate
generation, 3D component ranking and hot-core extraction
pipeline.

The proposed method improves the development Dice over the
Dev10 baseline, while remaining sensitive to physiological and
background FDG uptake.

The work therefore demonstrates a useful classical computer
vision baseline and identifies candidate localization and
false-positive discrimination as the principal remaining
challenges.

Future work should investigate larger multi-patient datasets,
anatomically informed PET-CT fusion, learned segmentation
models and stronger false-positive suppression.
"""

write_text(
    FINAL_DIR / "final_research_summary.txt",
    summary.strip() + "\n",
)

print("[CREATED] final_research_summary.txt")


# ============================================================
# 8. FINAL LIMITATIONS
# ============================================================

limitations = """
FINAL LIMITATIONS
=================

1. Small development cohort
---------------------------
The development analysis uses six cases. Therefore, feature
selection and ranking behavior may not generalize to a larger
population.

2. Limited validation
---------------------
The final validation analysis uses one validation case. Its
performance should therefore be interpreted as a case-level
external check rather than a population-level performance
estimate.

3. False positives
------------------
The principal limitation is physiological/background FDG uptake.
High SUV and high PET-quality scores can occur in components that
do not correspond to the tumor annotation.

4. High validation FPR
----------------------
The validation case produced a high false-positive rate despite
a Dice score of 0.304136. This indicates that the predicted
segmentation contains substantial non-tumor regions.

5. PET-only final ranking
-------------------------
The final Dev39 ranking intentionally uses PET-derived component
features. CT and anatomical information were investigated
diagnostically but were not added to the final ranking because
the available cohort was insufficient to justify another tuned
model.

6. Classical segmentation approach
-----------------------------------
The method is based on thresholding, connected components,
hand-crafted features and percentile-based core extraction.
It does not learn representations from data.

7. Dataset limitations
----------------------
The conclusions should not be generalized to all PET-CT
acquisition protocols, scanners, reconstruction settings or
cancer types.

8. Future work
--------------
A larger annotated cohort could support:
- PET-CT multimodal learning
- anatomical priors
- learned false-positive suppression
- 3D CNN/Transformer segmentation
- cross-validation
- patient-level statistical evaluation
"""

write_text(
    FINAL_DIR / "final_limitations.txt",
    limitations.strip() + "\n",
)

print("[CREATED] final_limitations.txt")


# ============================================================
# 9. REPRODUCIBILITY REPORT
# ============================================================

reproducibility = f"""
FINAL REPRODUCIBILITY REPORT
============================

Generated:
{datetime.now().isoformat(timespec="seconds")}

Frozen proposed method:
Dev39

Parameters:
{chr(10).join(f"- {k}: {v}" for k, v in METHOD.items())}

Source pipeline:
src/dev10_lite_hot_core.py
src/dev39_pet_quality_fpr_control.py

Important:
The source implementations above were frozen before final
packaging. This final package does not modify either source file.

Validation:
src/v4_validate_case7.py
src/dev53_final_validation.py

Diagnostic analysis:
src/dev54_validation_component_audit.py
src/dev55_validation_component_visual_audit.py
src/dev56_anatomical_context_diagnostic.py
src/dev57_component_spatial_audit.py
src/dev58_cohort_spatial_separation_audit.py

No new optimization is performed by Dev59.

Development benchmark:
Dice = 0.120468
IoU = 0.070577
Slice Recall = 0.376596
FPR = 0.085242

Validation:
Dice = 0.304136
IoU = 0.179340
Slice Recall = 0.521739
FPR = 0.845420

The validation result is case-level and must not be interpreted
as a statistically validated population-level estimate.
"""

write_text(
    FINAL_DIR / "final_reproducibility_report.txt",
    reproducibility.strip() + "\n",
)

print("[CREATED] final_reproducibility_report.txt")


# ============================================================
# 10. FINAL METHOD DESCRIPTION FOR PAPER
# ============================================================

methodology = """
FINAL METHODOLOGY DESCRIPTION
==============================

Candidate Generation
---------------------
The method operates on CT-aligned PET SUV volumes. Voxels with
SUV >= 2.25 are initially considered PET candidate voxels.

3D Component Formation
----------------------
Candidate voxels are grouped into connected 3D components.
Components smaller than 75 voxels or spanning fewer than two
slices are discarded.

PET Component Characterization
------------------------------
Each eligible component is characterized using PET-derived
features including mean SUV, maximum SUV, fraction of voxels
above the hot threshold, spatial persistence, longitudinal
extent and compactness.

PET Quality Ranking
-------------------
The Dev39 PET quality score combines normalized component
intensity and structural PET characteristics. Components are
ranked using this score with mean SUV as the secondary ranking
criterion.

False-Positive Control
----------------------
A hot-fraction threshold of 0.40 is applied before ranking.
This removes components with insufficiently concentrated hot
PET uptake.

Top-K Selection
---------------
The four highest-ranked eligible components are retained.

Hot-Core Extraction
-------------------
For each selected component, a P70 core is extracted using the
PET intensity distribution. The resulting cores are combined to
form the final predicted tumor mask.

Evaluation
----------
Performance is evaluated using Dice coefficient, intersection
over union, slice-level recall and false-positive rate.

Development and validation are kept separate. Ground-truth
annotations are used for evaluation only and are not required
for the final component-selection procedure.
"""

write_text(
    FINAL_DIR / "final_methodology.txt",
    methodology.strip() + "\n",
)

print("[CREATED] final_methodology.txt")


# ============================================================
# 11. CREATE FINAL MANIFEST
# ============================================================

print("\n" + "=" * 80)
print("11. FINAL MANIFEST")
print("=" * 80)

manifest_rows = []

for path in sorted(FINAL_DIR.rglob("*")):
    if path.is_file():
        manifest_rows.append({
            "File": str(path.relative_to(FINAL_DIR)),
            "Size_bytes": path.stat().st_size,
        })

manifest_df = pd.DataFrame(manifest_rows)

manifest_df.to_csv(
    FINAL_DIR / "final_manifest.csv",
    index=False,
)

print(f"[CREATED] final_manifest.csv")
print(f"Files packaged: {len(manifest_df)}")


# ============================================================
# 12. FINAL STATUS
# ============================================================

print("\n" + "=" * 80)
print("DEV59 COMPLETE")
print("=" * 80)

print(f"""
FINAL PACKAGE:
{FINAL_DIR}

FROZEN METHOD:
Dev39

Development Dice:
0.120468

Validation Dice:
0.304136

Validation FPR:
0.845420

No new optimization was performed.

PROJECT STATUS:
FINALIZED
""")