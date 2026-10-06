# 🩺 Automatic PET-CT Tumor Segmentation Using Computer Vision

> A classical Computer Vision framework for automated tumor candidate localization and segmentation from FDG PET-CT images using PET intensity, 3D connected-component analysis, longitudinal persistence, and quantitative feature-based ranking.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-blue?style=for-the-badge&logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple?style=for-the-badge&logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter)
![Medical Imaging](https://img.shields.io/badge/Domain-Medical%20Imaging-red?style=for-the-badge)

---

## 📌 Overview

Medical image segmentation is an important task in computer-aided medical image analysis. In FDG PET-CT imaging, metabolically active regions can provide useful information for identifying suspicious lesions, but physiological uptake and other high-intensity regions can also produce false positives.

This project investigates a **classical Computer Vision approach for PET-based tumor candidate segmentation**, using three-dimensional PET information rather than relying exclusively on 2D image processing.

The finalized method uses:

- PET SUV intensity thresholding
- 3D connected-component analysis
- Component-size and slice-span filtering
- Longitudinal persistence across axial slices
- PET intensity statistics
- Hot-voxel fraction
- Component compactness
- Feature-based component quality ranking
- Top-K component selection
- P70 intensity-core extraction

The project was developed as an experimental research pipeline, with extensive diagnostic and ablation experiments used to understand candidate localization, component selection, false positives, and validation behavior.

---

## 🎯 Research Objectives

The main objectives of this project are:

1. Develop a classical Computer Vision pipeline for PET-based tumor candidate localization.
2. Investigate the effectiveness of PET SUV intensity thresholds for candidate generation.
3. Analyze candidate regions using 3D connected components.
4. Incorporate spatial and longitudinal characteristics of candidate components.
5. Rank candidate components using quantitative PET-derived features.
6. Generate a compact tumor core from selected PET components.
7. Evaluate the segmentation using Dice, IoU, slice recall, and false-positive rate.
8. Analyze failure cases and limitations of the classical approach.
9. Produce a reproducible research package for further development.

---

## 🧠 Final Methodology

The final development method is based on the following pipeline:

```text
                 FDG PET Volume
                       │
                       ▼
              PET SUV Candidate Mask
                  SUV ≥ 2.25
                       │
                       ▼
            3D Connected Components
                       │
                       ▼
       ┌─────────────────────────────────┐
       │ Component Eligibility Filtering │
       │                                 │
       │ • ≥ 75 voxels                   │
       │ • ≥ 2 axial slices              │
       └─────────────────────────────────┘
                       │
                       ▼
             Component Feature Extraction
                       │
                       ▼
       ┌─────────────────────────────────┐
       │ PET Component Quality Score     │
       │                                 │
       │ Mean SUV              25%       │
       │ Maximum SUV           15%       │
       │ Hot Fraction          15%       │
       │ Persistence           20%       │
       │ Longest Z-run         15%       │
       │ Compactness           10%       │
       └─────────────────────────────────┘
                       │
                       ▼
              Hot-Fraction Gate
                   ≥ 0.40
                       │
                       ▼
                 Top-K Selection
                     K = 4
                       │
                       ▼
              P70 Intensity Core
                       │
                       ▼
             Final Tumor Prediction
```

---

## 🔬 Final Configuration

The finalized development configuration is:

| Parameter | Final Value |
|---|---:|
| PET candidate threshold | SUV ≥ 2.25 |
| Minimum component size | 75 voxels |
| Minimum slice span | 2 slices |
| Hot threshold | SUV ≥ 3.0 |
| Hot-fraction gate | ≥ 0.40 |
| Component selection | Top-K |
| K | 4 |
| Tumor core percentile | P70 |
| Evaluation | Exact Dev10 evaluator |

### Component Quality Score

Eligible PET components are ranked using:

```text
Quality Score =
    0.25 × normalized mean SUV
  + 0.15 × normalized maximum SUV
  + 0.15 × normalized hot fraction
  + 0.20 × normalized persistence
  + 0.15 × normalized longest Z-run
  + 0.10 × normalized compactness
```

where:

```text
Persistence =
Longest consecutive axial-slice run / Z-span
```

The longest-run contribution uses a logarithmic transformation to reduce the dominance of very large components.

---

## 📊 Development Results

The finalized method was evaluated on the development cohort.

### Development Benchmark

| Metric | Result |
|---|---:|
| Dice | **0.120468** |
| IoU | **0.070577** |
| Slice Recall | **0.376596** |
| False Positive Rate | **0.085242** |
| Prediction Voxels | **376,252** |

These results represent the **current verified development benchmark for the finalized Dev39 method**.

The development evaluation should be interpreted as a research benchmark rather than as evidence of clinical-level segmentation performance.

---

## 🧪 Independent Validation

An independent validation case was evaluated using the finalized method.

### Validation Case

```text
PETCT_0011f3deaf
```

### Image Dimensions

```text
CT / PET shape: (391, 512, 512)
```

### Validation Statistics

| Metric | Result |
|---|---:|
| Ground-truth voxels | 9,206 |
| Candidate voxels | 230,550 |
| Eligible components | 12 |
| Selected components | 4 |
| P70 prediction voxels | 52,000 |
| Dice | **0.304136** |
| IoU | **0.179340** |
| Slice Recall | **0.521739** |
| False Positive Rate | **0.845420** |

Selected component labels:

```text
[84, 310, 32, 302]
```

The validation result demonstrates that the method can localize and produce overlap with the target region, but the **high validation false-positive rate indicates substantial over-segmentation**.

Therefore, the validation result should be interpreted cautiously and should not be presented as evidence of robust generalization.

---

## ⚠️ Limitations

The finalized experiments identified several important limitations.

### 1. High False-Positive Behavior

Although the final development configuration reduced the false-positive rate compared with earlier experiments, false-positive regions remain a major challenge.

The independent validation case produced:

```text
FPR = 0.845420
```

indicating substantial non-tumor prediction.

### 2. PET Physiological Uptake

High FDG uptake can occur in normal physiological structures and inflammatory regions. Intensity-based methods therefore cannot reliably distinguish all tumors from non-malignant uptake.

### 3. Limited Development Cohort

The development experiments were performed on a limited number of verified PET-CT cases. The results should therefore not be interpreted as representative of a large clinical population.

### 4. Classical Computer Vision Limitations

The method does not use a learned segmentation model. Consequently, it lacks the representation-learning capability of modern deep learning approaches.

### 5. Validation Scope

The final validation analysis includes a dedicated validation case, but this should not be treated as a large-scale independent clinical validation study.

### 6. No Clinical Deployment

This project is intended for academic and research purposes only. It is **not a clinical diagnostic system**.

---

## 🔎 Research Diagnostics

A substantial portion of the project consists of diagnostic experiments used to understand where the segmentation pipeline succeeds and fails.

The experiments investigated:

- Candidate localization
- Candidate-generation failures
- CT contour failures
- PET component filtering
- PET/ground-truth spatial alignment
- Longitudinal component behavior
- Component ranking
- Top-K selection
- PET-only ranking
- Structural ranking
- False-positive behavior
- Anatomical and spatial separation
- Validation-case component analysis

The diagnostic development history is preserved in `src/` for research provenance and reproducibility.

---

## 📈 Key Research Findings

The experiments showed that:

- PET candidate generation can localize a substantial portion of tumor-positive slices.
- A significant number of failures occur before the final ranking stage.
- Some missed regions are spatially far from the anatomical structures recovered by the original CT-based pipeline.
- PET-only candidate generation provided a useful alternative to the original CT-constrained approach.
- Component quality ranking substantially improved PET-only segmentation performance.
- Adding additional CT structural ranking to the finalized PET ranking did not improve the development benchmark and was therefore not adopted.
- Spatial analysis of eligible components did not provide sufficient evidence for another ranking modification.
- The final method was therefore frozen rather than continuing experimental modification.

---

## 🧮 Evaluation Metrics

The project uses multiple metrics to evaluate segmentation quality.

### Dice Similarity Coefficient

```text
Dice = 2 × |Prediction ∩ Ground Truth|
       ---------------------------------
       |Prediction| + |Ground Truth|
```

### Intersection over Union

```text
IoU = |Prediction ∩ Ground Truth|
      ----------------------------
      |Prediction ∪ Ground Truth|
```

### Slice Recall

Measures the proportion of ground-truth-positive axial slices on which the predicted segmentation detects the target region.

### False Positive Rate

Measures the proportion of predicted/non-target regions contributing to false-positive segmentation behavior.

Using multiple metrics provides a more informative evaluation than relying on Dice alone.

---

## 📂 Project Structure

The repository is organized around the finalized research package, development data, source code, research artifacts, and reproducibility outputs.

```text
CV-Research-PET-CT-Tumor-Segmentation/
│
├── annotations/
│   ├── masks/
│   ├── polygons/
│   └── review_40_cases_progress.csv
│
├── archive/
│   └── validation_utilities/
│
├── development_cases/
│   ├── PETCT_0168f65af8/
│   ├── PETCT_04606080a0/
│   ├── PETCT_04ab5c61c9/
│   ├── PETCT_0b57b247b6/
│   ├── PETCT_11afab3485/
│   └── PETCT_185da4c8b6/
│
├── notebooks/
│
├── paper/
│
├── results/
│   └── final/
│       ├── final_research_summary.txt
│       ├── final_methodology.txt
│       ├── final_limitations.txt
│       ├── final_parameters.csv
│       ├── final_ablation_table.csv
│       ├── final_development_vs_validation.csv
│       ├── final_reproducibility_report.txt
│       ├── final_manifest.csv
│       ├── validation_metrics.csv
│       ├── validation_report.txt
│       ├── selected_components.csv
│       ├── selected_core_summary.csv
│       ├── tumor_mask.nii.gz
│       ├── axial_validation_overlay.png
│       └── figures/
│
├── src/
│   ├── dev10_lite_hot_core.py
│   ├── dev39_pet_quality_fpr_control.py
│   ├── dev45_final_package.py
│   ├── dev46_final_inference.py
│   ├── dev53_final_validation.py
│   ├── dev59_final_research_package.py
│   └── ...
│
├── verified_case/
│
├── Clinical-Metadata-FDG-PET_CT-Lesions.csv
├── review_40_cases.csv
├── select_autopet_subset.py
├── README.md
└── .gitignore
```

> Large local PET-CT datasets and working environments are intentionally excluded from Git tracking. The repository contains the research code, metadata, documentation, final results, and reproducibility artifacts necessary to understand the work.

---

## 🗃️ Dataset

The project uses FDG PET-CT data from the **AutoPET dataset** and associated research data used during development and validation.

Because medical imaging datasets are large and may have access or redistribution restrictions, raw PET-CT volumes are **not included directly in the Git repository**.

The repository instead preserves:

- Dataset metadata
- TCIA manifests
- Development-case organization
- Validation artifacts
- Annotation-related files
- Final segmentation results
- Research methodology
- Reproducibility documentation

Users should obtain the underlying dataset through its official distribution channel and comply with its licensing and usage requirements.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/priyanshuranjan02/CV-Research-PET-CT-Tumor-Segmentation.git
```

### 2. Enter the Project Directory

```bash
cd CV-Research-PET-CT-Tumor-Segmentation
```

### 3. Create a Virtual Environment

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Final Pipeline

The finalized development implementation is:

```text
src/dev39_pet_quality_fpr_control.py
```

The final validation implementation is:

```text
src/dev53_final_validation.py
```

The final research package generator is:

```text
src/dev59_final_research_package.py
```

### Development Evaluation

```bash
python src/dev39_pet_quality_fpr_control.py
```

### Validation

```bash
python src/dev53_final_validation.py
```

### Final Research Package

```bash
python src/dev59_final_research_package.py
```

> The exact input-data paths may need to be configured according to the locally available PET-CT dataset.

---

## 📦 Final Research Package

The final reproducibility package is located at:

```text
results/final/
```

It contains:

### Final Methodology

```text
final_methodology.txt
```

### Final Parameters

```text
final_parameters.csv
```

### Development vs. Validation Results

```text
final_development_vs_validation.csv
```

### Ablation Results

```text
final_ablation_table.csv
```

### Research Summary

```text
final_research_summary.txt
```

### Limitations

```text
final_limitations.txt
```

### Reproducibility Report

```text
final_reproducibility_report.txt
```

### Final Validation Outputs

```text
validation_metrics.csv
validation_report.txt
selected_components.csv
selected_core_summary.csv
tumor_mask.nii.gz
axial_validation_overlay.png
```

### Component Analysis

```text
all_component_scores.csv
dev58_all_eligible_components.csv
dev58_group_feature_summary.csv
dev58_case_summary.csv
```

---

## 🧪 Reproducibility

The repository preserves the progression from initial experimentation to the finalized method.

The research workflow can be broadly divided into:

```text
Initial Pipeline
      │
      ▼
Candidate Localization Analysis
      │
      ▼
Failure-Mode Diagnostics
      │
      ▼
PET Component Analysis
      │
      ▼
Ranking Experiments
      │
      ▼
PET Quality Ranking
      │
      ▼
False-Positive Control
      │
      ▼
Validation
      │
      ▼
Final Research Package
```

Historical experiments remain available in `src/` to preserve research provenance.

The final implementation and final results are explicitly documented in the `results/final/` directory.

---

## 📓 Research Documentation

The project also contains supporting research material under:

```text
paper/
```

and experimental notebooks under:

```text
notebooks/
```

These resources provide additional context regarding the development of the segmentation pipeline, experimentation, visualization, and research methodology.

---

## 🚧 Current Status

### Research Status: **Finalized**

The experimental phase of the project has been completed.

The final method has been selected based on the development experiments, validation analysis, ablation studies, and component-level diagnostics.

No further ranking or segmentation modifications are currently part of the finalized research package.

---

## 🔮 Future Work

Although the current classical Computer Vision pipeline has been finalized, several directions could be explored in future research:

- Deep learning-based 3D segmentation
- U-Net / 3D U-Net architectures
- Transformer-based medical image segmentation
- Multi-modal PET + CT fusion
- Larger independent validation cohorts
- More robust anatomical false-positive suppression
- Automatic lesion classification
- Radiomics feature extraction
- Explainable AI for medical image analysis
- Uncertainty estimation
- Cross-dataset evaluation
- Clinical expert validation

These are potential future research directions and are **not part of the finalized current methodology**.

---

## 📚 Research Motivation

This project was developed as an academic research study in **Computer Vision and Medical Image Analysis**.

The primary motivation was to investigate how far classical image-processing and feature-engineering techniques can be taken for PET-based tumor segmentation before moving toward more computationally expensive deep learning approaches.

The project emphasizes not only segmentation performance but also:

- Understanding failure modes
- Quantitative experimentation
- Reproducibility
- Component-level analysis
- Transparent evaluation
- Explicit documentation of limitations

---

## ⚠️ Disclaimer

This project is intended **strictly for academic and research purposes**.

It is not a medical device and should not be used for:

- Clinical diagnosis
- Treatment planning
- Patient management
- Medical decision-making

PET-CT image interpretation should always be performed by qualified medical professionals using appropriate clinical information and validated medical imaging systems.

---

## 🤝 Contributing

Contributions, suggestions, and research discussions are welcome.

If you would like to contribute:

1. Fork the repository.
2. Create a feature branch.

```bash
git checkout -b feature/your-feature
```

3. Commit your changes.

```bash
git commit -m "Add your feature"
```

4. Push the branch.

```bash
git push origin feature/your-feature
```

5. Open a Pull Request.

For major methodological changes, please describe the experimental motivation and evaluation results clearly.

---

## 👨‍💻 Author

**Priyanshu Ranjan**

B.Tech CSE (AI & ML)

📧 Email: **ranjanpriyanshu441@gmail.com**

🔗 LinkedIn:  
https://www.linkedin.com/in/priyanshu-ranjan-74170a227/

💻 GitHub:  
https://github.com/priyanshuranjan02

---

## ⭐ Support

If you found this research project useful or interesting, consider giving the repository a ⭐ on GitHub.

It helps support continued work in:

- Artificial Intelligence
- Computer Vision
- Medical Image Analysis
- Machine Learning Research

---

## 📌 Project Summary

```text
Project
│
├── Domain
│   └── Computer Vision / Medical Image Analysis
│
├── Imaging Modality
│   └── FDG PET-CT
│
├── Approach
│   └── Classical Computer Vision
│
├── Candidate Threshold
│   └── SUV ≥ 2.25
│
├── 3D Filtering
│   ├── ≥ 75 voxels
│   └── ≥ 2 slices
│
├── Hot Threshold
│   └── SUV ≥ 3.0
│
├── Component Ranking
│   ├── Mean SUV
│   ├── Maximum SUV
│   ├── Hot Fraction
│   ├── Persistence
│   ├── Longest Z-run
│   └── Compactness
│
├── Hot-Fraction Gate
│   └── ≥ 0.40
│
├── Selection
│   ├── Top-K = 4
│   └── P70 Core
│
├── Development Dice
│   └── 0.120468
│
├── Development IoU
│   └── 0.070577
│
├── Development Slice Recall
│   └── 0.376596
│
├── Development FPR
│   └── 0.085242
│
├── Validation Dice
│   └── 0.304136
│
├── Validation IoU
│   └── 0.179340
│
├── Validation Slice Recall
│   └── 0.521739
│
└── Validation FPR
    └── 0.845420
```

---

**⭐ Finalized academic research project — PET-CT tumor candidate segmentation using classical Computer Vision.**
