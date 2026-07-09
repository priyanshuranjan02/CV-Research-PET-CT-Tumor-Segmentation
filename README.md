# 🩺 Automatic PET-CT Tumor Segmentation using Computer Vision

> A Computer Vision-based medical image analysis project for automated tumor region detection from PET-CT images using image processing and feature extraction techniques.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-blue?style=for-the-badge&logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple?style=for-the-badge&logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter)

---

# 📌 Overview

Medical image segmentation is a critical task in assisting radiologists with identifying suspicious tumor regions. This project implements a classical Computer Vision pipeline to automatically process PET-CT images and identify potential tumor regions using image preprocessing, thresholding, morphological operations, contour detection, and feature extraction.

The project demonstrates how traditional image processing techniques can be applied to medical imaging before adopting deep learning-based segmentation models.

---

# 🎯 Objectives

- Perform preprocessing on PET-CT images.
- Apply thresholding-based segmentation.
- Remove image noise using morphological operations.
- Detect potential tumor regions using contour analysis.
- Extract geometric features such as:
  - Tumor Area
  - Tumor Perimeter
- Process an entire dataset using batch image processing.

---

# 🚀 Features

✅ Automatic PET-CT image preprocessing

✅ Adaptive & Global Thresholding

✅ Morphological Image Refinement

✅ Contour-based Tumor Region Detection

✅ Feature Extraction (Area & Perimeter)

✅ Batch Processing (200+ Images)

✅ CSV Report Generation

✅ Jupyter Notebook + Python Pipeline

---

# 🛠 Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Computer Vision | OpenCV |
| Numerical Computing | NumPy |
| Data Analysis | Pandas |
| Visualization | Matplotlib |
| Development | VS Code, Jupyter Notebook |
| Version Control | Git & GitHub |

---

# 📂 Project Structure

```
CV-Research-PET-CT-Tumor-Segmentation
│
├── data/
│   └── 2D Images/
│
├── notebooks/
│   └── Exp.ipynb
│
├── src/
│   └── pipeline.py
│
├── results/
│   ├── results.csv
│   └── output_images/
│
├── paper/
│   ├── Research_Paper.pdf
│   ├── figures/
│   └── references.bib
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# ⚙️ Workflow

```
PET-CT Image
      │
      ▼
Preprocessing
      │
      ▼
Thresholding
(Global + Adaptive)
      │
      ▼
Morphological Operations
      │
      ▼
Contour Detection
      │
      ▼
Tumor Candidate Selection
      │
      ▼
Feature Extraction
      │
      ▼
CSV Report Generation
```

---


# 📊 Output

The pipeline extracts quantitative features including:

| Image | Tumor Area | Perimeter |
|--------|-----------:|----------:|
| ID_0000_Z_0142 | XXXX | XXXX |

Results are automatically exported as:

```
results/results.csv
```

---

# 💻 Installation

Clone the repository

```bash
git clone https://github.com/priyanshuranjan02/CV-Research-PET-CT-Tumor-Segmentation.git
```

Move into project

```bash
cd CV-Research-PET-CT-Tumor-Segmentation
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Notebook

```bash
jupyter notebook
```

or execute

```bash
python src/pipeline.py
```

---

# 📈 Future Improvements

- Deep Learning based U-Net Segmentation
- 3D PET-CT Volume Processing
- Dice Score & IoU Evaluation
- Automatic Tumor Classification
- Radiomics Feature Extraction
- Explainable AI (Grad-CAM)

---

# 📚 Research Motivation

This project was developed as part of academic research in Computer Vision and Medical Image Analysis to explore classical segmentation techniques for PET-CT imaging. It establishes a strong baseline for future deep learning-based medical image segmentation research.

---

# 🤝 Contributing

Contributions are welcome!

If you'd like to improve the project, feel free to fork the repository and submit a Pull Request.

---

# 👨‍💻 Author

**Priyanshu Ranjan**

B.Tech CSE (AI & ML)

📧 Email: *ranjanpriyanshu441@gmail.com*

🔗 LinkedIn: [https://linkedin.com/in/your-profile](https://www.linkedin.com/in/priyanshu-ranjan-74170a227/)

💻 GitHub: https://github.com/priyanshuranjan02

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

It motivates me to continue building impactful AI & Computer Vision projects.
