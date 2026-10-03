<div align="center">

# 🔍 VisionGuard AI

### Industrial Anomaly Detection & Defect Localization

<p>
  <strong>Computer Vision • CNNs • TensorFlow • Keras • Image Reconstruction • Explainable Inspection</strong>
</p>

<p>
  A practical deep-learning pipeline for discovering visual defects in industrial products and understanding where anomalies occur.
</p>

<p>
  <a href="https://github.com/Maganpreet-Singh/VisionGuard-AI">
    <img src="https://img.shields.io/badge/GitHub-VisionGuard--AI-181717?style=for-the-badge&logo=github" alt="GitHub">
  </a>
  <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-Neural%20Networks-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/Python-Computer%20Vision-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
</p>

</div>

---

## 🧠 What is VisionGuard AI?

**VisionGuard AI** is a computer-vision project focused on **industrial anomaly detection and defect localization**.

The repository is organized as a complete experimentation workflow:

**Raw MVTec-style industrial images → dataset exploration → preprocessing → normalized/processed data → masks & defect statistics → visual analysis → TensorFlow-ready pipeline**

The project is designed around a reconstruction/anomaly-detection mindset: learn the visual characteristics of normal products, measure deviations, and use defect masks and visual diagnostics to understand anomalous regions.

> **Core idea:** teach the system what “normal” looks like, then investigate where the visual signal starts behaving differently.

---

## ✨ Project Highlights

| Capability | What it covers |
|---|---|
| 🔬 Dataset Exploration | Inventory, dimensions, formats, distributions and integrity checks |
| 🧹 Image Preprocessing | Image preparation, resizing, normalization and TensorFlow-ready transformations |
| 🧩 Defect Masks | Ground-truth mask organization and mask-level statistics |
| 📊 EDA | Defect distribution, coverage, resolution, sharpness, RGB statistics and correlations |
| 🖼️ Visual Diagnostics | Sample galleries, overlays, defect maps and preprocessing previews |
| 🧠 Deep Learning Ready | Structured for CNN/TensorFlow-based anomaly-detection experimentation |
| 📁 Reproducible Assets | Processed datasets, CSV summaries, notebooks and generated figures |

---

## 🏭 Dataset Coverage

The repository contains a focused **5-category subset** of the MVTec-style industrial anomaly dataset under:

`data/raw/mvtec_5_categories/`

The data is organized with the familiar structure:

```text
category/
├── train/
│   └── good/
├── test/
│   ├── good/
│   └── <defect_types>/
└── ground_truth/
    └── <defect_types>/
```

This structure keeps normal training examples, defective test examples, and localization masks clearly separated.

### Categories represented in the repository

- **Bottle**
- **Capsule**
- **Hazelnut**
- **Metal Nut**
- **Screw**

The repository also contains processed copies of the image data and generated mask assets under `data/processed/`.

---

## 🔄 Pipeline

```text
                    ┌──────────────────────┐
                    │   Industrial Images  │
                    │   5-category subset  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Dataset Exploration  │
                    │ Inventory & Quality  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Preprocessing     │
                    │ Resize / Normalize   │
                    │ TensorFlow pipeline │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Processed Images &   │
                    │ Ground-Truth Masks   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     EDA & Visuals    │
                    │ Defects / Masks /    │
                    │ Resolution / Quality │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Deep Learning Stage  │
                    │ Anomaly Detection &  │
                    │ Localization         │
                    └──────────────────────┘
```

---

## 📊 Exploratory Data Analysis

VisionGuard AI goes beyond simply loading images. The repository contains generated analysis for:

- Dataset inventory and file statistics
- Normal vs. defective distribution
- Defect-type distribution
- Defect diversity
- Image dimensions and resolution
- File formats and file sizes
- Brightness vs. contrast
- RGB channel statistics
- Image sharpness
- Aspect-ratio distribution
- Defect-mask coverage
- Mask metrics and mask pair analysis
- Defect centroid mapping
- Bounding-box coverage analysis
- Feature correlation analysis

### Visual dashboard

![VisionGuard EDA Dashboard](images/visionguard_eda_dashboard.png)

### Defect overlay gallery

![Defect Overlay Gallery](images/defect_overlay_gallery.png)

### Ground-truth masks

![Ground Truth Mask Gallery](images/ground_truth_mask_gallery.png)

---

## 🧪 TensorFlow Preprocessing

The repository includes a dedicated `preprocessing.ipynb` notebook for the TensorFlow-oriented preprocessing workflow.

The generated outputs include:

- TensorFlow augmentation previews
- TensorFlow normalization previews
- Resized mask visualizations
- Processed training samples
- Train/validation distribution views
- Test-balance visualizations
- TensorFlow preprocessing dashboards

These artifacts make it easier to inspect the transformation pipeline before moving into deeper model experimentation.

---

## 🗂️ Repository Structure

```text
VisionGuard-AI/
│
├── data/
│   ├── raw/
│   │   └── mvtec_5_categories/
│   │       ├── bottle/
│   │       ├── capsule/
│   │       ├── hazelnut/
│   │       ├── metal_nut/
│   │       └── screw/
│   │
│   └── processed/
│       ├── train/
│       └── masks/
│
├── eda_tables/
│   ├── dataset_inventory.csv
│   ├── dataset_statistics.csv
│   ├── defect_distribution.csv
│   ├── defect_diversity.csv
│   ├── file_formats.csv
│   ├── file_size_summary.csv
│   ├── image_dimensions.csv
│   ├── integrity_summary.csv
│   ├── mask_metrics.csv
│   ├── mask_pairs.csv
│   ├── normal_vs_defective.csv
│   ├── pixel_statistics.csv
│   ├── resolution_summary.csv
│   └── sharpness.csv
│
├── images/
│   ├── EDA figures
│   ├── defect visualizations
│   ├── mask galleries
│   └── analysis dashboards
│
├── outputs/
│   └── graphs/
│       ├── preprocessing outputs
│       ├── augmentation previews
│       ├── normalization previews
│       └── TensorFlow analysis
│
├── dataset_exploration.ipynb
├── preprocessing.ipynb
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Maganpreet-Singh/VisionGuard-AI.git
cd VisionGuard-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3. Install the main dependencies

```bash
pip install numpy pandas matplotlib seaborn pillow opencv-python jupyter tensorflow
```

> TensorFlow compatibility depends on your Python version and operating system. Use a Python/TensorFlow combination supported by the TensorFlow release you plan to run.

### 4. Launch Jupyter

```bash
jupyter notebook
```

Start with:

```text
dataset_exploration.ipynb
```

Then continue with:

```text
preprocessing.ipynb
```

---

## 📓 Notebooks

### `dataset_exploration.ipynb`

Used for understanding the raw dataset before modeling.

Typical outputs include:

- File and folder inventory
- Image statistics
- Category analysis
- Defect analysis
- Mask inspection
- Resolution and image-quality analysis
- EDA visualizations

### `preprocessing.ipynb`

Used for preparing image data for TensorFlow experiments.

It covers the preprocessing workflow and generates visual checks for:

- Resizing
- Normalization
- Augmentation
- Mask resizing
- Processed training samples
- Dataset balance and distributions

---

## 🧰 Tech Stack

| Technology | Role |
|---|---|
| **Python** | Core development language |
| **TensorFlow** | Deep-learning framework |
| **Keras** | Neural-network API |
| **NumPy** | Numerical computation |
| **Pandas** | Dataset and tabular analysis |
| **OpenCV** | Image processing |
| **Pillow** | Image I/O and manipulation |
| **Matplotlib** | Visualization |
| **Seaborn** | Statistical visualization |
| **Jupyter Notebook** | Experimentation and analysis |

---

## 🎯 Learning & Research Goals

VisionGuard AI is structured as a foundation for experimenting with industrial computer vision problems such as:

- **Anomaly detection**
- **Defect classification**
- **Defect localization**
- **Reconstruction-based detection**
- **Pixel-level error analysis**
- **CNN-based feature learning**
- **Explainable visual inspection**

The project emphasizes not only model training, but also **data understanding, preprocessing validation, localization masks, and visual interpretability**.

---

## 📈 Why the EDA Matters

Industrial anomaly detection is not just a model problem. Image quality, category imbalance, defect size, mask coverage, resolution, brightness, contrast and data distribution can strongly affect experiments.

That is why VisionGuard AI keeps its analysis artifacts visible in the repository instead of treating EDA as disposable notebook output.

The `eda_tables/`, `images/`, and `outputs/graphs/` directories provide a persistent record of those investigations.

---

## 🧠 Anomaly Detection Concept

A reconstruction-oriented anomaly detector can be thought of as:

```text
Normal Image
     │
     ▼
  Encoder
     │
     ▼
 Latent Representation
     │
     ▼
  Decoder
     │
     ▼
Reconstructed Image
     │
     ▼
Pixel / Feature Difference
     │
     ▼
Anomaly Score + Heatmap
```

Regions with larger reconstruction or feature discrepancies can become candidates for defect localization.

> The exact model architecture, thresholding strategy and evaluation metrics should be documented alongside the final trained model when those experiments are added.

---

## 📌 Current Project Status

**Status: Active development / experimentation**

The repository currently provides a strong end-to-end **data exploration and preprocessing foundation** for industrial anomaly-detection research, including raw/processed image assets, ground-truth masks, EDA tables, visualization outputs and TensorFlow preprocessing workflows.

Future experimentation can extend this foundation into trained anomaly-detection models, benchmark comparisons and quantitative localization metrics.

---

## 🛣️ Roadmap

- [x] Collect and organize the 5-category industrial dataset
- [x] Dataset exploration
- [x] Image and mask preprocessing
- [x] EDA tables and statistical summaries
- [x] Defect visualization and mask galleries
- [x] TensorFlow preprocessing workflow
- [ ] CNN/autoencoder model training
- [ ] Anomaly-score calibration
- [ ] Defect heatmap generation
- [ ] Localization metrics
- [ ] Model comparison and ablation studies
- [ ] Inference pipeline
- [ ] Lightweight deployment/demo interface

---

## ⚠️ Reproducibility Notes

The repository contains a substantial image dataset and generated analysis artifacts. For a clean experiment:

1. Keep the expected directory structure unchanged.
2. Run dataset exploration before modifying preprocessing assumptions.
3. Validate processed images and masks visually.
4. Record model configuration, thresholds and evaluation metrics with each experiment.
5. Avoid claiming model performance until it has been measured on a clearly defined evaluation split.

---

## 🙌 Credits & Dataset

This project uses a focused subset of the **MVTec Anomaly Detection** dataset for industrial visual inspection research and experimentation.

Please follow the original dataset's license and usage requirements. The category folders include their own dataset-provided license/readme files where applicable.

---

## 👨‍💻 Author

### Maganpreet Singh

Computer Science & Engineering student building projects across:

**AI/ML • Computer Vision • Deep Learning • Data Science**

GitHub:  
**https://github.com/Maganpreet-Singh**

---

## ⭐ Support the Project

If this repository helps you learn about industrial computer vision, preprocessing, anomaly detection or TensorFlow workflows, consider giving it a **⭐ Star**.

Built with curiosity, iteration, and a healthy amount of debugging. 🧠⚙️
