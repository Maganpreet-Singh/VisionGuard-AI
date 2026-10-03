# 🛡️ VisionGuard AI

<div align="center">

### Industrial Anomaly Detection & Defect Localization with TensorFlow/Keras

<p>
  <strong>Computer Vision</strong> •
  <strong>Convolutional Autoencoders</strong> •
  <strong>Reconstruction-Based Anomaly Detection</strong> •
  <strong>Pixel-Level Localization</strong> •
  <strong>Streamlit</strong>
</p>

<p>
  <a href="https://github.com/Maganpreet-Singh/VisionGuard-AI">
    <img src="https://img.shields.io/badge/GitHub-VisionGuard--AI-181717?style=for-the-badge&logo=github" alt="GitHub">
  </a>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-Neural%20Networks-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Streamlit-Interactive%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
</p>

<p>
  <em>Teach a model what normal looks like. Measure the reconstruction gap. Turn that gap into an interpretable inspection signal.</em>
</p>

</div>

---

## 📚 Table of Contents

- [Project at a Glance](#-project-at-a-glance)
- [Why VisionGuard AI Exists](#-why-visionguard-ai-exists)
- [Problem Definition](#-problem-definition)
- [Project Objectives](#-project-objectives)
- [Core Idea](#-core-idea)
- [How the System Works](#-how-the-system-works)
- [Dataset](#-dataset)
- [Selected Categories](#-selected-categories)
- [Data Organization](#-data-organization)
- [Repository Architecture](#-repository-architecture)
- [Notebook Workflow](#-notebook-workflow)
- [Model Architecture](#-model-architecture)
- [Model Contract](#-model-contract)
- [Training Strategy](#-training-strategy)
- [Anomaly Detection](#-anomaly-detection)
- [Defect Localization](#-defect-localization)
- [Threshold Calibration](#-threshold-calibration)
- [Evaluation Methodology](#-evaluation-methodology)
- [Inference Pipeline](#-inference-pipeline)
- [Streamlit Application](#-streamlit-application)
- [Python Scripts](#-python-scripts)
- [Configuration](#-configuration)
- [Installation](#-installation)
- [Google Colab Workflow](#-google-colab-workflow)
- [Local VS Code Workflow](#-local-vs-code-workflow)
- [Command Reference](#-command-reference)
- [Example Inference](#-example-inference)
- [Artifacts and Outputs](#-artifacts-and-outputs)
- [Reproducibility](#-reproducibility)
- [Data Integrity](#-data-integrity)
- [Troubleshooting](#-troubleshooting)
- [Performance Considerations](#-performance-considerations)
- [Known Repository Caveats](#-known-repository-caveats)
- [Security and Secret Hygiene](#-security-and-secret-hygiene)
- [Limitations](#-limitations)
- [Responsible Use](#-responsible-use)
- [Research Extensions](#-research-extensions)
- [Roadmap](#-roadmap)
- [Contribution Guide](#-contribution-guide)
- [Citation and Dataset Attribution](#-citation-and-dataset-attribution)
- [Author](#-author)
- [Acknowledgements](#-acknowledgements)
- [License Note](#-license-note)
- [Final Notes](#-final-notes)

---

## 🔎 Project at a Glance

**VisionGuard AI** is a computer-vision research and engineering project focused on **industrial anomaly detection and defect localization** using **TensorFlow/Keras** and the **MVTec AD** benchmark.

The repository is intentionally built as more than a single notebook. It contains a complete experimental workflow covering:

`Dataset Download → Preparation → Exploration → Preprocessing → CNN Autoencoder → Reconstruction Error → Anomaly Detection → Defect Localization → Evaluation → Inference → Streamlit`

The current repository is centered on five MVTec AD object categories:

- `screw`
- `bottle`
- `capsule`
- `metal_nut`
- `hazelnut`

The project uses a **reconstruction-based approach**. Instead of beginning with a conventional supervised image classifier, the system learns to reconstruct normal images and uses the reconstruction discrepancy as an anomaly signal.

That distinction matters. In industrial inspection, the useful question is often not merely:

> “Which class is this image?”

but:

> “Does this product look sufficiently different from what a normal product should look like, and where is the difference concentrated?”

VisionGuard AI is organized around that second question.

---

## 🎯 Why VisionGuard AI Exists

Industrial visual inspection is a practical computer-vision problem with a deceptively simple interface: take an image, inspect the object, decide whether something is wrong.

The engineering reality is harder.

Industrial images can contain:

- subtle scratches,
- dents,
- cracks,
- contamination,
- deformations,
- structural inconsistencies,
- surface defects,
- small localized anomalies,
- changes that occupy only a tiny fraction of the image.

A useful inspection system therefore needs more than a binary label. It needs a pipeline that can reason about visual normality and produce evidence for its decision.

VisionGuard AI is an attempt to build that pipeline in an educational and research-friendly way.

The repository therefore keeps the stages visible:

1. dataset preparation,
2. data inspection,
3. preprocessing,
4. representation learning,
5. reconstruction,
6. anomaly scoring,
7. localization,
8. quantitative evaluation,
9. interactive inference.

This makes the project useful both as a machine-learning implementation and as a study of the reasoning steps behind industrial anomaly detection.

---

## 🧩 Problem Definition

The central task is **industrial image anomaly detection**.

For a selected product category, the training set is dominated by `good` or normal examples. The model learns a reconstruction function:

[
f_	heta(x) ightarrow hat{x}
]

where:

- (x) is the input image,
- (hat{x}) is the reconstructed image,
- (f_	heta) is the learned convolutional autoencoder.

The image-level anomaly score is based on the reconstruction error.

For mean squared reconstruction error:

[
S(x)=rac{1}{HWC}sum_{i=1}^{H}sum_{j=1}^{W}sum_{c=1}^{C}(x_{ijc}-hat{x}_{ijc})^2
]

A category-specific threshold (T) can then be used for image-level classification:

[
	ext{Anomaly}(x)=
egin{cases}
1 & 	ext{if } S(x)>T \
0 & 	ext{otherwise}
end{cases}
]

The same reconstruction difference can be preserved at the pixel level:

[
M_{ij}=rac{1}{C}sum_{c=1}^{C}(x_{ijc}-hat{x}_{ijc})^2
]

where (M) is an anomaly-error map.

The project then transforms the continuous error map into a candidate defect region through thresholding and lightweight morphological processing.

This is the conceptual bridge from:

**image reconstruction → anomaly score → localization map → defect region**

---

## 🥅 Project Objectives

VisionGuard AI is designed around the following objectives.

### 1. Build a complete industrial computer-vision workflow

The repository should be executable as a sequence rather than a collection of disconnected experiments.

### 2. Understand the dataset before training

The EDA stage is treated as part of the model-development process, not as decoration.

### 3. Learn normal visual structure

The autoencoder is trained primarily on normal training images, supporting a reconstruction-based anomaly-detection formulation.

### 4. Produce an image-level anomaly signal

The system calculates reconstruction MSE/MAE and compares the MSE against a threshold.

### 5. Produce pixel-level evidence

The system generates an anomaly map from reconstruction differences.

### 6. Localize candidate defect regions

The anomaly map can be thresholded, morphologically cleaned, and converted into bounding regions.

### 7. Compare predictions against ground truth when available

The repository supports ground-truth mask comparison and localization metrics such as IoU and Dice.

### 8. Make the workflow accessible

Both notebook-driven experimentation and standalone Python entry points are included.

### 9. Expose the model through an interactive interface

The repository contains a Streamlit application for practical inference exploration.

---

## 🧠 Core Idea

The central intuition is simple:

> A model trained to reconstruct normal products should generally reconstruct normal visual structure more faithfully than unusual structure.

The project therefore follows this pattern:

```text
Normal Training Images
        |
        v
   CNN Autoencoder
        |
        v
Learn Normal Representation
        |
        v
      Test Image
        |
        v
    Reconstruction
        |
        v
Original - Reconstruction
        |
        +----------------------+
        |                      |
        v                      v
 Image-Level MSE          Pixel Error Map
        |                      |
        v                      v
   Thresholding          Thresholding
        |                      |
        v                      v
 Normal / Anomaly       Candidate Defect Regions
```

The approach is intentionally interpretable at the pipeline level. The model does not magically output a defect explanation. Instead, the system creates an evidence trail from reconstruction quality to spatial error.

---

# 🔬 How the System Works

## Stage 1 — Dataset acquisition

The project uses the MVTec AD dataset through a KaggleHub download workflow.

The preparation process is designed to:

- obtain the dataset,
- discover the actual dataset root dynamically,
- keep only the five selected categories,
- build a predictable project-local/Colab structure,
- validate expected folders,
- count available images,
- preview samples.

The project avoids relying on a single hard-coded KaggleHub cache path.

---

## Stage 2 — Dataset exploration

Before training, the EDA workflow examines the available data.

The repository's broader analysis includes areas such as:

- category inventory,
- normal/defective distributions,
- defect-type distribution,
- image dimensions,
- file formats,
- image statistics,
- pixel statistics,
- sharpness,
- resolution,
- mask metrics,
- mask coverage,
- defect centroids,
- bounding-box analysis,
- correlation-oriented summaries,
- visual sample inspection.

The objective is not to produce graphs for the sake of graphs.

The objective is to answer practical questions:

- Are files present where expected?
- Are categories complete?
- Are images readable?
- How are normal and defective samples distributed?
- How large are the images?
- How large are defects relative to the image?
- Are there quality differences that could influence training?
- Are localization masks available and aligned with the corresponding test images?

---

## Stage 3 — Preprocessing

The model contract currently operates on RGB images resized to:

`128 × 128 × 3`

Pixels are converted to floating point and normalized to:

`[0,1]`

A simplified preprocessing path is:

```text
Image File
   ↓
RGB Conversion
   ↓
Resize to 128×128
   ↓
Convert to float32
   ↓
Normalize to 0–1
   ↓
TensorFlow Dataset
   ↓
Batch
   ↓
Prefetch
   ↓
Model
```

The preprocessing notebook also provides augmentation and visualization stages so that transformations can be inspected before training.

---

## Stage 4 — Autoencoder training

The core neural network is a convolutional autoencoder.

The encoder progressively reduces spatial resolution while increasing channel depth.

The latent block is used as the learned internal representation.

The decoder then upsamples the latent representation and reconstructs the input image.

The training objective is reconstruction quality:

[
mathcal{L}_{MSE}
=
rac{1}{N}sum_{n=1}^{N}
(x_n-hat{x}_n)^2
]

The repository's standalone training implementation uses:

- Adam optimizer,
- learning rate `1e-3`,
- MSE loss,
- MAE metric,
- deterministic seed `42`,
- a 10% validation split,
- early stopping,
- learning-rate reduction,
- model checkpointing,
- CSV training logs.

The configured maximum epoch count is `40`.

---

# 🗃️ Dataset

VisionGuard AI works with the **MVTec Anomaly Detection (MVTec AD)** benchmark and is configured around a selected five-category subset.

The original dataset is designed for industrial visual anomaly detection and contains normal training data, test data with anomalous examples, and ground-truth annotations for supported defects.

For this repository, the selected working categories are:

```text
screw
bottle
capsule
metal_nut
hazelnut
```

### Dataset source used by the preparation workflow

```python
import kagglehub

path = kagglehub.dataset_download("ipythonx/mvtec-ad")
```

The project then identifies the actual `mvtec_anomaly_detection` root and prepares the requested five categories.

---

# 🧱 Selected Categories

## 🔩 Screw

A small mechanical fastener category.

Typical inspection challenges include subtle surface or structural differences that can be spatially localized.

## 🧴 Bottle

A manufactured container category.

Bottle imagery is useful for studying object structure, shape, and surface anomalies.

## 💊 Capsule

A pharmaceutical-style manufactured object.

Capsule imagery can expose changes in texture, geometry, contamination, or placement.

## 🔩 Metal Nut

A mechanical component with defined geometric structure.

This category is useful for evaluating whether a reconstruction model preserves relatively rigid object structure.

## 🌰 Hazelnut

A natural-object category with a more variable appearance than highly manufactured geometric parts.

This makes it an interesting stress case for understanding how normal variation can interact with reconstruction-based anomaly scoring.

---

# 📁 Data Organization

The prepared dataset follows the standard MVTec-style organization:

```text
visionguard_data/
├── bottle/
│   ├── train/
│   │   └── good/
│   ├── test/
│   │   ├── good/
│   │   └── <defect_type>/
│   └── ground_truth/
│       └── <defect_type>/
│
├── capsule/
│   ├── train/
│   ├── test/
│   └── ground_truth/
│
├── hazelnut/
│   ├── train/
│   ├── test/
│   └── ground_truth/
│
├── metal_nut/
│   ├── train/
│   ├── test/
│   └── ground_truth/
│
└── screw/
    ├── train/
    ├── test/
    └── ground_truth/
```

For training, the normal path of interest is:

```text
<dataset_root>/<category>/train/good/
```

For test-time inspection:

```text
<dataset_root>/<category>/test/
```

For defect localization references:

```text
<dataset_root>/<category>/ground_truth/
```

---

# 🏗️ Repository Architecture

The current repository contains the following major root-level components:

```text
VisionGuard-AI/
│
├── README.md
├── project_structure.md
├── requirements.txt
├── requirements_and_setup.md
├── .gitignore
├── .gitattributes
│
├── config.py
├── train.py
├── predict.py
├── app.py
│
├── dataset_download_and_preparation.ipynb
├── dataset_exploration_EDA.ipynb
├── preprocessing.ipynb
├── CNN_Autoencoder_Model.ipynb
├── anomaly_detection.ipynb
├── defect_localization.ipynb
├── model_evaluation.ipynb
├── inference_demo.ipynb
│
├── artifacts/
├── images/
├── logs/
├── models/
├── table/
└── visionguard_data/
```

> The repository also contains an `.agents/` tree used for project tooling. It is not part of the core ML methodology.

---

# 📓 Notebook Workflow

The notebooks form the research-facing side of the project.

Recommended conceptual order:

```text
1. Dataset Download & Preparation
              ↓
2. Dataset Exploration / EDA
              ↓
3. Preprocessing
              ↓
4. CNN Autoencoder
              ↓
5. Anomaly Detection
              ↓
6. Defect Localization
              ↓
7. Model Evaluation
              ↓
8. Inference Demo
```

## 1️⃣ `dataset_download_and_preparation.ipynb`

Purpose:

- download MVTec AD,
- find the actual dataset root,
- select the five categories,
- prepare `visionguard_data`,
- validate directory structure,
- inspect image counts,
- display representative samples.

This notebook is the correct place to rebuild the prepared dataset in Google Colab.

---

## 2️⃣ `dataset_exploration_EDA.ipynb`

Purpose:

- inspect dataset structure,
- compute image-level statistics,
- analyze defect distributions,
- inspect image quality,
- analyze mask information,
- generate visual summaries.

This notebook should be run before serious model interpretation because downstream observations are only as useful as the underlying data assumptions.

---

## 3️⃣ `preprocessing.ipynb`

Purpose:

- load images,
- resize images,
- normalize images,
- apply approved augmentation,
- prepare TensorFlow data pipelines,
- process localization masks where relevant,
- inspect transformed samples.

The repository's target image contract is `128×128 RGB`.

---

## 4️⃣ `CNN_Autoencoder_Model.ipynb`

Purpose:

- define the convolutional autoencoder,
- compile the model,
- inspect architecture,
- train on normal images,
- visualize reconstruction behavior,
- record training history.

The committed model architecture artifact indicates a compact CNN autoencoder with approximately **778k total parameters** for the documented architecture.

---

## 5️⃣ `anomaly_detection.ipynb`

Purpose:

- load the trained model,
- reconstruct test images,
- calculate image-level reconstruction error,
- calibrate or load thresholds,
- classify normal vs anomaly,
- compare category-level behavior.

---

## 6️⃣ `defect_localization.ipynb`

Purpose:

- compute pixel-level reconstruction differences,
- create anomaly maps,
- threshold anomalous pixels,
- apply morphological cleanup,
- identify candidate regions,
- compare against ground-truth masks,
- visualize localization results.

---

## 7️⃣ `model_evaluation.ipynb`

Purpose:

- aggregate predictions,
- calculate image-level metrics,
- calculate localization metrics,
- inspect errors,
- compare behavior across categories,
- preserve evaluation outputs.

---

## 8️⃣ `inference_demo.ipynb`

Purpose:

- select a test image,
- load the corresponding category model,
- reconstruct the image,
- calculate anomaly score,
- display localization evidence,
- compare to ground truth when available.

---

# 🧠 Model Architecture

The core model is a **convolutional autoencoder**.

The committed architecture is approximately:

```text
Input: 128 × 128 × 3
        │
        ▼
Conv2D 32
BatchNorm
MaxPool
        │
        ▼
Conv2D 64
BatchNorm
MaxPool
        │
        ▼
Conv2D 128
BatchNorm
MaxPool
        │
        ▼
Latent Conv2D 256
        │
        ▼
UpSampling
Conv2D 128
BatchNorm
        │
        ▼
UpSampling
Conv2D 64
BatchNorm
        │
        ▼
UpSampling
Conv2D 32
BatchNorm
        │
        ▼
Reconstruction Conv2D 3
Sigmoid
        │
        ▼
128 × 128 × 3
```

### Architecture characteristics

- Input shape: `(128, 128, 3)`
- Output shape: `(128, 128, 3)`
- Convolution-based encoder
- Convolution-based decoder
- Batch normalization
- Max pooling in the encoder
- UpSampling2D in the decoder
- ReLU hidden activations
- Sigmoid reconstruction output
- MSE training objective
- MAE reporting metric

### Documented model size

The committed screw model summary reports:

- Total parameters: **778,371**
- Trainable parameters: **777,475**
- Non-trainable parameters: **896**
- Approximate parameter memory: **2.97 MB**

These figures describe the committed model summary; they should not be treated as universal numbers for future modified architectures.

---

# 📐 Model Contract

The inference implementation explicitly validates that a category model matches the expected shape:

```text
Input : (None, 128, 128, 3)
Output: (None, 128, 128, 3)
```

This contract is important.

A model trained for a different resolution or number of channels should not silently be accepted by the inference pipeline.

The model-discovery logic prefers:

```text
models/<category>/best_model.keras
```

and can additionally discover compatible `.keras`, `.h5`, and `.hdf5` files associated with the selected category.

---

# 🎓 Training Strategy

The standalone trainer is configured around **category-specific autoencoders**.

Each selected category can have its own model directory:

```text
models/
├── screw/
├── bottle/
├── capsule/
├── metal_nut/
└── hazelnut/
```

This has an important conceptual advantage: the model learns the normal visual distribution of one object category rather than forcing five different object geometries into one reconstruction function.

### Current training configuration

| Configuration | Value |
|---|---:|
| Image height | 128 |
| Image width | 128 |
| Channels | 3 |
| Batch size | 32 |
| Maximum epochs | 40 |
| Validation fraction | 10% |
| Learning rate | 0.001 |
| Minimum learning rate | 0.000001 |
| Optimizer | Adam |
| Loss | MSE |
| Metric | MAE |
| Random seed | 42 |
| Early stopping patience | 7 |
| LR reduction patience | 3 |
| LR reduction factor | 0.5 |

The values above reflect the committed configuration in the repository's training implementation.

---

# 🧪 Training Callbacks

The standalone training script uses several safeguards intended to make repeated experiments cleaner.

### Model checkpointing

Best models are selected using validation loss.

The trainer can save:

```text
best_model.keras
best_weights.weights.h5
```

### Early stopping

Training can terminate when validation loss stops improving.

### ReduceLROnPlateau

The learning rate is reduced when validation loss plateaus, down to a configured minimum.

### CSV logging

Training history is written to CSV so experiments can be inspected without reopening the original training process.

### Seed control

The project uses seed `42` for Python/NumPy/TensorFlow randomness where supported.

Deterministic behavior is a goal, not an absolute guarantee across every hardware and software combination.

---

# 🚨 Anomaly Detection

The anomaly detector is reconstruction-based.

## Image reconstruction

For a test image (x):

[
hat{x}=f_	heta(x)
]

The reconstruction is clipped to ([0,1]).

---

## Image-level metrics

The inference code calculates at least:

### Mean Squared Error

[
MSE=rac{1}{HWC}sum(x-hat{x})^2
]

### Mean Absolute Error

[
MAE=rac{1}{HWC}sum|x-hat{x}|
]

### Maximum absolute pixel-channel error

[
E_{max}=max|x-hat{x}|
]

The primary image anomaly score is the reconstruction MSE.

---

## Normal vs anomaly decision

Conceptually:

```text
score <= threshold
        ↓
      NORMAL

score > threshold
        ↓
     ANOMALY
```

When a category threshold is not available as a stored artifact, the implementation can calibrate a fallback threshold using normal examples.

---

# 🗺️ Defect Localization

Image-level classification answers:

> Is there evidence of an anomaly?

Localization asks:

> Where is that evidence?

VisionGuard AI retains the spatial reconstruction error.

For every pixel location ((i,j)):

[
M_{ij}=rac{1}{3}sum_c(x_{ijc}-hat{x}_{ijc})^2
]

This produces an error map with shape approximately:

`128 × 128`

The localization stage can then perform:

1. thresholding,
2. morphological processing,
3. component filtering,
4. bounding-box extraction,
5. heatmap visualization.

The current configuration includes:

- pixel threshold percentile: `99.5` for the broader project configuration,
- morphology kernel size: `3`,
- minimum component area: `12`.

The exact numerical behavior depends on the threshold artifact and input being evaluated.

---

# 🎚️ Threshold Calibration

Threshold selection is one of the most important parts of anomaly detection.

A model can reconstruct images well and still perform poorly if its threshold is arbitrary.

VisionGuard AI therefore supports **normal-data-based calibration**.

## Image threshold

The configured image threshold percentile is:

`99.0`

The idea is:

1. select normal images,
2. reconstruct them,
3. calculate their reconstruction MSE,
4. estimate a high percentile of normal reconstruction error,
5. use that value as a reference threshold.

In the standalone inference pipeline, the fallback calibration uses up to `40` normal examples.

---

## Pixel threshold

The project also supports a pixel-level threshold calibrated from normal-image pixel-error values.

The configured percentile is:

`99.5`

The fallback calibration can use a smaller number of normal images for the pixel distribution.

---

## Important interpretation note

A percentile threshold is a **calibration strategy**, not a guarantee of optimal production performance.

Thresholds should be evaluated against a validation protocol appropriate to the intended deployment environment.

A threshold derived from a tiny or unrepresentative normal sample may be unstable.

---

# 📊 Evaluation Methodology

The evaluation stage should be split into two distinct levels.

## Image-level evaluation

The repository is structured to support metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC where meaningful
- Confusion matrix

These answer questions such as:

- How often are anomalies detected?
- How many normal images are incorrectly flagged?
- How well does the threshold separate the available normal and defective samples?

---

## Pixel-level evaluation

For localization, the project supports:

- IoU
- Dice score

### Intersection over Union

[
IoU=rac{|Pcap G|}{|Pcup G|}
]

where:

- (P) is the predicted mask,
- (G) is the ground-truth mask.

### Dice score

[
Dice=rac{2|Pcap G|}{|P|+|G|}
]

These metrics are more informative than simply drawing a heatmap because they quantify spatial overlap.

---

## Evaluation discipline

A serious experiment should always record:

- category,
- checkpoint/model version,
- threshold,
- number of evaluated images,
- class distribution,
- metric definitions,
- random seed,
- preprocessing configuration,
- whether thresholds were learned from a separate validation set.

Do not interpret one test image as proof of generalization.

---

# 🖥️ Inference Pipeline

The standalone prediction script is designed for single-image inference.

Pipeline:

```text
Input Image
     ↓
Category Selection
     ↓
Model Discovery
     ↓
Model Contract Validation
     ↓
RGB Conversion
     ↓
Resize 128×128
     ↓
Normalize 0–1
     ↓
Reconstruction
     ↓
MSE / MAE
     ↓
Threshold
     ↓
NORMAL / ANOMALY / UNKNOWN
     ↓
Pixel Error Map
     ↓
Morphological Cleanup
     ↓
Bounding Boxes
     ↓
PNG + JSON Report
```

---

# 🖥️ Streamlit Application

The repository includes an interactive `app.py`.

The application is designed around category-aware inference and provides a more practical inspection interface than a command-line-only workflow.

The Streamlit application supports concepts including:

- category selection,
- image upload,
- model discovery,
- model compatibility checks,
- reconstruction,
- anomaly score,
- threshold inspection,
- pixel-level anomaly map,
- defect-region visualization,
- ground-truth comparison when a known test image is selected,
- IoU and Dice display for reference comparisons,
- interactive Plotly charts,
- model/threshold status tables,
- dataset exploration.

### Run the app

```bash
streamlit run app.py
```

The actual UI behavior depends on the model and dataset artifacts available in the local environment.

---

# 🐍 Python Scripts

## `config.py`

Centralizes:

- project paths,
- dataset root,
- selected categories,
- image dimensions,
- batch size,
- epochs,
- learning rate,
- anomaly percentiles,
- model paths,
- artifact locations,
- random seed,
- runtime switches.

This reduces path and parameter duplication.

---

## `train.py`

Standalone category training pipeline.

Core responsibilities include:

- dataset discovery,
- data validation,
- normal-image enumeration,
- deterministic train/validation split,
- TensorFlow input pipeline,
- autoencoder construction,
- compilation,
- callback setup,
- training,
- history persistence,
- model summary generation,
- threshold reference outputs.

### Run

```bash
python train.py
```

The exact options supported by the current implementation can be inspected with:

```bash
python train.py --help
```

---

## `predict.py`

Standalone single-image inference.

Typical usage:

```bash
python predict.py --category screw --image path/to/image.png
```

With an explicit output file:

```bash
python predict.py \
  --category bottle \
  --image path/to/image.png \
  --output artifacts/predictions/bottle_result.png
```

List discoverable models:

```bash
python predict.py --list-models
```

The current implementation can produce JSON results and visual inference output.

---

## `app.py`

Streamlit interface for interactive inference and repository exploration.

---

# ⚙️ Configuration

The repository uses a central configuration module.

Important values include:

```python
CATEGORIES = [
    "screw",
    "bottle",
    "capsule",
    "metal_nut",
    "hazelnut",
]

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_CHANNELS = 3

BATCH_SIZE = 32
EPOCHS = 40
VALIDATION_SIZE = 0.10

LEARNING_RATE = 1e-3

IMAGE_THRESHOLD_PERCENTILE = 99.0
PIXEL_THRESHOLD_PERCENTILE = 99.5

RANDOM_SEED = 42
```

The prepared dataset is expected to be discoverable in either:

```text
/content/visionguard_data
```

or the repository-local:

```text
<project_root>/visionguard_data
```

depending on the script.

---

# 💻 Installation

## Prerequisites

Recommended baseline:

- Python 3.x
- Git
- VS Code or Jupyter
- TensorFlow/Keras
- enough local storage for image data and generated artifacts

GPU availability is optional for non-training steps.

---

## 1. Clone the repository

```bash
git clone https://github.com/Maganpreet-Singh/VisionGuard-AI.git
cd VisionGuard-AI
```

---

## 2. Create a virtual environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell execution policy blocks activation for the current process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Upgrade pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## 4. Install dependencies

The intended dependency set includes:

- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Plotly
- scikit-learn
- Pillow
- OpenCV
- KaggleHub
- Jupyter
- ipykernel
- Streamlit-related runtime packages used by `app.py`

Use:

```bash
python -m pip install -r requirements.txt
```

### ⚠️ Important

The current committed `requirements.txt` contains unresolved Git merge-conflict markers:

```text
<<<<<<< HEAD
...
=======
...
>>>>>>> ...
```

That file should be cleaned before treating `pip install -r requirements.txt` as the canonical setup command.

A temporary manual installation path is:

```bash
python -m pip install tensorflow numpy pandas matplotlib seaborn plotly scikit-learn Pillow opencv-python kagglehub jupyter ipykernel streamlit
```

---

# ☁️ Google Colab Workflow

Google Colab is a convenient environment for the dataset and TensorFlow experiments.

## Recommended order

```text
Open Colab
   ↓
Select GPU runtime for training
   ↓
Install required packages
   ↓
Run dataset preparation notebook
   ↓
Run EDA
   ↓
Run preprocessing
   ↓
Train model
   ↓
Run anomaly detection
   ↓
Run localization
   ↓
Evaluate
   ↓
Run inference demo
```

---

## GPU runtime check

```python
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
```

A result such as:

```text
GPUs: []
```

means TensorFlow is not currently exposing a GPU to that runtime.

This does not prevent all project work. Dataset preparation, EDA, and many utility operations can still run on CPU.

---

# 🧪 Local VS Code Workflow

## Verify the Python interpreter

```bash
python -c "import sys; print(sys.executable)"
```

Make sure it points to the intended `.venv`.

In VS Code:

```text
Ctrl + Shift + P
→ Python: Select Interpreter
→ choose .venv
```

---

## Install Jupyter kernel

```bash
python -m pip install jupyter ipykernel
```

Register a dedicated kernel:

```bash
python -m ipykernel install --user --name visionguard-ai --display-name "VisionGuard AI"
```

Then select the **VisionGuard AI** kernel in the notebook UI.

---

# 🧭 Command Reference

| Task | Command |
|---|---|
| Clone repository | `git clone https://github.com/Maganpreet-Singh/VisionGuard-AI.git` |
| Enter repository | `cd VisionGuard-AI` |
| Create venv | `python -m venv .venv` |
| Upgrade pip | `python -m pip install --upgrade pip setuptools wheel` |
| Install dependencies | `python -m pip install -r requirements.txt` |
| Train | `python train.py` |
| Train help | `python train.py --help` |
| Prediction help | `python predict.py --help` |
| List models | `python predict.py --list-models` |
| Streamlit | `streamlit run app.py` |
| TensorFlow check | `python -c "import tensorflow as tf; print(tf.__version__)"` |
| GPU check | `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"` |
| Git status | `git status` |

---

# 🔍 Example Inference

Suppose a compatible `screw` model exists at:

```text
models/screw/best_model.keras
```

and the input image is:

```text
sample.png
```

Run:

```bash
python predict.py --category screw --image sample.png
```

A typical logical output contains:

```text
Category
Model path
Input path
Anomaly score
Threshold
Prediction
Localization information
Output artifact paths
```

The implementation may report:

- `NORMAL`
- `ANOMALY`
- `UNKNOWN`

depending on whether a usable threshold is available.

The important part is that **the printed prediction is only as trustworthy as the calibration and evaluation behind that threshold**.

---

# 📦 Artifacts and Outputs

The repository includes an artifact-oriented organization.

Common locations include:

```text
artifacts/
├── figures/
├── heatmaps/
├── predictions/
├── evaluation/
└── reports/
```

## `artifacts/figures/`

Suitable for:

- training curves,
- model diagnostics,
- analysis figures,
- reconstruction comparisons.

## `artifacts/heatmaps/`

Suitable for:

- anomaly maps,
- heatmaps,
- localization visuals.

## `artifacts/predictions/`

Suitable for:

- image-level prediction reports,
- serialized inference outputs,
- saved visual inspection results.

## `artifacts/evaluation/`

Suitable for:

- confusion matrices,
- metrics,
- category-level evaluations,
- localization statistics.

## `artifacts/reports/`

Suitable for:

- experiment summaries,
- generated reports,
- structured evaluation notes.

## `logs/training/`

Suitable for:

- CSV history,
- training logs,
- experiment tracking outputs.

---

# 🖼️ Visualization Strategy

The repository is built around the principle that model behavior should be visible.

Useful visual comparisons include:

```text
Original Image
      ↓
Reconstructed Image
      ↓
Error / Difference Map
      ↓
Thresholded Anomaly Mask
      ↓
Bounding Regions
      ↓
Ground Truth Mask
```

This sequence is much more informative than looking only at a final class label.

A strong industrial anomaly-detection report should answer all three:

1. **Did the model flag the image?**
2. **How strong was the anomaly signal?**
3. **Where did the signal appear?**

---

# 🔁 Reproducibility

Reproducibility is treated as a first-class concern.

## Fixed configuration

Keep:

- image resolution,
- seed,
- split fraction,
- batch size,
- optimizer,
- thresholding policy,
- model path conventions

consistent for comparable experiments.

## Deterministic data splitting

The standalone trainer uses a deterministic shuffled split based on seed `42`.

## Preserve artifacts

Do not overwrite a valuable experiment without recording:

- model version,
- threshold,
- training history,
- evaluation results.

## Separate model training and threshold calibration

Ideally, threshold calibration should use a clearly defined validation subset that is distinct from the final evaluation set.

---

# 🧹 Data Integrity

Before training, verify:

### Category structure

```text
category/
├── train/
├── test/
└── ground_truth/
```

### Normal training images

```text
category/train/good/
```

### Test images

```text
category/test/<defect_type>/
```

### Ground-truth masks

```text
category/ground_truth/<defect_type>/
```

### Common structure mistake

Do not create:

```text
visionguard_data/screw/screw/train/
```

when the intended structure is:

```text
visionguard_data/screw/train/
```

Nested category duplication is a common source of `FileNotFoundError` and zero-image datasets.

---

# 🛠️ Troubleshooting

## `ModuleNotFoundError: No module named 'tensorflow'`

Check the interpreter:

```bash
python -c "import sys; print(sys.executable)"
```

Then install TensorFlow into that exact environment:

```bash
python -m pip install tensorflow
```

Do not assume the Python interpreter used by VS Code is the same one used by another terminal.

---

## TensorFlow installs but GPU is not detected

Check:

```bash
nvidia-smi
```

Then:

```python
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
```

A working NVIDIA driver does not automatically imply that a particular TensorFlow installation can use the GPU.

For a training workflow, verify the operating-system and TensorFlow compatibility for the environment you are actually using.

---

## `FileNotFoundError` for the dataset

Check:

```python
from pathlib import Path

root = Path("/content/visionguard_data")

print("Exists:", root.exists())
print("Is directory:", root.is_dir())

if root.exists():
    print([p.name for p in root.iterdir()])
```

On local VS Code, also check:

```text
<project_root>/visionguard_data
```

---

## Wrong dataset root

The preparation and inference workflows are designed to discover the actual dataset layout rather than blindly trusting a fixed cache path.

Do not hard-code a KaggleHub cache folder after an environment change.

---

## `BadZipFile`

A `BadZipFile` error does not automatically prove that the dataset source is invalid.

First inspect:

- the returned KaggleHub path,
- whether the expected root directory exists,
- whether the downloaded item is a directory or archive,
- whether the environment produced a partial download.

The repository's preparation workflow is designed to dynamically locate `mvtec_anomaly_detection`.

---

## Empty training set

If the trainer reports no images, verify:

```text
<dataset_root>/<category>/train/good/
```

and confirm image extensions such as:

```text
.png
.jpg
.jpeg
```

The standalone training script performs explicit filesystem validation before beginning training.

---

## Incompatible model

The inference pipeline validates that the model's input and output shape match:

```text
128 × 128 × 3
```

A model with another input shape should be rejected instead of silently producing invalid predictions.

---

## Missing threshold

If no saved image threshold is available, inference may attempt fallback normal-data calibration.

If no usable normal calibration images exist, classification may become `UNKNOWN` instead of pretending the threshold is reliable.

That behavior is intentional.

---

## Streamlit starts but inference fails

Check:

1. TensorFlow importability.
2. Model presence.
3. Model compatibility.
4. Dataset availability if ground-truth comparison is requested.
5. Required Python packages.
6. Correct selected category.

The application contains safe error handling so an unavailable artifact does not need to crash the entire dashboard.

---

# 🚀 Performance Considerations

Industrial image pipelines can become expensive very quickly.

## Batch size

Current configuration:

`32`

Reduce batch size if memory pressure occurs.

## Image size

Current model contract:

`128×128`

Increasing resolution can improve access to tiny spatial details, but it also increases compute and memory cost.

## Caching

The configuration intentionally does not enable dataset caching by default because image-heavy datasets can consume substantial RAM.

## Parallel input

TensorFlow uses `AUTOTUNE` for parallel mapping/prefetch where configured.

## GPU memory growth

The standalone trainer enables TensorFlow GPU memory growth where possible.

This can reduce aggressive up-front GPU memory allocation.

---

# ⚠️ Known Repository Caveats

This section is intentionally blunt because reproducibility depends on knowing what is imperfect.

## 1. `requirements.txt` contains merge markers

The committed file currently includes unresolved Git conflict markers.

That should be cleaned before relying on:

```bash
pip install -r requirements.txt
```

as a guaranteed setup path.

## 2. Root notebook naming is not perfectly uniform

The current repository uses both descriptive names such as:

```text
dataset_download_and_preparation.ipynb
```

and earlier documentation conventions that describe numbered notebook names.

The actual root-level filenames are the source of truth for this repository.

## 3. Repository contains both code and large data trees

The presence of `visionguard_data/` means storage and Git performance should be monitored carefully.

Large datasets and model files are better handled through appropriate storage/versioning strategies rather than blindly committing every generated artifact.

## 4. Performance numbers are not fabricated here

This README intentionally does not invent a final accuracy, precision, recall, IoU, or Dice score.

Those values should come from a reproducible evaluation run using the committed model/checkpoint and a clearly documented evaluation set.

---

# 🔐 Security and Secret Hygiene

Never commit:

```text
.env
API keys
passwords
private credentials
service-account keys
Kaggle tokens
temporary authentication files
```

The presence of a dataset-download workflow does not mean credentials belong in the repository.

For Kaggle authentication, use the supported environment/credential mechanism of the runtime rather than hard-coding secrets in notebooks.

---

# 📉 Limitations

VisionGuard AI is a research/learning system, not a certified industrial safety system.

Important limitations include:

### Domain dependence

The model is category-specific and trained on a limited benchmark subset.

### Threshold dependence

Reconstruction thresholds depend on the distribution used for calibration.

### False positives

Natural or legitimate variation can produce reconstruction errors.

### False negatives

A defect that the model reconstructs unusually well may produce a low anomaly score.

### Localization is approximate

Difference-map thresholding is not equivalent to a fully trained segmentation network.

### Benchmark-to-production gap

A controlled benchmark environment does not necessarily represent:

- real factory lighting,
- camera variation,
- motion blur,
- lens distortion,
- contamination,
- perspective changes,
- production-line vibration,
- unseen product variants.

### Model generalization

The system should not be assumed to generalize to categories or environments outside the data used during development.

---

# 🧭 Responsible Use

VisionGuard AI is best treated as:

- a computer-vision research project,
- an educational deep-learning implementation,
- a benchmark experimentation framework,
- a prototype inspection system.

For production use, additional work would be needed around:

- validation,
- calibration,
- fail-safe behavior,
- dataset shift monitoring,
- image-quality checks,
- logging,
- deployment testing,
- model governance,
- threshold governance,
- auditability.

A model that produces a heatmap is not automatically a production-grade inspection system.

---

# 🔬 Research Extensions

The current architecture creates a useful baseline for several deeper directions.

## 1. Improved autoencoder architectures

Possible experiments:

- deeper convolutional autoencoders,
- residual blocks,
- denoising autoencoders,
- variational autoencoders,
- perceptual losses.

---

## 2. Better anomaly scoring

Instead of MSE alone:

- SSIM-based discrepancy,
- feature-space distance,
- LPIPS-style perceptual metrics,
- hybrid image + feature scores,
- normalized category scores.

---

## 3. Stronger localization

Move from simple error-map thresholding toward:

- U-Net segmentation,
- attention-guided localization,
- patch-based anomaly maps,
- feature pyramid approaches,
- segmentation-aware post-processing.

---

## 4. Modern anomaly-detection baselines

For research comparisons, the project could be extended with methods such as:

- PatchCore-style feature memory,
- PaDiM-style statistical modeling,
- FastFlow-style normalizing-flow methods,
- teacher-student anomaly detection,
- transformer-based anomaly models.

The important principle is not to replace the autoencoder blindly. It is to compare methods under the same dataset split and evaluation protocol.

---

## 5. Threshold optimization

Instead of using one percentile policy:

- optimize thresholds on a validation set,
- compare ROC and precision-recall operating points,
- analyze per-category threshold stability,
- investigate calibration drift.

---

## 6. Explainability

Extend the evidence layer with:

- occlusion maps,
- saliency methods,
- feature visualization,
- reconstruction residual analysis,
- confidence calibration.

---

## 7. Deployment

Potential productionization layers:

```text
Camera
  ↓
Image Quality Gate
  ↓
Category Model
  ↓
Anomaly Score
  ↓
Localization
  ↓
Decision API
  ↓
Inspection UI
  ↓
Human Review / Audit Log
```

---

# 🛣️ Roadmap

## ✅ Foundation

- [x] MVTec AD dataset workflow
- [x] Five-category selection
- [x] Dataset preparation
- [x] EDA workflow
- [x] Image preprocessing
- [x] TensorFlow pipeline
- [x] CNN autoencoder architecture
- [x] Standalone training script
- [x] Standalone inference script
- [x] Streamlit application
- [x] Ground-truth localization comparison support

## 🔄 In Progress / Experimental

- [ ] Clean and pin final dependency versions
- [ ] Consolidate notebook naming conventions
- [ ] Produce reproducible category-by-category benchmark results
- [ ] Standardize threshold artifacts
- [ ] Consolidate model artifact naming

## 🚀 Future

- [ ] Stronger anomaly baselines
- [ ] Automated experiment tracking
- [ ] Model/version registry
- [ ] Better localization metrics
- [ ] Robust image-quality validation
- [ ] Production-oriented API
- [ ] Containerization
- [ ] CI tests for preprocessing and inference
- [ ] Automated regression tests for model contract
- [ ] Benchmark dashboard

---

# 🤝 Contribution Guide

Contributions should improve the reliability, clarity, or scientific usefulness of the project.

## Recommended workflow

```bash
git clone https://github.com/Maganpreet-Singh/VisionGuard-AI.git
cd VisionGuard-AI
git checkout -b feature/your-feature
```

Make a focused change.

Then:

```bash
git status
git add .
git commit -m "Describe the change"
git push -u origin feature/your-feature
```

Open a pull request with:

- what changed,
- why it changed,
- how it was tested,
- which notebooks/scripts were affected,
- whether outputs were regenerated.

---

# 🧪 Contribution Quality Checklist

Before submitting changes:

```text
[ ] Code runs from a clean environment
[ ] Dataset assumptions are documented
[ ] Paths are not hard-coded unnecessarily
[ ] No secrets are committed
[ ] Large generated files are handled appropriately
[ ] Notebook changes are reproducible
[ ] Model input/output contract is preserved
[ ] Evaluation claims are backed by actual runs
[ ] README instructions match the real repository
```

---

# 📚 Citation and Dataset Attribution

VisionGuard AI uses the **MVTec Anomaly Detection** benchmark.

Please consult and follow the dataset's official licensing, citation, and usage requirements.

The KaggleHub workflow references:

```text
ipythonx/mvtec-ad
```

This repository should not be interpreted as the original publisher of the MVTec AD dataset.

---

# 👨‍💻 Author

## Maganpreet Singh

Computer Science & Engineering student working across:

**Artificial Intelligence • Machine Learning • Deep Learning • Computer Vision • Data Science**

GitHub:

https://github.com/Maganpreet-Singh

Project:

https://github.com/Maganpreet-Singh/VisionGuard-AI

---

# 🙏 Acknowledgements

VisionGuard AI stands on the shoulders of the broader open-source machine-learning ecosystem.

Core technologies used in the project include:

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Plotly
- OpenCV
- Pillow
- scikit-learn
- KaggleHub
- Jupyter
- Streamlit

The dataset and benchmark methodology should always be attributed to their original source.

---

# 📜 License Note

The repository's source code, generated artifacts, model files, and third-party dataset components may not all share the same licensing terms.

In particular:

- the MVTec AD dataset has its own terms,
- third-party dependencies have their own licenses,
- generated model artifacts may have separate provenance,
- repository source files are distinct from dataset redistribution rights.

Review the applicable license/readme files included with the dataset before redistributing data.

---

# 📈 A Deeper Look at the Research Logic

This section summarizes the intellectual structure of the project.

## Normality as a learned reconstruction problem

The model does not need a separate class label for every possible defect.

Instead, it learns a representation of normal appearance.

That reframes the problem:

```text
Traditional supervised classifier:
image → defect class

VisionGuard reconstruction approach:
normal images → reconstruction model
test image → reconstruction error
reconstruction error → anomaly signal
```

This is especially useful conceptually when the set of possible defects is open-ended.

---

## Why category-specific models?

A screw, a bottle, a capsule, a metal nut, and a hazelnut do not share the same geometry.

A category-specific model can devote its representational capacity to one normal visual distribution.

This also makes threshold calibration category-aware.

The trade-off is operational complexity:

- more models,
- more model files,
- more deployment routing,
- more category-specific monitoring.

There is no free lunch.

---

## Why preserve the error map?

An image-level score compresses a rich spatial signal into one scalar.

Consider two images with the same MSE.

In one image, error could be spread weakly across the whole object.

In another, error could be concentrated into one small region.

The two situations can have similar totals but very different inspection implications.

That is why VisionGuard retains the pixel-level error map.

---

# 🧠 Practical Interpretation of Model Outputs

Suppose inference produces:

```text
MSE = 0.012
Threshold = 0.009
```

The arithmetic relationship says:

`0.012 > 0.009`

so the threshold rule flags the image as anomalous.

But that does **not** mean:

> “The model proved the product is defective.”

It means:

> “Under this category, preprocessing path, reconstruction model, and calibrated threshold, the image's reconstruction error exceeded the chosen operating point.”

That language matters.

The model emits a signal.

The inspection system turns that signal into a decision.

Those are related, but they are not identical.

---

# 🧮 Interpreting Reconstruction Error

A low reconstruction error can mean:

- the image resembles learned normal structure,
- the model is good at reconstructing that visual pattern,
- the image may be normal.

A high reconstruction error can mean:

- the image contains anomalous structure,
- the image has unusual but legitimate variation,
- the image quality differs from the training distribution,
- the model simply struggles with the particular appearance.

This is why a mature evaluation process must include both:

```text
Defective examples
+
Normal examples
+
Image quality analysis
+
Threshold calibration
+
Error analysis
```

---

# 🧪 Experimental Hygiene

For every serious experiment, create a record similar to:

```text
Experiment ID:
Date:
Category:
Dataset version:
Image size:
Training seed:
Batch size:
Epochs:
Learning rate:
Model checkpoint:
Threshold method:
Threshold value:
Validation size:
Test set definition:
Accuracy:
Precision:
Recall:
F1:
ROC-AUC:
IoU:
Dice:
Notes:
```

This small discipline dramatically improves reproducibility.

---

# 📦 Suggested Long-Term Artifact Layout

As the project matures, a clean experiment-oriented structure could look like:

```text
artifacts/
├── runs/
│   ├── 2026-10-03_screw_baseline/
│   │   ├── config.json
│   │   ├── history.csv
│   │   ├── metrics.json
│   │   ├── threshold.json
│   │   ├── confusion_matrix.png
│   │   └── examples/
│   │
│   └── ...
│
├── figures/
├── heatmaps/
├── predictions/
├── evaluation/
└── reports/
```

That layout separates reproducible experiments from ad-hoc outputs.

---

# 🔧 Engineering Principles Behind the Repository

The project benefits from a few simple rules.

## One source of truth for configuration

Paths and hyperparameters should live in configuration rather than being duplicated across ten notebooks.

## Fail early

A missing category should cause a clear error before training starts.

## Validate model contracts

Do not silently run inference with an incompatible architecture.

## Prefer explicit outputs

Write model summaries, histories, metrics, thresholds, and inference reports to known locations.

## Never manufacture evaluation numbers

A project becomes less credible the moment it invents a benchmark result.

## Visualize the model's reasoning

Reconstruction and error maps provide useful inspection evidence.

---

# 🧭 What This Project Is — and Is Not

### This project is:

- a TensorFlow/Keras computer-vision pipeline,
- an anomaly-detection prototype,
- an industrial inspection research project,
- a learning portfolio project,
- a foundation for more advanced experiments.

### This project is not yet:

- a certified manufacturing QA system,
- a guaranteed defect detector,
- a complete edge-deployment stack,
- a universal anomaly detector for arbitrary industrial images.

That boundary should remain explicit as the project evolves.

---

# 🌱 Where to Take VisionGuard AI Next

The most valuable next steps are not simply adding more code.

They are making the scientific loop tighter:

```text
Hypothesis
   ↓
Controlled Experiment
   ↓
Measured Result
   ↓
Error Analysis
   ↓
Improved Hypothesis
   ↓
New Experiment
```

For an anomaly-detection project, that loop is more important than a flashy UI.

A beautiful dashboard is useful.

A reproducible experiment is better.

A reproducible experiment with interpretable localization is better still.

---

# ⭐ Final Notes

VisionGuard AI is built around a straightforward idea:

> **Learn normal. Measure deviation. Localize the evidence. Evaluate honestly.**

The repository brings together:

```text
MVTec AD
   +
TensorFlow/Keras
   +
CNN Autoencoder
   +
Reconstruction Error
   +
Anomaly Thresholding
   +
Defect Localization
   +
Ground-Truth Evaluation
   +
Streamlit
```

The project is intentionally open-ended. The current autoencoder is a baseline, not the end of the story.

The strongest future version of VisionGuard AI will not be the one with the most files.

It will be the one with:

- cleaner data contracts,
- pinned dependencies,
- reproducible experiments,
- transparent threshold calibration,
- category-level benchmarks,
- stronger localization,
- automated testing,
- clear model provenance,
- and evidence-backed performance claims.

That is the direction from prototype toward engineering.

---

<div align="center">

### 🛡️ VisionGuard AI

**Industrial Anomaly Detection & Defect Localization**

Built with Python, TensorFlow, Keras, OpenCV, and a lot of debugging.

**Learn normal → detect deviation → localize defects → evaluate → iterate.**

⭐ [View the repository](https://github.com/Maganpreet-Singh/VisionGuard-AI)

</div>


---

# 🧠 Extended Technical Field Manual

This appendix expands the repository documentation into deeper implementation notes. It is intentionally detailed so the README can serve as a long-form reference for studying, extending, and auditing the project.

## 1. Dataset contract: screw

The dataset contract is the boundary between raw files and every downstream experiment. A stable contract makes failures visible early, keeps training assumptions explicit, and prevents accidental mixing of categories or splits. Applied specifically to **screw**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. The practical question is not whether this stage sounds sophisticated; it is whether its inputs, outputs, assumptions, and failure modes are visible enough that another engineer can reproduce the reasoning. For industrial inspection, small implementation details can change the decision boundary. Resize filters, normalization ranges, category routing, threshold provenance, and artifact selection should therefore be documented alongside the high-level method.

### Operational checklist

- Confirm that the screw directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for screw.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to screw and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For screw, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. The safest pattern is to fail loudly when a required artifact is missing, and to return an explicit unknown state when a decision cannot be supported by a reliable threshold or compatible model.

### Research angle

A useful experiment is to compare the baseline behavior for screw under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 2. Dataset contract: bottle

The dataset contract is the boundary between raw files and every downstream experiment. A stable contract makes failures visible early, keeps training assumptions explicit, and prevents accidental mixing of categories or splits. Applied specifically to **bottle**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. For industrial inspection, small implementation details can change the decision boundary. Resize filters, normalization ranges, category routing, threshold provenance, and artifact selection should therefore be documented alongside the high-level method. A robust workflow treats every generated score as conditional on the pipeline that produced it. Change the image size, preprocessing, checkpoint, calibration set, or metric definition and the meaning of the resulting number can change.

### Operational checklist

- Confirm that the bottle directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for bottle.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to bottle and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For bottle, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. Visual inspection of representative examples remains valuable even when quantitative metrics exist. Images reveal clipping, misalignment, over-smoothing, localization leakage, and other issues that a single aggregate metric can conceal.

### Research angle

A useful experiment is to compare the baseline behavior for bottle under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 3. Dataset contract: capsule

The dataset contract is the boundary between raw files and every downstream experiment. A stable contract makes failures visible early, keeps training assumptions explicit, and prevents accidental mixing of categories or splits. Applied specifically to **capsule**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. A robust workflow treats every generated score as conditional on the pipeline that produced it. Change the image size, preprocessing, checkpoint, calibration set, or metric definition and the meaning of the resulting number can change. The safest pattern is to fail loudly when a required artifact is missing, and to return an explicit unknown state when a decision cannot be supported by a reliable threshold or compatible model.

### Operational checklist

- Confirm that the capsule directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for capsule.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to capsule and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For capsule, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. The repository should evolve by controlled experiments: change one important assumption, record the configuration, rerun the relevant evaluation, compare the result, and preserve the evidence.

### Research angle

A useful experiment is to compare the baseline behavior for capsule under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 4. Dataset contract: metal_nut

The dataset contract is the boundary between raw files and every downstream experiment. A stable contract makes failures visible early, keeps training assumptions explicit, and prevents accidental mixing of categories or splits. Applied specifically to **metal_nut**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. The safest pattern is to fail loudly when a required artifact is missing, and to return an explicit unknown state when a decision cannot be supported by a reliable threshold or compatible model. Visual inspection of representative examples remains valuable even when quantitative metrics exist. Images reveal clipping, misalignment, over-smoothing, localization leakage, and other issues that a single aggregate metric can conceal.

### Operational checklist

- Confirm that the metal_nut directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for metal_nut.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to metal_nut and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For metal_nut, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. A portfolio-quality machine-learning repository becomes much stronger when the documentation distinguishes current implementation, intended workflow, experimental ideas, and future work rather than blending them together.

### Research angle

A useful experiment is to compare the baseline behavior for metal_nut under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 5. Dataset contract: hazelnut

The dataset contract is the boundary between raw files and every downstream experiment. A stable contract makes failures visible early, keeps training assumptions explicit, and prevents accidental mixing of categories or splits. Applied specifically to **hazelnut**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. Visual inspection of representative examples remains valuable even when quantitative metrics exist. Images reveal clipping, misalignment, over-smoothing, localization leakage, and other issues that a single aggregate metric can conceal. The repository should evolve by controlled experiments: change one important assumption, record the configuration, rerun the relevant evaluation, compare the result, and preserve the evidence.

### Operational checklist

- Confirm that the hazelnut directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for hazelnut.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to hazelnut and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For hazelnut, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. Category-specific behavior deserves its own analysis because five classes can have very different texture, geometry, background, and defect distributions even when the same model family is used.

### Research angle

A useful experiment is to compare the baseline behavior for hazelnut under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 6. Image normalization: screw

Normalization maps decoded pixel values into the numeric range expected by the reconstruction network. Keeping the transformation consistent across training, validation, threshold calibration, notebooks, scripts, and Streamlit inference is essential. Applied specifically to **screw**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. The repository should evolve by controlled experiments: change one important assumption, record the configuration, rerun the relevant evaluation, compare the result, and preserve the evidence. A portfolio-quality machine-learning repository becomes much stronger when the documentation distinguishes current implementation, intended workflow, experimental ideas, and future work rather than blending them together.

### Operational checklist

- Confirm that the screw directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for screw.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to screw and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For screw, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. Thresholds and localization masks should be versioned conceptually even when the repository does not yet have a formal experiment tracker. A threshold file without provenance is much less useful than a threshold tied to a documented calibration protocol.

### Research angle

A useful experiment is to compare the baseline behavior for screw under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 7. Image normalization: bottle

Normalization maps decoded pixel values into the numeric range expected by the reconstruction network. Keeping the transformation consistent across training, validation, threshold calibration, notebooks, scripts, and Streamlit inference is essential. Applied specifically to **bottle**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. A portfolio-quality machine-learning repository becomes much stronger when the documentation distinguishes current implementation, intended workflow, experimental ideas, and future work rather than blending them together. Category-specific behavior deserves its own analysis because five classes can have very different texture, geometry, background, and defect distributions even when the same model family is used.

### Operational checklist

- Confirm that the bottle directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for bottle.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to bottle and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For bottle, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. Deployment should never be allowed to silently fall back from a missing trained artifact to an arbitrary model. The current scripts deliberately search for compatible category models and validate the input/output contract before inference.

### Research angle

A useful experiment is to compare the baseline behavior for bottle under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 8. Image normalization: capsule

Normalization maps decoded pixel values into the numeric range expected by the reconstruction network. Keeping the transformation consistent across training, validation, threshold calibration, notebooks, scripts, and Streamlit inference is essential. Applied specifically to **capsule**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. Category-specific behavior deserves its own analysis because five classes can have very different texture, geometry, background, and defect distributions even when the same model family is used. Thresholds and localization masks should be versioned conceptually even when the repository does not yet have a formal experiment tracker. A threshold file without provenance is much less useful than a threshold tied to a documented calibration protocol.

### Operational checklist

- Confirm that the capsule directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for capsule.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to capsule and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For capsule, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. The practical question is not whether this stage sounds sophisticated; it is whether its inputs, outputs, assumptions, and failure modes are visible enough that another engineer can reproduce the reasoning.

### Research angle

A useful experiment is to compare the baseline behavior for capsule under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 9. Image normalization: metal_nut

Normalization maps decoded pixel values into the numeric range expected by the reconstruction network. Keeping the transformation consistent across training, validation, threshold calibration, notebooks, scripts, and Streamlit inference is essential. Applied specifically to **metal_nut**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. Thresholds and localization masks should be versioned conceptually even when the repository does not yet have a formal experiment tracker. A threshold file without provenance is much less useful than a threshold tied to a documented calibration protocol. Deployment should never be allowed to silently fall back from a missing trained artifact to an arbitrary model. The current scripts deliberately search for compatible category models and validate the input/output contract before inference.

### Operational checklist

- Confirm that the metal_nut directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for metal_nut.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to metal_nut and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For metal_nut, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. For industrial inspection, small implementation details can change the decision boundary. Resize filters, normalization ranges, category routing, threshold provenance, and artifact selection should therefore be documented alongside the high-level method.

### Research angle

A useful experiment is to compare the baseline behavior for metal_nut under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 10. Image normalization: hazelnut

Normalization maps decoded pixel values into the numeric range expected by the reconstruction network. Keeping the transformation consistent across training, validation, threshold calibration, notebooks, scripts, and Streamlit inference is essential. Applied specifically to **hazelnut**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. Deployment should never be allowed to silently fall back from a missing trained artifact to an arbitrary model. The current scripts deliberately search for compatible category models and validate the input/output contract before inference. The practical question is not whether this stage sounds sophisticated; it is whether its inputs, outputs, assumptions, and failure modes are visible enough that another engineer can reproduce the reasoning.

### Operational checklist

- Confirm that the hazelnut directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for hazelnut.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to hazelnut and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For hazelnut, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. A robust workflow treats every generated score as conditional on the pipeline that produced it. Change the image size, preprocessing, checkpoint, calibration set, or metric definition and the meaning of the resulting number can change.

### Research angle

A useful experiment is to compare the baseline behavior for hazelnut under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 11. Reconstruction error: screw

Reconstruction error is the bridge from the autoencoder to anomaly scoring. The project primarily uses mean squared error at the image level while retaining a spatial error map for localization. Applied specifically to **screw**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. The practical question is not whether this stage sounds sophisticated; it is whether its inputs, outputs, assumptions, and failure modes are visible enough that another engineer can reproduce the reasoning. For industrial inspection, small implementation details can change the decision boundary. Resize filters, normalization ranges, category routing, threshold provenance, and artifact selection should therefore be documented alongside the high-level method.

### Operational checklist

- Confirm that the screw directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for screw.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to screw and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For screw, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. The safest pattern is to fail loudly when a required artifact is missing, and to return an explicit unknown state when a decision cannot be supported by a reliable threshold or compatible model.

### Research angle

A useful experiment is to compare the baseline behavior for screw under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 12. Reconstruction error: bottle

Reconstruction error is the bridge from the autoencoder to anomaly scoring. The project primarily uses mean squared error at the image level while retaining a spatial error map for localization. Applied specifically to **bottle**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. For industrial inspection, small implementation details can change the decision boundary. Resize filters, normalization ranges, category routing, threshold provenance, and artifact selection should therefore be documented alongside the high-level method. A robust workflow treats every generated score as conditional on the pipeline that produced it. Change the image size, preprocessing, checkpoint, calibration set, or metric definition and the meaning of the resulting number can change.

### Operational checklist

- Confirm that the bottle directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for bottle.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to bottle and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For bottle, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. Visual inspection of representative examples remains valuable even when quantitative metrics exist. Images reveal clipping, misalignment, over-smoothing, localization leakage, and other issues that a single aggregate metric can conceal.

### Research angle

A useful experiment is to compare the baseline behavior for bottle under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 13. Reconstruction error: capsule

Reconstruction error is the bridge from the autoencoder to anomaly scoring. The project primarily uses mean squared error at the image level while retaining a spatial error map for localization. Applied specifically to **capsule**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. A robust workflow treats every generated score as conditional on the pipeline that produced it. Change the image size, preprocessing, checkpoint, calibration set, or metric definition and the meaning of the resulting number can change. The safest pattern is to fail loudly when a required artifact is missing, and to return an explicit unknown state when a decision cannot be supported by a reliable threshold or compatible model.

### Operational checklist

- Confirm that the capsule directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for capsule.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to capsule and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For capsule, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. The repository should evolve by controlled experiments: change one important assumption, record the configuration, rerun the relevant evaluation, compare the result, and preserve the evidence.

### Research angle

A useful experiment is to compare the baseline behavior for capsule under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 14. Reconstruction error: metal_nut

Reconstruction error is the bridge from the autoencoder to anomaly scoring. The project primarily uses mean squared error at the image level while retaining a spatial error map for localization. Applied specifically to **metal_nut**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. The safest pattern is to fail loudly when a required artifact is missing, and to return an explicit unknown state when a decision cannot be supported by a reliable threshold or compatible model. Visual inspection of representative examples remains valuable even when quantitative metrics exist. Images reveal clipping, misalignment, over-smoothing, localization leakage, and other issues that a single aggregate metric can conceal.

### Operational checklist

- Confirm that the metal_nut directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for metal_nut.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to metal_nut and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For metal_nut, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. A portfolio-quality machine-learning repository becomes much stronger when the documentation distinguishes current implementation, intended workflow, experimental ideas, and future work rather than blending them together.

### Research angle

A useful experiment is to compare the baseline behavior for metal_nut under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 15. Reconstruction error: hazelnut

Reconstruction error is the bridge from the autoencoder to anomaly scoring. The project primarily uses mean squared error at the image level while retaining a spatial error map for localization. Applied specifically to **hazelnut**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. Visual inspection of representative examples remains valuable even when quantitative metrics exist. Images reveal clipping, misalignment, over-smoothing, localization leakage, and other issues that a single aggregate metric can conceal. The repository should evolve by controlled experiments: change one important assumption, record the configuration, rerun the relevant evaluation, compare the result, and preserve the evidence.

### Operational checklist

- Confirm that the hazelnut directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for hazelnut.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to hazelnut and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For hazelnut, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. Category-specific behavior deserves its own analysis because five classes can have very different texture, geometry, background, and defect distributions even when the same model family is used.

### Research angle

A useful experiment is to compare the baseline behavior for hazelnut under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 16. Threshold calibration: screw

A threshold is an operating point, not an intrinsic truth about an image. It should be derived from clearly identified normal data and evaluated against a separate set of normal and anomalous examples. Applied specifically to **screw**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. The repository should evolve by controlled experiments: change one important assumption, record the configuration, rerun the relevant evaluation, compare the result, and preserve the evidence. A portfolio-quality machine-learning repository becomes much stronger when the documentation distinguishes current implementation, intended workflow, experimental ideas, and future work rather than blending them together.

### Operational checklist

- Confirm that the screw directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for screw.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to screw and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For screw, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. Thresholds and localization masks should be versioned conceptually even when the repository does not yet have a formal experiment tracker. A threshold file without provenance is much less useful than a threshold tied to a documented calibration protocol.

### Research angle

A useful experiment is to compare the baseline behavior for screw under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 17. Threshold calibration: bottle

A threshold is an operating point, not an intrinsic truth about an image. It should be derived from clearly identified normal data and evaluated against a separate set of normal and anomalous examples. Applied specifically to **bottle**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. A portfolio-quality machine-learning repository becomes much stronger when the documentation distinguishes current implementation, intended workflow, experimental ideas, and future work rather than blending them together. Category-specific behavior deserves its own analysis because five classes can have very different texture, geometry, background, and defect distributions even when the same model family is used.

### Operational checklist

- Confirm that the bottle directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for bottle.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to bottle and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For bottle, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. Deployment should never be allowed to silently fall back from a missing trained artifact to an arbitrary model. The current scripts deliberately search for compatible category models and validate the input/output contract before inference.

### Research angle

A useful experiment is to compare the baseline behavior for bottle under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 18. Threshold calibration: capsule

A threshold is an operating point, not an intrinsic truth about an image. It should be derived from clearly identified normal data and evaluated against a separate set of normal and anomalous examples. Applied specifically to **capsule**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. Category-specific behavior deserves its own analysis because five classes can have very different texture, geometry, background, and defect distributions even when the same model family is used. Thresholds and localization masks should be versioned conceptually even when the repository does not yet have a formal experiment tracker. A threshold file without provenance is much less useful than a threshold tied to a documented calibration protocol.

### Operational checklist

- Confirm that the capsule directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for capsule.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to capsule and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For capsule, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. The practical question is not whether this stage sounds sophisticated; it is whether its inputs, outputs, assumptions, and failure modes are visible enough that another engineer can reproduce the reasoning.

### Research angle

A useful experiment is to compare the baseline behavior for capsule under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 19. Threshold calibration: metal_nut

A threshold is an operating point, not an intrinsic truth about an image. It should be derived from clearly identified normal data and evaluated against a separate set of normal and anomalous examples. Applied specifically to **metal_nut**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. Thresholds and localization masks should be versioned conceptually even when the repository does not yet have a formal experiment tracker. A threshold file without provenance is much less useful than a threshold tied to a documented calibration protocol. Deployment should never be allowed to silently fall back from a missing trained artifact to an arbitrary model. The current scripts deliberately search for compatible category models and validate the input/output contract before inference.

### Operational checklist

- Confirm that the metal_nut directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for metal_nut.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to metal_nut and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For metal_nut, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. For industrial inspection, small implementation details can change the decision boundary. Resize filters, normalization ranges, category routing, threshold provenance, and artifact selection should therefore be documented alongside the high-level method.

### Research angle

A useful experiment is to compare the baseline behavior for metal_nut under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 20. Threshold calibration: hazelnut

A threshold is an operating point, not an intrinsic truth about an image. It should be derived from clearly identified normal data and evaluated against a separate set of normal and anomalous examples. Applied specifically to **hazelnut**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. Deployment should never be allowed to silently fall back from a missing trained artifact to an arbitrary model. The current scripts deliberately search for compatible category models and validate the input/output contract before inference. The practical question is not whether this stage sounds sophisticated; it is whether its inputs, outputs, assumptions, and failure modes are visible enough that another engineer can reproduce the reasoning.

### Operational checklist

- Confirm that the hazelnut directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for hazelnut.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to hazelnut and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For hazelnut, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. A robust workflow treats every generated score as conditional on the pipeline that produced it. Change the image size, preprocessing, checkpoint, calibration set, or metric definition and the meaning of the resulting number can change.

### Research angle

A useful experiment is to compare the baseline behavior for hazelnut under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 21. Localization: screw

Localization preserves spatial information that disappears when a full image is reduced to one scalar score. The error map can be thresholded and cleaned to obtain candidate defect regions. Applied specifically to **screw**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. The practical question is not whether this stage sounds sophisticated; it is whether its inputs, outputs, assumptions, and failure modes are visible enough that another engineer can reproduce the reasoning. For industrial inspection, small implementation details can change the decision boundary. Resize filters, normalization ranges, category routing, threshold provenance, and artifact selection should therefore be documented alongside the high-level method.

### Operational checklist

- Confirm that the screw directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for screw.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to screw and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For screw, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. The safest pattern is to fail loudly when a required artifact is missing, and to return an explicit unknown state when a decision cannot be supported by a reliable threshold or compatible model.

### Research angle

A useful experiment is to compare the baseline behavior for screw under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 22. Localization: bottle

Localization preserves spatial information that disappears when a full image is reduced to one scalar score. The error map can be thresholded and cleaned to obtain candidate defect regions. Applied specifically to **bottle**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. For industrial inspection, small implementation details can change the decision boundary. Resize filters, normalization ranges, category routing, threshold provenance, and artifact selection should therefore be documented alongside the high-level method. A robust workflow treats every generated score as conditional on the pipeline that produced it. Change the image size, preprocessing, checkpoint, calibration set, or metric definition and the meaning of the resulting number can change.

### Operational checklist

- Confirm that the bottle directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for bottle.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to bottle and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For bottle, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. Visual inspection of representative examples remains valuable even when quantitative metrics exist. Images reveal clipping, misalignment, over-smoothing, localization leakage, and other issues that a single aggregate metric can conceal.

### Research angle

A useful experiment is to compare the baseline behavior for bottle under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 23. Localization: capsule

Localization preserves spatial information that disappears when a full image is reduced to one scalar score. The error map can be thresholded and cleaned to obtain candidate defect regions. Applied specifically to **capsule**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. A robust workflow treats every generated score as conditional on the pipeline that produced it. Change the image size, preprocessing, checkpoint, calibration set, or metric definition and the meaning of the resulting number can change. The safest pattern is to fail loudly when a required artifact is missing, and to return an explicit unknown state when a decision cannot be supported by a reliable threshold or compatible model.

### Operational checklist

- Confirm that the capsule directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for capsule.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to capsule and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For capsule, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. The repository should evolve by controlled experiments: change one important assumption, record the configuration, rerun the relevant evaluation, compare the result, and preserve the evidence.

### Research angle

A useful experiment is to compare the baseline behavior for capsule under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 24. Localization: metal_nut

Localization preserves spatial information that disappears when a full image is reduced to one scalar score. The error map can be thresholded and cleaned to obtain candidate defect regions. Applied specifically to **metal_nut**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. The safest pattern is to fail loudly when a required artifact is missing, and to return an explicit unknown state when a decision cannot be supported by a reliable threshold or compatible model. Visual inspection of representative examples remains valuable even when quantitative metrics exist. Images reveal clipping, misalignment, over-smoothing, localization leakage, and other issues that a single aggregate metric can conceal.

### Operational checklist

- Confirm that the metal_nut directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for metal_nut.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to metal_nut and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For metal_nut, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. A portfolio-quality machine-learning repository becomes much stronger when the documentation distinguishes current implementation, intended workflow, experimental ideas, and future work rather than blending them together.

### Research angle

A useful experiment is to compare the baseline behavior for metal_nut under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 25. Localization: hazelnut

Localization preserves spatial information that disappears when a full image is reduced to one scalar score. The error map can be thresholded and cleaned to obtain candidate defect regions. Applied specifically to **hazelnut**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. Visual inspection of representative examples remains valuable even when quantitative metrics exist. Images reveal clipping, misalignment, over-smoothing, localization leakage, and other issues that a single aggregate metric can conceal. The repository should evolve by controlled experiments: change one important assumption, record the configuration, rerun the relevant evaluation, compare the result, and preserve the evidence.

### Operational checklist

- Confirm that the hazelnut directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for hazelnut.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to hazelnut and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For hazelnut, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. Category-specific behavior deserves its own analysis because five classes can have very different texture, geometry, background, and defect distributions even when the same model family is used.

### Research angle

A useful experiment is to compare the baseline behavior for hazelnut under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 26. Evaluation: screw

Evaluation converts a visually convincing prototype into an evidence-backed experiment. Image-level and pixel-level metrics answer different questions and should not be conflated. Applied specifically to **screw**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. The repository should evolve by controlled experiments: change one important assumption, record the configuration, rerun the relevant evaluation, compare the result, and preserve the evidence. A portfolio-quality machine-learning repository becomes much stronger when the documentation distinguishes current implementation, intended workflow, experimental ideas, and future work rather than blending them together.

### Operational checklist

- Confirm that the screw directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for screw.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to screw and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For screw, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. Thresholds and localization masks should be versioned conceptually even when the repository does not yet have a formal experiment tracker. A threshold file without provenance is much less useful than a threshold tied to a documented calibration protocol.

### Research angle

A useful experiment is to compare the baseline behavior for screw under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 27. Evaluation: bottle

Evaluation converts a visually convincing prototype into an evidence-backed experiment. Image-level and pixel-level metrics answer different questions and should not be conflated. Applied specifically to **bottle**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. A portfolio-quality machine-learning repository becomes much stronger when the documentation distinguishes current implementation, intended workflow, experimental ideas, and future work rather than blending them together. Category-specific behavior deserves its own analysis because five classes can have very different texture, geometry, background, and defect distributions even when the same model family is used.

### Operational checklist

- Confirm that the bottle directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for bottle.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to bottle and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For bottle, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. Deployment should never be allowed to silently fall back from a missing trained artifact to an arbitrary model. The current scripts deliberately search for compatible category models and validate the input/output contract before inference.

### Research angle

A useful experiment is to compare the baseline behavior for bottle under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 28. Evaluation: capsule

Evaluation converts a visually convincing prototype into an evidence-backed experiment. Image-level and pixel-level metrics answer different questions and should not be conflated. Applied specifically to **capsule**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. Category-specific behavior deserves its own analysis because five classes can have very different texture, geometry, background, and defect distributions even when the same model family is used. Thresholds and localization masks should be versioned conceptually even when the repository does not yet have a formal experiment tracker. A threshold file without provenance is much less useful than a threshold tied to a documented calibration protocol.

### Operational checklist

- Confirm that the capsule directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for capsule.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to capsule and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For capsule, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. The practical question is not whether this stage sounds sophisticated; it is whether its inputs, outputs, assumptions, and failure modes are visible enough that another engineer can reproduce the reasoning.

### Research angle

A useful experiment is to compare the baseline behavior for capsule under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 29. Evaluation: metal_nut

Evaluation converts a visually convincing prototype into an evidence-backed experiment. Image-level and pixel-level metrics answer different questions and should not be conflated. Applied specifically to **metal_nut**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. Thresholds and localization masks should be versioned conceptually even when the repository does not yet have a formal experiment tracker. A threshold file without provenance is much less useful than a threshold tied to a documented calibration protocol. Deployment should never be allowed to silently fall back from a missing trained artifact to an arbitrary model. The current scripts deliberately search for compatible category models and validate the input/output contract before inference.

### Operational checklist

- Confirm that the metal_nut directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for metal_nut.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to metal_nut and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For metal_nut, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. For industrial inspection, small implementation details can change the decision boundary. Resize filters, normalization ranges, category routing, threshold provenance, and artifact selection should therefore be documented alongside the high-level method.

### Research angle

A useful experiment is to compare the baseline behavior for metal_nut under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 30. Evaluation: hazelnut

Evaluation converts a visually convincing prototype into an evidence-backed experiment. Image-level and pixel-level metrics answer different questions and should not be conflated. Applied specifically to **hazelnut**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. Deployment should never be allowed to silently fall back from a missing trained artifact to an arbitrary model. The current scripts deliberately search for compatible category models and validate the input/output contract before inference. The practical question is not whether this stage sounds sophisticated; it is whether its inputs, outputs, assumptions, and failure modes are visible enough that another engineer can reproduce the reasoning.

### Operational checklist

- Confirm that the hazelnut directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for hazelnut.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to hazelnut and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For hazelnut, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. A robust workflow treats every generated score as conditional on the pipeline that produced it. Change the image size, preprocessing, checkpoint, calibration set, or metric definition and the meaning of the resulting number can change.

### Research angle

A useful experiment is to compare the baseline behavior for hazelnut under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 31. Deployment: screw

Deployment adds constraints that notebooks can ignore: startup time, artifact discovery, missing files, model compatibility, malformed inputs, user feedback, and repeatable output handling. Applied specifically to **screw**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. The practical question is not whether this stage sounds sophisticated; it is whether its inputs, outputs, assumptions, and failure modes are visible enough that another engineer can reproduce the reasoning. For industrial inspection, small implementation details can change the decision boundary. Resize filters, normalization ranges, category routing, threshold provenance, and artifact selection should therefore be documented alongside the high-level method.

### Operational checklist

- Confirm that the screw directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for screw.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to screw and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For screw, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. The safest pattern is to fail loudly when a required artifact is missing, and to return an explicit unknown state when a decision cannot be supported by a reliable threshold or compatible model.

### Research angle

A useful experiment is to compare the baseline behavior for screw under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 32. Deployment: bottle

Deployment adds constraints that notebooks can ignore: startup time, artifact discovery, missing files, model compatibility, malformed inputs, user feedback, and repeatable output handling. Applied specifically to **bottle**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. For industrial inspection, small implementation details can change the decision boundary. Resize filters, normalization ranges, category routing, threshold provenance, and artifact selection should therefore be documented alongside the high-level method. A robust workflow treats every generated score as conditional on the pipeline that produced it. Change the image size, preprocessing, checkpoint, calibration set, or metric definition and the meaning of the resulting number can change.

### Operational checklist

- Confirm that the bottle directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for bottle.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to bottle and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For bottle, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. Visual inspection of representative examples remains valuable even when quantitative metrics exist. Images reveal clipping, misalignment, over-smoothing, localization leakage, and other issues that a single aggregate metric can conceal.

### Research angle

A useful experiment is to compare the baseline behavior for bottle under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 33. Deployment: capsule

Deployment adds constraints that notebooks can ignore: startup time, artifact discovery, missing files, model compatibility, malformed inputs, user feedback, and repeatable output handling. Applied specifically to **capsule**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. A robust workflow treats every generated score as conditional on the pipeline that produced it. Change the image size, preprocessing, checkpoint, calibration set, or metric definition and the meaning of the resulting number can change. The safest pattern is to fail loudly when a required artifact is missing, and to return an explicit unknown state when a decision cannot be supported by a reliable threshold or compatible model.

### Operational checklist

- Confirm that the capsule directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for capsule.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to capsule and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For capsule, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. The repository should evolve by controlled experiments: change one important assumption, record the configuration, rerun the relevant evaluation, compare the result, and preserve the evidence.

### Research angle

A useful experiment is to compare the baseline behavior for capsule under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 34. Deployment: metal_nut

Deployment adds constraints that notebooks can ignore: startup time, artifact discovery, missing files, model compatibility, malformed inputs, user feedback, and repeatable output handling. Applied specifically to **metal_nut**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. The safest pattern is to fail loudly when a required artifact is missing, and to return an explicit unknown state when a decision cannot be supported by a reliable threshold or compatible model. Visual inspection of representative examples remains valuable even when quantitative metrics exist. Images reveal clipping, misalignment, over-smoothing, localization leakage, and other issues that a single aggregate metric can conceal.

### Operational checklist

- Confirm that the metal_nut directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for metal_nut.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to metal_nut and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For metal_nut, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. A portfolio-quality machine-learning repository becomes much stronger when the documentation distinguishes current implementation, intended workflow, experimental ideas, and future work rather than blending them together.

### Research angle

A useful experiment is to compare the baseline behavior for metal_nut under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 35. Deployment: hazelnut

Deployment adds constraints that notebooks can ignore: startup time, artifact discovery, missing files, model compatibility, malformed inputs, user feedback, and repeatable output handling. Applied specifically to **hazelnut**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. Visual inspection of representative examples remains valuable even when quantitative metrics exist. Images reveal clipping, misalignment, over-smoothing, localization leakage, and other issues that a single aggregate metric can conceal. The repository should evolve by controlled experiments: change one important assumption, record the configuration, rerun the relevant evaluation, compare the result, and preserve the evidence.

### Operational checklist

- Confirm that the hazelnut directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for hazelnut.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to hazelnut and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For hazelnut, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. Category-specific behavior deserves its own analysis because five classes can have very different texture, geometry, background, and defect distributions even when the same model family is used.

### Research angle

A useful experiment is to compare the baseline behavior for hazelnut under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 36. Reproducibility: screw

Reproducibility is the ability to recreate an experiment with the same assumptions. Seeds help, but the full contract also includes package versions, data version, split logic, model files, thresholds, and evaluation definitions. Applied specifically to **screw**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. The repository should evolve by controlled experiments: change one important assumption, record the configuration, rerun the relevant evaluation, compare the result, and preserve the evidence. A portfolio-quality machine-learning repository becomes much stronger when the documentation distinguishes current implementation, intended workflow, experimental ideas, and future work rather than blending them together.

### Operational checklist

- Confirm that the screw directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for screw.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to screw and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For screw, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. Thresholds and localization masks should be versioned conceptually even when the repository does not yet have a formal experiment tracker. A threshold file without provenance is much less useful than a threshold tied to a documented calibration protocol.

### Research angle

A useful experiment is to compare the baseline behavior for screw under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 37. Reproducibility: bottle

Reproducibility is the ability to recreate an experiment with the same assumptions. Seeds help, but the full contract also includes package versions, data version, split logic, model files, thresholds, and evaluation definitions. Applied specifically to **bottle**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. A portfolio-quality machine-learning repository becomes much stronger when the documentation distinguishes current implementation, intended workflow, experimental ideas, and future work rather than blending them together. Category-specific behavior deserves its own analysis because five classes can have very different texture, geometry, background, and defect distributions even when the same model family is used.

### Operational checklist

- Confirm that the bottle directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for bottle.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to bottle and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For bottle, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. Deployment should never be allowed to silently fall back from a missing trained artifact to an arbitrary model. The current scripts deliberately search for compatible category models and validate the input/output contract before inference.

### Research angle

A useful experiment is to compare the baseline behavior for bottle under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 38. Reproducibility: capsule

Reproducibility is the ability to recreate an experiment with the same assumptions. Seeds help, but the full contract also includes package versions, data version, split logic, model files, thresholds, and evaluation definitions. Applied specifically to **capsule**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. Category-specific behavior deserves its own analysis because five classes can have very different texture, geometry, background, and defect distributions even when the same model family is used. Thresholds and localization masks should be versioned conceptually even when the repository does not yet have a formal experiment tracker. A threshold file without provenance is much less useful than a threshold tied to a documented calibration protocol.

### Operational checklist

- Confirm that the capsule directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for capsule.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to capsule and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For capsule, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. The practical question is not whether this stage sounds sophisticated; it is whether its inputs, outputs, assumptions, and failure modes are visible enough that another engineer can reproduce the reasoning.

### Research angle

A useful experiment is to compare the baseline behavior for capsule under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 39. Reproducibility: metal_nut

Reproducibility is the ability to recreate an experiment with the same assumptions. Seeds help, but the full contract also includes package versions, data version, split logic, model files, thresholds, and evaluation definitions. Applied specifically to **metal_nut**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. Thresholds and localization masks should be versioned conceptually even when the repository does not yet have a formal experiment tracker. A threshold file without provenance is much less useful than a threshold tied to a documented calibration protocol. Deployment should never be allowed to silently fall back from a missing trained artifact to an arbitrary model. The current scripts deliberately search for compatible category models and validate the input/output contract before inference.

### Operational checklist

- Confirm that the metal_nut directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for metal_nut.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to metal_nut and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For metal_nut, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. For industrial inspection, small implementation details can change the decision boundary. Resize filters, normalization ranges, category routing, threshold provenance, and artifact selection should therefore be documented alongside the high-level method.

### Research angle

A useful experiment is to compare the baseline behavior for metal_nut under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 40. Reproducibility: hazelnut

Reproducibility is the ability to recreate an experiment with the same assumptions. Seeds help, but the full contract also includes package versions, data version, split logic, model files, thresholds, and evaluation definitions. Applied specifically to **hazelnut**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. Deployment should never be allowed to silently fall back from a missing trained artifact to an arbitrary model. The current scripts deliberately search for compatible category models and validate the input/output contract before inference. The practical question is not whether this stage sounds sophisticated; it is whether its inputs, outputs, assumptions, and failure modes are visible enough that another engineer can reproduce the reasoning.

### Operational checklist

- Confirm that the hazelnut directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for hazelnut.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to hazelnut and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For hazelnut, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. A robust workflow treats every generated score as conditional on the pipeline that produced it. Change the image size, preprocessing, checkpoint, calibration set, or metric definition and the meaning of the resulting number can change.

### Research angle

A useful experiment is to compare the baseline behavior for hazelnut under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 41. Error analysis: screw

Error analysis is where false positives and false negatives become actionable engineering information. Instead of hiding mistakes, inspect their shared visual and pipeline characteristics. Applied specifically to **screw**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. The practical question is not whether this stage sounds sophisticated; it is whether its inputs, outputs, assumptions, and failure modes are visible enough that another engineer can reproduce the reasoning. For industrial inspection, small implementation details can change the decision boundary. Resize filters, normalization ranges, category routing, threshold provenance, and artifact selection should therefore be documented alongside the high-level method.

### Operational checklist

- Confirm that the screw directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for screw.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to screw and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For screw, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. The safest pattern is to fail loudly when a required artifact is missing, and to return an explicit unknown state when a decision cannot be supported by a reliable threshold or compatible model.

### Research angle

A useful experiment is to compare the baseline behavior for screw under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 42. Error analysis: bottle

Error analysis is where false positives and false negatives become actionable engineering information. Instead of hiding mistakes, inspect their shared visual and pipeline characteristics. Applied specifically to **bottle**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. For industrial inspection, small implementation details can change the decision boundary. Resize filters, normalization ranges, category routing, threshold provenance, and artifact selection should therefore be documented alongside the high-level method. A robust workflow treats every generated score as conditional on the pipeline that produced it. Change the image size, preprocessing, checkpoint, calibration set, or metric definition and the meaning of the resulting number can change.

### Operational checklist

- Confirm that the bottle directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for bottle.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to bottle and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For bottle, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. Visual inspection of representative examples remains valuable even when quantitative metrics exist. Images reveal clipping, misalignment, over-smoothing, localization leakage, and other issues that a single aggregate metric can conceal.

### Research angle

A useful experiment is to compare the baseline behavior for bottle under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 43. Error analysis: capsule

Error analysis is where false positives and false negatives become actionable engineering information. Instead of hiding mistakes, inspect their shared visual and pipeline characteristics. Applied specifically to **capsule**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. A robust workflow treats every generated score as conditional on the pipeline that produced it. Change the image size, preprocessing, checkpoint, calibration set, or metric definition and the meaning of the resulting number can change. The safest pattern is to fail loudly when a required artifact is missing, and to return an explicit unknown state when a decision cannot be supported by a reliable threshold or compatible model.

### Operational checklist

- Confirm that the capsule directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for capsule.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to capsule and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For capsule, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. The repository should evolve by controlled experiments: change one important assumption, record the configuration, rerun the relevant evaluation, compare the result, and preserve the evidence.

### Research angle

A useful experiment is to compare the baseline behavior for capsule under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 44. Error analysis: metal_nut

Error analysis is where false positives and false negatives become actionable engineering information. Instead of hiding mistakes, inspect their shared visual and pipeline characteristics. Applied specifically to **metal_nut**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. The safest pattern is to fail loudly when a required artifact is missing, and to return an explicit unknown state when a decision cannot be supported by a reliable threshold or compatible model. Visual inspection of representative examples remains valuable even when quantitative metrics exist. Images reveal clipping, misalignment, over-smoothing, localization leakage, and other issues that a single aggregate metric can conceal.

### Operational checklist

- Confirm that the metal_nut directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for metal_nut.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to metal_nut and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For metal_nut, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. A portfolio-quality machine-learning repository becomes much stronger when the documentation distinguishes current implementation, intended workflow, experimental ideas, and future work rather than blending them together.

### Research angle

A useful experiment is to compare the baseline behavior for metal_nut under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 45. Error analysis: hazelnut

Error analysis is where false positives and false negatives become actionable engineering information. Instead of hiding mistakes, inspect their shared visual and pipeline characteristics. Applied specifically to **hazelnut**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. Visual inspection of representative examples remains valuable even when quantitative metrics exist. Images reveal clipping, misalignment, over-smoothing, localization leakage, and other issues that a single aggregate metric can conceal. The repository should evolve by controlled experiments: change one important assumption, record the configuration, rerun the relevant evaluation, compare the result, and preserve the evidence.

### Operational checklist

- Confirm that the hazelnut directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for hazelnut.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to hazelnut and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For hazelnut, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. Category-specific behavior deserves its own analysis because five classes can have very different texture, geometry, background, and defect distributions even when the same model family is used.

### Research angle

A useful experiment is to compare the baseline behavior for hazelnut under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 46. Engineering hygiene: screw

Engineering hygiene keeps a research repository understandable. Explicit paths, clear logs, dependency cleanup, safe artifact handling, and honest status reporting matter as much as model code. Applied specifically to **screw**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. The repository should evolve by controlled experiments: change one important assumption, record the configuration, rerun the relevant evaluation, compare the result, and preserve the evidence. A portfolio-quality machine-learning repository becomes much stronger when the documentation distinguishes current implementation, intended workflow, experimental ideas, and future work rather than blending them together.

### Operational checklist

- Confirm that the screw directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for screw.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to screw and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For screw, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. Thresholds and localization masks should be versioned conceptually even when the repository does not yet have a formal experiment tracker. A threshold file without provenance is much less useful than a threshold tied to a documented calibration protocol.

### Research angle

A useful experiment is to compare the baseline behavior for screw under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 47. Engineering hygiene: bottle

Engineering hygiene keeps a research repository understandable. Explicit paths, clear logs, dependency cleanup, safe artifact handling, and honest status reporting matter as much as model code. Applied specifically to **bottle**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. A portfolio-quality machine-learning repository becomes much stronger when the documentation distinguishes current implementation, intended workflow, experimental ideas, and future work rather than blending them together. Category-specific behavior deserves its own analysis because five classes can have very different texture, geometry, background, and defect distributions even when the same model family is used.

### Operational checklist

- Confirm that the bottle directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for bottle.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to bottle and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For bottle, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. Deployment should never be allowed to silently fall back from a missing trained artifact to an arbitrary model. The current scripts deliberately search for compatible category models and validate the input/output contract before inference.

### Research angle

A useful experiment is to compare the baseline behavior for bottle under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 48. Engineering hygiene: capsule

Engineering hygiene keeps a research repository understandable. Explicit paths, clear logs, dependency cleanup, safe artifact handling, and honest status reporting matter as much as model code. Applied specifically to **capsule**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. Category-specific behavior deserves its own analysis because five classes can have very different texture, geometry, background, and defect distributions even when the same model family is used. Thresholds and localization masks should be versioned conceptually even when the repository does not yet have a formal experiment tracker. A threshold file without provenance is much less useful than a threshold tied to a documented calibration protocol.

### Operational checklist

- Confirm that the capsule directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for capsule.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to capsule and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For capsule, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. The practical question is not whether this stage sounds sophisticated; it is whether its inputs, outputs, assumptions, and failure modes are visible enough that another engineer can reproduce the reasoning.

### Research angle

A useful experiment is to compare the baseline behavior for capsule under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 49. Engineering hygiene: metal_nut

Engineering hygiene keeps a research repository understandable. Explicit paths, clear logs, dependency cleanup, safe artifact handling, and honest status reporting matter as much as model code. Applied specifically to **metal_nut**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. Thresholds and localization masks should be versioned conceptually even when the repository does not yet have a formal experiment tracker. A threshold file without provenance is much less useful than a threshold tied to a documented calibration protocol. Deployment should never be allowed to silently fall back from a missing trained artifact to an arbitrary model. The current scripts deliberately search for compatible category models and validate the input/output contract before inference.

### Operational checklist

- Confirm that the metal_nut directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for metal_nut.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to metal_nut and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For metal_nut, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. For industrial inspection, small implementation details can change the decision boundary. Resize filters, normalization ranges, category routing, threshold provenance, and artifact selection should therefore be documented alongside the high-level method.

### Research angle

A useful experiment is to compare the baseline behavior for metal_nut under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.


## 50. Engineering hygiene: hazelnut

Engineering hygiene keeps a research repository understandable. Explicit paths, clear logs, dependency cleanup, safe artifact handling, and honest status reporting matter as much as model code. Applied specifically to **hazelnut**, this means the workflow should preserve the distinction between category-specific normality and generic visual similarity. Deployment should never be allowed to silently fall back from a missing trained artifact to an arbitrary model. The current scripts deliberately search for compatible category models and validate the input/output contract before inference. The practical question is not whether this stage sounds sophisticated; it is whether its inputs, outputs, assumptions, and failure modes are visible enough that another engineer can reproduce the reasoning.

### Operational checklist

- Confirm that the hazelnut directory exists and represents the intended dataset version.
- Confirm that training uses the expected normal-image path for hazelnut.
- Confirm that preprocessing produces RGB tensors with the documented shape.
- Confirm that the selected model belongs to hazelnut and passes the model contract.
- Confirm that thresholds are sourced from a documented calibration procedure.
- Confirm that outputs are written to the intended artifact directory.
- Confirm that evaluation results identify the category, sample count, and metric definition.

### Interpretation notes

For hazelnut, the anomaly signal should be interpreted relative to the normal visual distribution learned by the category-specific autoencoder. A higher reconstruction error is evidence of reconstruction disagreement, not by itself a proof of manufacturing failure. A localized error region is a candidate inspection area, not automatically a verified defect. A robust workflow treats every generated score as conditional on the pipeline that produced it. Change the image size, preprocessing, checkpoint, calibration set, or metric definition and the meaning of the resulting number can change.

### Research angle

A useful experiment is to compare the baseline behavior for hazelnut under controlled changes to one variable at a time: image resolution, calibration percentile, augmentation policy, latent capacity, reconstruction loss, or post-processing threshold. Keep the test protocol fixed so that differences can be attributed to the changed factor rather than to a hidden dataset or evaluation change.



# 🔭 Extended Research Playbook

The strongest way to extend VisionGuard AI is to turn every future improvement into a measurable hypothesis. Examples include: increasing image resolution may improve tiny-defect sensitivity but raise compute cost; a richer latent representation may improve reconstruction of normal geometry but risk reconstructing defects too faithfully; a stricter threshold may reduce false negatives at the expense of false positives; stronger localization post-processing may remove noise but also erase small real defects. Each hypothesis should be tested under a controlled protocol with the same category split, comparable calibration strategy, and explicitly recorded model artifact.

## Suggested experiment record

```text
experiment_id
category
dataset_version
model_version
image_size
batch_size
epochs
learning_rate
augmentation_policy
loss_function
threshold_method
threshold_value
pixel_threshold_method
pixel_threshold_value
evaluation_set
image_count
accuracy
precision
recall
f1
roc_auc
iou
dice
false_positive_examples
false_negative_examples
notes
```

## Review questions

1. Did the data pipeline use only the intended category and split?
2. Was the threshold calibrated without leaking final test labels?
3. Was the model artifact compatible with the inference contract?
4. Were the displayed metrics computed on the same population described in the report?
5. Do qualitative examples support the quantitative conclusions?
6. Are failures documented rather than removed from the narrative?
7. Could another engineer reconstruct the experiment from the repository alone?

## Final engineering principle

The goal is not to make the README look enormous. The goal is to make the project legible. Every important assumption should have a place where it can be inspected, challenged, and improved. That is the difference between a demo and a serious machine-learning repository.


> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.

> The repository should continue to prefer measured results over attractive but unsupported claims.

> This note reinforces the same implementation contract for maintainability, reproducibility, and careful evaluation.

> Keep the category boundary explicit, keep the preprocessing path consistent, and keep the evidence attached to the decision.

> When behavior changes, record the changed assumption rather than relying on memory or an undocumented notebook state.

> A stable artifact pipeline makes debugging faster because each failure can be localized to data, preprocessing, model loading, calibration, inference, localization, or evaluation.