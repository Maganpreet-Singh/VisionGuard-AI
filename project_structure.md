# VisionGuard AI — Project Structure

**Industrial Anomaly Detection & Defect Localization**

VisionGuard AI is a TensorFlow/Keras computer-vision project built around the MVTec AD dataset. The project is organized as a modular pipeline covering dataset preparation, exploration, preprocessing, model training, anomaly detection, defect localization, evaluation, inference, and deployment.

---

## 1. Complete Project Architecture

```text
VisionGuard-AI/
│
├── README.md
├── project_structure.md
├── requirements.txt
├── requirements_and_setup.md
├── .gitignore
│
├── config.py
├── utils.py
├── train.py
├── predict.py
├── app.py
│
├── notebooks/
│   │
│   ├── 01_dataset_download_and_preparation.ipynb
│   ├── 02_dataset_exploration_EDA.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_cnn_autoencoder_training.ipynb
│   ├── 05_anomaly_detection.ipynb
│   ├── 06_defect_localization.ipynb
│   ├── 07_model_evaluation.ipynb
│   └── 08_inference_demo.ipynb
│
├── data/
│   ├── README.md
│   └── .gitkeep
│
├── models/
│   ├── checkpoints/
│   ├── final/
│   └── README.md
│
├── artifacts/
│   ├── figures/
│   ├── heatmaps/
│   ├── predictions/
│   ├── evaluation/
│   └── reports/
│
├── logs/
│   └── training/
│
└── screenshots/
    └── README.md
```

---

# 2. Root Files

## `README.md`

Main project documentation.

Contains:

- Project overview
- Problem statement
- Objectives
- Dataset information
- Selected MVTec AD categories
- Model architecture
- Training workflow
- Evaluation methodology
- Inference instructions
- Deployment instructions
- Results
- Screenshots
- Future improvements
- Technologies used

This is the primary entry point for anyone visiting the GitHub repository.

---

## `project_structure.md`

This file.

It documents the complete repository architecture and explains the purpose of each major directory and file.

---

## `requirements.txt`

Contains Python dependencies required to run the project.

Typical dependencies include:

```text
tensorflow
numpy
pandas
matplotlib
seaborn
Pillow
opencv-python
scikit-learn
kagglehub
psutil
streamlit
```

Only dependencies actually required by the final implementation should be retained.

---

## `requirements_and_setup.md`

Contains environment setup instructions for:

- Google Colab
- Local VS Code
- Python environment
- TensorFlow
- GPU verification
- Dataset preparation
- Project installation
- Running training
- Running inference
- Running Streamlit

---

## `.gitignore`

Prevents large datasets, generated model files, caches, temporary files, and environment files from being committed accidentally.

Example categories to ignore:

```text
__pycache__/
.ipynb_checkpoints/
.venv/
venv/
.env

data/
models/checkpoints/
logs/

*.h5
*.keras
*.pkl
*.ckpt
```

Large generated artifacts should not be committed unless intentionally selected for the repository.

---

# 3. Configuration

## `config.py`

Central configuration file.

Responsible for project-wide settings such as:

```python
DATASET_ROOT
CATEGORIES
IMAGE_SIZE
BATCH_SIZE
EPOCHS
LEARNING_RATE
MODEL_PATH
CHECKPOINT_PATH
RANDOM_SEED
```

The selected MVTec AD categories are:

```python
CATEGORIES = [
    "screw",
    "bottle",
    "capsule",
    "metal_nut",
    "hazelnut"
]
```

The prepared dataset location is:

```text
/content/visionguard_data/
```

Centralizing configuration prevents paths and hyperparameters from being duplicated throughout the project.

---

# 4. Utility Functions

## `utils.py`

Reusable helper functions shared by training, inference, evaluation, and dataset processing.

Possible responsibilities include:

- Image loading
- Image validation
- Image resizing
- Normalization
- Dataset path handling
- Image-file detection
- Image counting
- Ground-truth mask handling
- Visualization
- Prediction utilities
- Heatmap generation
- File utilities

Functions should be reusable instead of duplicating the same logic across notebooks and scripts.

---

# 5. Training

## `train.py`

Standalone TensorFlow/Keras training script.

Responsible for:

1. Loading configuration
2. Loading prepared data
3. Creating the training pipeline
4. Creating the CNN/Convolutional Autoencoder
5. Compiling the model
6. Training the model
7. Running validation
8. Applying callbacks
9. Saving checkpoints
10. Saving the final model
11. Recording training history

Typical callbacks:

```text
EarlyStopping
ReduceLROnPlateau
ModelCheckpoint
```

---

# 6. Prediction

## `predict.py`

Standalone inference script.

Responsible for:

1. Loading the trained model
2. Loading an input image
3. Preprocessing the image
4. Running reconstruction
5. Calculating reconstruction error
6. Comparing the error against the anomaly threshold
7. Generating an anomaly map
8. Producing the prediction
9. Saving or displaying the result

Typical output:

```text
Prediction: ANOMALY
Anomaly Score: <score>
Threshold: <threshold>
```

---

# 7. Streamlit Application

## `app.py`

Web application for interactive inference.

The application should allow a user to:

```text
Upload Image
      ↓
Preprocess Image
      ↓
Load Model
      ↓
Reconstruct Image
      ↓
Calculate Anomaly Score
      ↓
Classify Normal / Anomaly
      ↓
Generate Heatmap
      ↓
Display Defect Region
```

The interface should display:

- Uploaded image
- Prediction
- Anomaly score
- Threshold
- Reconstruction
- Anomaly heatmap
- Defect localization

---

# 8. Notebook Pipeline

The notebooks form the research and experimentation pipeline.

```text
01 Dataset
     ↓
02 EDA
     ↓
03 Preprocessing
     ↓
04 Model Training
     ↓
05 Anomaly Detection
     ↓
06 Defect Localization
     ↓
07 Evaluation
     ↓
08 Inference
```

---

# 9. Dataset Preparation

## `01_dataset_download_and_preparation.ipynb`

Responsible for downloading and preparing MVTec AD.

Dataset source:

```python
import kagglehub

path = kagglehub.dataset_download("ipythonx/mvtec-ad")
```

Only these categories are used:

```text
screw
bottle
capsule
metal_nut
hazelnut
```

Prepared dataset:

```text
/content/visionguard_data/
```

Expected structure:

```text
visionguard_data/
│
├── screw/
│   ├── train/
│   ├── test/
│   └── ground_truth/
│
├── bottle/
│   ├── train/
│   ├── test/
│   └── ground_truth/
│
├── capsule/
│   ├── train/
│   ├── test/
│   └── ground_truth/
│
├── metal_nut/
│   ├── train/
│   ├── test/
│   └── ground_truth/
│
└── hazelnut/
    ├── train/
    ├── test/
    └── ground_truth/
```

The notebook dynamically locates the `mvtec_anomaly_detection` directory instead of assuming a fixed KaggleHub cache location.

---

# 10. Exploratory Data Analysis

## `02_dataset_exploration_EDA.ipynb`

Used to understand the prepared dataset before training.

Includes:

- Dataset statistics
- Category distribution
- Train/test distribution
- Defect distribution
- Image dimensions
- File formats
- Pixel statistics
- Missing-file detection
- Corrupt-file detection
- Visual exploration
- Tables
- Graphs

The EDA notebook should use the actual files on disk rather than hardcoded dataset counts.

---

# 11. Preprocessing

## `03_preprocessing.ipynb`

Responsible for preparing images for TensorFlow.

Pipeline:

```text
Raw Image
    ↓
Image Loading
    ↓
Resize
    ↓
Normalization
    ↓
Augmentation
    ↓
TensorFlow Dataset
    ↓
Batching
    ↓
Caching / Prefetching
    ↓
Model
```

Responsibilities include:

- Image loading
- Image resizing
- Normalization
- Data augmentation
- Mask processing
- Train/validation split
- `tf.data` pipelines
- Batching
- Prefetching
- Caching where appropriate
- Verification visualizations

---

# 12. CNN Autoencoder Training

## `04_cnn_autoencoder_training.ipynb`

Contains the primary deep-learning training workflow.

Main components:

```text
Input Image
     ↓
Encoder
     ↓
Latent Representation
     ↓
Decoder
     ↓
Reconstructed Image
```

Training is performed primarily on normal/good samples so that reconstruction error can be used for anomaly detection.

Includes:

- CNN architecture
- Encoder
- Decoder
- Model compilation
- Training
- Validation
- Early stopping
- Learning-rate scheduling
- Checkpointing
- Reconstruction loss
- Training curves
- Model saving

---

# 13. Anomaly Detection

## `05_anomaly_detection.ipynb`

Uses reconstruction error to distinguish normal and anomalous images.

Pipeline:

```text
Input
  ↓
Trained Autoencoder
  ↓
Reconstruction
  ↓
Pixel Difference
  ↓
Reconstruction Error
  ↓
Threshold
  ↓
Normal / Anomaly
```

Includes:

- Model loading
- Image reconstruction
- Reconstruction error
- Threshold calculation
- Normal/anomaly classification
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC where appropriate

---

# 14. Defect Localization

## `06_defect_localization.ipynb`

Responsible for identifying where an anomaly occurs.

Pipeline:

```text
Original Image
      ↓
Reconstructed Image
      ↓
Absolute Difference
      ↓
Anomaly Map
      ↓
Thresholding
      ↓
Morphological Processing
      ↓
Defect Region
      ↓
Heatmap / Bounding Region
```

Includes:

- Reconstruction-difference maps
- Pixel-level anomaly maps
- Ground-truth comparison
- Thresholding
- Morphological operations
- Bounding regions
- Heatmaps
- Defect localization visualizations

---

# 15. Model Evaluation

## `07_model_evaluation.ipynb`

Provides detailed evaluation of the trained system.

### Image-level metrics

```text
Accuracy
Precision
Recall
F1-score
ROC-AUC
```

### Pixel-level metrics

```text
IoU
Dice Score
```

where appropriate.

Also includes:

- Confusion matrix
- Error analysis
- Normal/anomaly examples
- Localization analysis
- Category-level evaluation

---

# 16. Inference Demo

## `08_inference_demo.ipynb`

Demonstrates how the final trained system works on test images.

Workflow:

```text
Select Test Image
       ↓
Load Model
       ↓
Preprocess
       ↓
Prediction
       ↓
Anomaly Score
       ↓
Heatmap
       ↓
Defect Localization
       ↓
Ground Truth Comparison
```

The notebook should clearly display:

- Input image
- Reconstruction
- Prediction
- Anomaly score
- Heatmap
- Predicted defect region
- Ground-truth mask when available

---

# 17. Data Directory

## `data/`

Stores project-specific prepared data when local storage is used.

Recommended structure:

```text
data/
└── README.md
```

The complete MVTec AD dataset should generally not be committed to GitHub because of its size.

For Google Colab, the working prepared dataset is:

```text
/content/visionguard_data/
```

---

# 18. Model Directory

## `models/`

Stores trained model artifacts.

Structure:

```text
models/
│
├── checkpoints/
│   └── ...
│
├── final/
│   └── ...
│
└── README.md
```

### `models/checkpoints/`

Contains intermediate model checkpoints generated during training.

### `models/final/`

Contains the final selected trained model.

Possible formats include:

```text
.keras
.h5
```

depending on the final implementation.

Large model files should be handled carefully when pushing to GitHub.

---

# 19. Artifacts

## `artifacts/`

Contains generated outputs from experiments and inference.

```text
artifacts/
│
├── figures/
├── heatmaps/
├── predictions/
├── evaluation/
└── reports/
```

### `figures/`

Stores:

- Training curves
- Dataset graphs
- EDA visualizations
- Reconstruction comparisons

### `heatmaps/`

Stores generated anomaly heatmaps and localization visualizations.

### `predictions/`

Stores inference results.

### `evaluation/`

Stores:

- Confusion matrices
- Metric outputs
- Evaluation tables
- Error-analysis outputs

### `reports/`

Stores generated experiment or evaluation reports.

---

# 20. Logs

## `logs/`

Stores training and experiment logs.

```text
logs/
└── training/
```

Possible contents:

```text
training_history.csv
TensorBoard logs
experiment metadata
```

---

# 21. Screenshots

## `screenshots/`

Contains screenshots used in the GitHub README.

Possible screenshots:

```text
screenshots/
│
├── dataset_samples.png
├── training_curves.png
├── anomaly_detection.png
├── defect_heatmap.png
├── streamlit_app.png
└── README.md
```

Only screenshots that actually exist should be referenced by the final README.

---

# 22. Selected Dataset Categories

VisionGuard AI uses only five MVTec AD categories:

```text
1. screw
2. bottle
3. capsule
4. metal_nut
5. hazelnut
```

The project must not intentionally copy or process unrelated MVTec categories during dataset preparation.

---

# 23. End-to-End Workflow

The complete project follows this workflow:

```text
                MVTec AD
                   │
                   ▼
       Dataset Download & Preparation
                   │
                   ▼
                  EDA
                   │
                   ▼
             Preprocessing
                   │
                   ▼
          TensorFlow Dataset
                   │
                   ▼
        CNN Convolutional
             Autoencoder
                   │
                   ▼
              Training
                   │
                   ▼
          Reconstruction Error
                   │
                   ▼
        Anomaly Classification
                   │
                   ▼
       Difference / Anomaly Map
                   │
                   ▼
        Defect Localization
                   │
                   ▼
             Evaluation
                   │
                   ▼
               Inference
                   │
                   ▼
          Streamlit Deployment
```

---

# 24. Recommended Development Order

The project should be developed in this order:

```text
01_dataset_download_and_preparation.ipynb
                    ↓
02_dataset_exploration_EDA.ipynb
                    ↓
03_preprocessing.ipynb
                    ↓
04_cnn_autoencoder_training.ipynb
                    ↓
05_anomaly_detection.ipynb
                    ↓
06_defect_localization.ipynb
                    ↓
07_model_evaluation.ipynb
                    ↓
08_inference_demo.ipynb
                    ↓
config.py + utils.py
                    ↓
train.py + predict.py
                    ↓
app.py
                    ↓
README.md
```

This order keeps data preparation and validation ahead of model development and deployment.

---

# 25. GitHub Repository View

The final GitHub repository should present the project approximately as:

```text
VisionGuard-AI
│
├── notebooks/
│   ├── 01_dataset_download_and_preparation.ipynb
│   ├── 02_dataset_exploration_EDA.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_cnn_autoencoder_training.ipynb
│   ├── 05_anomaly_detection.ipynb
│   ├── 06_defect_localization.ipynb
│   ├── 07_model_evaluation.ipynb
│   └── 08_inference_demo.ipynb
│
├── models/
├── artifacts/
├── data/
├── logs/
├── screenshots/
│
├── app.py
├── train.py
├── predict.py
├── config.py
├── utils.py
│
├── requirements.txt
├── requirements_and_setup.md
├── project_structure.md
└── README.md
```

---

# 26. Important Repository Rules

### Dataset

Do not commit the complete MVTec AD dataset to GitHub.

### Model files

Avoid committing unnecessarily large trained models. Use Git LFS or an appropriate model-hosting solution when required.

### Generated artifacts

Commit only useful, reasonably sized artifacts that improve the portfolio value of the repository.

### Credentials

Never commit:

```text
API keys
tokens
passwords
.env files
private credentials
```

### Reproducibility

Keep configuration, preprocessing, training, and inference logic reproducible.

### Paths

Avoid machine-specific paths such as:

```text
C:\Users\...
/home/user/...
/root/.cache/...
```

The Colab prepared dataset path is intentionally:

```text
/content/visionguard_data
```

---

# 27. Project Design Principle

VisionGuard AI follows a modular architecture:

```text
DATA
  ↓
PREPROCESSING
  ↓
MODEL
  ↓
DETECTION
  ↓
LOCALIZATION
  ↓
EVALUATION
  ↓
INFERENCE
  ↓
DEPLOYMENT
```

Each stage should have a clear responsibility, making the project easier to debug, reproduce, extend, and present as an academic and portfolio computer-vision project.