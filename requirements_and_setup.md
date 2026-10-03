# 🛡️ VisionGuard AI — Requirements & Setup

## Industrial Anomaly Detection & Defect Localization

This guide explains how to set up VisionGuard AI in:

- Google Colab
- Local VS Code
- A Python virtual environment
- TensorFlow/Keras
- GPU-enabled environments
- Streamlit deployment

VisionGuard AI uses TensorFlow/Keras with the MVTec AD dataset and is organized as a modular pipeline from dataset preparation through training, anomaly detection, defect localization, evaluation, inference, and deployment. fileciteturn4file0L1-L18

---

# 1. Project Requirements

## Recommended software

```text
Python
TensorFlow / Keras
Jupyter
VS Code
Git
GitHub
```

The repository also contains:

```text
requirements.txt
config.py
utils.py
train.py
predict.py
app.py
```

The project configuration is intended to centralize dataset paths, image size, batch size, epochs, learning rate, model paths, categories, and random seed. fileciteturn4file1L49-L75

---

# 2. Python Environment

## Check Python

Open a terminal in VS Code:

```bash
python --version
```

Also check:

```bash
python -m pip --version
```

Recommended practice:

- Use a dedicated virtual environment.
- Do not install project dependencies directly into the global Python environment.
- Keep the environment isolated from unrelated projects.

---

# 3. Local VS Code Setup

## Step 1 — Clone the repository

```bash
git clone https://github.com/Maganpreet-Singh/VisionGuard-AI.git
cd VisionGuard-AI
```

## Step 2 — Create a virtual environment

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

If PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

After activation, your terminal should show something similar to:

```text
(.venv)
```

---

# 4. Upgrade pip

Run:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Verify:

```bash
python -m pip --version
```

---

# 5. Install Project Dependencies

The project dependency set includes packages for TensorFlow, numerical computation, data analysis, image processing, visualization, dataset downloading, system utilities, and Streamlit. fileciteturn1file3L1-L24

Install everything from:

```bash
python -m pip install -r requirements.txt
```

If you are setting up only the core ML environment manually:

```bash
python -m pip install tensorflow numpy pandas scikit-learn Pillow opencv-python matplotlib seaborn kagglehub psutil streamlit
```

Do not install the complete dependency list repeatedly if the environment is already configured.

---

# 6. Verify Core Imports

Run:

```bash
python -c "import tensorflow as tf; import numpy; import pandas; import cv2; import PIL; import sklearn; print('Core imports OK'); print('TensorFlow:', tf.__version__)"
```

Expected final message:

```text
Core imports OK
TensorFlow: <installed-version>
```

If an import fails, install the missing package inside the active `.venv`.

---

# 7. TensorFlow Verification

Run:

```python
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
```

The project documentation specifically uses `tf.config.list_physical_devices("GPU")` for GPU verification and does not require GPU availability for dataset preparation. fileciteturn4file2L75-L91

A result such as:

```text
GPUs: []
```

means TensorFlow is currently not seeing a GPU.

That does **not** automatically mean the installation is broken. CPU execution can still be used for preprocessing, EDA, and other non-training tasks.

---

# 8. Google Colab Setup

VisionGuard AI is designed to run directly in Google Colab.

The intended workflow does not require:

- Google Drive
- Manual dataset uploads
- Manual dataset extraction

The dataset-preparation notebook downloads the dataset through KaggleHub and dynamically locates the MVTec root. fileciteturn4file2L75-L91

## Open Google Colab

Open the repository notebook in Colab or upload the required notebook.

For training, enable a GPU runtime:

```text
Runtime
→ Change runtime type
→ Hardware accelerator
→ GPU
```

Then verify:

```python
import tensorflow as tf

print(tf.config.list_physical_devices("GPU"))
```

---

# 9. Google Colab Dependencies

For the dataset/preprocessing/training workflow, the project uses packages such as:

```text
kagglehub
numpy
pandas
matplotlib
Pillow
opencv-python
scikit-learn
tensorflow
seaborn
psutil
```

Not every notebook needs every package. The project specification explicitly recommends installing only what the selected workflow actually needs. fileciteturn4file3L1-L34

A general Colab installation cell can be:

```python
%pip install -q -U kagglehub numpy pandas matplotlib Pillow opencv-python scikit-learn seaborn psutil tensorflow
```

For Streamlit-related work:

```python
%pip install -q -U streamlit
```

After changing major TensorFlow dependencies, restart the Colab runtime if Python reports that a restart is required.

---

# 10. Dataset

VisionGuard AI uses:

```python
import kagglehub

path = kagglehub.dataset_download("ipythonx/mvtec-ad")
```

Only these five MVTec AD categories are used:

```python
CATEGORIES = [
    "screw",
    "bottle",
    "capsule",
    "metal_nut",
    "hazelnut"
]
```

The prepared dataset is:

```text
/content/visionguard_data/
```

Expected structure:

```text
visionguard_data/
├── screw/
│   ├── train/
│   ├── test/
│   └── ground_truth/
├── bottle/
│   ├── train/
│   ├── test/
│   └── ground_truth/
├── capsule/
│   ├── train/
│   ├── test/
│   └── ground_truth/
├── metal_nut/
│   ├── train/
│   ├── test/
│   └── ground_truth/
└── hazelnut/
    ├── train/
    ├── test/
    └── ground_truth/
```

The project specification requires the notebook to dynamically search for `mvtec_anomaly_detection` rather than assuming a particular KaggleHub cache location. fileciteturn4file5L1-L38

---

# 11. Dataset Preparation

Run:

```text
01_dataset_download_and_preparation.ipynb
```

The preparation workflow should:

1. Download MVTec AD through KaggleHub.
2. Locate `mvtec_anomaly_detection`.
3. Check the five requested categories.
4. Prepare `/content/visionguard_data`.
5. Copy only the requested categories.
6. Validate `train`, `test`, and `ground_truth`.
7. Count images.
8. Generate dataset statistics.
9. Visualize training samples.
10. Visualize normal and defective samples.
11. Display ground-truth masks.
12. Perform final verification.

The preparation must be idempotent and must not create nested duplicates such as:

```text
/content/visionguard_data/screw/screw/
```

The original KaggleHub dataset should not be modified. fileciteturn4file5L1-L38

---

# 12. Project Execution Order

Run the notebooks in this order:

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
```

This order keeps dataset preparation and validation ahead of model development and deployment. fileciteturn4file9L1-L23

---

# 13. Configuration

Project-wide configuration belongs in:

```text
config.py
```

Typical configuration values include:

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

The selected categories are:

```python
CATEGORIES = [
    "screw",
    "bottle",
    "capsule",
    "metal_nut",
    "hazelnut"
]
```

The prepared Colab dataset path is:

```python
DATASET_ROOT = "/content/visionguard_data"
```

fileciteturn4file1L49-L75

---

# 14. Preprocessing

Run:

```text
03_preprocessing.ipynb
```

The preprocessing stage handles:

- Image loading
- Image resizing
- Normalization
- Data augmentation
- Mask processing
- Train/validation split
- TensorFlow datasets
- Batching
- Prefetching
- Caching where appropriate
- Verification visualizations

The intended pipeline is:

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

fileciteturn4file1L80-L108

---

# 15. Autoencoder Training

Run:

```text
04_cnn_autoencoder_training.ipynb
```

The training workflow uses a CNN/Convolutional Autoencoder:

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

Training is primarily performed on normal/good samples so reconstruction error can be used for anomaly detection.

The training stage includes:

- Model creation
- Compilation
- Training
- Validation
- Early stopping
- Learning-rate scheduling
- Checkpointing
- Reconstruction loss monitoring
- Training curves
- Model saving

fileciteturn4file1L109-L143

---

# 16. Anomaly Detection

Run:

```text
05_anomaly_detection.ipynb
```

The workflow is:

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

The notebook should calculate:

- Reconstruction error
- Anomaly threshold
- Normal/anomaly classification
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC where appropriate

fileciteturn4file1L145-L174

---

# 17. Defect Localization

Run:

```text
06_defect_localization.ipynb
```

The localization pipeline is:

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

The notebook handles:

- Reconstruction-difference maps
- Pixel-level anomaly maps
- Ground-truth comparison
- Thresholding
- Morphological operations
- Bounding regions
- Heatmaps
- Localization visualization

fileciteturn4file1L176-L205

---

# 18. Model Evaluation

Run:

```text
07_model_evaluation.ipynb
```

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

Also include:

```text
Confusion Matrix
Error Analysis
Category-level Evaluation
Localization Analysis
```

These metrics are part of the intended VisionGuard evaluation pipeline. fileciteturn4file1L207-L240

---

# 19. Inference

Run:

```text
08_inference_demo.ipynb
```

The inference workflow is:

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

The inference notebook should display:

- Input image
- Reconstruction
- Prediction
- Anomaly score
- Heatmap
- Predicted defect region
- Ground-truth mask when available

---

# 20. Standalone Training

The repository also contains:

```text
train.py
```

It is responsible for:

- Loading configuration
- Loading prepared data
- Creating the training pipeline
- Creating the autoencoder
- Compiling the model
- Training
- Validation
- Callbacks
- Checkpoints
- Saving the final model
- Recording training history

Typical callbacks:

```text
EarlyStopping
ReduceLROnPlateau
ModelCheckpoint
```

fileciteturn4file1L109-L143

Run from the project root:

```bash
python train.py
```

---

# 21. Standalone Prediction

The repository contains:

```text
predict.py
```

Its intended workflow is:

```text
Load Model
    ↓
Load Image
    ↓
Preprocess
    ↓
Reconstruct
    ↓
Calculate Reconstruction Error
    ↓
Compare With Threshold
    ↓
Generate Anomaly Map
    ↓
Return Prediction
```

Run:

```bash
python predict.py
```

The exact command-line arguments depend on the implementation in `predict.py`.

---

# 22. Streamlit

VisionGuard AI includes:

```text
app.py
```

The application is intended to provide:

- Image upload
- Model inference
- Normal/anomaly prediction
- Anomaly score
- Threshold
- Reconstruction
- Heatmap
- Defect localization

fileciteturn4file2L94-L117

From the project root:

```bash
streamlit run app.py
```

Streamlit should open the application in your browser.

---

# 23. GPU Troubleshooting

## Check NVIDIA GPU

On Windows:

```bash
nvidia-smi
```

If an NVIDIA GPU is detected, this command should display GPU and driver information.

## Check TensorFlow

```python
import tensorflow as tf

print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))
```

If TensorFlow returns:

```text
[]
```

then TensorFlow is not currently exposing a GPU to Python.

Do not confuse:

```text
NVIDIA GPU detected by Windows
```

with:

```text
TensorFlow successfully using that GPU
```

They are separate checks.

---

# 24. Google Colab GPU Troubleshooting

Check:

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU available:")
    for gpu in gpus:
        print(gpu)
else:
    print("No TensorFlow GPU detected.")
```

If no GPU is detected:

1. Check the Colab runtime type.
2. Make sure a GPU accelerator is selected.
3. Restart the runtime if dependencies were changed.
4. Run the verification cell again.

Dataset preparation should remain usable even without a GPU. fileciteturn4file2L75-L91

---

# 25. TensorFlow Environment Rule

Keep TensorFlow isolated inside the project's virtual environment.

Do not mix packages from:

```text
Global Python
Anaconda
System Python
.venv
```

without knowing which interpreter VS Code is using.

In VS Code:

```text
Ctrl + Shift + P
→ Python: Select Interpreter
→ Select .venv
```

Then verify in the VS Code terminal:

```bash
python -c "import sys; print(sys.executable)"
```

The printed path should point to the project's `.venv`.

---

# 26. Jupyter / VS Code

Install the Jupyter kernel inside the virtual environment:

```bash
python -m pip install jupyter ipykernel
```

Register it:

```bash
python -m ipykernel install --user --name visionguard-ai --display-name "VisionGuard AI"
```

In VS Code, select:

```text
VisionGuard AI
```

as the notebook kernel.

---

# 27. Git Setup

Check Git:

```bash
git --version
```

Check repository status:

```bash
git status
```

Add project files:

```bash
git add .
```

Commit:

```bash
git commit -m "Add VisionGuard AI setup and utility files"
```

Push:

```bash
git push
```

Do **not** commit:

```text
MVTec AD dataset
large generated datasets
temporary caches
API keys
passwords
.env files
unnecessary large model files
```

The repository architecture explicitly recommends excluding large datasets and generated model artifacts from normal Git commits. fileciteturn4file9L25-L51

---

# 28. Recommended `.gitignore`

Use a `.gitignore` similar to:

```gitignore
__pycache__/
*.py[cod]

.ipynb_checkpoints/

.venv/
venv/
env/

.env
.env.*

data/
logs/

*.h5
*.keras
*.ckpt
*.pkl

models/checkpoints/

*.tmp
*.cache

.DS_Store
Thumbs.db
```

Do not ignore source files such as:

```text
config.py
utils.py
train.py
predict.py
app.py
```

---

# 29. Common Errors

## `ModuleNotFoundError`

Example:

```text
ModuleNotFoundError: No module named 'tensorflow'
```

Check the active interpreter:

```bash
python -c "import sys; print(sys.executable)"
```

Then install into that exact environment:

```bash
python -m pip install tensorflow
```

---

## `FileNotFoundError`

Check the expected prepared dataset:

```text
/content/visionguard_data/
```

Verify:

```python
from pathlib import Path

root = Path("/content/visionguard_data")

print("Exists:", root.exists())
print("Directory:", root.is_dir())

if root.exists():
    print(list(root.iterdir()))
```

The project intentionally uses `/content/visionguard_data` as the prepared Colab dataset path. fileciteturn4file1L49-L75

---

## Wrong dataset structure

Expected:

```text
visionguard_data/
└── screw/
    ├── train/
    ├── test/
    └── ground_truth/
```

Not:

```text
visionguard_data/
└── screw/
    └── screw/
        └── train/
```

The dataset-preparation workflow is explicitly designed to prevent these duplicate nested category directories. fileciteturn4file5L25-L38

---

## `BadZipFile`

If a downloaded archive is reported as an invalid ZIP, do not assume the MVTec dataset itself is corrupt.

First check:

```python
from pathlib import Path

print("Downloaded path:", path)
print("Exists:", Path(path).exists())
```

Then inspect the returned KaggleHub directory rather than hardcoding a cache path.

The project specification requires dynamic discovery of `mvtec_anomaly_detection`. fileciteturn4file5L25-L38

---

# 30. Final Installation Verification

Run:

```python
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import sklearn

print("=" * 60)
print("VisionGuard AI Environment Check")
print("=" * 60)

print("Python:", sys.version)
print("TensorFlow:", tf.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("OpenCV:", cv2.__version__)
print("Scikit-learn:", sklearn.__version__)

print("\nTensorFlow GPUs:")
print(tf.config.list_physical_devices("GPU"))

print("\nEnvironment check completed.")
```

---

# 31. Final Project Checklist

Before training:

```text
[ ] Virtual environment created
[ ] Correct VS Code interpreter selected
[ ] requirements.txt installed
[ ] TensorFlow imports successfully
[ ] GPU checked
[ ] Dataset downloaded
[ ] mvtec_anomaly_detection located
[ ] Five requested categories found
[ ] /content/visionguard_data created
[ ] train/ directories verified
[ ] test/ directories verified
[ ] ground_truth/ directories verified
[ ] Dataset summary generated
```

Before deployment:

```text
[ ] Autoencoder trained
[ ] Final model saved
[ ] Anomaly threshold generated
[ ] Anomaly detection tested
[ ] Heatmaps tested
[ ] Defect localization tested
[ ] Evaluation completed
[ ] Inference tested
[ ] app.py tested
[ ] Large files excluded from Git
[ ] No credentials committed
```

---

# 32. Complete Workflow

```text
SETUP
  ↓
Virtual Environment
  ↓
Dependencies
  ↓
GPU Verification
  ↓
DATASET
  ↓
KaggleHub Download
  ↓
MVTec Root Detection
  ↓
Five Categories
  ↓
Dataset Validation
  ↓
EDA
  ↓
PREPROCESSING
  ↓
Resize
  ↓
Normalize
  ↓
Augment
  ↓
TensorFlow Dataset
  ↓
MODEL
  ↓
CNN Autoencoder
  ↓
Training
  ↓
Checkpoint
  ↓
Final Model
  ↓
DETECTION
  ↓
Reconstruction
  ↓
Reconstruction Error
  ↓
Threshold
  ↓
Normal / Anomaly
  ↓
LOCALIZATION
  ↓
Difference Map
  ↓
Heatmap
  ↓
Defect Region
  ↓
EVALUATION
  ↓
Image-Level Metrics
  ↓
Pixel-Level Metrics
  ↓
INFERENCE
  ↓
Streamlit
```

---

# 33. Important Notes

- Do not commit the complete MVTec AD dataset to GitHub.
- Do not hardcode KaggleHub cache paths.
- Do not hardcode dataset counts.
- Do not invent defect names.
- Do not modify the original downloaded dataset during preparation.
- Keep configuration centralized.
- Keep reusable functionality in `utils.py`.
- Use deterministic file ordering where practical.
- Validate filesystem paths before training.
- Keep model artifacts separate from source code.
- Never commit credentials or `.env` files.

VisionGuard AI is intended to remain modular, reproducible, debuggable, and suitable for an academic/portfolio computer-vision project. fileciteturn4file0L1-L18
