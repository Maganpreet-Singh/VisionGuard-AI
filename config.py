"""
VisionGuard AI — Central Configuration
Industrial Anomaly Detection & Defect Localization

This module centralizes project-wide paths, dataset settings,
model hyperparameters, training callbacks, and runtime options.

Designed to stay compatible with the existing VisionGuard AI
CNN/Convolutional Autoencoder workflow.
"""

from pathlib import Path


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# Google Colab prepared dataset path required by the project.
DATASET_ROOT = Path("/content/visionguard_data")

# Local fallback for projects copied to another environment.
LOCAL_DATASET_ROOT = PROJECT_ROOT / "data"

# Model/artifact directories.
MODEL_ROOT = PROJECT_ROOT / "models"
CHECKPOINT_ROOT = MODEL_ROOT / "checkpoints"
FINAL_MODEL_ROOT = MODEL_ROOT / "final"

ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_ROOT / "figures"
HEATMAPS_DIR = ARTIFACTS_ROOT / "heatmaps"
PREDICTIONS_DIR = ARTIFACTS_ROOT / "predictions"
EVALUATION_DIR = ARTIFACTS_ROOT / "evaluation"
REPORTS_DIR = ARTIFACTS_ROOT / "reports"

LOG_ROOT = PROJECT_ROOT / "logs"
TRAINING_LOG_DIR = LOG_ROOT / "training"

# Legacy/research notebook output directories.
TABLE_DIR = PROJECT_ROOT / "table"
IMAGE_DIR = PROJECT_ROOT / "images"


# ============================================================
# Dataset configuration
# ============================================================

CATEGORIES = [
    "screw",
    "bottle",
    "capsule",
    "metal_nut",
    "hazelnut",
]

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
}

# Case-insensitive extension checking should always use suffix.lower().

# Input image configuration.
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_CHANNELS = 3

# Convenient tuple used by PIL, OpenCV, and inference code.
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

# Autoencoder training uses only normal/good training images.
NORMAL_CLASS_NAME = "good"


# ============================================================
# Training configuration
# ============================================================

BATCH_SIZE = 32
EPOCHS = 40

# Validation fraction taken from normal training images.
VALIDATION_SIZE = 0.10

# Optimizer.
LEARNING_RATE = 1e-3

# Learning-rate scheduler.
MIN_LEARNING_RATE = 1e-6
LR_REDUCTION_FACTOR = 0.5
LR_REDUCTION_PATIENCE = 3

# Early stopping.
EARLY_STOPPING_PATIENCE = 7

# Reproducibility.
RANDOM_SEED = 42
SEED = RANDOM_SEED


# ============================================================
# Model configuration
# ============================================================

MODEL_NAME = "VisionGuard_CNN_Autoencoder"

# Keras model format.
MODEL_EXTENSION = ".keras"

BEST_MODEL_FILENAME = "best_model.keras"
FINAL_MODEL_FILENAME = "final_model.keras"
BEST_WEIGHTS_FILENAME = "best_weights.weights.h5"

# Default/shared paths.
MODEL_PATH = MODEL_ROOT / BEST_MODEL_FILENAME
CHECKPOINT_PATH = CHECKPOINT_ROOT

# These names mirror the existing training notebook workflow.
LOSS_FUNCTION = "mse"
OPTIMIZER_NAME = "adam"
METRIC_NAME = "mae"

# Autoencoder architecture.
ENCODER_FILTERS = (32, 64, 128)
LATENT_FILTERS = 256
DECODER_FILTERS = (128, 64, 32)
KERNEL_SIZE = 3
POOL_SIZE = 2
UPSAMPLE_SIZE = 2
ACTIVATION = "relu"
OUTPUT_ACTIVATION = "sigmoid"


# ============================================================
# Anomaly detection configuration
# ============================================================

# Reconstruction-error thresholds are calibrated from normal
# validation images by the anomaly-detection pipeline.
IMAGE_THRESHOLD_PERCENTILE = 99.0
PIXEL_THRESHOLD_PERCENTILE = 99.5

# Maximum number of normal images used for deterministic
# threshold calibration.
CALIBRATION_LIMIT = 40
PIXEL_CALIBRATION_LIMIT = 10

# Defect-localization post-processing.
MORPH_KERNEL_SIZE = 3
MIN_COMPONENT_AREA = 12


# ============================================================
# TensorFlow / data pipeline configuration
# ============================================================

SHUFFLE_BUFFER_SIZE = 1000
NUM_PARALLEL_CALLS = "AUTOTUNE"
PREFETCH = "AUTOTUNE"

# Whether to cache datasets in memory. Disabled by default to
# avoid unnecessary RAM pressure with image-heavy workloads.
CACHE_DATASETS = False

# TensorFlow mixed precision is disabled by default for maximum
# compatibility. Enable explicitly when the hardware/runtime is
# known to support it safely.
ENABLE_MIXED_PRECISION = False

# GPU memory growth.
ENABLE_GPU_MEMORY_GROWTH = True


# ============================================================
# Output configuration
# ============================================================

SAVE_BEST_MODEL = True
SAVE_FINAL_MODEL = True
SAVE_BEST_WEIGHTS = True
SAVE_TRAINING_HISTORY = True
SAVE_TRAINING_PLOTS = True

TRAINING_HISTORY_FILENAME = "training_history.csv"
TRAINING_LOG_FILENAME = "training_log.csv"

# Threshold artifact used by downstream anomaly detection.
THRESHOLD_FILENAME = "reference_reconstruction_thresholds.csv"


# ============================================================
# Utility functions
# ============================================================

def get_category_root(category: str) -> Path:
    """Return the prepared dataset root for one MVTec category."""
    return DATASET_ROOT / category


def get_train_root(category: str) -> Path:
    """Return the training directory for one category."""
    return get_category_root(category) / "train"


def get_normal_train_root(category: str) -> Path:
    """Return the normal/good training-image directory."""
    return get_train_root(category) / NORMAL_CLASS_NAME


def get_validation_root(category: str) -> Path:
    """Return the optional project validation directory."""
    return DATASET_ROOT / "validation" / category / NORMAL_CLASS_NAME


def get_test_root(category: str) -> Path:
    """Return the test directory for one category."""
    return get_category_root(category) / "test"


def get_ground_truth_root(category: str) -> Path:
    """Return the ground-truth mask directory for one category."""
    return get_category_root(category) / "ground_truth"


def get_category_model_dir(category: str) -> Path:
    """Return the model directory used by train.py/app.py."""
    return MODEL_ROOT / category


def get_best_model_path(category: str) -> Path:
    """Return the best complete Keras model path for a category."""
    return get_category_model_dir(category) / BEST_MODEL_FILENAME


def get_final_model_path(category: str) -> Path:
    """Return the final Keras model path for a category."""
    return get_category_model_dir(category) / FINAL_MODEL_FILENAME


def get_best_weights_path(category: str) -> Path:
    """Return the best weights path for a category."""
    return get_category_model_dir(category) / BEST_WEIGHTS_FILENAME


def get_category_checkpoint_dir(category: str) -> Path:
    """Return the checkpoint directory for a category."""
    return CHECKPOINT_ROOT / category


def ensure_output_directories() -> None:
    """Create project output directories when needed."""
    directories = [
        MODEL_ROOT,
        CHECKPOINT_ROOT,
        FINAL_MODEL_ROOT,
        FIGURES_DIR,
        HEATMAPS_DIR,
        PREDICTIONS_DIR,
        EVALUATION_DIR,
        REPORTS_DIR,
        TRAINING_LOG_DIR,
        TABLE_DIR,
        IMAGE_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def validate_configuration() -> None:
    """Validate configuration values before training/inference."""
    if len(CATEGORIES) != 5:
        raise ValueError(
            "VisionGuard AI must be configured with exactly five "
            "MVTec AD categories."
        )

    if IMAGE_HEIGHT <= 0 or IMAGE_WIDTH <= 0:
        raise ValueError("IMAGE_HEIGHT and IMAGE_WIDTH must be positive.")

    if IMAGE_CHANNELS != 3:
        raise ValueError("VisionGuard AI expects RGB images with 3 channels.")

    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be greater than zero.")

    if EPOCHS <= 0:
        raise ValueError("EPOCHS must be greater than zero.")

    if not 0.0 < VALIDATION_SIZE < 1.0:
        raise ValueError("VALIDATION_SIZE must be between 0 and 1.")

    if LEARNING_RATE <= 0:
        raise ValueError("LEARNING_RATE must be greater than zero.")

    if MIN_LEARNING_RATE <= 0:
        raise ValueError("MIN_LEARNING_RATE must be greater than zero.")

    if not 0.0 < IMAGE_THRESHOLD_PERCENTILE <= 100.0:
        raise ValueError("IMAGE_THRESHOLD_PERCENTILE must be in (0, 100].")

    if not 0.0 < PIXEL_THRESHOLD_PERCENTILE <= 100.0:
        raise ValueError("PIXEL_THRESHOLD_PERCENTILE must be in (0, 100].")

    if not IMAGE_EXTENSIONS:
        raise ValueError("IMAGE_EXTENSIONS cannot be empty.")


# Validate constants when config.py is imported.
validate_configuration()


if __name__ == "__main__":
    ensure_output_directories()

    print("VisionGuard AI configuration")
    print("-" * 50)
    print(f"Project root:     {PROJECT_ROOT}")
    print(f"Dataset root:     {DATASET_ROOT}")
    print(f"Categories:       {', '.join(CATEGORIES)}")
    print(f"Image size:       {IMAGE_SIZE}")
    print(f"Batch size:       {BATCH_SIZE}")
    print(f"Epochs:            {EPOCHS}")
    print(f"Learning rate:    {LEARNING_RATE}")
    print(f"Validation size:  {VALIDATION_SIZE}")
    print(f"Random seed:      {RANDOM_SEED}")
    print(f"Model root:       {MODEL_ROOT}")
    print("Configuration validation: PASSED")
