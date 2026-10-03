"""
VisionGuard AI — Standalone TensorFlow/Keras Training Script

Industrial Anomaly Detection & Defect Localization
MVTec AD | CNN / Convolutional Autoencoder

This script mirrors the project's existing CNN autoencoder notebook:
- 5 selected MVTec AD categories
- Normal/"good" training images only
- 128 x 128 RGB inputs
- 10% deterministic validation split
- CNN convolutional autoencoder
- Adam optimizer, learning rate 1e-3
- MSE loss + MAE metric
- ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
- Best model + final model + best weights
- Training history, plots, model report, and 99th-percentile reference thresholds

Default prepared dataset:
    Auto-detected: /content/visionguard_data (Colab) or
    <project_folder>/visionguard_data (local VS Code)

Expected category layout:
    /content/visionguard_data/
    ├── screw/train/good/
    ├── bottle/train/good/
    ├── capsule/train/good/
    ├── metal_nut/train/good/
    └── hazelnut/train/good/
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError as exc:
    raise ImportError(
        "TensorFlow is required to run train.py. "
        "Install it with: pip install tensorflow"
    ) from exc


# ============================================================
# Project configuration
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

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_CHANNELS = 3
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

BATCH_SIZE = 32
EPOCHS = 40
VALIDATION_SIZE = 0.10

LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-6

EARLY_STOPPING_PATIENCE = 7
LR_REDUCTION_PATIENCE = 3
LR_REDUCTION_FACTOR = 0.5

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent

# The project is designed to run in both Google Colab and local VS Code.
# Colab uses /content/visionguard_data, while the VS Code project in this
# repository keeps visionguard_data beside train.py.
def resolve_default_dataset_root() -> Path:
    candidates = [
        Path("/content/visionguard_data"),
        PROJECT_ROOT / "visionguard_data",
        Path.cwd() / "visionguard_data",
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    # Keep the local-project path as the final fallback so error messages
    # point at the path a VS Code user is most likely expecting.
    return PROJECT_ROOT / "visionguard_data"


DEFAULT_DATASET_ROOT = resolve_default_dataset_root()
MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
FIGURE_DIR = ARTIFACT_DIR / "figures"
EVALUATION_DIR = ARTIFACT_DIR / "evaluation"
LOG_DIR = PROJECT_ROOT / "logs" / "training"

for directory in (
    MODEL_DIR,
    FIGURE_DIR,
    EVALUATION_DIR,
    LOG_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

ImageFile.LOAD_TRUNCATED_IMAGES = False


# ============================================================
# Reproducibility / TensorFlow runtime
# ============================================================

def set_global_seed(seed: int = SEED) -> None:
    """Set deterministic seeds where supported by the runtime."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


def configure_gpu() -> list:
    """Enable memory growth when a GPU is available."""
    gpus = tf.config.list_physical_devices("GPU")

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # Memory growth must be set before GPU initialization.
            pass

    return gpus


# ============================================================
# Filesystem helpers
# ============================================================

def is_image_file(path: Path) -> bool:
    """Return True for supported image files, case-insensitively."""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def image_files(folder: Path) -> list[Path]:
    """Return sorted image files under a directory."""
    if not folder.is_dir():
        return []

    return sorted(
        path
        for path in folder.rglob("*")
        if is_image_file(path)
    )


def discover_good_images(dataset_root: Path, category: str) -> list[Path]:
    """
    Locate normal training images for a category.

    The project contract is category/train/good, but a small amount of
    path robustness is retained by checking the category recursively
    for a directory named exactly 'good' beneath train.
    """
    exact_good = dataset_root / category / "train" / "good"
    if exact_good.is_dir():
        files = image_files(exact_good)
        if files:
            return files

    train_root = dataset_root / category / "train"
    if train_root.is_dir():
        candidates = [
            path
            for path in train_root.rglob("*")
            if path.is_dir() and path.name.casefold() == "good"
        ]
        for candidate in sorted(candidates):
            files = image_files(candidate)
            if files:
                return files

    return []


def validate_dataset_root(dataset_root: Path, categories: Sequence[str]) -> dict[str, list[Path]]:
    """Validate the prepared MVTec structure and return category image lists."""
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {dataset_root}\n"
            "Prepare the dataset first with "
            "01_dataset_download_and_preparation.ipynb."
        )

    if not dataset_root.is_dir():
        raise NotADirectoryError(
            f"Dataset root is not a directory: {dataset_root}"
        )

    category_data: dict[str, list[Path]] = {}
    missing_categories: list[str] = []

    print("\nDataset validation")
    print("-" * 72)

    for category in categories:
        category_root = dataset_root / category
        train_root = category_root / "train"
        good_root = train_root / "good"
        files = discover_good_images(dataset_root, category)

        if not category_root.is_dir():
            print(f"{category:<12} -> CATEGORY MISSING")
            missing_categories.append(category)
            continue

        if not train_root.is_dir():
            print(f"{category:<12} -> train/ MISSING")
            missing_categories.append(category)
            continue

        if not good_root.is_dir() and not files:
            print(f"{category:<12} -> train/good/ MISSING")
            missing_categories.append(category)
            continue

        if not files:
            print(f"{category:<12} -> NO IMAGE FILES FOUND")
            missing_categories.append(category)
            continue

        category_data[category] = files
        print(f"{category:<12} -> FOUND ({len(files):,} normal images)")

    if missing_categories:
        missing_text = ", ".join(missing_categories)
        raise FileNotFoundError(
            "Required training data is missing for: "
            f"{missing_text}\n"
            f"Checked prepared dataset: {dataset_root}\n"
            "Verify that each category contains train/good images."
        )

    return category_data


def validate_images(paths: Iterable[Path], category: str) -> None:
    """Validate image readability and basic metadata before training."""
    invalid: list[str] = []

    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()

            with Image.open(path) as image:
                if image.width <= 0 or image.height <= 0:
                    raise ValueError("Image has invalid dimensions.")
        except Exception as exc:
            invalid.append(f"{path.name}: {type(exc).__name__}: {exc}")

    if invalid:
        preview = "\n".join(invalid[:10])
        suffix = (
            "\n..."
            if len(invalid) > 10
            else ""
        )
        raise ValueError(
            f"{category}: {len(invalid)} invalid training image(s) found.\n"
            f"{preview}{suffix}"
        )


# ============================================================
# Train / validation split
# ============================================================

def split_paths(
    paths: Sequence[Path],
    validation_size: float = VALIDATION_SIZE,
    seed: int = SEED,
) -> tuple[list[Path], list[Path]]:
    """
    Deterministically split normal images into train/validation sets.

    This avoids depending on sklearn in the standalone training script
    while keeping the same 10% shuffled split behavior.
    """
    paths = sorted(paths)

    if len(paths) < 2:
        raise ValueError(
            "At least two normal images are required to create a "
            "training/validation split."
        )

    rng = np.random.default_rng(seed)
    indices = np.arange(len(paths))
    rng.shuffle(indices)

    validation_count = max(
        1,
        int(round(len(paths) * validation_size)),
    )
    validation_count = min(
        validation_count,
        len(paths) - 1,
    )

    validation_indices = set(
        indices[:validation_count].tolist()
    )

    validation_paths = [
        paths[index]
        for index in indices[:validation_count]
    ]

    train_paths = [
        paths[index]
        for index in indices
        if index not in validation_indices
    ]

    return (
        sorted(train_paths),
        sorted(validation_paths),
    )


# ============================================================
# TensorFlow input pipeline
# ============================================================

def load_image(path: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Read, decode, resize, normalize, and duplicate an image for AE training."""
    data = tf.io.read_file(path)

    image = tf.io.decode_image(
        data,
        channels=IMAGE_CHANNELS,
        expand_animations=False,
    )

    image.set_shape(
        [None, None, IMAGE_CHANNELS]
    )

    image = tf.image.convert_image_dtype(
        image,
        tf.float32,
    )

    image = tf.image.resize(
        image,
        [IMAGE_HEIGHT, IMAGE_WIDTH],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=True,
    )

    image = tf.clip_by_value(
        image,
        0.0,
        1.0,
    )

    image = tf.ensure_shape(
        image,
        [
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            IMAGE_CHANNELS,
        ],
    )

    tf.debugging.assert_all_finite(
        image,
        "Image contains NaN or Inf.",
    )

    return image, image


def make_dataset(
    paths: Sequence[Path],
    training: bool = False,
    batch_size: int = BATCH_SIZE,
) -> tf.data.Dataset:
    """Build a TensorFlow dataset for autoencoder training."""
    path_strings = [str(path) for path in paths]

    if not path_strings:
        raise ValueError("No image paths were supplied.")

    dataset = tf.data.Dataset.from_tensor_slices(
        path_strings
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(
                max(len(path_strings), 1),
                1000,
            ),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(
        batch_size,
        drop_remainder=False,
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# CNN / Convolutional Autoencoder
# ============================================================

def build_autoencoder() -> keras.Model:
    """Create the VisionGuard CNN convolutional autoencoder."""
    inputs = keras.Input(
        shape=[
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            IMAGE_CHANNELS,
        ],
        name="image",
    )

    # Encoder
    x = layers.Conv2D(
        32,
        3,
        padding="same",
        activation="relu",
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        2,
        padding="same",
    )(x)

    x = layers.Conv2D(
        64,
        3,
        padding="same",
        activation="relu",
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        2,
        padding="same",
    )(x)

    x = layers.Conv2D(
        128,
        3,
        padding="same",
        activation="relu",
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        2,
        padding="same",
    )(x)

    # Latent representation
    x = layers.Conv2D(
        256,
        3,
        padding="same",
        activation="relu",
        name="latent_space",
    )(x)

    # Decoder
    x = layers.UpSampling2D(
        2
    )(x)

    x = layers.Conv2D(
        128,
        3,
        padding="same",
        activation="relu",
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D(
        2
    )(x)

    x = layers.Conv2D(
        64,
        3,
        padding="same",
        activation="relu",
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D(
        2
    )(x)

    x = layers.Conv2D(
        32,
        3,
        padding="same",
        activation="relu",
    )(x)

    x = layers.BatchNormalization()(x)

    outputs = layers.Conv2D(
        IMAGE_CHANNELS,
        3,
        padding="same",
        activation="sigmoid",
        name="reconstruction",
    )(x)

    return keras.Model(
        inputs,
        outputs,
        name="VisionGuard_CNN_Autoencoder",
    )


def create_model() -> keras.Model:
    """Build and compile a fresh autoencoder."""
    model = build_autoencoder()

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="mse",
        metrics=[
            keras.metrics.MeanAbsoluteError(
                name="mae"
            )
        ],
    )

    return model


# ============================================================
# Training callbacks
# ============================================================

def build_callbacks(
    category: str,
    category_model_dir: Path,
) -> list[keras.callbacks.Callback]:
    """Create production-style callbacks aligned with the notebook."""
    best_model_file = category_model_dir / "best_model.keras"
    best_weights_file = (
        category_model_dir / "best_weights.weights.h5"
    )
    training_log_file = (
        LOG_DIR / f"{category}_training_log.csv"
    )

    # Remove stale outputs so a fresh run cannot accidentally reuse them.
    for path in (
        best_model_file,
        best_weights_file,
        training_log_file,
    ):
        if path.exists():
            path.unlink()

    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_file),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_weights_file),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=LR_REDUCTION_FACTOR,
            patience=LR_REDUCTION_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            filename=str(training_log_file),
            append=False,
        ),
    ]


# ============================================================
# Validation and reporting
# ============================================================

def calculate_reconstruction_errors(
    model: keras.Model,
    paths: Sequence[Path],
) -> np.ndarray:
    """Calculate per-image MSE on normal validation images."""
    dataset = make_dataset(
        paths,
        training=False,
    )

    values: list[float] = []

    for batch_x, _ in dataset:
        reconstruction = model(
            batch_x,
            training=False,
        )

        error = tf.reduce_mean(
            tf.square(
                batch_x - reconstruction
            ),
            axis=[1, 2, 3],
        )

        values.extend(
            error.numpy().astype(np.float64).tolist()
        )

    return np.asarray(
        values,
        dtype=np.float64,
    )


def save_history_artifacts(
    category: str,
    history: keras.callbacks.History,
) -> dict:
    """Save per-epoch history CSV and training-loss figure."""
    history_df = pd.DataFrame(
        history.history
    )

    history_df.insert(
        0,
        "Epoch",
        np.arange(
            1,
            len(history_df) + 1,
        ),
    )

    history_file = (
        LOG_DIR / f"{category}_training_history.csv"
    )
    history_df.to_csv(
        history_file,
        index=False,
    )

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        ax.plot(
            history_df["Epoch"],
            history_df["loss"],
            label="Training Loss",
            linewidth=2,
        )

        if "val_loss" in history_df:
            ax.plot(
                history_df["Epoch"],
                history_df["val_loss"],
                label="Validation Loss",
                linewidth=2,
            )

        ax.set_title(
            f"VisionGuard AI — {category} CNN Autoencoder Training Loss"
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Squared Error")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()

        figure_file = (
            FIGURE_DIR / f"{category}_training_loss.png"
        )

        fig.savefig(
            figure_file,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(fig)
    except ImportError:
        figure_file = None

    return {
        "history_file": history_file,
        "figure_file": figure_file,
        "history_df": history_df,
    }


def write_model_summary(
    model: keras.Model,
    category: str,
) -> Path:
    """Write a text summary for the trained architecture."""
    summary_file = (
        MODEL_DIR / category / "model_summary.txt"
    )

    lines: list[str] = []

    model.summary(
        print_fn=lines.append
    )

    summary_file.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return summary_file


# ============================================================
# Per-category training
# ============================================================

def train_category(
    category: str,
    dataset_root: Path,
    all_paths: Sequence[Path],
) -> dict:
    """Train, validate, save, and report one category-specific autoencoder."""
    print("\n" + "=" * 72)
    print(f"VISIONGUARD AI — TRAINING: {category.upper()}")
    print("=" * 72)

    validate_images(
        all_paths,
        category,
    )

    train_paths, validation_paths = split_paths(
        all_paths,
        validation_size=VALIDATION_SIZE,
        seed=SEED,
    )

    print(f"Category:            {category}")
    print(f"Dataset root:        {dataset_root}")
    print(f"Normal images:       {len(all_paths):,}")
    print(f"Training images:     {len(train_paths):,}")
    print(f"Validation images:   {len(validation_paths):,}")
    print(f"Image size:          {IMAGE_HEIGHT} x {IMAGE_WIDTH} x {IMAGE_CHANNELS}")
    print(f"Batch size:          {BATCH_SIZE}")
    print(f"Maximum epochs:      {EPOCHS}")
    print(f"Learning rate:       {LEARNING_RATE}")

    train_dataset = make_dataset(
        train_paths,
        training=True,
    )

    validation_dataset = make_dataset(
        validation_paths,
        training=False,
    )

    # One-batch pipeline sanity check.
    train_x, train_y = next(iter(train_dataset))
    validation_x, validation_y = next(
        iter(validation_dataset)
    )

    expected_shape = (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        IMAGE_CHANNELS,
    )

    if tuple(train_x.shape[1:]) != expected_shape:
        raise ValueError(
            f"{category}: training batch shape is "
            f"{tuple(train_x.shape)}, expected (?, {expected_shape})."
        )

    if tuple(validation_x.shape[1:]) != expected_shape:
        raise ValueError(
            f"{category}: validation batch shape is "
            f"{tuple(validation_x.shape)}, expected (?, {expected_shape})."
        )

    if not tf.reduce_all(
        tf.math.is_finite(train_x)
    ):
        raise ValueError(
            f"{category}: training batch contains non-finite values."
        )

    if not (
        0.0 <= float(tf.reduce_min(train_x)) <= 1.0
        and 0.0 <= float(tf.reduce_max(train_x)) <= 1.0
    ):
        raise ValueError(
            f"{category}: training pixels are outside the [0, 1] range."
        )

    if tuple(train_x.shape) != tuple(train_y.shape):
        raise ValueError(
            f"{category}: autoencoder input/target shapes do not match."
        )

    if tuple(validation_x.shape) != tuple(validation_y.shape):
        raise ValueError(
            f"{category}: validation input/target shapes do not match."
        )

    category_model_dir = MODEL_DIR / category
    category_model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_model_file = (
        category_model_dir / "final_model.keras"
    )

    if final_model_file.exists():
        final_model_file.unlink()

    model = create_model()

    # Gradient smoke test before expensive training.
    with tf.GradientTape() as tape:
        smoke_prediction = model(
            train_x,
            training=True,
        )
        smoke_loss = tf.reduce_mean(
            tf.square(
                train_y - smoke_prediction
            )
        )

    gradients = tape.gradient(
        smoke_loss,
        model.trainable_variables,
    )

    gradient_count = sum(
        gradient is not None
        for gradient in gradients
    )

    if gradient_count == 0:
        raise RuntimeError(
            f"{category}: no gradients were produced by the model."
        )

    if not np.isfinite(
        float(smoke_loss.numpy())
    ):
        raise RuntimeError(
            f"{category}: initial loss is not finite."
        )

    print(
        f"Pipeline check passed | batch={tuple(train_x.shape)} "
        f"| gradient tensors={gradient_count}"
    )

    model.summary()

    callbacks = build_callbacks(
        category,
        category_model_dir,
    )

    start_time = time.time()

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        verbose=1,
        callbacks=callbacks,
    )

    elapsed_minutes = (
        time.time() - start_time
    ) / 60.0

    best_model_file = (
        category_model_dir / "best_model.keras"
    )
    best_weights_file = (
        category_model_dir / "best_weights.weights.h5"
    )

    # Save the final in-memory model. EarlyStopping restores best weights,
    # so final_model.keras corresponds to the best validation state.
    model.save(
        final_model_file
    )

    if not best_model_file.exists():
        raise RuntimeError(
            f"{category}: best_model.keras was not created."
        )

    if not final_model_file.exists():
        raise RuntimeError(
            f"{category}: final_model.keras was not created."
        )

    if not best_weights_file.exists():
        raise RuntimeError(
            f"{category}: best_weights.weights.h5 was not created."
        )

    # Reload best model to validate the serialized artifact.
    best_model = keras.models.load_model(
        best_model_file,
        compile=False,
    )

    if tuple(best_model.input_shape[1:]) != (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        IMAGE_CHANNELS,
    ):
        raise ValueError(
            f"{category}: saved best model input shape is "
            f"{best_model.input_shape}."
        )

    if tuple(best_model.output_shape[1:]) != (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        IMAGE_CHANNELS,
    ):
        raise ValueError(
            f"{category}: saved best model output shape is "
            f"{best_model.output_shape}."
        )

    # Validate that the saved model can reconstruct a batch.
    reconstruction = best_model.predict(
        validation_x[:2],
        verbose=0,
    )

    if reconstruction.shape != validation_x[:2].shape:
        raise ValueError(
            f"{category}: saved model reconstruction shape mismatch: "
            f"{reconstruction.shape} vs {validation_x[:2].shape}"
        )

    if not np.isfinite(reconstruction).all():
        raise ValueError(
            f"{category}: saved model produced non-finite output."
        )

    reconstruction = np.clip(
        reconstruction,
        0.0,
        1.0,
    )

    history_artifacts = save_history_artifacts(
        category,
        history,
    )

    summary_file = write_model_summary(
        best_model,
        category,
    )

    history_df = history_artifacts["history_df"]

    best_index = int(
        history_df["val_loss"].idxmin()
    )
    best_row = history_df.iloc[best_index]

    validation_errors = calculate_reconstruction_errors(
        best_model,
        validation_paths,
    )

    if len(validation_errors) != len(validation_paths):
        raise RuntimeError(
            f"{category}: reconstruction error count mismatch."
        )

    threshold_99 = float(
        np.percentile(
            validation_errors,
            99,
        )
    )

    threshold_95 = float(
        np.percentile(
            validation_errors,
            95,
        )
    )

    result = {
        "Category": category,
        "Train Images": len(train_paths),
        "Validation Images": len(validation_paths),
        "Epochs Completed": len(history_df),
        "Best Epoch": int(best_row["Epoch"]),
        "Best Validation Loss": float(
            best_row["val_loss"]
        ),
        "Best Validation MAE": float(
            best_row.get("val_mae", np.nan)
        ),
        "Mean Normal Validation Error": float(
            np.mean(validation_errors)
        ),
        "95th Percentile Error": threshold_95,
        "99th Percentile Error": threshold_99,
        "Training Time (minutes)": float(
            elapsed_minutes
        ),
        "Parameters": int(
            best_model.count_params()
        ),
        "Best Model": str(best_model_file),
        "Final Model": str(final_model_file),
        "Best Weights": str(best_weights_file),
        "History CSV": str(
            history_artifacts["history_file"]
        ),
        "Model Summary": str(summary_file),
        "Status": "SUCCESS",
    }

    print(f"\n{category}: training completed successfully.")
    print(f"Best epoch:           {result['Best Epoch']}")
    print(f"Best val loss:        {result['Best Validation Loss']:.8f}")
    print(f"Best val MAE:         {result['Best Validation MAE']:.8f}")
    print(f"99th percentile MSE:  {threshold_99:.8f}")
    print(f"Training time:        {elapsed_minutes:.2f} minutes")
    print(f"Best model:           {best_model_file}")
    print(f"Final model:          {final_model_file}")

    return result


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Train VisionGuard AI category-specific CNN autoencoders "
            "on normal MVTec AD images."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=(
            "Prepared dataset root. "
            f"Default (auto-detected): {DEFAULT_DATASET_ROOT}"
        ),
    )

    parser.add_argument(
        "--category",
        choices=CATEGORIES,
        default=None,
        help=(
            "Train only one requested category. "
            "By default, all five categories are trained."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Maximum training epochs. Default: {EPOCHS}",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size. Default: {BATCH_SIZE}",
    )

    parser.add_argument(
        "--validation-size",
        type=float,
        default=VALIDATION_SIZE,
        help=(
            f"Validation fraction. Default: {VALIDATION_SIZE}"
        ),
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help=f"Adam learning rate. Default: {LEARNING_RATE}",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed. Default: {SEED}",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point for standalone execution."""
    args = parse_args()

    global EPOCHS
    global BATCH_SIZE
    global VALIDATION_SIZE
    global LEARNING_RATE
    global SEED

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")

    if not 0.0 < args.validation_size < 1.0:
        raise ValueError(
            "--validation-size must be between 0 and 1."
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "--learning-rate must be > 0."
        )

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    VALIDATION_SIZE = args.validation_size
    LEARNING_RATE = args.learning_rate
    SEED = args.seed

    set_global_seed(SEED)
    gpus = configure_gpu()

    selected_categories = (
        [args.category]
        if args.category
        else list(CATEGORIES)
    )

    dataset_root = args.dataset_root.expanduser().resolve()

    print("=" * 72)
    print("VISIONGUARD AI — STANDALONE TRAINING")
    print("=" * 72)
    print(f"TensorFlow:       {tf.__version__}")
    print(f"GPU count:        {len(gpus)}")
    print(
        "Runtime:          "
        + ("GPU" if gpus else "CPU")
    )
    print(f"Dataset root:     {dataset_root}")
    print(
        "Categories:       "
        + ", ".join(selected_categories)
    )
    print(f"Image size:       {IMAGE_HEIGHT}x{IMAGE_WIDTH}x{IMAGE_CHANNELS}")
    print(f"Batch size:       {BATCH_SIZE}")
    print(f"Epochs:           {EPOCHS}")
    print(f"Validation size:  {VALIDATION_SIZE}")
    print(f"Learning rate:    {LEARNING_RATE}")
    print(f"Seed:             {SEED}")
    print("=" * 72)

    category_data = validate_dataset_root(
        dataset_root,
        selected_categories,
    )

    results: list[dict] = []
    failed_categories: list[str] = []

    for category in selected_categories:
        try:
            result = train_category(
                category=category,
                dataset_root=dataset_root,
                all_paths=category_data[category],
            )
            results.append(result)
        except Exception as exc:
            failed_categories.append(category)
            print("\n" + "!" * 72)
            print(f"ERROR while training category '{category}'")
            print(f"{type(exc).__name__}: {exc}")
            print("This category was skipped; the remaining selected")
            print("categories will continue.")
            print("!" * 72)

    if not results:
        raise RuntimeError(
            "No category completed training successfully."
        )

    results_df = pd.DataFrame(results)

    report_file = (
        EVALUATION_DIR / "training_report.csv"
    )
    results_df.to_csv(
        report_file,
        index=False,
    )

    threshold_df = results_df[
        [
            "Category",
            "Mean Normal Validation Error",
            "95th Percentile Error",
            "99th Percentile Error",
        ]
    ].copy()

    threshold_df = threshold_df.rename(
        columns={
            "Mean Normal Validation Error": "Mean Error",
            "95th Percentile Error": "95th Percentile",
            "99th Percentile Error": "99th Percentile",
        }
    )

    threshold_file = (
        EVALUATION_DIR /
        "reference_reconstruction_thresholds.csv"
    )
    threshold_df.to_csv(
        threshold_file,
        index=False,
    )

    print("\n" + "=" * 72)
    print("VISIONGUARD AI — TRAINING SUMMARY")
    print("=" * 72)

    print(
        results_df[
            [
                "Category",
                "Train Images",
                "Validation Images",
                "Epochs Completed",
                "Best Epoch",
                "Best Validation Loss",
                "Best Validation MAE",
                "99th Percentile Error",
            ]
        ].to_string(index=False)
    )

    print("\nReports saved:")
    print(f"  Training report:     {report_file}")
    print(f"  Thresholds:          {threshold_file}")
    print(f"  Models:              {MODEL_DIR}")
    print(f"  Training logs:       {LOG_DIR}")
    print(f"  Figures:             {FIGURE_DIR}")

    if failed_categories:
        print(
            "\nCompleted with category failures: "
            + ", ".join(failed_categories)
        )
        raise SystemExit(1)

    if len(results) == len(selected_categories):
        print("\nAll selected categories trained successfully.")
    else:
        print(
            "\nTraining completed for the categories that "
            "successfully passed validation."
        )

    print("=" * 72)


if __name__ == "__main__":
    main()
