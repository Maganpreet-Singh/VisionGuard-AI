"""
VisionGuard AI — Standalone Inference / Prediction Script

Industrial Anomaly Detection & Defect Localization
MVTec AD + TensorFlow/Keras + CNN Autoencoder

Usage examples:

    python predict.py --category screw --image path/to/image.png

    python predict.py \
        --category bottle \
        --image path/to/image.png \
        --output artifacts/predictions/bottle_result.png

    python predict.py --list-models

The script:
    1. Finds a compatible category model.
    2. Loads and preprocesses the input image.
    3. Reconstructs the image using the CNN autoencoder.
    4. Calculates reconstruction MSE / MAE.
    5. Loads a category-specific anomaly threshold when available.
    6. Classifies the image as NORMAL / ANOMALY / UNKNOWN.
    7. Generates a pixel-level anomaly map.
    8. Performs basic morphological cleanup.
    9. Detects defect bounding boxes.
   10. Saves a visual inference report and JSON results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError as exc:
    raise SystemExit(
        "ERROR: TensorFlow is not installed.\n"
        "Install it with: pip install tensorflow"
    ) from exc


# ============================================================
# Configuration
# ============================================================

SEED = 42

CATEGORIES = [
    "screw",
    "bottle",
    "capsule",
    "metal_nut",
    "hazelnut",
]

IMAGE_SIZE = (128, 128)
IMAGE_CHANNELS = 3

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
}

# Used only when no saved pixel threshold is available.
DEFAULT_PIXEL_PERCENTILE = 99.0

# Used as a fallback for image-level threshold calibration.
IMAGE_THRESHOLD_PERCENTILE = 99.0

# Number of normal training images used for fallback calibration.
CALIBRATION_LIMIT = 40

MORPH_KERNEL_SIZE = 3
MIN_COMPONENT_AREA = 12

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_DATASET_ROOT = Path("/content/visionguard_data")

MODEL_SEARCH_ROOTS = [
    PROJECT_ROOT,
    Path.cwd(),
]

THRESHOLD_FILENAMES = {
    "primary_thresholds.csv",
    "thresholds.csv",
    "anomaly_thresholds.csv",
    "reference_reconstruction_thresholds.csv",
}

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "predictions"
)

DEFAULT_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# Console helpers
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_error(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)


# ============================================================
# Filesystem helpers
# ============================================================

def is_image_file(path: Path) -> bool:
    """
    Return True when path is a supported image file.
    Extension matching is case-insensitive.
    """
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_files(folder: Path) -> list[Path]:
    """
    Return sorted image files under a directory.
    """
    if not folder.is_dir():
        return []

    return sorted(
        path
        for path in folder.rglob("*")
        if is_image_file(path)
    )


def safe_stem(path: Path) -> str:
    """
    Create a filesystem-safe output filename stem.
    """
    value = path.stem

    allowed = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "_-"
    )

    cleaned = "".join(
        character if character in allowed else "_"
        for character in value
    )

    cleaned = cleaned.strip("_")

    return cleaned or "prediction"


# ============================================================
# Dataset discovery
# ============================================================

def find_dataset_root(
    explicit_root: Path | None = None,
) -> Path | None:
    """
    Find the prepared VisionGuard dataset.

    Priority:
        1. Explicit --dataset-root argument.
        2. /content/visionguard_data
        3. Project-local visionguard_data
        4. Current working directory / visionguard_data
    """

    candidates: list[Path] = []

    if explicit_root is not None:
        candidates.append(explicit_root)

    candidates.extend(
        [
            DEFAULT_DATASET_ROOT,
            PROJECT_ROOT / "visionguard_data",
            Path.cwd() / "visionguard_data",
        ]
    )

    seen: set[Path] = set()

    for candidate in candidates:
        candidate = candidate.resolve()

        if candidate in seen:
            continue

        seen.add(candidate)

        if not candidate.is_dir():
            continue

        available_categories = [
            category
            for category in CATEGORIES
            if (candidate / category).is_dir()
        ]

        if available_categories:
            return candidate

    return None


# ============================================================
# Model discovery
# ============================================================

def find_model_candidates(
    category: str,
) -> list[Path]:
    """
    Find trained models associated with a category.

    Preferred location:
        models/<category>/best_model.keras

    Additional supported formats:
        .keras
        .h5
        .hdf5
    """

    category_lower = category.casefold()

    candidates: list[Path] = []

    preferred = (
        PROJECT_ROOT
        / "models"
        / category
        / "best_model.keras"
    )

    if preferred.is_file():
        candidates.append(preferred)

    seen: set[Path] = set()

    for root in MODEL_SEARCH_ROOTS:
        root = root.resolve()

        if not root.exists():
            continue

        try:
            for pattern in (
                "best_model.keras",
                "*.keras",
                "*.h5",
                "*.hdf5",
            ):
                for candidate in root.rglob(pattern):
                    if not candidate.is_file():
                        continue

                    candidate = candidate.resolve()

                    if candidate in seen:
                        continue

                    seen.add(candidate)

                    parent_name = (
                        candidate.parent.name.casefold()
                    )

                    stem_name = (
                        candidate.stem.casefold()
                    )

                    if (
                        category_lower in parent_name
                        or category_lower in stem_name
                    ):
                        candidates.append(candidate)

        except OSError:
            continue

    # Prefer explicitly named best_model.keras files.
    candidates = sorted(
        set(candidates),
        key=lambda path: (
            path.name.casefold() != "best_model.keras",
            str(path).casefold(),
        ),
    )

    return candidates


def validate_model(
    model: keras.Model,
    category: str,
) -> None:
    """
    Validate the basic model input/output contract.
    """

    expected_shape = (
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        IMAGE_CHANNELS,
    )

    input_shape = tuple(model.input_shape)
    output_shape = tuple(model.output_shape)

    if input_shape[-3:] != expected_shape:
        raise ValueError(
            f"Model input shape {input_shape} is incompatible with "
            f"expected {expected_shape} for category '{category}'."
        )

    if output_shape[-3:] != expected_shape:
        raise ValueError(
            f"Model output shape {output_shape} is incompatible with "
            f"expected {expected_shape} for category '{category}'."
        )


def load_category_model(
    category: str,
    model_path: Path | None = None,
) -> tuple[keras.Model, Path]:
    """
    Load a category-specific TensorFlow/Keras model.
    """

    if category not in CATEGORIES:
        raise ValueError(
            f"Unsupported category '{category}'. "
            f"Expected one of: {', '.join(CATEGORIES)}"
        )

    if model_path is not None:
        selected_model = model_path

        if not selected_model.is_file():
            raise FileNotFoundError(
                f"Specified model does not exist:\n"
                f"{selected_model}"
            )
    else:
        candidates = find_model_candidates(category)

        if not candidates:
            raise FileNotFoundError(
                f"No trained model was found for category '{category}'.\n"
                f"Expected a model such as:\n"
                f"  models/{category}/best_model.keras\n"
                f"or another .keras/.h5/.hdf5 model containing "
                f"'{category}' in its name or parent directory."
            )

        selected_model = candidates[0]

    print(f"Loading model: {selected_model}")

    try:
        model = keras.models.load_model(
            selected_model,
            compile=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load model:\n"
            f"{selected_model}\n\n"
            f"Original error: {exc}"
        ) from exc

    validate_model(
        model,
        category,
    )

    return model, selected_model


# ============================================================
# Image preprocessing
# ============================================================

def load_image(
    image_path: Path,
) -> tuple[np.ndarray, Image.Image]:
    """
    Load an image as RGB, resize to 128x128, and normalize to 0-1.

    Returns:
        normalized_array
        resized_pil_image
    """

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Input image does not exist:\n{image_path}"
        )

    if not is_image_file(image_path):
        raise ValueError(
            f"Unsupported image format: {image_path.suffix}\n"
            f"Supported extensions: {sorted(IMAGE_EXTENSIONS)}"
        )

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")

            resized = image.resize(
                IMAGE_SIZE,
                Image.Resampling.LANCZOS,
            )

            array = np.asarray(
                resized,
                dtype=np.float32,
            )

    except Exception as exc:
        raise RuntimeError(
            f"Could not read image:\n{image_path}\n"
            f"Original error: {exc}"
        ) from exc

    array /= 255.0

    array = np.clip(
        array,
        0.0,
        1.0,
    )

    return array, resized


# ============================================================
# Reconstruction
# ============================================================

def reconstruct_image(
    model: keras.Model,
    image_array: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct a normalized RGB image using the autoencoder.
    """

    batch = image_array[None, ...]

    reconstruction = model.predict(
        batch,
        verbose=0,
    )[0]

    reconstruction = np.asarray(
        reconstruction,
        dtype=np.float32,
    )

    reconstruction = np.clip(
        reconstruction,
        0.0,
        1.0,
    )

    return reconstruction


# ============================================================
# Reconstruction metrics
# ============================================================

def calculate_reconstruction_metrics(
    original: np.ndarray,
    reconstruction: np.ndarray,
) -> dict[str, Any]:
    """
    Calculate image-level and pixel-level reconstruction errors.

    Image anomaly score:
        Mean squared reconstruction error.

    Pixel anomaly map:
        Mean squared RGB error at each pixel.
    """

    difference = (
        original - reconstruction
    )

    absolute_error = np.abs(
        difference
    )

    squared_error = np.square(
        difference
    )

    mae_map = np.mean(
        absolute_error,
        axis=-1,
    )

    mse_map = np.mean(
        squared_error,
        axis=-1,
    )

    return {
        "mse": float(
            np.mean(squared_error)
        ),
        "mae": float(
            np.mean(absolute_error)
        ),
        "max_abs_error": float(
            np.max(absolute_error)
        ),
        "mae_map": mae_map,
        "mse_map": mse_map,
    }


# ============================================================
# Threshold discovery
# ============================================================

def find_threshold_files() -> list[Path]:
    """
    Find supported threshold CSV files in the project.
    """

    candidates: list[Path] = []

    seen: set[Path] = set()

    for root in MODEL_SEARCH_ROOTS:
        root = root.resolve()

        if not root.exists():
            continue

        try:
            for candidate in root.rglob("*.csv"):
                if (
                    candidate.name
                    not in THRESHOLD_FILENAMES
                ):
                    continue

                candidate = candidate.resolve()

                if candidate not in seen:
                    seen.add(candidate)
                    candidates.append(candidate)

        except OSError:
            continue

    return sorted(candidates)


def load_saved_threshold(
    category: str,
) -> float | None:
    """
    Load a category-specific image anomaly threshold.

    Supported CSV columns:
        Threshold
        99th Percentile Error
        99th Percentile
    """

    try:
        import pandas as pd
    except ImportError:
        return None

    for threshold_file in find_threshold_files():

        try:
            frame = pd.read_csv(
                threshold_file
            )

        except Exception:
            continue

        if "Category" not in frame.columns:
            continue

        value_column = None

        for column in (
            "Threshold",
            "99th Percentile Error",
            "99th Percentile",
        ):
            if column in frame.columns:
                value_column = column
                break

        if value_column is None:
            continue

        matches = frame[
            frame["Category"]
            .astype(str)
            .str.casefold()
            .eq(category.casefold())
        ]

        if matches.empty:
            continue

        try:
            value = float(
                matches.iloc[0][value_column]
            )

            if np.isfinite(value):
                return value

        except (TypeError, ValueError):
            continue

    return None


# ============================================================
# Fallback image threshold calibration
# ============================================================

def select_normal_images(
    dataset_root: Path | None,
    category: str,
    limit: int = CALIBRATION_LIMIT,
) -> list[Path]:
    """
    Select deterministic normal/good images for threshold calibration.
    """

    if dataset_root is None:
        return []

    candidates = []

    validation_roots = [
        dataset_root
        / "validation"
        / category
        / "good",

        dataset_root
        / "val"
        / category
        / "good",

        dataset_root
        / category
        / "train"
        / "good",
    ]

    for root in validation_roots:
        files = image_files(root)

        if files:
            candidates.extend(files)

            # Prefer validation data when available.
            if (
                "validation"
                in root.parts
                or root.parent.name == "val"
            ):
                break

    candidates = sorted(
        set(candidates)
    )

    if len(candidates) <= limit:
        return candidates

    rng = np.random.default_rng(
        SEED
    )

    indices = rng.choice(
        len(candidates),
        size=limit,
        replace=False,
    )

    return sorted(
        candidates[index]
        for index in indices
    )


def calibrate_image_threshold(
    model: keras.Model,
    dataset_root: Path | None,
    category: str,
) -> float | None:
    """
    Calculate a fallback image threshold using normal samples.

    Threshold:
        99th percentile of normal-image reconstruction MSE.
    """

    images = select_normal_images(
        dataset_root,
        category,
    )

    if not images:
        return None

    scores: list[float] = []

    for image_path in images:
        try:
            image_array, _ = load_image(
                image_path
            )

            reconstruction = reconstruct_image(
                model,
                image_array,
            )

            metrics = calculate_reconstruction_metrics(
                image_array,
                reconstruction,
            )

            scores.append(
                metrics["mse"]
            )

        except Exception as exc:
            print(
                f"WARNING: Calibration skipped "
                f"for {image_path.name}: {exc}"
            )

    if not scores:
        return None

    threshold = float(
        np.percentile(
            np.asarray(
                scores,
                dtype=np.float64,
            ),
            IMAGE_THRESHOLD_PERCENTILE,
        )
    )

    return threshold


# ============================================================
# Pixel threshold
# ============================================================

def calibrate_pixel_threshold(
    model: keras.Model,
    dataset_root: Path | None,
    category: str,
    limit: int = 10,
) -> float | None:
    """
    Calculate a pixel-level threshold from normal images.

    Threshold:
        99.5th percentile of normal pixel MSE values.
    """

    images = select_normal_images(
        dataset_root,
        category,
        limit=limit,
    )

    if not images:
        return None

    maps = []

    for image_path in images:
        try:
            image_array, _ = load_image(
                image_path
            )

            reconstruction = reconstruct_image(
                model,
                image_array,
            )

            metrics = calculate_reconstruction_metrics(
                image_array,
                reconstruction,
            )

            maps.append(
                metrics["mse_map"].reshape(-1)
            )

        except Exception as exc:
            print(
                f"WARNING: Pixel calibration skipped "
                f"for {image_path.name}: {exc}"
            )

    if not maps:
        return None

    values = np.concatenate(
        maps
    )

    return float(
        np.percentile(
            values,
            99.5,
        )
    )


# ============================================================
# Anomaly classification
# ============================================================

def classify_anomaly(
    score: float,
    threshold: float | None,
) -> str:
    """
    Classify an image using its category-specific threshold.
    """

    if threshold is None:
        return "UNKNOWN"

    return (
        "ANOMALY"
        if score > threshold
        else "NORMAL"
    )


# ============================================================
# Anomaly map and localization
# ============================================================

def normalize_map(
    anomaly_map: np.ndarray,
) -> np.ndarray:
    """
    Normalize an anomaly map to 0-1.
    """

    anomaly_map = np.asarray(
        anomaly_map,
        dtype=np.float32,
    )

    minimum = float(
        np.nanmin(anomaly_map)
    )

    maximum = float(
        np.nanmax(anomaly_map)
    )

    if (
        not np.isfinite(minimum)
        or not np.isfinite(maximum)
        or maximum <= minimum
    ):
        return np.zeros_like(
            anomaly_map,
            dtype=np.float32,
        )

    normalized = (
        anomaly_map - minimum
    ) / (
        maximum - minimum
    )

    return np.clip(
        normalized,
        0.0,
        1.0,
    )


def create_anomaly_mask(
    mse_map: np.ndarray,
    pixel_threshold: float | None,
) -> np.ndarray:
    """
    Convert the pixel anomaly map into a binary anomaly mask.
    """

    if pixel_threshold is not None:
        mask = (
            mse_map
            > pixel_threshold
        )

    else:
        threshold = np.percentile(
            mse_map,
            DEFAULT_PIXEL_PERCENTILE,
        )

        mask = (
            mse_map
            > threshold
        )

    return mask.astype(
        np.uint8
    )


def clean_anomaly_mask(
    mask: np.ndarray,
) -> np.ndarray:
    """
    Perform lightweight morphological cleanup.

    Uses OpenCV when available. Falls back to PIL when OpenCV
    is unavailable.
    """

    mask = (
        np.asarray(mask)
        .astype(np.uint8)
    )

    try:
        import cv2

        kernel = np.ones(
            (
                MORPH_KERNEL_SIZE,
                MORPH_KERNEL_SIZE,
            ),
            dtype=np.uint8,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        return mask

    except ImportError:
        image = Image.fromarray(
            mask * 255
        )

        image = image.filter(
            ImageFilter.MedianFilter(
                size=3
            )
        )

        cleaned = np.asarray(
            image
        ) > 127

        return cleaned.astype(
            np.uint8
        )


def bounding_boxes(
    mask: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """
    Find connected defect regions.

    Returns boxes as:
        (x1, y1, x2, y2)
    """

    mask = (
        np.asarray(mask)
        .astype(np.uint8)
    )

    try:
        import cv2

        num_labels, _, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8,
            )
        )

        boxes = []

        for label in range(
            1,
            num_labels,
        ):
            x = int(
                stats[label, cv2.CC_STAT_LEFT]
            )

            y = int(
                stats[label, cv2.CC_STAT_TOP]
            )

            width = int(
                stats[label, cv2.CC_STAT_WIDTH]
            )

            height = int(
                stats[label, cv2.CC_STAT_HEIGHT]
            )

            area = int(
                stats[label, cv2.CC_STAT_AREA]
            )

            if area < MIN_COMPONENT_AREA:
                continue

            boxes.append(
                (
                    x,
                    y,
                    x + width,
                    y + height,
                )
            )

        return sorted(
            boxes,
            key=lambda box: (
                box[1],
                box[0],
            ),
        )

    except ImportError:
        return []


# ============================================================
# Visualization
# ============================================================

def array_to_image(
    array: np.ndarray,
) -> Image.Image:
    """
    Convert a normalized image array to RGB PIL image.
    """

    array = np.clip(
        array,
        0.0,
        1.0,
    )

    return Image.fromarray(
        (
            array * 255
        ).astype(np.uint8)
    ).convert("RGB")


def create_heatmap(
    anomaly_map: np.ndarray,
) -> Image.Image:
    """
    Create a heatmap-like RGB visualization using PIL.

    No external plotting package is required.
    """

    normalized = normalize_map(
        anomaly_map
    )

    values = (
        normalized * 255
    ).astype(np.uint8)

    try:
        heatmap = Image.fromarray(
            values,
            mode="L",
        ).convert("RGB")

        # Apply a standard PIL palette-like transformation
        # through channel operations.
        red = values

        green = (
            255
            - np.abs(
                values.astype(np.int16)
                - 128
            ) * 2
        ).clip(
            0,
            255,
        ).astype(np.uint8)

        blue = (
            255
            - values
        ).astype(np.uint8)

        heatmap_array = np.stack(
            [
                red,
                green,
                blue,
            ],
            axis=-1,
        )

        heatmap = Image.fromarray(
            heatmap_array,
            mode="RGB",
        )

        return heatmap

    except Exception:
        return Image.fromarray(
            values,
            mode="L",
        ).convert("RGB")


def overlay_heatmap(
    original: Image.Image,
    heatmap: Image.Image,
    alpha: float = 0.45,
) -> Image.Image:
    """
    Blend anomaly heatmap with the input image.
    """

    original = original.convert(
        "RGB"
    ).resize(
        IMAGE_SIZE
    )

    heatmap = heatmap.convert(
        "RGB"
    ).resize(
        IMAGE_SIZE
    )

    return Image.blend(
        original,
        heatmap,
        alpha=alpha,
    )


def draw_bounding_boxes(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
) -> Image.Image:
    """
    Draw predicted defect regions.
    """

    result = image.convert(
        "RGB"
    ).copy()

    draw = ImageDraw.Draw(
        result
    )

    for index, box in enumerate(
        boxes,
        start=1,
    ):
        draw.rectangle(
            box,
            outline=(255, 0, 0),
            width=2,
        )

        label = f"Defect {index}"

        text_position = (
            box[0] + 2,
            max(
                0,
                box[1] - 14,
            ),
        )

        draw.text(
            text_position,
            label,
            fill=(255, 0, 0),
        )

    return result


def create_report_image(
    original: Image.Image,
    reconstruction: Image.Image,
    heatmap: Image.Image,
    localization: Image.Image,
) -> Image.Image:
    """
    Create a 2x2 inference report image.
    """

    width, height = IMAGE_SIZE

    report = Image.new(
        "RGB",
        (
            width * 2,
            height * 2,
            ),
        "white",
    )

    report.paste(
        original,
        (0, 0),
    )

    report.paste(
        reconstruction,
        (width, 0),
    )

    report.paste(
        heatmap,
        (0, height),
    )

    report.paste(
        localization,
        (width, height),
    )

    return report


# ============================================================
# JSON serialization
# ============================================================

def save_json(
    output_path: Path,
    result: dict[str, Any],
) -> None:
    """
    Save inference metadata as JSON.
    """

    serializable = {}

    for key, value in result.items():

        if isinstance(
            value,
            Path,
        ):
            serializable[key] = str(
                value
            )

        elif isinstance(
            value,
            np.integer,
        ):
            serializable[key] = int(
                value
            )

        elif isinstance(
            value,
            np.floating,
        ):
            serializable[key] = float(
                value
            )

        else:
            serializable[key] = value

    output_path.write_text(
        json.dumps(
            serializable,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# Main inference pipeline
# ============================================================

def predict(
    category: str,
    image_path: Path,
    dataset_root: Path | None = None,
    model_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """
    Run complete VisionGuard AI inference.
    """

    print_header(
        "VisionGuard AI — Inference"
    )

    print(
        f"Category: {category}"
    )

    print(
        f"Input image: {image_path}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, selected_model = (
        load_category_model(
            category,
            model_path,
        )
    )

    print(
        f"Model parameters: "
        f"{model.count_params():,}"
    )

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    image_array, resized_image = (
        load_image(
            image_path
        )
    )

    # --------------------------------------------------------
    # Reconstruction
    # --------------------------------------------------------

    print(
        "Running reconstruction..."
    )

    reconstruction = (
        reconstruct_image(
            model,
            image_array,
        )
    )

    metrics = (
        calculate_reconstruction_metrics(
            image_array,
            reconstruction,
        )
    )

    score = metrics["mse"]

    # --------------------------------------------------------
    # Image threshold
    # --------------------------------------------------------

    threshold = (
        load_saved_threshold(
            category
        )
    )

    threshold_source = (
        "saved"
        if threshold is not None
        else "calibrated"
    )

    if threshold is None:
        print(
            "Saved image threshold not found."
        )

        if dataset_root is not None:
            print(
                "Calibrating threshold from normal images..."
            )

        threshold = (
            calibrate_image_threshold(
                model,
                dataset_root,
                category,
            )
        )

    prediction = (
        classify_anomaly(
            score,
            threshold,
        )
    )

    # --------------------------------------------------------
    # Pixel threshold
    # --------------------------------------------------------

    print(
        "Preparing pixel-level anomaly map..."
    )

    pixel_threshold = (
        calibrate_pixel_threshold(
            model,
            dataset_root,
            category,
        )
    )

    anomaly_map = (
        metrics["mse_map"]
    )

    mask = create_anomaly_mask(
        anomaly_map,
        pixel_threshold,
    )

    mask = clean_anomaly_mask(
        mask
    )

    boxes = bounding_boxes(
        mask
    )

    # --------------------------------------------------------
    # Visual outputs
    # --------------------------------------------------------

    reconstruction_image = (
        array_to_image(
            reconstruction
        )
    )

    heatmap = create_heatmap(
        anomaly_map
    )

    heatmap_overlay = (
        overlay_heatmap(
            resized_image,
            heatmap,
        )
    )

    localization = (
        draw_bounding_boxes(
            heatmap_overlay,
            boxes,
        )
    )

    report = create_report_image(
        resized_image,
        reconstruction_image,
        heatmap,
        localization,
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = safe_stem(
        image_path
    )

    base_name = (
        f"{category}_{stem}"
    )

    original_output = (
        output_dir
        / f"{base_name}_original.png"
    )

    reconstruction_output = (
        output_dir
        / f"{base_name}_reconstruction.png"
    )

    heatmap_output = (
        output_dir
        / f"{base_name}_heatmap.png"
    )

    localization_output = (
        output_dir
        / f"{base_name}_localization.png"
    )

    report_output = (
        output_dir
        / f"{base_name}_report.png"
    )

    json_output = (
        output_dir
        / f"{base_name}_result.json"
    )

    resized_image.save(
        original_output
    )

    reconstruction_image.save(
        reconstruction_output
    )

    heatmap.save(
        heatmap_output
    )

    localization.save(
        localization_output
    )

    report.save(
        report_output
    )

    # --------------------------------------------------------
    # Result metadata
    # --------------------------------------------------------

    threshold_margin = None
    threshold_ratio = None

    if threshold is not None:

        threshold_margin = (
            score - threshold
        )

        if threshold != 0:
            threshold_ratio = (
                score / threshold
            )

    result = {
        "project": "VisionGuard AI",
        "category": category,
        "input_image": image_path,
        "model": selected_model,
        "image_size": [
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
        ],
        "prediction": prediction,
        "anomaly_score_mse": score,
        "reconstruction_mae": metrics[
            "mae"
        ],
        "maximum_absolute_error": metrics[
            "max_abs_error"
        ],
        "image_threshold": threshold,
        "threshold_source": threshold_source,
        "threshold_ratio": threshold_ratio,
        "threshold_margin": threshold_margin,
        "pixel_threshold": pixel_threshold,
        "detected_regions": len(
            boxes
        ),
        "bounding_boxes": [
            list(box)
            for box in boxes
        ],
        "outputs": {
            "original": original_output,
            "reconstruction": reconstruction_output,
            "heatmap": heatmap_output,
            "localization": localization_output,
            "report": report_output,
            "json": json_output,
        },
    }

    save_json(
        json_output,
        result,
    )

    # --------------------------------------------------------
    # Console result
    # --------------------------------------------------------

    print_header(
        "VisionGuard AI — Prediction Result"
    )

    print(
        f"Category:          {category}"
    )

    print(
        f"Prediction:        {prediction}"
    )

    print(
        f"Anomaly Score:     {score:.8f}"
    )

    if threshold is not None:
        print(
            f"Threshold:         {threshold:.8f}"
        )

        print(
            f"Score / Threshold: "
            f"{threshold_ratio:.2f}x"
        )

        print(
            f"Threshold Margin:  "
            f"{threshold_margin:.8f}"
        )

    else:
        print(
            "Threshold:         Unavailable"
        )

    print(
        f"Reconstruction MAE: "
        f"{metrics['mae']:.8f}"
    )

    print(
        f"Max Absolute Error: "
        f"{metrics['max_abs_error']:.8f}"
    )

    if pixel_threshold is not None:
        print(
            f"Pixel Threshold:   "
            f"{pixel_threshold:.8f}"
        )

    print(
        f"Detected Regions:  "
        f"{len(boxes)}"
    )

    print()
    print(
        "Saved outputs:"
    )

    for key, path in result[
        "outputs"
    ].items():
        print(
            f"  {key}: {path}"
        )

    print()
    print(
        "Inference completed successfully."
    )

    return result


# ============================================================
# Model listing
# ============================================================

def list_models() -> None:
    """
    Display discovered models for all project categories.
    """

    print_header(
        "VisionGuard AI — Available Models"
    )

    for category in CATEGORIES:

        candidates = (
            find_model_candidates(
                category
            )
        )

        print(
            f"\n{category}:"
        )

        if not candidates:
            print(
                "  No model found."
            )
            continue

        for candidate in candidates:
            print(
                f"  - {candidate}"
            )


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "VisionGuard AI standalone "
            "industrial anomaly detection inference."
        )
    )

    parser.add_argument(
        "--category",
        choices=CATEGORIES,
        help=(
            "MVTec category associated with "
            "the trained autoencoder."
        ),
    )

    parser.add_argument(
        "--image",
        type=Path,
        help=(
            "Path to the input image."
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "Optional explicit model path. "
            "If omitted, the script discovers "
            "a category-specific model."
        ),
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Optional prepared dataset root. "
            "Used for threshold calibration when "
            "a saved threshold is unavailable."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory where prediction artifacts "
            "will be saved."
        ),
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help=(
            "List discovered category models "
            "and exit."
        ),
    )

    return parser


def main() -> int:
    parser = build_parser()

    args = parser.parse_args()

    if args.list_models:
        list_models()
        return 0

    if args.category is None:
        parser.error(
            "--category is required unless "
            "--list-models is used."
        )

    if args.image is None:
        parser.error(
            "--image is required unless "
            "--list-models is used."
        )

    dataset_root = (
        find_dataset_root(
            args.dataset_root
        )
    )

    if dataset_root is not None:
        print(
            f"Prepared dataset detected: "
            f"{dataset_root}"
        )
    else:
        print(
            "WARNING: Prepared dataset was not detected. "
            "Inference can still run if a saved threshold "
            "exists, but fallback threshold calibration "
            "will be unavailable."
        )

    try:
        predict(
            category=args.category,
            image_path=args.image,
            dataset_root=dataset_root,
            model_path=args.model,
            output_dir=args.output,
        )

        return 0

    except KeyboardInterrupt:
        print_error(
            "Inference interrupted by user."
        )
        return 130

    except Exception as exc:
        print_error(
            str(exc)
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )