
import io
import re
import math
import random
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance, UnidentifiedImageError

import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)

try:
    import tensorflow as tf
    from tensorflow import keras
except Exception:
    tf = None
    keras = None


# ============================================================
# VisionGuard AI — Streamlit Application
# ============================================================

st.set_page_config(
    page_title="VisionGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
BATCH_SIZE = 32

IMAGE_THRESHOLD_PERCENTILE = 99.0
PIXEL_THRESHOLD_PERCENTILE = 99.5

MORPH_KERNEL_SIZE = 3
MIN_COMPONENT_AREA = 12

CALIBRATION_LIMIT = 40
PIXEL_CALIBRATION_LIMIT = 10

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGES_DIR = PROJECT_ROOT / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    ".cache",
    "node_modules",
    ".venv",
    "venv",
    "env",
}


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }
    .subtitle {
        color: #667085;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.35rem;
        font-weight: 750;
        margin-top: 0.8rem;
        margin-bottom: 0.6rem;
    }
    .status-card {
        padding: 1rem;
        border: 1px solid #e4e7ec;
        border-radius: 0.8rem;
        background: #fcfcfd;
    }
    .small-muted {
        color: #667085;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Safe naming and artifact helpers
# ============================================================

def safe_title(title: str) -> str:
    """Convert a figure title into a stable, filesystem-safe filename stem."""
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(title)).strip("_")
    value = re.sub(r"_+", "_", value)
    return value[:180] if value else "visionguard_figure"


def safe_error_text(exc: Exception) -> str:
    """Return an error description without exposing filesystem locations."""
    return f"{type(exc).__name__}. Check that the required VisionGuard artifacts are available."


def save_matplotlib_figure(fig, title: str, dpi: int = 220) -> bool:
    """Save a Matplotlib figure under images using its title."""
    try:
        filename = safe_title(title) + ".png"
        output_file = IMAGES_DIR / filename
        fig.savefig(
            output_file,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        return True
    except Exception:
        return False


def save_plotly_figure(
    fig,
    title: str,
    width: int = 1400,
    height: int = 850,
    scale: int = 2,
) -> dict:
    """Save Plotly HTML and PNG artifacts using the figure title."""
    filename = safe_title(title)
    html_ok = False
    png_ok = False

    try:
        fig.write_html(
            IMAGES_DIR / f"{filename}.html",
            include_plotlyjs="cdn",
        )
        html_ok = True
    except Exception:
        html_ok = False

    try:
        fig.write_image(
            IMAGES_DIR / f"{filename}.png",
            format="png",
            width=width,
            height=height,
            scale=scale,
        )
        png_ok = True
    except Exception:
        png_ok = False

    return {"HTML": html_ok, "PNG": png_ok}


def finalize_matplotlib(fig, title: str) -> None:
    saved = save_matplotlib_figure(fig, title)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    if not saved:
        st.warning("This visualization could not be exported as a PNG.")


def finalize_plotly(fig, title: str) -> None:
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=55, r=35, t=85, b=55),
    )
    result = save_plotly_figure(fig, title)
    st.plotly_chart(fig, use_container_width=True)
    if not result["PNG"]:
        st.info("The interactive Plotly graph is available, but PNG export needs the Kaleido package.")


# ============================================================
# Filesystem discovery
# ============================================================

def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def image_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    try:
        return sorted(
            path
            for path in folder.rglob("*")
            if is_image_file(path)
        )
    except Exception:
        return []


def iter_candidate_dirs(root: Path):
    yield root
    try:
        for child in root.rglob("*"):
            if not child.is_dir():
                continue
            if any(part in EXCLUDED_DIRS for part in child.parts):
                continue
            yield child
    except Exception:
        return


def dataset_contract_score(candidate: Path) -> int:
    score = 0
    for category in CATEGORIES:
        category_root = candidate / category
        if category_root.is_dir():
            score += 1
        for split in ("train", "test", "ground_truth"):
            if (category_root / split).is_dir():
                score += 1
    return score


@st.cache_data(show_spinner=False)
def discover_dataset_root():
    candidates = []

    possible_roots = [
        PROJECT_ROOT,
        Path.cwd(),
    ]

    seen = set()

    for base in possible_roots:
        try:
            for candidate in iter_candidate_dirs(base):
                if candidate in seen:
                    continue
                seen.add(candidate)

                if candidate.name.casefold() == "visionguard_data":
                    if any(
                        (candidate / category / "test").is_dir()
                        for category in CATEGORIES
                    ):
                        candidates.append(candidate)
        except Exception:
            continue

    if not candidates:
        return None

    return max(candidates, key=dataset_contract_score)


def find_model_candidates(category: str) -> list[Path]:
    candidates = []

    preferred = PROJECT_ROOT / "models" / category / "best_model.keras"
    if preferred.is_file():
        candidates.append(preferred)

    roots = [PROJECT_ROOT]
    if Path.cwd() != PROJECT_ROOT:
        roots.append(Path.cwd())

    seen = set()

    for root in roots:
        try:
            for pattern in ("best_model.keras", "*.keras", "*.h5", "*.hdf5"):
                for candidate in root.rglob(pattern):
                    if not candidate.is_file():
                        continue
                    if candidate in seen:
                        continue
                    seen.add(candidate)

                    parent_name = candidate.parent.name.casefold()
                    stem_name = candidate.stem.casefold()
                    category_name = category.casefold()

                    if category_name in parent_name or category_name in stem_name:
                        candidates.append(candidate)
        except Exception:
            continue

    return sorted(set(candidates))


# ============================================================
# TensorFlow runtime
# ============================================================

def tensorflow_ready() -> bool:
    return tf is not None and keras is not None


def runtime_backend() -> str:
    if not tensorflow_ready():
        return "Unavailable"
    try:
        return "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    except Exception:
        return "CPU"


# ============================================================
# Model loading and validation
# ============================================================

@st.cache_resource(show_spinner=False)
def load_category_model(model_file: str):
    if not tensorflow_ready():
        raise RuntimeError("TensorFlow is unavailable.")
    return keras.models.load_model(model_file, compile=False)


@st.cache_data(show_spinner=False)
def model_status_table():
    rows = []

    for category in CATEGORIES:
        candidates = find_model_candidates(category)

        if candidates:
            preferred = [
                item
                for item in candidates
                if item.name == "best_model.keras"
            ]
            selected = preferred[0] if preferred else candidates[0]

            try:
                model = load_category_model(str(selected))
                input_shape = tuple(model.input_shape)
                output_shape = tuple(model.output_shape)

                valid = (
                    input_shape[-3:] == (
                        IMAGE_SIZE[0],
                        IMAGE_SIZE[1],
                        IMAGE_CHANNELS,
                    )
                    and output_shape[-3:] == (
                        IMAGE_SIZE[0],
                        IMAGE_SIZE[1],
                        IMAGE_CHANNELS,
                    )
                )

                rows.append(
                    {
                        "Category": category,
                        "Available": True,
                        "Compatible": valid,
                        "Layers": len(model.layers),
                        "Parameters": int(model.count_params()),
                    }
                )
            except Exception:
                rows.append(
                    {
                        "Category": category,
                        "Available": True,
                        "Compatible": False,
                        "Layers": np.nan,
                        "Parameters": np.nan,
                    }
                )
        else:
            rows.append(
                {
                    "Category": category,
                    "Available": False,
                    "Compatible": False,
                    "Layers": np.nan,
                    "Parameters": np.nan,
                }
            )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def available_model_categories() -> list[str]:
    table = model_status_table()
    return table.loc[
        table["Available"].eq(True)
        & table["Compatible"].eq(True),
        "Category",
    ].tolist()


# ============================================================
# Dataset status and test manifest
# ============================================================

def dataset_status_table(root: Path | None):
    rows = []

    for category in CATEGORIES:
        category_root = root / category if root else None

        rows.append(
            {
                "Category": category,
                "Available": bool(category_root and category_root.is_dir()),
                "Train": bool(
                    category_root
                    and (category_root / "train").is_dir()
                ),
                "Test": bool(
                    category_root
                    and (category_root / "test").is_dir()
                ),
                "Ground Truth": bool(
                    category_root
                    and (category_root / "ground_truth").is_dir()
                ),
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def build_test_manifest(root: Path | None):
    if root is None:
        return pd.DataFrame()

    records = []

    for category in CATEGORIES:
        test_root = root / category / "test"

        if not test_root.is_dir():
            continue

        defect_dirs = sorted(
            directory
            for directory in test_root.iterdir()
            if directory.is_dir()
        )

        for defect_dir in defect_dirs:
            defect_type = defect_dir.name
            files = image_files(defect_dir)

            for index, image_path in enumerate(files, start=1):
                records.append(
                    {
                        "Category": category,
                        "Defect Type": defect_type,
                        "Class": (
                            "Normal"
                            if defect_type.casefold() == "good"
                            else "Defective"
                        ),
                        "Sample": f"Sample {index}",
                        "Image Name": image_path.stem,
                        "_image_path": str(image_path),
                    }
                )

    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def test_distribution(manifest: pd.DataFrame):
    if manifest.empty:
        return pd.DataFrame()

    return (
        manifest.groupby(
            ["Category", "Class"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "Images"})
    )


@st.cache_data(show_spinner=False)
def defect_distribution(manifest: pd.DataFrame):
    if manifest.empty:
        return pd.DataFrame()

    return (
        manifest.groupby(
            ["Category", "Defect Type"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "Images"})
    )


# ============================================================
# Threshold loading and deterministic calibration
# ============================================================

def candidate_threshold_files():
    names = {
        "primary_thresholds.csv",
        "thresholds.csv",
        "anomaly_thresholds.csv",
        "reference_reconstruction_thresholds.csv",
    }

    candidates = []

    for root in [PROJECT_ROOT, Path.cwd()]:
        try:
            for candidate in root.rglob("*.csv"):
                if candidate.name in names:
                    candidates.append(candidate)
        except Exception:
            continue

    return sorted(set(candidates))


@st.cache_data(show_spinner=False)
def load_reference_thresholds():
    mapping = {}

    for candidate in candidate_threshold_files():
        try:
            frame = pd.read_csv(candidate)

            if not {"Category"}.issubset(frame.columns):
                continue

            if "Threshold" in frame.columns:
                values = frame["Threshold"]
            elif "99th Percentile Error" in frame.columns:
                values = frame["99th Percentile Error"]
            elif "99th Percentile" in frame.columns:
                values = frame["99th Percentile"]
            else:
                continue

            for category, value in zip(
                frame["Category"].astype(str),
                values,
            ):
                try:
                    mapping[category] = float(value)
                except Exception:
                    continue

            if mapping:
                return mapping
        except Exception:
            continue

    return mapping


def select_normal_calibration_images(
    root: Path | None,
    category: str,
    limit: int = CALIBRATION_LIMIT,
) -> list[Path]:
    if root is None:
        return []

    candidates = []

    validation_roots = [
        root / "validation" / category / "good",
        root / "val" / category / "good",
    ]

    for validation_root in validation_roots:
        candidates.extend(image_files(validation_root))

    if not candidates:
        candidates = image_files(
            root / category / "train" / "good"
        )

    candidates = sorted(set(candidates))

    if len(candidates) <= limit:
        return candidates

    rng = np.random.default_rng(SEED)
    indices = rng.choice(
        len(candidates),
        size=limit,
        replace=False,
    )

    return sorted(
        candidates[index]
        for index in indices
    )


def load_image_array(source, image_size=IMAGE_SIZE) -> np.ndarray:
    if isinstance(source, (str, Path)):
        with Image.open(source) as image:
            image = image.convert("RGB")
            image = image.resize(
                image_size,
                Image.Resampling.LANCZOS,
            )
            array = np.asarray(
                image,
                dtype=np.float32,
            )
    else:
        image = Image.open(io.BytesIO(source))
        image = image.convert("RGB")
        image = image.resize(
            image_size,
            Image.Resampling.LANCZOS,
        )
        array = np.asarray(
            image,
            dtype=np.float32,
        )

    array /= 255.0
    return np.clip(array, 0.0, 1.0)


def reconstruction_metrics(
    original: np.ndarray,
    reconstruction: np.ndarray,
):
    difference = original - reconstruction
    absolute_error = np.abs(difference)
    squared_error = np.square(difference)

    mae_map = np.mean(
        absolute_error,
        axis=-1,
    )
    mse_map = np.mean(
        squared_error,
        axis=-1,
    )

    return {
        "mse": float(np.mean(squared_error)),
        "mae": float(np.mean(absolute_error)),
        "max_abs_error": float(np.max(absolute_error)),
        "mae_map": mae_map,
        "mse_map": mse_map,
    }


def predict_array(model, image_array):
    reconstruction = model.predict(
        image_array[None, ...],
        verbose=0,
    )[0]

    reconstruction = np.clip(
        reconstruction,
        0.0,
        1.0,
    )

    metrics = reconstruction_metrics(
        image_array,
        reconstruction,
    )

    return reconstruction, metrics


@st.cache_data(show_spinner=False)
def calibrate_image_threshold(root: Path | None, category: str):
    model_candidates = find_model_candidates(category)
    if not model_candidates:
        return np.nan

    model_file = model_candidates[0]

    try:
        model = load_category_model(str(model_file))
    except Exception:
        return np.nan

    images = select_normal_calibration_images(
        root,
        category,
        CALIBRATION_LIMIT,
    )

    if not images:
        return np.nan

    scores = []

    for image_path in images:
        try:
            image_array = load_image_array(image_path)
            _, metrics = predict_array(model, image_array)
            scores.append(metrics["mse"])
        except Exception:
            continue

    if not scores:
        return np.nan

    return float(
        np.percentile(
            np.asarray(scores, dtype=np.float64),
            IMAGE_THRESHOLD_PERCENTILE,
        )
    )


@st.cache_data(show_spinner=False)
def calibrate_pixel_threshold(
    root: Path | None,
    category: str,
):
    model_candidates = find_model_candidates(category)
    if not model_candidates:
        return np.nan

    try:
        model = load_category_model(str(model_candidates[0]))
    except Exception:
        return np.nan

    images = select_normal_calibration_images(
        root,
        category,
        PIXEL_CALIBRATION_LIMIT,
    )

    if not images:
        return np.nan

    samples = []

    for image_path in images:
        try:
            image_array = load_image_array(image_path)
            _, metrics = predict_array(model, image_array)
            samples.append(
                metrics["mse_map"].reshape(-1)
            )
        except Exception:
            continue

    if not samples:
        return np.nan

    values = np.concatenate(samples)

    return float(
        np.percentile(
            values,
            PIXEL_THRESHOLD_PERCENTILE,
        )
    )


@st.cache_data(show_spinner=False)
def threshold_table(root: Path | None):
    reference_map = load_reference_thresholds()
    rows = []

    for category in CATEGORIES:
        threshold = reference_map.get(category, np.nan)
        method = "Reference artifact"

        if not np.isfinite(threshold):
            threshold = calibrate_image_threshold(
                root,
                category,
            )
            method = (
                f"Normal calibration P{IMAGE_THRESHOLD_PERCENTILE:g}"
            )

        rows.append(
            {
                "Category": category,
                "Threshold": threshold,
                "Method": method
                if np.isfinite(threshold)
                else "Unavailable",
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Ground-truth handling and localization
# ============================================================

def find_ground_truth_mask(
    root: Path | None,
    category: str,
    defect_type: str,
    image_path: Path,
):
    if root is None:
        return None

    if defect_type.casefold() == "good":
        return None

    ground_truth_root = (
        root
        / category
        / "ground_truth"
        / defect_type
    )

    if not ground_truth_root.is_dir():
        return None

    stem = image_path.stem

    preferred = [
        ground_truth_root / f"{stem}_mask.png",
        ground_truth_root / f"{stem}_mask.jpg",
        ground_truth_root / f"{stem}_mask.jpeg",
    ]

    for candidate in preferred:
        if candidate.is_file():
            return candidate

    candidates = image_files(ground_truth_root)

    matches = [
        candidate
        for candidate in candidates
        if candidate.stem.casefold().startswith(
            stem.casefold()
        )
    ]

    return matches[0] if matches else None


def load_mask(mask_path):
    if mask_path is None:
        return None

    try:
        with Image.open(mask_path) as mask:
            mask = mask.convert("L")
            mask = mask.resize(
                IMAGE_SIZE,
                Image.Resampling.NEAREST,
            )
            return np.asarray(mask) > 0
    except Exception:
        return None


def clean_anomaly_mask(binary_mask):
    binary_uint8 = (
        np.asarray(binary_mask, dtype=np.uint8) * 255
    )

    try:
        import cv2
    except Exception:
        return binary_uint8 > 0

    kernel = np.ones(
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
        dtype=np.uint8,
    )

    cleaned = cv2.morphologyEx(
        binary_uint8,
        cv2.MORPH_OPEN,
        kernel,
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel,
    )

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned,
        connectivity=8,
    )

    filtered = np.zeros_like(cleaned)

    for label in range(1, component_count):
        area = int(
            stats[label, cv2.CC_STAT_AREA]
        )
        if area >= MIN_COMPONENT_AREA:
            filtered[labels == label] = 255

    return filtered > 0


def bounding_boxes(binary_mask):
    try:
        import cv2
    except Exception:
        ys, xs = np.where(binary_mask)
        if len(xs) == 0:
            return []
        return [
            (
                int(xs.min()),
                int(ys.min()),
                int(xs.max()),
                int(ys.max()),
            )
        ]

    binary = (
        np.asarray(binary_mask, dtype=np.uint8)
        * 255
    )

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    boxes = []

    for label in range(1, component_count):
        area = int(
            stats[label, cv2.CC_STAT_AREA]
        )
        if area < MIN_COMPONENT_AREA:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        boxes.append(
            (
                x,
                y,
                x + w - 1,
                y + h - 1,
            )
        )

    return boxes


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.42,
):
    result = (
        np.asarray(image, dtype=np.float32)
        .copy()
    )

    overlay = np.zeros_like(result)
    overlay[..., 0] = 1.0
    overlay[..., 1] = 0.18
    overlay[..., 2] = 0.18

    mask_bool = np.asarray(mask).astype(bool)
    result[mask_bool] = (
        (1.0 - alpha) * result[mask_bool]
        + alpha * overlay[mask_bool]
    )

    return np.clip(result, 0.0, 1.0)


def iou_score(pred_mask, true_mask):
    pred = np.asarray(pred_mask).astype(bool)
    true = np.asarray(true_mask).astype(bool)

    union = np.logical_or(pred, true).sum()
    if union == 0:
        return 1.0

    intersection = np.logical_and(pred, true).sum()
    return float(intersection / union)


def dice_score(pred_mask, true_mask):
    pred = np.asarray(pred_mask).astype(bool)
    true = np.asarray(true_mask).astype(bool)

    denominator = pred.sum() + true.sum()
    if denominator == 0:
        return 1.0

    intersection = np.logical_and(pred, true).sum()
    return float(
        2.0 * intersection / denominator
    )


def localization_from_metrics(
    metrics,
    pixel_threshold: float,
):
    if not np.isfinite(pixel_threshold):
        return None, None, None

    anomaly_map = metrics["mse_map"]

    raw_mask = (
        anomaly_map > pixel_threshold
    )

    clean_mask = clean_anomaly_mask(
        raw_mask
    )

    boxes = bounding_boxes(clean_mask)

    return (
        anomaly_map,
        clean_mask,
        boxes,
    )


# ============================================================
# Plot builders
# ============================================================

def plot_reconstruction_gallery(
    original,
    reconstruction,
    anomaly_map,
    title,
):
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(reconstruction)
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")

    image = axes[2].imshow(
        anomaly_map,
        cmap="magma",
    )
    axes[2].set_title("Reconstruction Error Map")
    axes[2].axis("off")
    figure.colorbar(
        image,
        ax=axes[2],
        fraction=0.046,
        pad=0.04,
    )

    figure.suptitle(title, fontsize=15)
    figure.tight_layout()
    finalize_matplotlib(
        figure,
        title,
    )


def plot_localization_result(
    original,
    anomaly_map,
    predicted_mask,
    ground_truth,
    boxes,
    title,
):
    figure, axes = plt.subplots(
        1,
        4 if ground_truth is not None else 3,
        figsize=(
            19 if ground_truth is not None else 15,
            5,
        ),
    )

    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(
        anomaly_map,
        cmap="magma",
    )
    axes[1].set_title("Anomaly Heatmap")
    axes[1].axis("off")

    overlay = overlay_mask(
        original,
        predicted_mask,
    )

    axes[2].imshow(overlay)

    for x1, y1, x2, y2 in boxes:
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1 + 1,
            y2 - y1 + 1,
            fill=False,
            linewidth=2.2,
        )
        axes[2].add_patch(rect)

    axes[2].set_title(
        f"Predicted Regions ({len(boxes)})"
    )
    axes[2].axis("off")

    if ground_truth is not None:
        axes[3].imshow(ground_truth, cmap="gray")
        axes[3].set_title("Ground Truth")
        axes[3].axis("off")

    figure.suptitle(
        title,
        fontsize=15,
    )
    figure.tight_layout()
    finalize_matplotlib(
        figure,
        title,
    )


# ============================================================
# Main dashboard
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ VisionGuard AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Industrial Anomaly Detection & Defect Localization</div>',
    unsafe_allow_html=True,
)

dataset_root = discover_dataset_root()
dataset_df = dataset_status_table(dataset_root)
manifest_df = build_test_manifest(dataset_root)
model_df = model_status_table()
threshold_df = threshold_table(dataset_root)

ready_categories = available_model_categories()

with st.sidebar:
    st.header("VisionGuard Controls")

    st.caption(
        "Five-category MVTec AD inference dashboard"
    )

    backend = runtime_backend()
    st.metric(
        "Inference Backend",
        backend,
    )

    st.divider()

    page = st.radio(
        "Workspace",
        [
            "Dashboard",
            "Analyze Image",
            "Dataset Explorer",
            "Model & Thresholds",
            "About",
        ],
    )

    st.divider()

    st.caption(
        "The application automatically discovers the prepared dataset and compatible category models."
    )


# ============================================================
# Dashboard
# ============================================================

if page == "Dashboard":
    st.markdown(
        '<div class="section-title">System Overview</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Categories",
            len(ready_categories),
        )

    with col2:
        total_tests = (
            int(len(manifest_df))
            if not manifest_df.empty
            else 0
        )
        st.metric(
            "Test Images",
            f"{total_tests:,}",
        )

    with col3:
        if not threshold_df.empty:
            valid_thresholds = int(
                threshold_df["Threshold"]
                .apply(np.isfinite)
                .sum()
            )
        else:
            valid_thresholds = 0

        st.metric(
            "Thresholds Ready",
            f"{valid_thresholds}/{len(CATEGORIES)}",
        )

    with col4:
        st.metric(
            "Image Resolution",
            "128 × 128",
        )

    st.divider()

    left, right = st.columns([1.05, 1])

    with left:
        st.markdown(
            '<div class="section-title">Model Readiness</div>',
            unsafe_allow_html=True,
        )

        display_model_df = model_df.copy()
        display_model_df["Status"] = np.where(
            display_model_df["Available"]
            & display_model_df["Compatible"],
            "Ready",
            np.where(
                display_model_df["Available"],
                "Incompatible",
                "Missing",
            ),
        )

        st.dataframe(
            display_model_df[
                [
                    "Category",
                    "Status",
                    "Layers",
                    "Parameters",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.markdown(
            '<div class="section-title">Dataset Readiness</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(
            dataset_df[
                [
                    "Category",
                    "Available",
                    "Train",
                    "Test",
                    "Ground Truth",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    if not manifest_df.empty:
        st.divider()
        st.markdown(
            '<div class="section-title">Test Distribution</div>',
            unsafe_allow_html=True,
        )

        distribution_df = test_distribution(
            manifest_df
        )

        fig = px.bar(
            distribution_df,
            x="Category",
            y="Images",
            color="Class",
            barmode="group",
            text="Images",
            title="VisionGuard AI — Normal vs Defective Test Distribution",
        )
        fig.update_traces(textposition="outside")

        finalize_plotly(
            fig,
            "VisionGuard AI — Normal vs Defective Test Distribution",
        )

        defect_df = defect_distribution(
            manifest_df
        )

        fig2 = px.bar(
            defect_df,
            x="Category",
            y="Images",
            color="Defect Type",
            barmode="stack",
            text="Images",
            title="VisionGuard AI — Test Images by Defect Type",
        )
        fig2.update_traces(textposition="inside")

        finalize_plotly(
            fig2,
            "VisionGuard AI — Test Images by Defect Type",
        )


# ============================================================
# Analyze Image
# ============================================================

elif page == "Analyze Image":
    st.markdown(
        '<div class="section-title">Inference & Defect Localization</div>',
        unsafe_allow_html=True,
    )

    if not ready_categories:
        st.error(
            "No compatible trained category models are available."
        )
        st.stop()

    input_mode = st.radio(
        "Input source",
        [
            "Upload image",
            "Use an MVTec test sample",
        ],
        horizontal=True,
    )

    selected_category = st.selectbox(
        "Category",
        ready_categories,
    )

    source_bytes = None
    selected_image_path = None
    ground_truth = None
    defect_type = "Unknown"

    if input_mode == "Upload image":
        uploaded = st.file_uploader(
            "Upload an industrial image",
            type=["png", "jpg", "jpeg"],
        )

        if uploaded is not None:
            source_bytes = uploaded.getvalue()
            defect_type = "Unknown"

    else:
        if manifest_df.empty:
            st.warning(
                "No test images were discovered."
            )
            st.stop()

        category_manifest = manifest_df[
            manifest_df["Category"].eq(
                selected_category
            )
        ].copy()

        if category_manifest.empty:
            st.warning(
                "No test samples are available for this category."
            )
            st.stop()

        defect_types = sorted(
            category_manifest["Defect Type"]
            .unique()
            .tolist()
        )

        selected_defect = st.selectbox(
            "Defect type",
            defect_types,
        )

        sample_manifest = category_manifest[
            category_manifest["Defect Type"].eq(
                selected_defect
            )
        ].copy()

        selected_sample = st.selectbox(
            "Test sample",
            sample_manifest["Sample"].tolist(),
        )

        selected_row = sample_manifest[
            sample_manifest["Sample"].eq(
                selected_sample
            )
        ].iloc[0]

        selected_image_path = Path(
            selected_row["_image_path"]
        )
        defect_type = str(
            selected_row["Defect Type"]
        )

        ground_truth_path = find_ground_truth_mask(
            dataset_root,
            selected_category,
            defect_type,
            selected_image_path,
        )

        ground_truth = load_mask(
            ground_truth_path
        )

    if (source_bytes is None) and (
        selected_image_path is None
    ):
        st.info(
            "Choose an image source to start inference."
        )
    else:
        try:
            model_candidates = find_model_candidates(
                selected_category
            )

            if not model_candidates:
                st.error(
                    "The selected category model is unavailable."
                )
                st.stop()

            model = load_category_model(
                str(model_candidates[0])
            )

            image_array = load_image_array(
                source_bytes
                if source_bytes is not None
                else selected_image_path
            )

            with st.spinner(
                "Running VisionGuard inference..."
            ):
                reconstruction, metrics = predict_array(
                    model,
                    image_array,
                )

            row = threshold_df[
                threshold_df["Category"].eq(
                    selected_category
                )
            ]

            if row.empty:
                image_threshold = np.nan
            else:
                image_threshold = float(
                    row["Threshold"].iloc[0]
                )

            score = metrics["mse"]

            if np.isfinite(image_threshold):
                prediction = (
                    "ANOMALY"
                    if score > image_threshold
                    else "NORMAL"
                )
                ratio = (
                    score / image_threshold
                    if image_threshold != 0
                    else np.nan
                )
                margin = (
                    score - image_threshold
                )
            else:
                prediction = "UNKNOWN"
                ratio = np.nan
                margin = np.nan

            pixel_threshold = calibrate_pixel_threshold(
                dataset_root,
                selected_category,
            )

            anomaly_map, predicted_mask, boxes = (
                localization_from_metrics(
                    metrics,
                    pixel_threshold,
                )
            )

            if predicted_mask is None:
                predicted_mask = (
                    metrics["mse_map"]
                    > np.nanpercentile(
                        metrics["mse_map"],
                        99.0,
                    )
                )
                predicted_mask = clean_anomaly_mask(
                    predicted_mask
                )
                boxes = bounding_boxes(
                    predicted_mask
                )
                anomaly_map = metrics["mse_map"]

            st.divider()

            m1, m2, m3, m4, m5 = st.columns(5)

            with m1:
                st.metric(
                    "Prediction",
                    prediction,
                )

            with m2:
                st.metric(
                    "Anomaly Score",
                    f"{score:.7f}",
                )

            with m3:
                if np.isfinite(image_threshold):
                    st.metric(
                        "Threshold",
                        f"{image_threshold:.7f}",
                    )
                else:
                    st.metric(
                        "Threshold",
                        "Unavailable",
                    )

            with m4:
                if np.isfinite(ratio):
                    st.metric(
                        "Score / Threshold",
                        f"{ratio:.2f}×",
                    )
                else:
                    st.metric(
                        "Score / Threshold",
                        "—",
                    )

            with m5:
                st.metric(
                    "Regions",
                    len(boxes),
                )

            status_text = (
                f"Category: {selected_category}  •  "
                f"Sample type: {defect_type}"
            )

            st.caption(status_text)

            if prediction == "ANOMALY":
                st.error(
                    "VisionGuard AI classified this image as anomalous."
                )
            elif prediction == "NORMAL":
                st.success(
                    "VisionGuard AI classified this image as normal."
                )
            else:
                st.warning(
                    "An image-level threshold is not currently available."
                )

            if ground_truth is not None:
                st.divider()
                gt_iou = iou_score(
                    predicted_mask,
                    ground_truth,
                )
                gt_dice = dice_score(
                    predicted_mask,
                    ground_truth,
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "Ground Truth IoU",
                        f"{gt_iou:.3f}",
                    )

                with c2:
                    st.metric(
                        "Ground Truth Dice",
                        f"{gt_dice:.3f}",
                    )

                with c3:
                    ground_truth_label = (
                        "Normal"
                        if defect_type.casefold()
                        == "good"
                        else "Defective"
                    )
                    st.metric(
                        "Ground Truth Class",
                        ground_truth_label,
                    )

            st.divider()

            plot_reconstruction_gallery(
                image_array,
                reconstruction,
                metrics["mse_map"],
                f"VisionGuard AI — {selected_category} Reconstruction Analysis",
            )

            plot_localization_result(
                image_array,
                anomaly_map,
                predicted_mask,
                ground_truth,
                boxes,
                f"VisionGuard AI — {selected_category} Defect Localization",
            )

            comparison = pd.DataFrame(
                {
                    "Metric": [
                        "Reconstruction MSE",
                        "Reconstruction MAE",
                        "Maximum Absolute Error",
                        "Image Threshold",
                        "Threshold Ratio",
                        "Threshold Margin",
                        "Pixel Threshold",
                        "Detected Regions",
                    ],
                    "Value": [
                        score,
                        metrics["mae"],
                        metrics["max_abs_error"],
                        image_threshold,
                        ratio,
                        margin,
                        pixel_threshold,
                        len(boxes),
                    ],
                }
            )

            st.markdown(
                '<div class="section-title">Inference Details</div>',
                unsafe_allow_html=True,
            )

            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True,
            )

        except Exception as exc:
            st.error(
                f"Inference failed safely: {safe_error_text(exc)}"
            )


# ============================================================
# Dataset Explorer
# ============================================================

elif page == "Dataset Explorer":
    st.markdown(
        '<div class="section-title">MVTec Test Explorer</div>',
        unsafe_allow_html=True,
    )

    if manifest_df.empty:
        st.warning(
            "No test images are currently available."
        )
    else:
        category_filter = st.selectbox(
            "Category filter",
            ["All"] + CATEGORIES,
        )

        filtered = manifest_df.copy()

        if category_filter != "All":
            filtered = filtered[
                filtered["Category"].eq(
                    category_filter
                )
            ]

        st.dataframe(
            filtered[
                [
                    "Category",
                    "Defect Type",
                    "Class",
                    "Sample",
                    "Image Name",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        distribution_df = (
            filtered.groupby(
                ["Category", "Defect Type"],
                as_index=False,
            )
            .size()
            .rename(columns={"size": "Images"})
        )

        if not distribution_df.empty:
            fig = px.bar(
                distribution_df,
                x="Defect Type",
                y="Images",
                color="Category",
                barmode="group",
                text="Images",
                title="VisionGuard AI — Filtered Test Defect Distribution",
            )
            fig.update_traces(
                textposition="outside"
            )

            finalize_plotly(
                fig,
                "VisionGuard AI — Filtered Test Defect Distribution",
            )

        if category_filter != "All":
            selected_rows = filtered[
                filtered["Category"].eq(category_filter)
            ]

            defect_options = sorted(
                selected_rows["Defect Type"]
                .unique()
                .tolist()
            )

            chosen_defect = st.selectbox(
                "Preview defect type",
                defect_options,
            )

            chosen_rows = selected_rows[
                selected_rows["Defect Type"].eq(
                    chosen_defect
                )
            ]

            if not chosen_rows.empty:
                sample = chosen_rows.iloc[0]
                sample_path = Path(
                    sample["_image_path"]
                )

                try:
                    display_image = Image.open(
                        sample_path
                    ).convert("RGB")

                    st.image(
                        display_image,
                        caption=(
                            f"{category_filter} • "
                            f"{chosen_defect}"
                        ),
                        use_container_width=True,
                    )
                except Exception:
                    st.warning(
                        "The selected preview image could not be displayed."
                    )


# ============================================================
# Model & Thresholds
# ============================================================

elif page == "Model & Thresholds":
    st.markdown(
        '<div class="section-title">Model Contract</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        model_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<div class="section-title">Category-Specific Thresholds</div>',
        unsafe_allow_html=True,
    )

    threshold_display = threshold_df.copy()
    threshold_display["Threshold"] = threshold_display[
        "Threshold"
    ].apply(
        lambda value: (
            f"{value:.8f}"
            if np.isfinite(value)
            else "Unavailable"
        )
    )

    st.dataframe(
        threshold_display,
        use_container_width=True,
        hide_index=True,
    )

    finite_thresholds = threshold_df[
        threshold_df["Threshold"].apply(np.isfinite)
    ].copy()

    if not finite_thresholds.empty:
        fig = px.bar(
            finite_thresholds,
            x="Category",
            y="Threshold",
            text="Threshold",
            title="VisionGuard AI — Category-Specific Anomaly Thresholds",
        )

        fig.update_traces(
            texttemplate="%{y:.7f}",
            textposition="outside",
        )

        finalize_plotly(
            fig,
            "VisionGuard AI — Category-Specific Anomaly Thresholds",
        )


# ============================================================
# About
# ============================================================

else:
    st.markdown(
        '<div class="section-title">About VisionGuard AI</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        VisionGuard AI is a reconstruction-based industrial anomaly
        detection application built around category-specific CNN
        autoencoders trained on normal MVTec AD images.

        The application supports:

        - Category-specific TensorFlow/Keras model inference
        - Reconstruction-error anomaly scoring
        - Category-specific image-level thresholds
        - Pixel-level anomaly maps
        - Morphological cleanup
        - Defect region bounding boxes
        - Ground-truth comparison for available test samples
        - IoU and Dice localization metrics
        - Interactive Plotly dashboard visualizations
        - PNG and HTML figure export using title-derived filenames
        """
    )

    st.divider()

    st.markdown(
        '<div class="section-title">Inference Contract</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Component": "Input",
                    "Value": "RGB industrial image",
                },
                {
                    "Component": "Resolution",
                    "Value": "128 × 128",
                },
                {
                    "Component": "Normalization",
                    "Value": "Pixel values scaled to 0–1",
                },
                {
                    "Component": "Image Score",
                    "Value": "Mean squared reconstruction error",
                },
                {
                    "Component": "Image Threshold",
                    "Value": "Category-specific normal-image P99",
                },
                {
                    "Component": "Pixel Threshold",
                    "Value": "Category-specific normal-pixel P99.5",
                },
                {
                    "Component": "Localization",
                    "Value": "Error-map threshold + morphology",
                },
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "The app does not require manual absolute paths. It searches the project environment for the prepared dataset, compatible category models, and available threshold artifacts."
    )

# ============================================================
# Footer
# ============================================================

st.divider()
st.caption(
    "VisionGuard AI • TensorFlow/Keras • MVTec AD • Reconstruction-based anomaly detection"
)
