"""Run the crop classifier and, for confident Guava predictions, the Guava quality model.

Usage:
    python src/predict.py path/to/image.jpg

The crop model predicts Banana, Guava, Maize, Rice, or Wheat.
Quality grading is currently available for Guava only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]

CROP_MODEL = ROOT / "model" / "CROP_MODEL_CHAMPION_91_76_TEST.keras"
QUALITY_MODEL = ROOT / "model" / "CROP_QUALITY_MODEL_CHAMPION_78_27_TEST.keras"

CROP_CLASSES = ("Banana", "Guava", "Maize", "Rice", "Wheat")
QUALITY_CLASSES = ("A", "B", "C", "Reject")

IMAGE_SIZE = (224, 224)
MIN_GUAVA_CONFIDENCE = 50.0
LFS_POINTER_HEADER = b"version https://git-lfs.github.com/spec/v1"


def _load_runtime() -> tuple[Any, Any, Any, Any]:
    """Load heavy inference dependencies only when prediction is requested."""
    try:
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.utils import img_to_array, load_img
    except ImportError as exc:
        raise RuntimeError(
            "Inference dependencies are missing. Install them with "
            "'python -m pip install -r requirements.txt'."
        ) from exc

    return np, tf, img_to_array, load_img


def _require_model_file(path: Path) -> None:
    """Check that a model exists and its Git LFS object has been downloaded."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {path}. Check the repository model path."
        )

    with path.open("rb") as model_file:
        header = model_file.read(len(LFS_POINTER_HEADER))

    if header == LFS_POINTER_HEADER:
        raise RuntimeError(
            f"{path.name} is only a Git LFS pointer. Install Git LFS and run "
            "'git lfs pull' from the repository directory."
        )


def load_image(
    image_path: str | Path,
    size: tuple[int, int] = IMAGE_SIZE,
) -> Any:
    """Load an image and return a batched float32 array."""
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    np, _, img_to_array, load_img = _load_runtime()
    try:
        image = load_img(str(path), target_size=size)
        array = img_to_array(image).astype("float32")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read image '{path}': {exc}") from exc

    return np.expand_dims(array, axis=0)


def _top_prediction(
    scores: Sequence[float],
    classes: Sequence[str],
    model_name: str,
) -> tuple[str, float]:
    """Map one model output row to a label and percentage confidence."""
    if len(scores) != len(classes):
        raise ValueError(
            f"{model_name} returned {len(scores)} scores for "
            f"{len(classes)} configured classes."
        )

    if len(scores) == 0:
        raise ValueError(f"{model_name} returned no prediction scores.")

    try:
        values = [float(score) for score in scores]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{model_name} returned non-numeric scores.") from exc

    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError(f"{model_name} returned invalid prediction scores.")

    best_index = max(range(len(values)), key=values.__getitem__)
    return classes[best_index], round(values[best_index] * 100, 2)


def _predict_one(model: Any, image: Any, classes: Sequence[str], name: str) -> tuple[str, float]:
    output = model.predict(image, verbose=0)
    try:
        scores = output[0]
    except (IndexError, TypeError) as exc:
        raise ValueError(f"{name} returned an invalid output shape.") from exc

    return _top_prediction(scores, classes, name)


def predict(image_path: str | Path) -> dict[str, str | float | None]:
    """Classify a crop and grade Guava when the crop prediction is confident."""
    image = Path(image_path).expanduser()
    if not image.is_file():
        raise FileNotFoundError(f"Image not found: {image}")

    _require_model_file(CROP_MODEL)

    np, tf, img_to_array, load_img = _load_runtime()
    try:
        loaded_image = load_img(str(image), target_size=IMAGE_SIZE)
        image_array = np.expand_dims(
            img_to_array(loaded_image).astype("float32"),
            axis=0,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read image '{image}': {exc}") from exc

    try:
        crop_model = tf.keras.models.load_model(CROP_MODEL)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Could not load crop model '{CROP_MODEL}': {exc}") from exc

    crop, crop_confidence = _predict_one(
        crop_model, image_array, CROP_CLASSES, "Crop classifier"
    )

    result: dict[str, str | float | None] = {
        "crop": crop,
        "crop_confidence": crop_confidence,
        "quality": None,
        "quality_confidence": None,
    }

    if crop != "Guava" or crop_confidence < MIN_GUAVA_CONFIDENCE:
        return result

    _require_model_file(QUALITY_MODEL)
    try:
        quality_model = tf.keras.models.load_model(QUALITY_MODEL)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Could not load Guava quality model '{QUALITY_MODEL}': {exc}") from exc

    quality, quality_confidence = _predict_one(
        quality_model, image_array, QUALITY_CLASSES, "Guava quality model"
    )
    result["quality"] = quality
    result["quality_confidence"] = quality_confidence
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify a crop image and grade Guava when supported."
    )
    parser.add_argument("image", type=Path, help="Path to an image file")
    args = parser.parse_args(argv)

    try:
        result = predict(args.image)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
