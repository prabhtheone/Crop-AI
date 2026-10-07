"""Inference helper for the Crop AI two-stage pipeline.

Usage:
    python src/predict.py path/to/image.jpg

The crop model predicts:
    Banana, Guava, Maize, Rice, Wheat

The quality model predicts the supported crop-quality class using the
18-class EfficientNetB2 model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array, load_img

ROOT = Path(__file__).resolve().parents[1]

CROP_MODEL = ROOT / "model" / "CROP_MODEL_CHAMPION_91_76_TEST.keras"
QUALITY_MODEL = ROOT / "model" / "QUALITY_MODEL_B2_260_BEST.keras"

CROP_CLASSES = ["Banana", "Guava", "Maize", "Rice", "Wheat"]

QUALITY_CLASSES = [
    "Banana_A", "Banana_B", "Banana_D",
    "Guava_A", "Guava_B", "Guava_D",
    "Maize_A", "Maize_B", "Maize_C", "Maize_D",
    "Rice_A", "Rice_B", "Rice_C", "Rice_D",
    "Wheat_A", "Wheat_B", "Wheat_C", "Wheat_D",
]

CROP_IMG_SIZE = (224, 224)
QUALITY_IMG_SIZE = (260, 260)


def load_image(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    image = load_img(path, target_size=size)
    array = img_to_array(image).astype("float32")
    return np.expand_dims(array, axis=0)


def predict(image_path: str | Path) -> dict:
    crop_model = tf.keras.models.load_model(CROP_MODEL)
    quality_model = tf.keras.models.load_model(QUALITY_MODEL)

    crop_array = load_image(image_path, CROP_IMG_SIZE)

    crop_probs = crop_model.predict(crop_array, verbose=0)[0]
    crop_index = int(np.argmax(crop_probs))
    crop = CROP_CLASSES[crop_index]
    crop_confidence = float(crop_probs[crop_index] * 100)

    quality_array = load_image(image_path, QUALITY_IMG_SIZE)
    quality_probs = quality_model.predict(quality_array, verbose=0)[0]
    quality_index = int(np.argmax(quality_probs))
    quality_label = QUALITY_CLASSES[quality_index]
    quality_confidence = float(quality_probs[quality_index] * 100)

    quality_crop, quality_grade = quality_label.rsplit("_", 1)

    result = {
        "crop": crop,
        "crop_confidence": round(crop_confidence, 2),
        "quality": quality_grade if quality_crop == crop else None,
        "quality_crop": quality_crop,
        "quality_confidence": round(quality_confidence, 2),
    }

    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python src/predict.py path/to/image.jpg")

    image = Path(sys.argv[1])
    if not image.exists():
        raise SystemExit(f"Image not found: {image}")

    print(predict(image))
