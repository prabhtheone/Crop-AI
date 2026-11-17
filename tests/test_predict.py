from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import src.predict as predictor


class FakeArray(list):
    def astype(self, dtype: str) -> "FakeArray":
        return self


class FakeNumpy:
    @staticmethod
    def expand_dims(array: FakeArray, axis: int) -> list[FakeArray]:
        return [array]


class FakeModel:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls = 0

    def predict(self, image: object, verbose: int = 0) -> list[list[float]]:
        self.calls += 1
        return [self.scores]


class FakeModels:
    def __init__(
        self,
        crop_path: Path,
        quality_path: Path,
        crop_model: FakeModel,
        quality_model: FakeModel,
    ) -> None:
        self.crop_path = crop_path
        self.quality_path = quality_path
        self.crop_model = crop_model
        self.quality_model = quality_model
        self.loaded: list[Path] = []

    def load_model(self, path: Path) -> FakeModel:
        model_path = Path(path)
        self.loaded.append(model_path)
        if model_path == self.crop_path:
            return self.crop_model
        if model_path == self.quality_path:
            return self.quality_model
        raise AssertionError(f"Unexpected model path: {model_path}")


class FakeTensorFlow:
    def __init__(self, models: FakeModels) -> None:
        self.keras = type("Keras", (), {"models": models})()


class PredictionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.image_path = root / "sample.jpg"
        self.crop_path = root / "crop.keras"
        self.quality_path = root / "quality.keras"
        self.image_path.write_bytes(b"image-placeholder")
        self.crop_path.write_bytes(b"crop-model-placeholder")
        self.quality_path.write_bytes(b"quality-model-placeholder")

    def runtime(
        self,
        crop_scores: list[float],
        quality_scores: list[float],
    ) -> tuple[tuple[object, object, object, object], FakeModels, FakeModel, FakeModel]:
        crop_model = FakeModel(crop_scores)
        quality_model = FakeModel(quality_scores)
        models = FakeModels(
            self.crop_path,
            self.quality_path,
            crop_model,
            quality_model,
        )
        tensorflow = FakeTensorFlow(models)
        runtime = (
            FakeNumpy(),
            tensorflow,
            lambda image: FakeArray([1.0, 2.0, 3.0]),
            lambda path, target_size: object(),
        )
        return runtime, models, crop_model, quality_model

    def test_crop_and_guava_quality_prediction(self) -> None:
        runtime, models, crop_model, quality_model = self.runtime(
            [0.05, 0.82, 0.04, 0.04, 0.05],
            [0.10, 0.15, 0.65, 0.10],
        )

        with (
            patch.object(predictor, "CROP_MODEL", self.crop_path),
            patch.object(predictor, "QUALITY_MODEL", self.quality_path),
            patch.object(predictor, "_load_runtime", return_value=runtime),
        ):
            result = predictor.predict(self.image_path)

        self.assertEqual(result["crop"], "Guava")
        self.assertEqual(result["crop_confidence"], 82.0)
        self.assertEqual(result["quality"], "C")
        self.assertEqual(result["quality_confidence"], 65.0)
        self.assertEqual(crop_model.calls, 1)
        self.assertEqual(quality_model.calls, 1)
        self.assertEqual(len(models.loaded), 2)

    def test_non_guava_prediction_does_not_load_quality_model(self) -> None:
        runtime, models, _, quality_model = self.runtime(
            [0.90, 0.02, 0.03, 0.03, 0.02],
            [0.25, 0.25, 0.25, 0.25],
        )

        with (
            patch.object(predictor, "CROP_MODEL", self.crop_path),
            patch.object(predictor, "QUALITY_MODEL", self.quality_path),
            patch.object(predictor, "_load_runtime", return_value=runtime),
        ):
            result = predictor.predict(self.image_path)

        self.assertEqual(result["crop"], "Banana")
        self.assertIsNone(result["quality"])
        self.assertIsNone(result["quality_confidence"])
        self.assertEqual(len(models.loaded), 1)
        self.assertEqual(quality_model.calls, 0)

    def test_low_confidence_guava_does_not_run_quality_model(self) -> None:
        runtime, models, _, quality_model = self.runtime(
            [0.10, 0.45, 0.15, 0.15, 0.15],
            [0.10, 0.20, 0.60, 0.10],
        )

        with (
            patch.object(predictor, "CROP_MODEL", self.crop_path),
            patch.object(predictor, "QUALITY_MODEL", self.quality_path),
            patch.object(predictor, "_load_runtime", return_value=runtime),
        ):
            result = predictor.predict(self.image_path)

        self.assertEqual(result["crop"], "Guava")
        self.assertEqual(result["crop_confidence"], 45.0)
        self.assertIsNone(result["quality"])
        self.assertEqual(len(models.loaded), 1)
        self.assertEqual(quality_model.calls, 0)

    def test_wrong_number_of_scores_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "5 configured classes"):
            predictor._top_prediction([0.2, 0.8], predictor.CROP_CLASSES, "Crop classifier")

    def test_empty_scores_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no prediction scores"):
            predictor._top_prediction([], predictor.CROP_CLASSES, "Crop classifier")

    def test_non_finite_scores_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid prediction scores"):
            predictor._top_prediction(
                [0.2, float("nan"), 0.1, 0.3, 0.4],
                predictor.CROP_CLASSES,
                "Crop classifier",
            )

    def test_missing_image_is_reported_before_loading_runtime(self) -> None:
        missing = Path(self.temp_dir.name) / "missing.jpg"
        with patch.object(predictor, "_load_runtime") as load_runtime:
            with self.assertRaisesRegex(FileNotFoundError, "Image not found"):
                predictor.predict(missing)
        load_runtime.assert_not_called()

    def test_lfs_pointer_is_not_mistaken_for_a_downloaded_model(self) -> None:
        pointer = Path(self.temp_dir.name) / "model.keras"
        pointer.write_text(
            "version https://git-lfs.github.com/spec/v1\n"
            "oid sha256:" + "a" * 64 + "\n"
            "size 12345\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "git lfs pull"):
            predictor._require_model_file(pointer)


if __name__ == "__main__":
    unittest.main()
