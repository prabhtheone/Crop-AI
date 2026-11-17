# Crop AI

A two-stage image-classification project built with TensorFlow/Keras:

1. Classify an image as Banana, Guava, Maize, Rice, or Wheat.
2. When the crop prediction is Guava with at least 50% confidence, run the Guava quality model and return A, B, C, or Reject.

Quality grading is currently available for Guava only. The repository does not contain the later proposed 18-class multi-crop quality model.

## Models

| Model | Architecture | Classes | Reported result |
|---|---|---|---:|
| Crop classifier | EfficientNetB0 | 5 crops | 91.76% test accuracy |
| Guava quality classifier | EfficientNetB0 | A / B / C / Reject | 78.27% test accuracy |

Model files are stored with Git LFS:

- `model/CROP_MODEL_CHAMPION_91_76_TEST.keras`
- `model/CROP_QUALITY_MODEL_CHAMPION_78_27_TEST.keras`

The reported results are from the project's training/evaluation records. The raw datasets and training notebooks are not included here, so the results cannot be independently reproduced from this repository alone. They are not guarantees of real-world performance.

## Setup

Requirements: Python 3.11 is used by CI, Git LFS, and a compatible TensorFlow installation.

```bash
git clone https://github.com/prabhtheone/Crop-AI.git
cd Crop-AI

git lfs install
git lfs pull

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run inference

```bash
python src/predict.py path/to/your/image.jpg
```

The command prints a JSON result containing the crop label and confidence. For Guava predictions at or above 50% confidence, it also returns the Guava quality label and confidence. Other crops, or lower-confidence Guava predictions, return `null` for quality.

The prediction script checks that model files are present and that Git LFS pointers have been replaced by downloaded model objects. When a pointer is detected, run `git lfs pull`.

A notebook walkthrough is available at [notebook/Crop_AI_Inference_Demo.ipynb](notebook/Crop_AI_Inference_Demo.ipynb).

## Tests

Run the repository checks with:

```bash
python -m py_compile src/predict.py
python -m unittest discover -s tests -v
python -m json.tool notebook/Crop_AI_Inference_Demo.ipynb > /dev/null
```

The unit tests use small fake models to check the prediction logic without downloading TensorFlow or the large model files. They do not measure model accuracy or execute the trained Keras models. GitHub Actions runs these checks on pushes and pull requests.

## Dataset notes

The crop-classification dataset is described in the project records as 14,194 images:

| Crop | Images |
|---|---:|
| Wheat | 4,000 |
| Rice | 4,000 |
| Maize | 4,000 |
| Banana | 1,194 |
| Guava | 1,000 |
| **Total** | **14,194** |

The quality model was evaluated on a reported Guava test set of 520 images. The underlying datasets are not included in this repository.

## Limitations

- Quality grading is only implemented for Guava.
- `Reject` is a project-defined label, not an official agricultural grade.
- Confidence values are model scores, not calibrated probabilities.
- Lighting, background, camera, crop variety, and image source can affect results.
- This is a learning/research prototype, not a certified agricultural inspection tool.

Do not use the output as the sole basis for commercial grading, food-safety decisions, or financial decisions.

## Repository layout

```text
Crop-AI/
├── model/                 # Git LFS model files
├── notebook/              # Inference walkthrough
├── src/predict.py         # Two-stage inference
├── tests/                 # Dependency-free prediction tests
├── .github/workflows/ci.yml
├── requirements.txt
├── CONTRIBUTING.md
├── SECURITY.md
└── CITATION.cff
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the test and pull request workflow.

## License

MIT. See [LICENSE](LICENSE).
