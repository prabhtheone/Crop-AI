# 🌾 Crop AI — Intelligent Crop Classification & Quality Grading

### Computer Vision for Crop Recognition + Guava Quality Analysis 🤖🌱

Crop AI is a deep-learning agricultural computer-vision project using a two-stage pipeline:

1. **Crop classification:** Banana, Guava, Maize, Rice, or Wheat.
2. **Quality grading:** when the predicted crop is Guava, a second model predicts **A / B / C / Reject**.

> **Important:** Quality grading is currently available for Guava only we are workint to increase the number of crops on grading model.

## 🚀 Pipeline

```text
📷 Crop Image
      ↓
🖼️ 224 × 224
      ↓
🧠 EfficientNetB0 Crop Classifier
      ↓
🌾 Banana / Guava / Maize / Rice / Wheat
      ↓
🍈 If Guava → EfficientNetB0 Quality Model
      ↓
🏷️ A / B / C / Reject
```

## 🧠 Released Models

| Model | Architecture | Classes | Recorded Test Accuracy |
|---|---|---|---:|
| Crop Classification Champion | EfficientNetB0 | 5 crops | **91.76%** |
| Guava Quality Champion | EfficientNetB0 | A / B / C / Reject | **78.27%** |

### Crop Classification Champion

`model/CROP_MODEL_CHAMPION_91_76_TEST.keras`

- Input: 224 × 224 × 3
- Classes: Banana, Guava, Maize, Rice, Wheat
- Recorded final test accuracy: **91.76%**
- Uses EfficientNetB0 transfer learning with augmentation and a 5-class softmax head.

### Guava Quality Champion

`model/CROP_QUALITY_MODEL_CHAMPION_78_27_TEST.keras`

- Input: 224 × 224 × 3
- Classes: A, B, C, Reject
- Recorded final test accuracy: **78.27%**
- Reported test set: 520 images
- Uses EfficientNetB0 with augmentation and a 4-class softmax head.

## 📊 Dataset

### Crop Classification Dataset

| Crop | Images |
|---|---:|
| Wheat | 4,000 |
| Rice | 4,000 |
| Maize | 4,000 |
| Banana | 1,194 |
| Guava | 1,000 |
| **Total** | **14,194** |

The split is **80% train / 10% validation / 10% test**, stratified with `random_state=42`:

- Train: 11,355
- Validation: 1,419
- Test: 1,420

The raw dataset is not included in this repository.

### Quality Dataset

The quality workflow uses labelled Guava images with four classes: **A, B, C, Reject**.

## 🧪 Quick Inference

```bash
git clone https://github.com/prabhtheone/Crop-AI.git
cd Crop-AI
pip install -r requirements.txt
python src/predict.py path/to/your/image.jpg
```

A small notebook demo is also included at:

`notebook/Crop_AI_Inference_Demo.ipynb`

## 📁 Repository Structure

```text
Crop-AI/
├── model/
│   ├── CROP_MODEL_CHAMPION_91_76_TEST.keras
│   └── CROP_QUALITY_MODEL_CHAMPION_78_27_TEST.keras
├── notebook/
│   └── Crop_AI_Inference_Demo.ipynb
├── src/
│   └── predict.py
├── .github/workflows/ci.yml
├── README.md
├── LICENSE
├── requirements.txt
├── CITATION.cff
├── CONTRIBUTING.md
├── SECURITY.md
├── .gitattributes
└── .gitignore
```

The `.keras` files are managed with **Git LFS**. Raw datasets and private credentials are excluded from Git.

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- EfficientNetB0
- NumPy
- Pillow
- Pandas
- Scikit-learn
- Matplotlib
- Google Colab
- Git LFS

## 📈 Evaluation Notes

The reported accuracies are test-set evaluation results from the project. They are not guarantees of real-world accuracy. Performance can change with lighting, backgrounds, camera quality, crop varieties, image source, and other domain-shift conditions.

## ⚠️ Limitations

- Quality grading is currently designed for Guava only.
- This is a research/learning prototype, not a professional agricultural diagnosis system.
- Larger and more diverse real-world testing is still needed.

## 🔮 Roadmap

- [x] Multi-crop classification
- [x] EfficientNetB0 transfer learning
- [x] Crop evaluation
- [x] Guava quality classification
- [x] A/B/C/Reject grading
- [x] Two-stage crop + quality inference
- [ ] Larger real-world robustness evaluation
- [ ] Quality labels for additional crops
- [ ] Web/mobile deployment
- [ ] Explainable AI / visual attention

## 🤝 Contributing

Bug reports, experiments, documentation improvements, and pull requests are welcome. See `CONTRIBUTING.md`.

## 📄 License

MIT License — see `LICENSE`.

## ⭐ Support

If you find Crop AI useful for learning or experimentation, consider starring the repository and sharing constructive feedback.

**Built as a student project exploring AI, computer vision, and agriculture. 🌾🤖**
