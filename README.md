# 🌾 Crop AI — Intelligent Crop Classification & Quality Grading

### Computer Vision for 5-Crop Recognition + Multi-Crop Quality Grading 🤖🌱

Crop AI is a deep-learning agricultural computer-vision project using a two-stage pipeline:

1. **Crop classification:** Banana, Guava, Maize, Rice, or Wheat.
2. **Quality grading:** a second EfficientNetB2 model grades supported crop images into **A / B / C / D** quality classes.

> **Current quality-model release:** EfficientNetB2 at 260×260, trained on 4,295 labelled images across 18 available crop-quality classes. The best validation accuracy recorded during training was **91.46%**.

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
🖼️ 260 × 260
      ↓
🧠 EfficientNetB2 Quality Model
      ↓
🏷️ A / B / C / D
```

## 🧠 Released Models

| Model | Architecture | Classes | Recorded Accuracy |
|---|---|---|---:|
| Crop Classification Champion | EfficientNetB0 | 5 crops | **91.76% test** |
| Multi-Crop Quality Champion | EfficientNetB2 | 18 available crop-quality classes | **91.46% validation** |

### Crop Classification Champion

`model/CROP_MODEL_CHAMPION_91_76_TEST.keras`

- Input: 224 × 224 × 3
- Classes: Banana, Guava, Maize, Rice, Wheat
- Recorded test accuracy: **91.76%**
- EfficientNetB0 transfer learning with augmentation and a 5-class softmax head.

### Multi-Crop Quality Champion

`model/QUALITY_MODEL_B2_260_BEST.keras`

- Architecture: EfficientNetB2
- Input: 260 × 260 × 3
- Best recorded validation accuracy: **91.46%**
- Dataset: **4,295 images**
- Available quality classes:
  - Banana: A / B / D
  - Guava: A / B / D
  - Maize: A / B / C / D
  - Rice: A / B / C / D
  - Wheat: A / B / C / D
- **Banana-C and Guava-C are not present in the current training dataset**, so this release is an 18-class model rather than a complete 20-class (5 crops × 4 grades) model.

Quality labels are project-defined visual grading categories, not official agricultural certification grades.

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

Split: **80% train / 10% validation / 10% test**, stratified with `random_state=42`.

- Train: 11,355
- Validation: 1,419
- Test: 1,420

The raw dataset is not included in this repository.

### Quality Dataset

The current quality training set contains **4,295 images** across 18 available crop-quality classes. The current distribution is intentionally capped at up to 250 images per class where available; Banana and Guava do not currently have Class C images in the prepared dataset.

## 🧪 Quick Inference

```bash
git clone https://github.com/prabhtheone/Crop-AI.git
cd Crop-AI
pip install -r requirements.txt
python src/predict.py path/to/your/image.jpg
```

A notebook demo is also included at:

`notebook/Crop_AI_Inference_Demo.ipynb`

## 📁 Repository Structure

```text
Crop-AI/
├── model/
│   ├── CROP_MODEL_CHAMPION_91_76_TEST.keras
│   └── QUALITY_MODEL_B2_260_BEST.keras
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
- EfficientNetB2
- NumPy
- Pillow
- Pandas
- Scikit-learn
- Matplotlib
- Google Colab
- Git LFS

## 📈 Evaluation Notes

The crop-classification **91.76%** figure is a recorded test-set accuracy.

The quality-model **91.46%** figure is the best **validation accuracy recorded during the 18-class training run**. It should not be presented as a held-out test accuracy.

Real-world performance can change with lighting, backgrounds, camera quality, crop varieties, image source, and other domain-shift conditions.

## ⚠️ Limitations

- The current quality model has 18 available crop-quality classes because Banana-C and Guava-C are missing from the prepared dataset.
- A complete 20-class A/B/C/D model requires real Banana-C and Guava-C labelled images.
- Quality grades are project-defined visual categories, not official agricultural certification.
- This is a research/learning prototype, not a professional agricultural diagnosis system.
- Larger and more diverse real-world testing is still needed.

## 🔮 Roadmap

- [x] Multi-crop classification
- [x] EfficientNetB0 transfer learning
- [x] Crop evaluation
- [x] Multi-crop quality model
- [x] A/B/C/D grading scheme
- [x] EfficientNetB2 260×260 quality model
- [ ] Add real Banana-C and Guava-C data
- [ ] Train complete 20-class A/B/C/D model
- [ ] Held-out test evaluation for quality model
- [ ] Larger real-world robustness evaluation
- [ ] Web/mobile deployment
- [ ] Explainable AI / visual attention

## 🤝 Contributing

Bug reports, experiments, documentation improvements, and pull requests are welcome. See `CONTRIBUTING.md`.

## 📄 License

MIT License — see `LICENSE`.

## ⭐ Support

If you find Crop AI useful for learning or experimentation, consider starring the repository and sharing constructive feedback.

**Built as a student project exploring AI, computer vision, and agriculture. 🌾🤖**
