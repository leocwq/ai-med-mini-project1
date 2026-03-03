# Caltech-101 Mini Project (Image Classification)

This repository contains a complete mini-project pipeline for **Caltech-101 image classification**, implemented in Jupyter notebook format.

The main file is:

- `mini_proj1_completed.ipynb`

It includes:

- dataset scanning and split generation
- a classical ML baseline (**HOG + LinearSVC**)
- transfer learning baselines (**ResNet18**, **EfficientNet-B0**)
- unified evaluation metrics
- ablation experiments
- confusion matrices and training curves export

---

## Project Goal

Compare classical and transfer-learning approaches on the same Caltech-101 split, then analyze:

1. how much transfer learning improves over hand-crafted features
2. which deep model performs better under the same budget
3. whether resolution or augmentation matters more (ablation)

---

## Dataset Layout

This repo expects the Caltech-101 class folders directly under the project root (for example `accordion/`, `airplanes/`, ...).

The notebook uses folder names as class labels and builds a stratified split:

- Train: 70%
- Validation: 15%
- Test: 15%

---

## Methods Implemented

### 1) Classical baseline
- **HOG + LinearSVC**
- Grayscale + resize
- StandardScaler + linear SVM

### 2) Deep transfer learning
- **ResNet18** (pretrained)
- **EfficientNet-B0** (pretrained)
- Full fine-tuning

### 3) Ablation (EfficientNet-B0)
- `image_size=64, aug=off` (baseline)
- `image_size=128, aug=off`
- `image_size=64, aug=on`

---

## Metrics

The notebook reports:

- Top-1 Accuracy
- Top-5 Accuracy
- Mean Per-class Accuracy
- Macro Precision / Recall / F1
- Weighted Precision / Recall / F1
- Confusion Matrix

---

## How to Run

Open and run:

`mini_proj1_completed.ipynb`

in Jupyter/VSCode/Colab.

Recommended execution order: run all cells from top to bottom once.

---

## Environment / Dependencies

Main Python packages used:

- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `Pillow`, `tqdm`
- `scikit-learn`, `scikit-image`
- `torch`, `torchvision`

Install example:

```bash
pip install numpy pandas matplotlib seaborn pillow tqdm scikit-learn scikit-image torch torchvision
```

---

## Notes

- The notebook automatically detects device (`cuda`, `mps`, or `cpu`).
- Training epochs are intentionally short for mini-project runtime.
- For best reproducibility, keep the same random seed and split logic.

