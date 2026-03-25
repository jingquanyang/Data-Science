# Oxford-IIIT Pet Breed Classification

A deep learning project that classifies 37 pet breeds (cats and dogs) from the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) using transfer learning with ResNet-18.

---

## Project Summary

| Item | Details |
|------|---------|
| Dataset | Oxford-IIIT Pet Dataset |
| Task | 37-class fine-grained breed classification |
| Total Images | 7,349 |
| Final Model | ResNet-18 (ImageNet pretrained, fine-tuned) |
| Test Accuracy | **88.03%** |
| Baseline (Random Forest) | 9.46% |

---

## Problem Statement

Fine-grained visual recognition of pet breeds is a challenging task due to high intra-class variation and inter-class similarity. This project explores whether a pre-trained CNN can be effectively fine-tuned to distinguish 37 distinct breeds from the Oxford-IIIT Pet Dataset, and compares deep learning to a traditional handcrafted-feature baseline.

---

## Approach

### 1. Data Wrangling & EDA
- Parsed official `trainval.txt` and `test.txt` split files
- Built a master DataFrame with image paths, trimap paths, XML paths, and breed/species labels
- Extracted 11 handcrafted image features: dimensions, aspect ratio, brightness, contrast, RGB channel means, edge strength, pet area ratio, unknown ratio
- Verified zero train/test overlap and zero missing images

### 2. Baseline Model
- **Random Forest** trained on 11 handcrafted image features
- Result: 100% train accuracy, only **9.46% test accuracy** (severe overfitting)
- Conclusion: Low-level image statistics are insufficient for fine-grained classification

### 3. Deep Learning Model — ResNet-18 Transfer Learning
- Replaced the final fully connected layer with `Linear(512 → 37)`
- Applied ImageNet normalization and standard augmentation (horizontal flip, color jitter)
- Stratified 80/20 train/val split; evaluated on the held-out test set
- Selected best checkpoint by validation accuracy

---

## Results

| Epoch | Train Loss | Train Acc | Val Acc |
|-------|-----------|-----------|---------|
| 1 | 1.8164 | 90.96% | 86.01% |
| 2 | 0.5383 | 96.64% | 88.86% |
| 3 | 0.2556 | 98.98% | 90.49% |
| 5 | 0.0800 | 99.76% | **92.26%** |
| 8 | 0.0229 | 100.00% | 92.26% |

**Final Test Accuracy: 88.03%**

---

## Key Figures

1. **Breed Distribution** — Near-uniform class distribution (imbalance ratio 1.09), confirming a well-balanced dataset
2. **Training Curves** — Steady improvement in validation accuracy from 86% → 92% over 8 epochs
3. **Baseline vs. Deep Model Comparison** — 9.46% (Random Forest) vs. 88.03% (ResNet-18)
4. **Train/Test Split Quality** — Max per-breed proportion difference < 0.003, confirming representative splits

---

## Recommendations

1. **Deploy ResNet-18 as a production pet breed identifier** — 88% accuracy on 37 breeds is strong for a real-world pet app, and the model can be served with a simple REST API.
2. **Extend to larger backbones for marginal gains** — ResNet-50 or EfficientNet-B2 with longer training and learning rate scheduling could push test accuracy above 93%.
3. **Integrate breed classification into veterinary or shelter workflows** — Accurate automated breed identification can assist shelters with intake forms, insurance claims, and breed-specific care recommendations.

---

## Project Structure

```
├── The_Oxford-IIIT_Pet_ModeL.ipynb   # Full pipeline notebook
├── model_metrics.csv                  # Model parameters & performance
└── Capstone_Final_Report.pdf          # Project report
```

---

## Setup

```bash
conda create -n pets_torch python=3.10
conda activate pets_torch
pip install torch torchvision scikit-learn pandas pillow matplotlib
```

---

## Dataset

Download from: https://www.robots.ox.ac.uk/~vgg/data/pets/

Update `DATA_DIR` in the notebook to your local path.
