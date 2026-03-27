# DeepLearning-MPW-CNN

The goal is to apply the concepts seen in the lectures and practical sessions, and to document your experimentation process and results in a report.

This project trains and evaluates CNN models of progressively increasing complexity on the **iCoSimal V3** animal-classification dataset (10 classes, 30,000 images).

---

## Dataset

Download the **iCoSimal V3** dataset from:  
<https://drive.switch.ch/index.php/s/NTiYe8mamrgys3M>

Expected directory layout after extraction:

```
icosimal_img_class_03/data_uniform_224_224_sets/
├── train/
│   ├── cat/
│   ├── chicken/
│   ├── cow/
│   ├── dog/
│   ├── elephant/
│   ├── horse/
│   ├── rabbit/
│   ├── sheep/
│   ├── squirrel/
│   └── zebra/
└── validate/
    ├── cat/
    └── ...
```

| Property | Value |
|---|---|
| Classes | 10 (cat, chicken, cow, dog, elephant, horse, rabbit, sheep, squirrel, zebra) |
| Total images | 30,000 |
| Training images | 24,000 |
| Validation images | 6,000 |
| Image resolution | 224 × 224 px |
| Class distribution | Balanced |

---

## Project structure

```
.
├── src/
│   ├── data_loader.py          # Dataset loading & transforms (supports resize)
│   ├── models.py               # SimpleCNN, MediumCNN, DeepCNN architectures
│   ├── train.py                # Training loop with configurable optimizer/scheduler
│   ├── evaluate.py             # Metrics, confusion matrix, and comparison plots
│   └── hyperparameter_tuning.py# Grid-search sweep utility
├── notebooks/
│   └── cnn_experiments.ipynb  # Interactive Jupyter notebook walkthrough
├── main.py                     # CLI experiment runner
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick start

### Architecture depth comparison (Objective a)

```bash
python main.py \
    --data_root /path/to/icosimal_img_class_03/data_uniform_224_224_sets \
    --image_size 64 \
    --num_epochs 20
```

This trains **SimpleCNN** (2 conv layers), **MediumCNN** (4 conv layers), and **DeepCNN** (6 conv layers + BatchNorm) and saves comparison plots to `results/`.

### Hyperparameter tuning sweep (Objective b)

```bash
python main.py \
    --data_root /path/to/... \
    --sweep \
    --output_dir results/
```

Add `--sweep_full` to run the larger grid (image sizes 64 & 128, more LR / optimizer combinations).

### Full run

```bash
python main.py \
    --data_root /path/to/... \
    --image_size 128 \
    --num_epochs 30 \
    --sweep_full \
    --output_dir results/
```

---

## Jupyter Notebook

Open `notebooks/cnn_experiments.ipynb` for a step-by-step interactive walkthrough of both experiments, including visualisations of training curves, confusion matrices, and hyperparameter effect plots.

---

## CNN Architectures

| Name | Conv layers | BatchNorm | Dropout | ~Params (64×64 input) |
|---|---|---|---|---|
| SimpleCNN | 2 | ✗ | ✓ | ~3 M |
| MediumCNN | 4 | ✗ | ✓ | ~9 M |
| DeepCNN | 6 | ✓ | ✓ | ~1 M (AdaptiveAvgPool) |

### SimpleCNN
A minimal two-block baseline to establish a lower-bound on performance.

### MediumCNN
Adds two extra conv blocks with increasing channel depth (32 → 64 → 128 → 256) over SimpleCNN.

### DeepCNN
Uses paired conv layers, BatchNormalization for better gradient flow, and AdaptiveAvgPool so the architecture works at any input resolution.

---

## Hyperparameters explored

| Hyperparameter | Values explored |
|---|---|
| Image size | 64, 128 (also 224 in final run) |
| Batch size | 32, 64 |
| Learning rate | 1e-3, 3e-4, 1e-4 |
| Optimizer | Adam, SGD (momentum 0.9) |
| LR scheduler | Cosine annealing, Step decay |
| Dropout | 0.3, 0.5 |
| Weight decay | 1e-4 |
