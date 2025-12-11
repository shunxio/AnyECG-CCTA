# Fine-tuning an ECG Foundation Model to Predict Coronary CT Angiography Outcomes

## Table of Contents

* [Introduction](#introduction)
* [Key Contributions](#key-contributions)
* [Model Performance](#model-performance)
* [Model Architecture & Training Strategy](#model-architecture--training-strategy)
* [Project Structure](#project-structure)
* [Workflow Overview](#workflow-overview)
* [Data Format](#data-format)
* [Running the Project](#running-the-project)
* [Training Example](#training-example)
* [Citation](#citation)
* [License](#license)

---

## Introduction

This repository presents an **interpretable AI-ECG model** designed to directly predict **severe stenosis** or **complete occlusion** of the four major coronary arteries — **RCA, LM, LAD, LCX** — as defined by **coronary CT angiography (CCTA)**, using only 12-lead ECG signals.

By fine-tuning **ECGFounder**, the world’s largest ECG foundation model, the project demonstrates AI-ECG's potential as a **non-invasive, low-cost screening tool** for occult CAD, even among patients with clinically normal ECGs.

---

## Key Contributions

### 1. Direct prediction of vessel-level CCTA lesions

The first interpretable AI-ECG system capable of **direct vessel-specific prediction** of severe stenosis or complete occlusion.

### 2. Screening for occult CAD

The model maintains strong performance in the **normal ECG subgroup**, revealing underlying disease patterns.

### 3. Dynamic risk stratification

Combining predicted risk with incidence curves enables **prospective coronary event risk stratification**.

### 4. Waveform-level interpretability

Waveform comparison highlights **key electrophysiological regions** that differentiate high-risk vs. low-risk groups.

---

## Model Performance

AUC scores across internal, external, and normal-ECG validation.

| Vessel  | Internal AUC | External AUC | Normal-ECG AUC |
| ------- | ------------ | ------------ | -------------- |
| **RCA** | 0.794        | 0.749        | 0.793          |
| **LM**  | 0.818        | 0.971        | 0.896          |
| **LAD** | 0.744        | 0.667        | 0.707          |
| **LCX** | 0.755        | 0.727        | 0.765          |

> Note: LM performance may be inflated due to low lesion prevalence.

---

## Model Architecture & Training Strategy

### Foundation Model

* **ECGFounder**, pretrained for myocardial infarction prediction.

### Backbone Network

* Modified **Net1D** for ECG representation.

### Multi-Task Learning

Joint prediction of the four arteries to enhance shared feature extraction.

### Optimization Techniques

* **Uncertainty-based adaptive task weighting**
* **PCGrad** to resolve gradient conflicts in multi-task training

---

## Project Structure

```
project_root/
├── train.py                  # Main training script
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
├── data/
│   ├── ecg/                  # Raw ECG signals
│   ├── labels/               # CCTA labels
│   └── processed/            # Preprocessed data
├── models/
│   ├── ecgfounder.py         # Foundation model loader
│   ├── net1d.py              # Net1D architecture
│   ├── multitask_model.py    # Multi-task classifier
│   └── utils.py              # Model utilities
├── utils/
│   ├── preprocessing.py      # Normalization, filtering
│   ├── dataset.py            # Dataset & dataloader
│   ├── metrics.py            # AUC and evaluation metrics
│   └── visualization.py      # ECG visualization tools
└── outputs/
    ├── checkpoints/          # Saved models
    ├── logs/                 # Training logs
    └── figures/              # Visualization outputs
```

---

## Workflow Overview

```
ECG Signals
    ↓
Preprocessing (Z-score, filtering)
    ↓
ECGFounder Backbone
    ↓
Net1D Multi-Task Classifier
    ↓
Risk Prediction
    ↓
Interpretability (waveform analysis)
```

---

## Data Format

### ECG Data (.npy)

* Shape: `(T, 12)`
* T: time steps
* 12: ECG leads

### Label File (.csv)

```
patient_id, RCA_label, LM_label, LAD_label, LCX_label
0001, 1, 0, 0, 1
```

---

## Running the Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

* Organize ECG waveforms and CCTA labels.
* Apply **lead-wise Z-score normalization**.
* Ensure proper alignment of ECG samples and labels.

### 3. Train the Model

```bash
python train.py
```

---

## Training Example

```bash
python train.py \
    --batch_size 64 \
    --lr 1e-4 \
    --epochs 50 \
    --save_dir ./outputs/checkpoints
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{xiao2025fine,
  title={Fine-tuning an ECG Foundation Model to Predict Coronary CT Angiography Outcomes},
  author={Xiao, Yujie and Tang, Gongzhen and Zhang, Deyun and Li, Jun and Nie, Guangkun and Wang, Haoyu and Huang, Shun and Liu, Tong and Zhao, Qinghao and Chen, Kangyin and others},
  journal={arXiv preprint arXiv:2512.05136},
  year={2025}
}
```

