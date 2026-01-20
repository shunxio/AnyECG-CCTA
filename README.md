# Fine-tuning an ECG Foundation Model to Predict Coronary CT Angiography Outcomes

## Overview

This repository provides the official implementation of an **explainable AI-enabled electrocardiography (AI-ECG) system** for **vessel-level prediction of severe coronary artery stenosis and total occlusion**, using **12-lead ECG signals** with **coronary CT angiography (CCTA)** as the reference standard.

By fine-tuning **ECGFounder**, a large-scale ECG base model, which we then fine-tuned on independent datasets for the new myocardial infarction task, this project demonstrates that AI-ECG can serve as a **non-invasive, low-cost, and scalable screening tool** for identifying **occult coronary artery disease (CAD)**, even among individuals with **clinically normal ECGs**.

The model outputs **continuous risk probabilities** for the four major coronary arteries:

- Right coronary artery (**RCA**)
- Left main coronary artery (**LM**)
- Left anterior descending artery (**LAD**)
- Left circumflex artery (**LCX**)

and supports **longitudinal risk stratification**, **decision curve analysis**, and **waveform-level interpretability**.

---

## Key Contributions

1.  **End-to-End Prediction**: Developed an explainable AI-ECG model capable of directly predicting CCTA-defined severe coronary artery stenosis and total occlusion from raw ECG signals.
2.  **Occult CAD Screening**: Demonstrated that the model maintains stable performance in individuals with **apparently normal ECGs**, highlighting its potential for screening occult CAD in asymptomatic populations.
3.  **Risk Stratification**: Combined model predictions with epidemiological data to assess the cumulative risk of myocardial infarction (via Kaplan–Meier curves), enabling longitudinal risk assessment.
4.  **Clinical Utility**: Identified optimal operating points through sensitivity analyses to support opportunistic screening in clinical settings.
5.  **Explainability**: Characterized ECG waveform differences between high-risk and low-risk populations to provide electrophysiological insights into coronary artery stenosis.

---

## Model Performance

### Discrimination Performance (AUC)

| Vessel | Internal Validation | External Validation | Normal-ECG Subgroup |
|------|-------------------|-------------------|-------------------|
| **RCA** | 0.744 | 0.714 | 0.693 |
| **LM**  | 0.706 | 0.713 | 0.659 |
| **LAD** | 0.716 | 0.700 | 0.673 |
| **LCX** | 0.736 | 0.673 | 0.716 |

> **Note**: LM performance should be interpreted cautiously due to extremely low lesion prevalence (~0.1%), consistent with real-world epidemiology.

---

## Model Architecture & Training Strategy

### Foundation Model
- **ECGFounder**: large-scale ECG foundation model pretrained on >10 million ECG recordings.

### Backbone Network
- Modified **Net1D** architecture for 12-lead ECG representation learning.

### Multi-Task Learning
Simultaneous prediction of four coronary arteries using a shared encoder.

### Optimization Techniques
- Uncertainty-based adaptive task weighting
- PCGrad (Projected Conflicting Gradient)
- AdamW optimizer with warmup and cosine decay
- Online ECG data augmentation:
  - Temporal shifting
  - Amplitude scaling
  - Gaussian noise injection
  - Local signal occlusion

### Cross-Validation Strategy
- Five-fold stratified grouped cross-validation
- Patient ID used as grouping variable
- Best checkpoint selected by validation macro-AUC

---

## Project Structure

```text
project_root/
├── train.py                  # Main training script
├── analysis.ipynb            # Evaluation code
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── data/
│   └── sample.csv            # Example input table with ECG paths and labels
├── Utils/
│   ├── net1d.py              # ECG backbone network
│   ├── ECGDataset.py         # Dataset and preprocessing
└── outputs/
    ├── checkpoints/          # Saved model weights
    ├── logs/                 # Training logs
    └── figures/              # ROC, KM, DCA, waveform visualizations
```
---

## Workflow Overview

```
ECG Signals
    ↓
Preprocessing (Z-score, filtering)
    ↓
ECGFounder
    ↓
Risk Prediction
    ↓
Explainability (waveform analysis)
```

---

## Data Format

### ECG Data (.npy)

* Shape: `(12, T)`
* T: time steps, should be 5000 (500Hz * 10s)
* 12: ECG leads

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

