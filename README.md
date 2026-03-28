# SOH-ZeroShot-DeepLearning

**State of Health Estimation of Lithium-Ion Batteries: A Comparative Study of LSTM and CNN-LSTM Architectures**

> SYSC 5108W — Deep Learning | Carleton University | Winter 2026  
> Jalal AMAROF (101373872)

---

## Overview

This repository contains the complete source code, training logs, and reproducibility artifacts for a comparative study evaluating two deep learning architectures for battery State of Health (SOH) estimation:

| Model | Input | Architecture | Zero-Shot RMSE (B0007) | R² |
|---|---|---|---|---|
| **Baseline LSTM** | 14 engineered features | LSTM(64, 1 layer) | **0.1451 Ah** | **+18.29%** |
| Hybrid CNN-LSTM | 1,300 raw voltage points | Conv1D(16,k=7)→MaxPool(2)→LSTM(64, 2 layers) | 0.1699 Ah | −12.06% |

**Key finding:** In limited-data regimes with cross-battery domain shift, domain-informed feature engineering (14 points) significantly outperforms automated CNN feature extraction (1,300 points) for SOH prediction on unseen batteries.

## Evaluation Protocol

Unlike most SOH studies that test on the *same* battery used for training, this project uses a strict **zero-shot cross-battery** protocol:

- **Training:** B0005 + B0006 (336 cycles)
- **Validation:** B0018 (132 cycles) — Early Stopping only
- **Test:** B0007 (168 cycles) — **never seen**, no capacity calibration

## Repository Structure

```
SOH-ZeroShot-DeepLearning/
├── README.md
├── requirements.txt
├── extract2csv.py                  # NASA .mat → CSV conversion
├── tune_lstm_gpu.py                # LSTM grid search (12 configs)
├── tune_cnnlstm_gpu.py             # CNN-LSTM grid search (72 configs)
├── test_lstm_generalization_gpu.py  # LSTM champion → zero-shot B0007
├── test_cnnlstm_generalization_gpu.py # CNN-LSTM champion → zero-shot B0007
├── data/
│   ├── *.mat                        # Raw NASA battery .mat files
│   └── dataset_soh_nasa.csv         # Extracted CSV (via extract2csv.py)
├── logs/
│   ├── tune_lstm_gpu_20260324_005504.txt
│   ├── tune_cnnlstm_gpu_20260323_235217.txt
│   ├── test_lstm_generalization_gpu_20260324_010509.txt
│   └── test_cnnlstm_generalization_gpu_20260324_010549.txt
├── plots/
│   └── all_batteries_capacity_drop_overlay.png
├── b0007_champion_lstm_plot.png
├── b0007_champion_cnnlstm_plot.png
└── models/                          # Saved .pth weights (generated at training)
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset
Place the NASA `.mat` files in `data/`, then run:
```bash
python extract2csv.py
```

### 3. Run the grid searches
```bash
python tune_lstm_gpu.py          # ~5 min on GPU
python tune_cnnlstm_gpu.py      # ~45 min on GPU
```

### 4. Evaluate champions on unseen B0007
```bash
python test_lstm_generalization_gpu.py
python test_cnnlstm_generalization_gpu.py
```

## Training Configuration

| Parameter | LSTM | CNN-LSTM |
|---|---|---|
| Loss function | Huber (δ=1.0) | Huber (δ=1.0) |
| Optimizer | Adam | Adam |
| Learning rate | 1×10⁻² | 1×10⁻³ |
| Batch size | 16 | 16 |
| Max epochs | 250 | 250 |
| Early stopping patience | 20 | 20 |
| Dropout | 0.0 | 0.1 (between LSTM layers) |
| Random seed | 42 | 42 |

## Dataset

NASA Ames Prognostics Center of Excellence (PCoE) — Li-ion Battery Aging Dataset:
- **Source:** [NASA PCoE Data Repository](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets)
- **Batteries:** B0005, B0006, B0007, B0018
- **Protocol:** Constant current discharge (2A) at 24°C
- **Features:** 1,300 voltage measurements per discharge cycle

## Feature Engineering (Tiane et al., 2023)

The 1,300 raw voltage readings are compressed into 14 features using a sliding window (W=100):
- 13 non-overlapping window averages
- 1 terminal voltage anchor point

Reference: Tiane et al., "Adversarial Defensive Framework for State-of-Health Prediction of Lithium Batteries," *IEEE Trans. Power Electronics*, 2023.

## Hardware

Training executed on **NVIDIA GeForce RTX 5090** (33.7 GB VRAM) via [TensorDock](https://www.tensordock.com) cloud GPU rental.

## Citation

If you use this code or methodology, please cite:
```
@misc{amarof2026soh,
  author = {Amarof, Jalal},
  title = {SOH-ZeroShot-DeepLearning: A Comparative Study of LSTM and CNN-LSTM for Battery Health Estimation},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/amarof/SOH-ZeroShot-DeepLearning}
}
```

## License

This project is released for academic purposes under the MIT License.
