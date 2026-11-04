# Generative Adversarial Network Data Augmentation for Network Intrusion Detection Under Severe Class Imbalance

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![License: Academic Research](https://img.shields.io/badge/License-Academic_Research-green.svg)](#attribution--references)

Official PyTorch implementation of our research on **Generative Adversarial Network (GAN) Data Augmentation for Network Intrusion Detection Systems (NIDS)** under severe class imbalance on the CICIDS2017 benchmark.

---

## Authors & Affiliation
- **Someshwar Sahu** (`202451155@iiitvadodara.ac.in`)
- **Ameer Hamza Khan** (`202451015@iiitvadodara.ac.in`)
- **Nagendra Yadav** (`202451112@iiitvadodara.ac.in`)

*Department of Computer Science and Engineering*  
*Indian Institute of Information Technology Vadodara, Gandhinagar, Gujarat, India*

---

## 1. Overview

In production computer networks, legitimate background traffic constitutes over $90\%$ to $99\%$ of continuous flow telemetry, while critical security threats like botnets, web exploits, and brute force intrusions appear in fewer than $1\%$ of flows. 

When deep neural networks (DNNs) are trained on raw skewed data, the majority class dominates the loss gradient, causing models to guess "benign" and miss stealthy attacks. While heuristic oversampling methods like **SMOTE** boost minority recall, their linear interpolation mechanism creates unrealistic synthetic samples in empty or overlapping feature spaces. In high-speed enterprise networks, this causes decision boundaries to expand into legitimate traffic space, generating thousands of operational false alarms.

```
Raw PCAP Flows (2.8M) ──► Stratified 70/30 Split ──► 13x6 Spatial Feature Reshape
                                                                │
                                                                ▼
Test Set (710k Flows) ◄── Balanced Downstream DNN ◄── TMG-GAN Multi-Generator
   [Strictly Untouched]       (3 Seeds: 0, 1, 2)       + Algorithm 2 Quality Gate
```

### Core Architecture & Innovations
1. **Spatial-Structural Flow Representation**: Retains all 78 statistical flow features from CICIDS2017 without lossy column dropping, mapping flow vectors into $13 \times 6 \times 1$ spatial matrices to leverage 2D convolutional weight sharing.
2. **Multi-Generator Architecture with Spectral Stability**: Deploys dedicated generators ($G_k$) for each distinct minority attack class, tied to a shared convolutional discriminator stabilized by Spectral Normalization to bound the 1-Lipschitz constant.
3. **Algorithm 2 Quality Filtering Gate**: Enforces a dual-stage post-generation acceptance gate based on auxiliary classification confidence ($\tau_{\text{conf}} \ge 0.50$) and discriminator realism, discarding out-of-distribution noise flows ($98.10\%$ acceptance rate).
4. **Benign False-Alarm Suppression**: Tested on $710{,}976$ held-out test flows across 3 random seeds, our framework suppresses the Benign False-Positive Rate down to **$2.31\%$**, eliminating **2,038 operational false alarms** on normal traffic compared to matched-capacity SMOTE.

---

## 2. Benchmark Evaluation

Evaluated across $710{,}976$ untouched test flows from CICIDS2017 ($621{,}677$ Benign, $58{,}119$ DoS, $27{,}208$ PortScan, $2{,}745$ BruteForce, $643$ WebAttack, $584$ Bot) over 3 independent random seeds ($0, 1, 2$):

| Method | Training Set Size | Macro Precision (%) | Macro Recall (%) | Macro F1-Score | Benign FPR (%) | Benign False Alarms |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Classifier** | 145,149 flows (Unbalanced) | $71.25 \pm 2.01$ | $96.71 \pm 1.81$ | $0.7813 \pm 0.0117$ | $2.4716 \pm 0.0664$ | 15,365 |
| **SMOTE Baseline** | 210,444 flows (Matched) | $68.29 \pm 0.77$ | $\mathbf{99.19 \pm 0.02}$ | $0.7518 \pm 0.0102$ | $2.6366 \pm 0.0798$ | 16,391 |
| **Proposed TMG-GAN** | 210,444 flows (Augmented) | $\mathbf{73.41 \pm 3.28}$ | $95.23 \pm 2.57$ | $\mathbf{0.7940 \pm 0.0237}$ | $\mathbf{2.3088 \pm 0.2161}$ | **14,353** |

### Per-Class Performance Breakdown

| Class | Baseline F1 | SMOTE F1 | **TMG-GAN F1 (Proposed)** | TMG-GAN Precision |
| :--- | :---: | :---: | :---: | :---: |
| **Benign** | 0.987 | 0.987 | **0.988** | 99.96% |
| **DoS** | 0.976 | 0.979 | **0.980** | 96.19% |
| **PortScan** | 0.852 | 0.852 | **0.852** | 74.25% |
| **BruteForce** | 0.963 | 0.976 | **0.978** | 96.24% |
| **WebAttack** | 0.354 | 0.336 | **0.402** | 25.53% |
| **Bot** | 0.555 | 0.381 | **0.564** | **48.27%** *(vs. SMOTE 23.81%)* |

---

## 3. Repository Architecture

```
├── README.md               # Project documentation & execution guide
├── requirements.txt        # Python package dependencies
│
├── src/                    # Core deep learning modules
│   ├── __init__.py
│   ├── config.py           # Hyperparameters, paths, and class labels
│   ├── dataset.py          # PyTorch dataset & data loaders
│   ├── models.py           # Conv2D CDModel (Discriminator/Classifier) & Generator
│   ├── tmg_gan.py          # Multi-Generator training & Algorithm 2 filter
│   ├── dnn.py              # Downstream classifier architecture
│   └── inference.py        # Unified real-time multi-model inference engine
│
├── pipeline/               # Step-by-step reproduction pipeline
│   ├── 01_preprocess.py    # Data cleaning, 70/30 stratified split & zero-leakage scaling
│   ├── 02_train_gan.py     # 2,000-epoch TMG-GAN training with Spectral Normalization
│   ├── 03_generate.py      # Algorithm 2 quality-filtered minority sample synthesis
│   ├── 04_train_smote.py   # Matched-size SMOTE dataset generation
│   ├── 05_train_dnn.py     # Multi-seed downstream classifier training (3 seeds)
│   ├── 06_evaluate.py      # Comprehensive evaluation over 710,976 held-out test flows
│   ├── generate_figures.py # Generates publication-ready figures
│   ├── plot_imbalance.py   # Dataset class distribution analysis
│   └── verify_health.py    # Generator and discriminator health verification
│
├── dashboard/              # Interactive Defense Studio Web Application
│   ├── run.py              # Launch script: python dashboard/run.py
│   ├── index.html          # Web UI interface
│   ├── styles.css          # Dark-mode security theme styling
│   ├── app.js              # Live multi-model API client & visualizer
│   ├── build_test_pool.py  # 300-flow live testing pool generator
│   └── extract_demo_samples.py # Curates key challenge discrepancy flows
│
├── models/                 # Model checkpoints & serialized normalization metadata
│   └── scaler.json         # Min-max parameters fitted strictly on training data
│
├── results/                # Empirical evaluation results
│   ├── summary.csv         # Macro metrics comparison across seeds
│   ├── per_class.csv       # Per-class precision, recall, and F1 values
│   └── *.png               # Confusion matrices and Benign FPR comparison charts
│
└── data/                   # Data directory (raw 2.8M CSVs are gitignored)
    ├── datasets/           # Storage location for CICIDS2017 CSV files
    ├── plots/              # Figures and plots served to the dashboard
    └── samples/            # Preloaded challenge flows & random test pools
```

---

## 4. Quick Start & Interactive Demo

### Setup Environment
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install PyTorch and required packages
pip install -r requirements.txt
```

### Launch the Interactive Defense Studio
Run the self-contained demonstration server:
```powershell
python dashboard/run.py
```
Open **`http://localhost:8050`** in your browser to access:
- **Live Flow Classifier**: Test arbitrary real-world flows from the $710{,}976$ test set against all three models simultaneously, observing where SMOTE generates false alarms.
- **Adversarial Stress-Test Lab**: Synthesize attack vectors live on GPU and observe the Algorithm 2 confidence and realism filtering gate in real time.
- **Benchmark & FPR Analysis**: Visualizations of Benign FPR trade-offs, per-class recall, and confusion matrices.

---

## 5. End-to-End Pipeline Reproduction

To execute the research pipeline from raw PCAP flows from scratch:

```powershell
# 1. Preprocess CICIDS2017 (clean, 70/30 stratified split, zero-leakage normalization)
python pipeline/01_preprocess.py

# 2. Train TMG-GAN Multi-Generator framework (2,000 epochs with Spectral Normalization)
python pipeline/02_train_gan.py

# 3. Generate synthetic minority attack flows with Algorithm 2 quality filtering
python pipeline/03_generate.py

# 4. Generate matched-size SMOTE baseline dataset
python pipeline/04_train_smote.py

# 5. Train 9 downstream classifiers (Baseline, SMOTE, TMG-GAN across seeds 0, 1, 2)
python pipeline/05_train_dnn.py

# 6. Evaluate all models on the 710,976 untouched test flows
python pipeline/06_evaluate.py

# 7. Generate publication figures
python pipeline/generate_figures.py
```

---

## 6. Attribution & References
- **Base Architecture**: Ding et al., *"TMG-GAN: Generative Adversarial Networks-Based Imbalanced Learning for Network Intrusion Detection"*, **IEEE Transactions on Information Forensics and Security (TIFS)**, vol. 19, pp. 3421–3435, 2024. DOI: [10.1109/TIFS.2023.3331240](https://doi.org/10.1109/TIFS.2023.3331240).
- **Dataset**: Sharafaldin et al., *"Toward Generating a New Dataset for Network Intrusion Detection: Analysis and Issues"*, **Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)**, pp. 108–116, 2018.
- **Resampling Baseline**: Chawla et al., *"SMOTE: Synthetic Minority Over-sampling Technique"*, **Journal of Artificial Intelligence Research**, vol. 16, pp. 321–357, 2002.
