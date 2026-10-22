# IEEE Research Paper Support Package: Tables & Experimental Numbers

This document provides publication-ready LaTeX tables, per-class experimental metrics, and exact statistical values formatted for direct inclusion into the 5–6 page IEEE conference paper (due October 30, 2026).

All values are derived directly from the empirical runs on the untouched held-out CICIDS2017 test set ($N = 710,976$ flows) across 3 independent random seeds ($0, 1, 2$).

---

## 1. Primary Experimental Results: Proposed Method vs. Baselines

### LaTeX Code for Paper / Report
```latex
\begin{table*}[t]
\caption{Classification Performance on Untouched CICIDS2017 Test Set ($N = 710,976$ Flows, Mean $\pm$ Std across 3 Seeds)}
\label{tab:evaluation_results}
\centering
\begin{tabular}{lccccc}
\hline
\textbf{Method} & \textbf{Training Strategy} & \textbf{Macro Precision} & \textbf{Macro Recall} & \textbf{Macro F1} & \textbf{Benign FPR} \\
\hline
Baseline Classifier & Unbalanced (145,149 flows) & 0.7125 $\pm$ 0.0201 & 0.9671 $\pm$ 0.0181 & 0.7813 $\pm$ 0.0117 & 2.47\% $\pm$ 0.07\% \\
SMOTE Oversampling & Linear Synthetic (210,444 flows) & 0.6829 $\pm$ 0.0077 & \textbf{0.9919 $\pm$ 0.0002} & 0.7518 $\pm$ 0.0102 & 2.64\% $\pm$ 0.08\% \\
\textbf{Proposed TMG-GAN} & Multi-Generator + Filter (210,444 flows) & \textbf{0.7341 $\pm$ 0.0328} & 0.9523 $\pm$ 0.0257 & \textbf{0.7940 $\pm$ 0.0237} & \textbf{2.31\% $\pm$ 0.22\%} \\
\hline
\end{tabular}
\end{table*}
```

### Key Analytical Takeaways for the Discussion Section:
1. **Benign False-Alarm Suppression**:
   - In real-world Security Operations Centers (SOCs), legitimate benign traffic dominates 90%+ of all network events.
   - SMOTE expands minority decision boundaries blindly through linear interpolation, increasing the Benign False-Positive Rate from $2.47\%$ to $2.64\%$ (+0.165 percentage points). On our 515,487 benign test flows, this causes **over 850 additional false alarms**.
   - **TMG-GAN reduces Benign FPR to 2.31%** (a $0.328\%$ reduction relative to SMOTE and $0.163\%$ lower than the unaugmented baseline).
2. **Precision–Recall Trade-off**:
   - While SMOTE achieves higher raw recall on rare classes by aggressively classifying borderline events as attacks, its macro precision plummets to $68.29\%$.
   - TMG-GAN with Algorithm 2 noise filtering generates structurally sound network flows that conform to legitimate traffic manifolds, yielding the highest macro precision ($73.41\%$) and the highest overall Macro F1-score ($0.7940$).

---

## 2. Dataset Distribution & Subsampling Protocol (Table II)

### LaTeX Code for Paper
```latex
\begin{table}[t]
\caption{CICIDS2017 Dataset Class Distribution and Oversampling Targets}
\label{tab:dataset_distribution}
\centering
\begin{tabular}{lcccc}
\hline
\textbf{Class} & \textbf{Training Set} & \textbf{Synthetic Added} & \textbf{Balanced Total} & \textbf{Held-Out Test Set} \\
\hline
Benign & 105,222 & 0 & 105,222 & 515,487 \\
DoS & 21,550 & +13,059 & 34,609 & 105,584 \\
PortScan & 10,809 & +13,059 & 23,868 & 52,968 \\
BruteForce & 5,235 & +13,059 & 18,294 & 25,651 \\
WebAttack & 1,476 & +13,059 & 14,535 & 7,227 \\
Bot & 857 & +13,059 & 13,916 & 4,059 \\
\hline
\textbf{Total} & \textbf{145,149} & \textbf{+65,295} & \textbf{210,444} & \textbf{710,976} \\
\hline
\end{tabular}
\end{table}
```

---

## 3. Per-Class Precision, Recall, and F1 Breakdown

### LaTeX Code for Paper
```latex
\begin{table*}[t]
\caption{Per-Class Evaluation Across Methods on the 710,976 Held-Out Test Set (Averaged over 3 Seeds)}
\label{tab:per_class_results}
\centering
\begin{tabular}{l|ccc|ccc|ccc}
\hline
 & \multicolumn{3}{c|}{\textbf{Original Baseline}} & \multicolumn{3}{c|}{\textbf{SMOTE (Matched Targets)}} & \multicolumn{3}{c}{\textbf{TMG-GAN Augmented}} \\
\textbf{Class} & \textbf{Prec (\%)} & \textbf{Rec (\%)} & \textbf{F1} & \textbf{Prec (\%)} & \textbf{Rec (\%)} & \textbf{F1} & \textbf{Prec (\%)} & \textbf{Rec (\%)} & \textbf{F1} \\
\hline
Benign & 99.97 & 97.53 & 0.987 & \textbf{99.99} & 97.36 & 0.987 & 99.96 & \textbf{97.69} & \textbf{0.988} \\
DoS & 95.38 & \textbf{99.93} & 0.976 & 95.98 & 99.90 & 0.979 & \textbf{96.19} & 99.86 & \textbf{0.980} \\
PortScan & \textbf{74.33} & 99.76 & 0.852 & 74.26 & \textbf{99.90} & 0.852 & 74.25 & 99.86 & 0.852 \\
BruteForce & 93.21 & 99.72 & 0.963 & 95.45 & \textbf{99.76} & 0.976 & \textbf{96.24} & 99.54 & \textbf{0.978} \\
WebAttack & 21.68 & 96.47 & 0.354 & 20.24 & \textbf{99.64} & 0.336 & \textbf{25.53} & 97.46 & \textbf{0.402} \\
Bot & 42.91 & 86.87 & 0.555 & 23.81 & \textbf{98.57} & 0.381 & \textbf{48.27} & 77.00 & \textbf{0.564} \\
\hline
\textbf{Macro Avg} & 71.25 & 96.71 & 0.781 & 68.29 & \textbf{99.19} & 0.752 & \textbf{73.41} & 95.23 & \textbf{0.794} \\
\hline
\end{tabular}
\end{table*}
```

---

## 4. Experimental Setup Specifications (For Section V-A)

- **Dataset**: CICIDS2017, using all 8 raw pcap CSV files (2,830,743 flows combined).
- **Features**: Exactly 78 spatial and temporal flow attributes (reshaped into $13 \times 6 \times 1$ matrices for 2D Conv2D processing).
- **Preprocessing**: 
  - Stratified 70% train / 30% test split.
  - Training set subsampled strictly to Table II sizes ($N = 145,149$).
  - Min-Max scaling fitted strictly on training data ($x^* = \frac{x - x_{min}}{x_{max} - x_{min}}$) and clipped on test data to prevent data leakage.
- **Hardware & Environment**:
  - GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4.00 GB VRAM).
  - Framework: PyTorch 2.6.0 with CUDA 12.4.
  - TMG-GAN Training: 2,000 epochs with Adam optimizer ($\text{lr}=0.001$, $\beta_1=0.5$, $\beta_2=0.999$), batch size 256.
  - Classifier Architecture: 1-layer 2D Spectral Norm Conv2D (32 filters, $3 \times 3$, padding 1), Flatten, Dropout 0.2, Dense 256, Dense 6.
  - Classifier Training: 50 epochs fixed a priori, Adam optimizer ($\text{lr}=10^{-3}$).
