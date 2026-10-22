# Project Decisions and Assumptions

This document records all methodological decisions, assumptions, and deliberate divergences from the base paper (*Ding et al., IEEE TIFS 2024*).

---

## 1. Train / Test Split
- **Paper status**: The paper does not state the train/test split ratio, how the test set was formed, or whether cross-validation was used.
- **Our decision**:
  - We use a **70% train / 30% test stratified split** (`test_size=0.3`, `random_state=0`) performed once on the cleaned dataset.
  - The test set is **never resampled, oversampled, or tuned on**. It remains strictly unseen until final evaluation in Step 5.

---

## 2. Training Set Subsampling (Matching Table II)
- **Paper status**: Table II lists explicit training set sizes for CICIDS2017:
  - Benign: 105,222
  - DoS: 21,550
  - Port Scan: 10,809
  - Brute Force: 5,235
  - Web Attack: 1,476
  - Bot: 857
  - **Total**: 145,149 training samples.
- **Our decision**:
  - After the 70/30 stratified split, each class in the 70% training pool has at least the count required by Table II.
  - We randomly subsample the training split to match these exact Table II numbers using fixed seed `random_state=0`.
  - The held-out 30% test set preserves the natural test distribution of the cleaned data.

---

## 3. Data Cleaning: Missing Values, Infs, and Duplicates
- **Paper status**: The paper does not document how NaN/inf values or duplicate flow records were addressed.
- **Our decision**:
  - Replace `+inf` and `-inf` with `NaN`.
  - Drop all rows containing any `NaN` values.
  - Drop all duplicate rows across the 78 features and label. Duplicate flows in CICIDS2017 are known artifacts of pcap replay; removing them prevents synthetic memorization of repeated samples.

---

## 4. Retaining All 78 Features & Handling Constant Columns
- **Paper status**: The paper states that 78 feature attributes are used and reshaped into $13 \times 6 = 78$ spatial representations for 2D convolution (Figure 6a).
- **Our decision**:
  - Our previous baseline script (`preprocess_cicids.py`) dropped 8 constant columns, leaving 70 features. This is incompatible with Figure 6(a).
  - We **retain all 78 feature attributes**.
  - For the constant columns where $x_{max} - x_{min} = 0$, setting the denominator to 1.0 scales these constant attributes to $0.0$ without division-by-zero errors or NaNs.

---

## 5. Min-Max Normalization Scope (Deliberate Difference from Paper)
- **Paper status**: Equation 10 defines standard min-max scaling:
  $$x^* = \frac{x - x_{min}}{x_{max} - x_{min}}$$
  It does not specify whether $x_{min}$ and $x_{max}$ were computed over the training set or over the entire dataset (including test).
- **Our decision**:
  - We fit the scaler ($x_{min}$ and $x_{max}$) **strictly on the training set**.
  - The test set is transformed using the training set's minimum and range, and clipped to $[0.0, 1.0]$.
  - **Rationale**: Computing scaling parameters across the entire dataset constitutes data leakage from test into train, artificially inflating test set metrics. Computing solely on the training set ensures rigorous experimental validity.
  - The fitted scaler parameters are serialized to `artifacts/scaler.json` for reproducibility and inference.

---

## 6. Class Grouping and Exclusions
- **Paper status**: Table II includes 6 classes: Benign, DoS, Port Scan, Brute Force, Web Attack, and Bot.
- **Our decision**:
  - Drop classes not evaluated in the paper: `DDoS`, `Infiltration`, and `Heartbleed`.
  - Map raw attack sub-types into the 6 groups:
    - `BENIGN` $\to$ Benign
    - `DoS Hulk`, `DoS GoldenEye`, `DoS slowloris`, `DoS Slowhttptest` $\to$ DoS
    - `PortScan` $\to$ PortScan
    - `FTP-Patator`, `SSH-Patator` $\to$ BruteForce
    - `Bot` $\to$ Bot
    - Prefix match `Web Attack` (handling corrupt dash characters in raw CSVs like `Web Attack  Brute Force`) $\to$ WebAttack.

---

## 7. Single GAN Training Run
- **Paper status**: The paper does not report standard deviations over multiple GAN training runs due to high compute cost.
- **Our decision**:
  - The TMG-GAN model is trained once with fixed seed 0.
  - Downstream DNN classifiers and SMOTE baselines are trained and evaluated across seeds 0, 1, and 2 to quantify classification variance. This limitation is explicitly recorded.

---

## 8. Matched SMOTE Target Sample Sizes
- **Paper status**: The paper compares against other methods but does not provide a matched-size SMOTE baseline.
- **Our decision**:
  - To isolate the generative advantage of TMG-GAN from simple dataset expansion, SMOTE oversampling is configured with target class sizes identical to Table II:
    - Benign: 105,222
    - DoS: 34,609
    - Port Scan: 23,868
    - Brute Force: 18,294
    - Web Attack: 14,535
    - Bot: 13,916
  - This produces exactly 210,444 samples in both SMOTE and TMG-GAN training sets, allowing a completely fair comparison.

---

## 9. Classifier Training Protocol & Epochs
- **Paper status**: The paper uses DNN as the downstream classifier but does not detail early stopping or hyperparameter search on the test set.
- **Our decision**:
  - In accordance with Hard Rule 3 ("No tuning on the test set"), the DNN classifier is trained for a **fixed 50 epochs** with batch size 256 and Adam optimizer (`lr=1e-3`), chosen a priori without test set feedback.
  - Exactly 9 models are trained: 3 methods (`original`, `smote`, `tmg_gan`) $\times$ 3 random seeds (`0`, `1`, `2`).

---

## 10. Evaluation Protocol & Benign False-Positive Rate (FPR)
- **Paper status**: Table VI in Ding et al. reports Precision, Recall, F1, and Accuracy on CICIDS2017. However, the paper does NOT report the Benign False-Positive Rate (FPR), obscuring whether synthetic minority augmentation causes legitimate flows to be misclassified as attacks.
- **Our decision**:
  - We evaluate all 9 trained DNN models on the complete, untouched held-out test set (710,976 flows).
  - Metrics computed:
    - Per-class Precision, Recall, and F1 across all 6 classes.
    - Macro-averaged Precision, Recall, F1, and Overall Accuracy across seeds 0, 1, 2 (reported as mean $\pm$ std).
    - **Benign False-Positive Rate (FPR)**: defined as the fraction of actual Benign flows in the test set predicted as any attack class ($\frac{FP_{benign}}{N_{benign}} = 1 - \text{Recall}_{benign}$).
  - All metrics are saved to `results/per_class.csv`, `results/summary.csv`, and compared to Table VI in `results/paper_vs_ours.csv`.

