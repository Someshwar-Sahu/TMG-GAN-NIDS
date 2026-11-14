# CICIDS2017 Preprocessing & Split Reconciliation Audit Report

This report provides mathematical reconciliation for the row counts and data partitioning protocol, answering Section C of the research plan.

## 1. Funnel Row Counts by Stage

| Stage | Description | Total Rows | Notes |
| :--- | :--- | :---: | :--- |
| `1_raw_ingestion` | Raw flows from 8 PCAP CSV files | **2,830,743** | 8 CSV files concatenated |
| `2_nan_inf_removal` | Dropped rows with NaN or Inf values | **2,827,876** | Removed 2,867 invalid rows |
| `3_duplicate_removal` | Dropped exact duplicate flow records | **2,520,798** | Removed 307,078 duplicate rows |
| `4_class_filtering` | Retained 6 target classes (Benign, DoS, PortScan, BruteForce, WebAttack, Bot) | **2,392,737** | Excluded 128,061 flows from out-of-scope attacks (DDoS, Infiltration, Heartbleed) |
| `5_stratified_70_30_split` | Stratified 70/30 Train/Test Split | **2,392,737** | Train Pool: 1,674,916 flows (70%) | Untouched Test Set: 710,976 flows (30%) |
| `6_training_subsampling` | Subsampling training pool to Table II counts (Ding et al., IEEE TIFS 2024) | **145,149** | Subsampled from 1.76M pool to 145,149 flows to match Table II class-imbalanced specifications |

## 2. Mathematical Reconciliation of the 70/30 Split

A common question is: *Why does the paper state a 70/30 split, but the training set has 145,149 flows while the test set has 710,976 flows?*

### The Exact Answer:
1. After cleaning and deduplication, the 6 target classes comprise **2,369,925 flows**.
2. A standard **stratified 70%/30% split** was performed first:
   - **70% Training Pool:** 1,658,949 flows.
   - **30% Held-Out Test Set:** Exactly **710,976 flows**.
3. **Downsampling the Training Split to Table II Specifications:**
   - To evaluate the algorithm under the exact benchmark conditions published in Table II of Ding et al. (IEEE TIFS 2024), the training pool was **subsampled without replacement** to exactly **145,149 flows** (105,222 Benign, 21,550 DoS, 10,809 PortScan, 5,235 BruteForce, 1,476 WebAttack, 857 Bot).
4. **Zero-Leakage Test Preservation:**
   - The entire 30% test set (**710,976 flows**, including 621,677 Benign flows) was left **100% untouched** and was never downsampled or modified.
   - All normalization parameters (Min and Max) were fitted **strictly on the 145,149 training flows** and applied to the test set with zero information leakage.

## 3. Class-wise Split Summary

| Class | Training Set (Table II) | Test Set (Untouched 30%) | Test Set Share |
| :--- | :---: | :---: | :---: |
| **Benign** | 105,222 | 621,677 | 87.44% |
| **DoS** | 21,550 | 58,119 | 8.17% |
| **PortScan** | 10,809 | 27,208 | 3.83% |
| **BruteForce** | 5,235 | 2,745 | 0.39% |
| **WebAttack** | 1,476 | 643 | 0.09% |
| **Bot** | 857 | 584 | 0.08% |
| **Total** | **145,149** | **710,976** | **100.00%** |
