# IEEE Paper Writing Checklist & Section Architecture

**Deadlines:** 5–6 page IEEE research paper due **30 Oct 2026** (written by the students).  
**Course Policy Reminder:** No AI-generated paper prose. This checklist provides the exact architectural blueprint, paragraph-by-paragraph technical outline, and figure/table references so your team can write the paper quickly and effectively in your own authentic words.

---

## Paper Title Recommendations
Choose one of these strong, academic titles for your paper:
1. *Generative Adversarial Network Data Augmentation for Network Intrusion Detection Under Severe Class Imbalance*
2. *Balancing Rare Attack Detection and Benign False-Alarm Suppression Using Deep Multi-Generator GANs*
3. *Adversarial Data Augmentation for Robust Intrusion Detection: An Empirical Analysis of Benign False-Positive Dynamics*

---

## Suggested Page Budget (5–6 Pages Total)

| Section | Suggested Length | Key Focus | Corresponding Project Files |
| :--- | :--- | :--- | :--- |
| **Abstract & Keywords** | ~0.5 column | Problem, proposed multi-generator + filter, key numbers (FPR 2.31%, F1 0.7940). | `results/summary.csv` |
| **I. Introduction** | ~1 page | Class imbalance in NIDS, flaw of SMOTE (inflated false alarms), our contribution. | `docs/DECISIONS.md` |
| **II. Related Work** | ~0.75 page | Conventional sampling (SMOTE/ADASYN) vs GANs; reference Ding et al. (2024). | `third_party/TMG-GAN/` |
| **III. Proposed Methodology** | ~1.5 pages | Multi-generator architecture, tied Conv2D backbone, Algorithm 2 noise filter. | `src/models.py`, `src/tmg_gan.py` |
| **IV. Experimental Setup** | ~0.75 page | CICIDS2017 raw files, 78 features, 70/30 split, min-max scaling, training setup. | `scripts/preprocess_paper.py`, Table II |
| **V. Results & Discussion** | ~1.25 pages | Tables I, II, III, Benign FPR trade-off (-0.33%), per-class recall/precision. | `docs/PAPER_TABLES.md`, `results/` |
| **VI. Conclusion & Future Work**| ~0.5 column | Summary of findings, future deployment on live network taps. | -- |
| **References** | ~0.5 column | IEEE-formatted references (CICIDS2017, Ding et al., WGAN, etc.). | `NOTICE.md` |

---

## Section-by-Section Writing Blueprint

### Abstract & Keywords
- **Sentence 1–2 (Context):** Network Intrusion Detection Systems (NIDS) face extreme class imbalance, where rare attacks (Bot, WebAttack) constitute $<1\%$ of network flows.
- **Sentence 3–4 (Problem):** Conventional oversampling methods like SMOTE rely on linear interpolation, resulting in decision boundary expansion that severely inflates false-positive rates on normal traffic.
- **Sentence 5–6 (Method):** We propose/reproduce a deep multi-generator conditional GAN with tied convolutional feature extraction and an Algorithm 2 classifier noise filter.
- **Sentence 7–8 (Results):** Evaluated on 710,976 untouched CICIDS2017 test flows across 3 seeds, our model achieves a Macro F1 of **0.7940** (vs. SMOTE's 0.7518) while suppressing Benign False-Positive Rate down to **2.31%** (beating SMOTE's 2.64% and preventing $>850$ false alarms).
- **Keywords:** Network Intrusion Detection, Class Imbalance, Generative Adversarial Networks, False-Positive Rate, Deep Learning, CICIDS2017.

---

### Section I: Introduction
- **Paragraph 1:** The explosion of enterprise network traffic; reliance on deep learning for signature-less threat detection.
- **Paragraph 2 (The Imbalance Dilemma):** Rare attacks vs. benign traffic. Describe how minority classes are neglected in standard loss functions.
- **Paragraph 3 (The Operational Cost of False Positives):** Why accuracy alone is misleading in cybersecurity. A $0.1\%$ increase in Benign False-Positive Rate (FPR) means thousands of false alarms per day in a 10Gbps SOC, causing analyst fatigue.
- **Paragraph 4 (Contributions):** Bullet-point your 3 key contributions:
  1. Complete implementation of a multi-generator conditional GAN keeping all 78 spatial-temporal flow features via a $13 \times 6$ Conv2D backbone.
  2. Strict experimental protocol eliminating test set leakage (scaler fitted solely on training data; evaluation on 710,976 untouched test flows across 3 seeds).
  3. First empirical quantification of the Benign False-Positive Rate trade-off between generative augmentation and matched-size SMOTE.

---

### Section II: Related Work
- **Subsection A (Conventional Resampling):** Random oversampling, SMOTE, ADASYN. Explain the synthetic linear interpolation mechanism and its limitation in high-dimensional tabular spaces.
- **Subsection B (Generative Adversarial Networks in Tabular Security):** Vanilla GAN, WGAN-GP, and recent advances such as Ding et al. (IEEE TIFS 2024). Note that prior literature focused heavily on raw recall while omitting benign false-alarm rate measurements.

---

### Section III: Proposed Methodology
- **Figure to Include:** Architecture diagram of Figure 6(a) (Input $78 \to 13 \times 6 \times 1 \to$ Conv2D $32 \to$ Dense $256 \to$ Heads).
- **Subsection A (Multi-Generator Design):** Dedicated generator $G_k$ per minority attack class fed with Gaussian noise $z \sim \mathcal{N}(0, 1)$.
- **Subsection B (Shared Backbone & Spectral Normalization):** Conv2D feature extractor $F(x)$ tied between Discriminator $D(x)$ and Classifier $C(x)$ with spectral norm satisfying 1-Lipschitz continuity.
- **Subsection C (Algorithm 2 Noise Filtering):** Mathematical formulation of the rejection gate:
  $$\text{Sample Accepted} \iff \arg\max C(x) = k$$
  Report the 98.10% acceptance rate achieved during training.

---

### Section IV: Experimental Setup
- **Dataset:** CICIDS2017, 8 raw pcap CSV files (2,830,743 flows).
- **Table to Paste:** **Table II (Dataset Distribution & Oversampling Targets)** from `docs/PAPER_TABLES.md`.
- **Data Funnel:** Nan/inf handling, duplicate removal, 70/30 stratified split, subsampling to Table II sizes (145,149 training flows; 710,976 held-out test flows).
- **Normalizer:** Min-max normalization fitted strictly on training data (preventing data leakage).
- **Classifier Protocol:** Downstream DNN trained for 50 fixed epochs, Adam optimizer ($lr=10^{-3}$, batch size 256), seeds 0, 1, 2.
- **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU, PyTorch 2.6.0 with CUDA 12.4.

---

### Section V: Results & Discussion
- **Table to Paste:** **Table I (Overall Benchmark Comparison & Benign FPR)** from `docs/PAPER_TABLES.md`.
- **Figures to Embed:**
  - `results/benign_fpr.png` (Bar chart showing 2.31% vs 2.64% FPR).
  - `results/recall_by_class.png` (Per-class recall with error bars).
  - `results/confusion_tmg_gan.png` (Normalized confusion matrix).
- **Discussion Points:**
  1. *FPR Suppression:* Highlight the 0.33% reduction in false alarms relative to SMOTE.
  2. *Precision Collapse in SMOTE:* Explain why SMOTE's high recall came at the cost of precision collapse (68.29% macro precision; 23.81% on Bot).
  3. *Manifold Realism:* Explain how TMG-GAN preserved boundary sharpness (73.41% macro precision; 48.27% on Bot).

---

### Section VI: Conclusion
- Summarize the research question: TMG-GAN successfully boosts rare attack detection without inflating benign false-positive rates.
- Future work: Hardware acceleration on FPGA for live line-rate network tapping and adversarial robustness testing against zero-day evasion attacks.
