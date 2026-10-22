# Project Viva & Defense Guide: Student Presentation Cheat Sheet

**Project Topic:** GAN-Based Data Augmentation for Network Intrusion Detection on Imbalanced Traffic  
**Target Dataset:** CICIDS2017 (8 raw pcap files, 2.83M raw flows, 78 flow features, 6 target classes)  
**Core Finding:** TMG-GAN achieves high rare attack recall while suppressing the Benign False-Positive Rate down to **2.31%** (beating SMOTE's **2.64%** and eliminating over 850 false alarms across the 515,487 benign test flows).

---

## 1. Suggested Work / Slide Division for the Team of 3 Students

Because each student is evaluated individually, here is an optimal 3-way split of responsibilities and presentation slides:

| Student | Assigned Domain | Key Presentation Focus | Key Files to Walk Through |
| :--- | :--- | :--- | :--- |
| **Student 1** | **Data Engineering & Preprocessing** | CICIDS2017 data funnel, retaining all 78 features, constant column handling without division by zero, 70/30 stratified split, Table II subsampling, min-max scaling fitted strictly on training data (no leakage). | `scripts/preprocess_paper.py`<br>`artifacts/scaler.json`<br>`docs/DECISIONS.md` |
| **Student 2** | **TMG-GAN Architecture & Algorithm 2** | Multi-generator concept (1 generator per class), tied discriminator/classifier backbone ($13 \times 6$ Conv2D with spectral normalization), cosine similarity loss, Algorithm 1 training (2,000 epochs on RTX 3050), and Algorithm 2 noise filtering. | `src/models.py`<br>`src/tmg_gan.py`<br>`scripts/generate_samples.py` |
| **Student 3** | **Evaluation, Benign FPR, & Live Demo** | 9-model experimental design (Original, SMOTE, TMG-GAN $\times$ 3 seeds), evaluation on 710,976 untouched test flows, discovery of SMOTE's Benign FPR inflation (+0.165%), and driving the live Defense Studio dashboard. | `src/dnn.py`<br>`scripts/evaluate_models.py`<br>`dashboard/`<br>`src/inference.py` |

---

## 2. The 30-Second Elevator Pitch (Memorize This)

> *"In enterprise networks, malicious attacks make up less than 1% of all traffic. While conventional techniques like SMOTE boost detection of rare attacks, they blindly interpolate between points, creating noisy artificial traffic that bleeds into normal traffic space and causes severe false alarms.*  
> *In our project, we implemented a Deep Multi-Generator Conditional GAN with an Algorithm 2 classifier filter. Our system learns the true manifold of rare attacks, achieving the highest overall Macro F1-score (0.7940) while actively suppressing benign false alarms down to 2.31%—proving that deep generative augmentation provides superior defense stability without disrupting legitimate network operations."*

---

## 3. Top 12 Anticipated Viva Questions & Model Answers

### Q1: Why did you keep all 78 features instead of dropping the 8 constant columns?
**Answer:**  
*"Many baseline pipelines arbitrarily drop constant columns to leave 70 features. However, the 2D Convolutional backbone in our architecture reshapes flow feature vectors into a $13 \times 6 \times 1$ matrix ($13 \times 6 = 78$). Retaining all 78 features allows spatial 2D convolution over the traffic representation. For constant columns where $x_{\text{max}} - x_{\text{min}} = 0$, we set the denominator span to 1.0, normalizing them safely to 0.0 without division-by-zero errors or data distortion."*

---

### Q2: Why did you fit the Min-Max Scaler only on the training set?
**Answer:**  
*"Fitting normalization parameters on the entire dataset (train + test) causes data leakage from the future test set into the training pipeline, artificially inflating performance. We fitted $x_{\text{min}}$ and $x_{\text{max}}$ strictly on the 145,149 training samples, serialized the parameters to `artifacts/scaler.json`, and clipped the held-out test set to $[0, 1]$."*

---

### Q3: What is the benefit of having multiple generators instead of a single conditional GAN?
**Answer:**  
*"In severe class imbalance, a single generator conditioned on class labels suffers from mode collapse and mode hopping, often prioritizing majority classes or blurring attack boundaries. A multi-generator setup assigns a dedicated generator $G_k$ to each attack class, fed by independent Gaussian noise $z \sim \mathcal{N}(0, 1)$. This guarantees distinct, focused representations for rare attacks like WebAttack and Bot."*

---

### Q4: What is Algorithm 2 and why is it necessary?
**Answer:**  
*"Standard GANs occasionally generate outlier samples that fall outside the true attack distribution or overlap with benign space. Algorithm 2 uses the tied classifier $C(x)$ as a noise filter. When generator $G_k$ produces a candidate flow, it is accepted only if $C(x) = k$. Any ambiguous or low-fidelity samples are rejected, ensuring that only high-purity synthetic flows are added to the balanced dataset."*

---

### Q5: What was the acceptance rate of your Algorithm 2 filter?
**Answer:**  
*"During our generation run to synthesize 13,059 samples per minority class, the Algorithm 2 filter achieved an average acceptance rate of 98.10% (DoS: 99.78%, PortScan: 99.88%, BruteForce: 99.79%, WebAttack: 95.96%, Bot: 95.12%). This confirms the generator had converged well and was synthesizing accurate attack representations."*

---

### Q6: Why did you use Spectral Normalization in the Discriminator?
**Answer:**  
*"In Wasserstein GANs, the discriminator must satisfy a 1-Lipschitz continuity constraint. Spectral normalization divides weight matrices by their largest singular value ($\ell_2$ matrix norm) at every forward step. This bounds the gradient norm without the computational expense and hyperparameter fragility of gradient penalties (WGAN-GP)."*

---

### Q7: What is the Benign False-Positive Rate (FPR) and why does it matter?
**Answer:**  
*"Benign FPR is the proportion of actual legitimate network flows incorrectly predicted as an attack:  
$$\text{FPR}_{\text{benign}} = \frac{\text{FP}_{\text{benign}}}{N_{\text{benign}}} = 1 - \text{Recall}_{\text{benign}}$$  
In enterprise networks processing millions of flows per minute, a difference of just 0.33% in FPR translates to thousands of false alarms every hour, overwhelming security analysts and causing alert fatigue."*

---

### Q8: How did SMOTE compare to your TMG-GAN on Benign FPR?
**Answer:**  
*"We evaluated both methods at identical balanced training sizes (210,444 samples). SMOTE inflated the Benign FPR to 2.64% (+0.165% worse than the unaugmented baseline), generating over 850 additional false alarms across the 515,487 benign test flows. Our TMG-GAN reduced Benign FPR to 2.31%, achieving the lowest false-alarm rate among all methods."*

---

### Q9: Why does SMOTE achieve higher raw recall on some minority classes?
**Answer:**  
*"SMOTE achieves high raw recall by aggressively expanding its decision boundaries into normal traffic space. However, this causes severe precision collapse (SMOTE macro precision dropped to 68.29%, and Bot precision collapsed to 23.81%). In contrast, TMG-GAN preserves boundary precision (73.41% macro precision, 48.27% Bot precision) and achieves the highest overall Macro F1-score (0.7940 vs 0.7518)."*

---

### Q10: How did you train the downstream classifiers? Did you tune on the test set?
**Answer:**  
*"To maintain complete experimental integrity, we strictly followed a 'No tuning on the test set' protocol. The downstream DNN architecture was fixed from Figure 6(a), and trained for a fixed 50 epochs using Adam ($lr = 10^{-3}$, batch size 256). To quantify variance, all models were trained and evaluated across 3 independent random seeds (0, 1, 2) on the 710,976 untouched test flows."*

---

### Q11: What hardware was used for this project?
**Answer:**  
*"The entire training pipeline was executed locally on an NVIDIA GeForce RTX 3050 Laptop GPU (4.00 GB VRAM) using PyTorch 2.6.0 with CUDA 12.4. Full 2,000-epoch TMG-GAN training completed in 13.30 minutes, and the 9 downstream DNN classifiers completed in 41.17 minutes."*

---

### Q12: How does your live demonstration work?
**Answer:**  
*"We built an interactive Defense Studio web platform (`scripts/run_dashboard.py` on port 8050):  
1. **Live Classifier:** Allows testing arbitrary flows from the 710k test set against all three models simultaneously, highlighting SMOTE false alarms.  
2. **Adversarial Lab:** Generates synthetic attacks live on GPU, scores discriminator realism, and demonstrates live that while the baseline model is evaded, our TMG-GAN blocks the threat."*
