# Project brief: GAN augmentation for imbalanced intrusion detection (CICIDS2017)

Read this whole file first. Work one step at a time. At the end of each step, stop, show the real printed output, and wait for approval.

## 1. Context

- College design project, team of 3. Each student is evaluated individually, so every file must be simple enough for a student to explain.
- Deadlines: 5-6 page IEEE research paper (written by the students) due **30 Oct 2026**; whole project due **5 Nov 2026**. Today is 22 Oct 2026.
- Project topic (from the form): *GAN-Based Data Augmentation for Network Intrusion Detection on Imbalanced Traffic*. A multi-generator conditional GAN generates synthetic minority-class attack samples to balance training data, improving recall on rare attacks without hurting the normal-traffic false-positive rate.
- Base work: Ding et al., "TMG-GAN: Generative Adversarial Networks-Based Imbalanced Learning for Network Intrusion Detection", IEEE TIFS, vol. 19, 2024, DOI 10.1109/TIFS.2023.3331240. PDF: `docs/TMG-GAN_paper.pdf`. Authors' code: https://github.com/universuen/TMG-GAN
- Plan: **Phase 1** follow the paper's pipeline on CICIDS2017. **Phase 2** our addition. The addition is measurement, not a new architecture: the paper never reports false-positive rate and never compares against SMOTE at matched class sizes.

**Research question:** Does TMG-GAN augmentation raise recall on rare attack classes of CICIDS2017 without raising the benign false-positive rate, compared with SMOTE at the same class sizes?

## 2. Hard rules

1. **Never invent, estimate or recall a result.** Every number you report must come from a script run in this folder, with its log saved in `logs/`. If a run failed or is unfinished, say so.
2. **Do not write paper text** (abstract, introduction, related work, discussion). Course rule: no AI-generated paper content. Allowed: code, CSVs, figures, tables, and a factual run log.
3. **No tuning on the test set.** If results differ from the paper, report them as they are.
4. Do not change the scope in section 3 without asking.
5. **Attribution.** Clone the authors' repo into `third_party/TMG-GAN` (read-only reference). Copy only what is needed into `src/`. Keep `NOTICE.md` listing every file adapted from that repo and what was changed. Read the repo's LICENSE and record it there.
6. Use float32, fixed seeds, and print GPU status at the start of each training script. Environment: Windows, PowerShell, venv `.venv` at the project root. Keep `requirements.txt` updated. Commit to git after each step (`git init` if needed).
7. After each step report: files created or changed, the exact command run, and the real printed output. Then stop.

## 3. Scope

**IN:** CICIDS2017 with 6 classes; TMG-GAN training; classifier-based filtering of generated samples; DNN classifier; SMOTE baseline; evaluation including benign false-positive rate; saving the scaler and model weights.

**OUT (do not build):** UNSW-NB15, NSL-KDD, ablation study, sampling-ratio experiments, MAGNETO, TACGAN-IDS, ADASYN, plain GAN / WGAN / SNGAN baselines, image-dataset code (STL10), sbatch files, any UI or demo (until told).

## 4. What is already in the project folder

- `data/datasets/CICIDS2017/raw/`: the 8 original CSV files.
- `scripts/preprocess_cicids.py`, `plot_class_imbalance.py`, `balance_baseline.py`, `train_baseline_dnn.py`, `make_demo_assets.py`: our earlier pipeline. Read them and reuse logic where useful. Known problems with them:
  - they drop 8 constant columns (70 features left), which is **not compatible with the paper** (see section 5);
  - they cap classes at 200k rows;
  - their SMOTE targets were arbitrary, so their results are **superseded**.
- `docs/TMG-GAN_paper.pdf`.

Treat the data as raw. Clean and balance it again following Step 1.

## 5. Facts from the paper (the PDF wins over this summary; verify each)

**Data (Sec. V-A, Table II).** CICIDS2017, 78 feature attributes plus a label. Min-max scaling to [0, 1] (Eq. 10). Six classes, labels 0-5:

| Label | Class | Training set | After oversampling (OS) |
|---|---|---|---|
| 0 | Benign | 105,222 | 105,222 |
| 1 | DoS | 21,550 | 34,609 |
| 2 | Port Scan | 10,809 | 23,868 |
| 3 | Brute Force | 5,235 | 18,294 |
| 4 | Web Attack | 1,476 | 14,535 |
| 5 | Bot | 857 | 13,916 |

Not documented in the paper: the train/test split, duplicate removal, NaN/inf handling. Make a choice, write it down in `docs/DECISIONS.md`, and state it as our assumption.

**Model (Sec. IV, Fig. 4, Fig. 6a).**
- N generators G1..GN, one per class (Benign included), each fed Gaussian noise N(0,1).
- A discriminator D and a classifier C that share hidden layers (tied parameters).
- A feature extractor F, which is the last hidden layer of C.
- Losses: WGAN objective, classifier cross-entropy (Eq. 6), cosine-similarity loss (Eq. 2-4), combined objective (Eq. 5), D update (Eq. 7), G update (Eq. 8), spectral normalization (Eq. 9).
- Training: Algorithm 1 (td discriminator/classifier steps, then tg generator steps per round). Paper uses 2000 epochs.
- Sample generation: Algorithm 2. Keep a generated sample from G_k only if the classifier predicts class k. Generate until the needed count per class is reached.
- Fig. 6(a), as I read it: the discriminator/classifier and the final DNN reshape the 78 features to 13x6x1 and apply a 3x3 convolution with 32 filters, then flatten, dropout 0.2, dense 256, softmax. 13 x 6 = 78, which is why **all 78 features must be kept**. Verify against the figure and ask if anything is ambiguous.

**Paper results to compare against (Table VI, CICIDS2017 multi-class, macro average):**
TMG-IDS precision 0.9963, recall 0.9981, F1 0.9972. DNN only: 0.9125 / 0.6501 / 0.6900. Read the per-class values from the PDF.

## 6. Steps

### Step 0: setup and repo audit
- Clone the repo into `third_party/TMG-GAN`. Check that PyTorch sees the GPU.
- Write `docs/NOTES_repo.md` containing:
  - which repo files and lines are specific to images or STL10;
  - every difference you find between the repo and the paper's Algorithm 1 and Eq. 2-9 (for example when the cosine loss is switched on, how often generator steps run, how the filter is done). A first read suggested the cosine loss may be enabled only late in training and that `.squeeze()` breaks on batches of size 1. Verify; do not trust this.
- Report, then stop.

### Step 1: data
- Create `scripts/preprocess_paper.py`, based on our `preprocess_cicids.py`, with these changes:
  - **keep all 78 features**; handle constant columns without dividing by zero;
  - keep the 6 classes (Benign, DoS, PortScan, BruteForce, WebAttack, Bot) and drop DDoS, Infiltration and Heartbleed, as the paper does;
  - fix the broken dash in the "Web Attack" labels by matching on the prefix;
  - stratified split into train and test; the test set is **never** resampled;
  - subsample the **training** set to the Table II "Training set" sizes;
  - min-max scaling with minimum and maximum from the training set only (a deliberate difference from the paper, note it in `docs/DECISIONS.md`);
  - save the scaler to `artifacts/scaler.json`.
- Print the class counts of train and test, the funnel of rows removed at each step, and the feature count (must be 78).

### Step 2: TMG-GAN on CICIDS
- Adapt the repo's code into `src/` for 78-dimensional input following Fig. 6(a). Do not rewrite it as plain Dense layers unless the figure shows that.
- Smoke test first: a few epochs, check shapes and that all losses are finite.
- Then the full run (target 2000 epochs). Save a checkpoint every 100 epochs, log all losses to `logs/gan_losses.csv`, save final weights to `models/`.
- Produce a t-SNE plot of real versus generated samples per class (reuse the repo's visualization approach).

### Step 3: generation and filtering
- Implement Algorithm 2 in `scripts/generate_samples.py`. Target the OS counts in section 5 (generated = OS - training size per class; Benign is not augmented).
- Guard against infinite loops. Report the acceptance rate of the filter per class.
- Save `data/balanced/gan_train.csv`.

### Step 4: baselines and classifier
- `scripts/make_smote_baseline.py`: SMOTE oversampling of the same training set to the same OS counts, saved separately.
- `src/dnn.py`: the DNN from Fig. 6(a), third column.
- Train the DNN on three training sets: (a) original, (b) SMOTE, (c) TMG-GAN. Use seeds 0, 1, 2 for each. The GAN itself is trained once, which is a limitation to record in `docs/DECISIONS.md`. If GPU time allows, retrain the GAN with seeds 1 and 2 and say so.
- Fixed number of epochs, chosen once and documented. Do not select anything using the test set.

### Step 5: evaluation (our addition)
Compute on the untouched test set, for each method and seed:
- per-class precision, recall and F1;
- macro precision, recall and F1, and accuracy (as in the paper's Table VI);
- **benign false-positive rate**: the fraction of benign test flows predicted as any attack class;
- confusion matrices.

Outputs:
- `results/per_class.csv` with columns seed, method, class, precision, recall, f1;
- `results/summary.csv` with mean and standard deviation over seeds per method;
- `results/paper_vs_ours.csv`, macro values only, with the paper's numbers copied from the PDF;
- `results/recall_by_class.png` (error bars), `results/benign_fpr.png`, `results/confusion_<method>.png`.

### Step 6 (do not start until told): demo
Only the artifacts that make a demo possible: `artifacts/scaler.json` and the saved model weights. Do not build any app yet.

## 7. Target dates

| Dates | Work |
|---|---|
| 22-23 Oct | Steps 0 and 1 |
| 23-25 Oct | Step 2 (GAN training) |
| 25-26 Oct | Steps 3 and 4 |
| 27 Oct | Step 5, final numbers |
| 28-30 Oct | Students write the paper |

If Step 2 is not working by 25 Oct, stop and report what you see so the team can decide.
