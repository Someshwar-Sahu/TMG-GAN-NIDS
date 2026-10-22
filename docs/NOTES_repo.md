# Repository Audit Notes: TMG-GAN Upstream vs Paper Analysis

**Audit Date:** 22 Oct 2026  
**Upstream Repository:** `third_party/TMG-GAN` (Commit `545cb993ec9033f52e7dbef357150a4639a7aea8`)  
**Base Paper:** Ding et al., *"TMG-GAN: Generative Adversarial Networks-Based Imbalanced Learning for Network Intrusion Detection"*, IEEE TIFS, vol. 19, 2024.

---

## 1. Summary of Upstream Codebase Status

A comprehensive review of the commit history and source code reveals that the upstream repository is in a **hybrid / image-experiment state**:
1. Commit `6ec4c27b4a6c` ("MINIST") and `545cb993ec90` ("img_backup") shifted the active generator, discriminator, and dataset loaders from tabular intrusion detection data to image generation on MNIST and STL-10 (`3 x 32 x 32` RGB images).
2. Earlier commits (e.g. `bde3d64`, `9aa2dcf`, `a8ec6d7`) had tabular MLPs for `GeneratorModel` and `CDModel`, but the final committed state on `main` contains image convolutions (`Conv2d`, `ConvTranspose2d`) and hardcoded STL-10 dataset logic.
3. Therefore, the repository cannot be run as-is for CICIDS2017 without adapting the network architectures to the 78-feature tabular format documented in the paper (specifically Figure 6a).

---

## 2. Repo Files and Lines Specific to Images or STL-10

| File | Line Numbers | Description of Image/STL-10 Specific Code |
|---|---|---|
| `src/datasets/__init__.py` | Lines 2, 8–14 | Imports `torchvision.transforms` and defines an image pipeline: `ToTensor()`, `Resize(32)`, `Normalize(mean=0, std=1)`. |
| `src/datasets/__init__.py` | Lines 16–21 | Directly instantiates and downloads `datasets.STL10(split='train', download=True)`. |
| `src/datasets/__init__.py` | Lines 29–34 | Directly instantiates and downloads `datasets.STL10(split='test', download=True)`. |
| `src/models/generator_model.py` | Lines 12–26 | `self.main_model` consists of 2D transposed convolutions (`nn.ConvTranspose2d`) with channels $z\_size \to 64 \to 32 \to 16$, built for $16 \times 16$ spatial feature maps. |
| `src/models/generator_model.py` | Lines 28–34 | `self.last_layer` uses `nn.ConvTranspose2d(16, 3, 4, 2, 1)`, `BatchNorm2d(3)`, and `Tanh()`, producing a 3-channel $32 \times 32$ image tensor. |
| `src/models/generator_model.py` | Line 42 | `x.view(-1, self.z_size, 1, 1)` reshapes latent vector $z$ into spatial dimensions `[B, z_size, 1, 1]` for transposed 2D convolution. |
| `src/models/generator_model.py` | Line 45 | Hardcoded output reshape: `torch.reshape(self.last_layer(x), [-1, 3, 32, 32])`. |
| `src/models/cd_model.py` | Lines 11–25 | `self.main_model` consists of 4 spectral-norm 2D convolutions: `Conv2d(3, 16)`, `Conv2d(16, 32)`, `Conv2d(32, 64)`, `Conv2d(64, 128)` expecting $3 \times 32 \times 32$ image inputs. Ignores `in_features`. |
| `src/models/sngan_d.py` | Lines 11–25, 33–36 | Duplicate 2D convolutional architecture as `cd_model.py` for $3 \times 32 \times 32$ images. |
| `src/tmg_gan.py` | Lines 102–115 | Periodic image grid visualization: hardcodes 10 classes (`range(10)`), calls `images[i*10 + j].permute(1, 2, 0)` for `imshow`, and saves `tmg_{e}.jpg`. Crashes on non-image or non-10-class data. |
| `scripts/visualize_gan_output.py` | Lines 19–30 | Assumes 10 classes of image data, calls `.permute(1, 2, 0)` and renders a $10 \times 10$ image grid. |
| `tests/test_gan.py`, `tests/test_sngan.py`, `tests/test_wgan.py`, `tests/test_tmg_gan.py` | Various | Test routines that produce 2D image output plots (`gan.jpg`, `sngan.jpg`, `wgan.jpg`, `tmg_gan.jpg`). |

---

## 3. Detailed Differences Between Repo and Paper

### 3.1 Cosine Loss Activation (Eq. 4 & Algorithm 1 Line 11)
- **Paper**: The cosine loss $O_k$ (Eq. 4) between real samples and generated samples of class $k$, combined with divergence from other classes $j \ne k$, is part of the generator training objective throughout training.
- **Repository (`src/tmg_gan.py`, lines 76–77)**:
  ```python
  if e < 1000:
      cd_hidden_loss = 0
  ```
  **Verification**: Confirmed. The cosine similarity loss on hidden representations (`cd_hidden_loss`) is artificially zeroed out for the first 1,000 epochs (`e < 1000`). It is only activated halfway through training (epochs 1001 to 2000).

### 3.2 Generator Loss and Dual Update Mechanism (Eq. 2, 3, 4, 8 & Algorithm 1 Lines 8–13)
- **Paper**: Algorithm 1 specifies that in each round, the generator parameters $\theta(G_k)$ are updated with the combined gradient (Eq. 8, 12):
  $$\nabla_{\theta(G_k)} = \frac{1}{m}\frac{1}{N}\sum_{i=1}^m \sum_{k=1}^N [-D(\tilde{x}_i^k) + C(\tilde{x}_i^k) + O_i^k]$$
  where $O_i^k = \downarrow O_k(F(\tilde{x}_k), F(\tilde{x}_j)) - \uparrow O_k(F(\tilde{x}_k), F(x_k))$ is computed on feature extractor $F(x)$ (the last hidden layer before classification).
- **Repository (`src/tmg_gan.py`, lines 61–100)**:
  The repo executes **two completely separate gradient steps** on the generators:
  1. *Per-class update (Lines 61–80)*: For each class, computes $g\_loss = -score\_generated + loss\_label + cd\_hidden\_loss$ and executes `g_loss.backward()` and `g_optimizers[target_label].step()`. Here `cd_hidden_loss` is only the similarity to real samples of the same class ($\uparrow O_k$), and it is disabled for $e < 1000$.
  2. *Cross-generator diversity update (Lines 81–100)*: Outside the class loop, generates 3 samples (`i.generate_samples(3)`), computes pairwise cosine similarity between generator internal layers `self.generators[i].hidden_status` (NOT on the classifier feature extractor $F$ as in Eq. 3), averages them, scales by `datasets.feature_num`, and runs a separate `g_hidden_loss.backward()` and optimizer step across all generators.

### 3.3 Softmax Applied Twice / Cross-Entropy on Probabilities (Eq. 6)
- **Paper (Eq. 6)**: Defines standard cross-entropy $L_C = -\frac{1}{m}\sum \sum y_{ik} \log(p_{ik})$.
- **Repository (`src/models/cd_model.py` Line 29 & `src/tmg_gan.py` Line 53)**:
  - `CDModel.c_last_layer` terminates with `nn.Softmax(dim=1)`:
    ```python
    self.c_last_layer = nn.Sequential(
        nn.Linear(128, label_num),
        nn.Softmax(dim=1),
    )
    ```
  - In `tmg_gan.py`, the output `predicted_labels` (which already underwent softmax) is passed directly into PyTorch's `cross_entropy`:
    ```python
    c_loss = cross_entropy(
        input=predicted_labels,
        target=torch.full([len(predicted_labels)], target_label, device=config.device),
    )
    ```
  - PyTorch's `cross_entropy` internally applies `log_softmax`. Passing softmax probabilities into `cross_entropy` is mathematically incorrect, leads to compressed gradients, and can cause training instability.

### 3.4 The `.squeeze()` Dimension Collapse Bug on Batch Size 1
- **Repository (`src/models/cd_model.py`, line 39)**:
  ```python
  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
      x = self.main_model(x)
      self.hidden_status = x
      x = x.squeeze()
      return self.d_last_layer(x), self.c_last_layer(x)
  ```
- **Verification**: Confirmed. When a single sample is passed (`batch_size = 1`), `x` before squeezing has shape `[1, 128, 1, 1]`. Calling unqualified `.squeeze()` collapses **all** dimensions of size 1, producing a 1D tensor of shape `[128]`.
- When passed to `self.c_last_layer(x)`, the final layer `Softmax(dim=1)` fails with:
  `IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)`.
- Proper implementation should flatten spatial dimensions while preserving batch dimension: `torch.flatten(x, 1)`.

### 3.5 Noise Sample Filtering & Fallback Behavior (Algorithm 2)
- **Paper (Algorithm 2, lines 6–12)**: Keep generated sample $\tilde{x}_k$ if and only if $C(\tilde{x}_k) == \tilde{Y}_k$. If the classifier rejects the sample, it is discarded, ensuring only high-confidence samples enter the augmented dataset.
- **Repository (`src/tmg_gan.py`, lines 141–152)**:
  ```python
  def generate_qualified_samples(self, target_label: int, num: int):
      result = []
      patience = 10
      while len(result) < num:
          sample = self.generators[target_label].generate_samples(1)
          label = torch.argmax(self.cd(sample)[1])
          if label == target_label or patience == 0:
              result.append(sample.cpu().detach())
              patience = 10
          else:
              patience -= 1
      return torch.cat(result)
  ```
  **Critical findings**:
  1. **Silent Fallback**: When 10 consecutive candidates are rejected (`patience == 0`), the function appends the rejected/misclassified sample anyway! This defeats the purpose of noise filtering.
  2. **Extreme Inefficiency**: Generates only 1 sample at a time in Python (`generate_samples(1)`), resulting in massive overhead.
  3. **Vulnerable to `.squeeze()` bug**: Since `num=1` is passed, `self.cd(sample)` crashes due to the dimension squeeze bug noted in Section 3.4.

### 3.6 Target Oversampling Counts (Table II vs `train_tmg_gan.py`)
- **Paper (Table II)**: Specific target sample sizes per class:
  - Benign: 105,222 (no oversampling)
  - DoS: 21,550 $\to$ 34,609
  - Port Scan: 10,809 $\to$ 23,868
  - Brute Force: 5,235 $\to$ 18,294
  - Web Attack: 1,476 $\to$ 14,535
  - Bot: 857 $\to$ 13,916
- **Repository (`scripts/train_tmg_gan.py`, lines 51–60)**:
  Balances every minority class up to `max_cnt = max([len(samples[i]) ...])` (full class equalization to the majority class size). This differs completely from Table II.

### 3.7 Evaluation Pipeline and Classifier Architecture (Fig. 6a vs `train_tmg_gan.py`)
- **Paper (Fig. 6a, Column 3 & Table VI)**:
  - TMG-GAN is used to generate synthetic samples.
  - The augmented dataset is then used to train an independent **DNN classifier** consisting of: `Reshape(13x6x1)` $\to$ `Conv(13x6x32, 3x3)` $\to$ `MaxPooling(1,1)` $\to$ `Flatten` $\to$ `Dropout(0.2)` $\to$ `Dense(256)` $\to$ `Softmax(6)`.
- **Repository (`scripts/train_tmg_gan.py`, lines 73–75)**:
  - Sets `clf.model = tmg_gan.cd`. Instead of training an independent classifier from scratch, it reuses the GAN's internal discriminator/classifier model and fine-tunes it.

### 3.8 Output Activation & Normalization Range (Eq. 10 vs `generator_model.py`)
- **Paper (Eq. 10 & Fig. 6a)**:
  - Features are normalized between $0.0$ and $1.0$ using Min-Max scaling.
  - In Fig. 6a, the Generator output layer terminates with **Sigmoid**, naturally bounding generated feature values to $[0, 1]$.
- **Repository (`src/models/generator_model.py`, line 33)**:
  - Uses `nn.Tanh()`, producing outputs in $[-1, 1]$. In earlier commit `a8ec6d78`, it had `Sigmoid()` on tabular data.

### 3.9 Weight Initialization Bug in `init_weights`
- **Repository (`src/utils.py`, lines 91–99)**:
  ```python
  def init_weights(layer: nn.Module):
      if type(layer) == nn.Linear:
          nn.init.normal_(layer.weight, 0.0, 0.02)
          ...
      elif type(layer) == nn.BatchNorm1d:
          nn.init.normal_(layer.weight, 1.0, 0.02)
          ...
  ```
  - For image models (`Conv2d`, `ConvTranspose2d`, `BatchNorm2d`), `init_weights` was a no-op because it only handled `nn.Linear` and `nn.BatchNorm1d`.
  - When adapting Fig. 6(a), `Conv2d` and `Linear` layers must have proper initialization.

---

## 4. Key Takeaways and Architectural Blueprint for Step 2

1. **Do not use the upstream repo's STL-10 models directly.** We must construct the clean, paper-accurate architectures specified in Fig. 6(a) for CICIDS2017:
   - **Feature shape**: 78 features reshaped to $(1, 13, 6)$.
   - **Discriminator/Classifier**: Shared hidden layers with Spectral Normalization:
     - `Reshape(1, 13, 6)`
     - `spectral_norm(Conv2d(1, 32, kernel_size=(3, 3), padding=1))`
     - `MaxPooling2d(kernel_size=(1, 1))` (identity spatial pooling)
     - `Flatten` $\to$ `Dropout(0.2)`
     - `spectral_norm(Linear(2496, 256))` $\to$ `LeakyReLU(0.2)` (Feature Extractor $F$)
     - D head: `spectral_norm(Linear(256, 1))`
     - C head: `Linear(256, 6)` (outputting logits for numerical stability)
   - **Generator**:
     - Latent $z \sim \mathcal{N}(0, 1)$ of size 128 (or 100)
     - `Linear(z_dim, 512)` $\to$ `BatchNorm1d` $\to$ `LeakyReLU(0.2)`
     - `Linear(512, 128)` $\to$ `BatchNorm1d` $\to$ `LeakyReLU(0.2)`
     - `Linear(128, 32)` $\to$ `BatchNorm1d` $\to$ `LeakyReLU(0.2)`
     - `Linear(32, 78)` $\to$ `Sigmoid` (ensuring $[0, 1]$ output matching min-max scaler)
2. **Loss stability**: Output logits from the classifier head; compute `F.cross_entropy(logits, target)`.
3. **Filtering**: Fix `generate_qualified_samples` to generate in efficient batches (e.g. 512 or 1024), filter with `argmax(logits) == target_label`, replace `.squeeze()` with `torch.flatten(x, 1)`, and remove the `patience == 0` fallback so only genuinely qualified samples are admitted.
