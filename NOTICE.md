# Attribution and Third-Party Notices

## Upstream Repository
- **Project**: TMG-GAN: Generative Adversarial Networks-Based Imbalanced Learning for Network Intrusion Detection
- **Authors**: Y. Ding, Y. Chen, S. Su, X. Cui et al. (IEEE TIFS 2024, DOI: 10.1109/TIFS.2023.3331240)
- **Repository URL**: https://github.com/universuen/TMG-GAN
- **Cloned Location**: `third_party/TMG-GAN` (read-only reference)
- **License**: The upstream GitHub repository does not include a `LICENSE` file (unlicensed / all rights reserved by author `universuen`). Used under academic fair use for academic reproduction, evaluation, and research comparison.

## Files Adapted into `src/`

1. **`src/models.py`**
   - Adapted from upstream `src/models/cd_model.py`, `src/models/generator_model.py`, and `src/utils.py`.
   - **Modifications**:
     - Converted `CDModel` from 3-channel 32x32 image convolutions to Fig. 6(a) architecture: sequence data (78 features) reshaped to `(B, 1, 13, 6)`, `Conv2d(1, 32, (3,3))` with spectral normalization, `MaxPool2d(1,1)`, `Flatten(1)` (fixing the batch_size=1 `.squeeze()` bug), `Dropout(0.2)`, `Linear(2496, 256)` with spectral normalization.
     - Separated Discriminator output (`Linear(256, 1)`) and Classifier logits (`Linear(256, 6)`) without premature Softmax to ensure stable cross-entropy gradients.
     - Implemented `GeneratorModel` following Fig. 6(a): `Linear(z, 512)` -> `BatchNorm1d` -> `Linear(512, 128)` -> `Linear(128, 32)` -> `Linear(32, 78)` with `Sigmoid()` output matching Min-Max scaled features $[0, 1]$.
     - Fixed `init_weights` to support both `Conv2d` and `Linear` layers.

2. **`src/tmg_gan.py`**
   - Adapted from upstream `src/tmg_gan.py`.
   - **Modifications**:
     - Removed hardcoded STL-10 image rendering loop and 10-class assumption.
     - Replaced individual sample-at-a-time loop in `generate_qualified_samples` with efficient batched generation.
     - Removed the unsafe `patience == 0` fallback that previously accepted misclassified samples, ensuring strict compliance with Algorithm 2 noise filtering.
     - Restored the cosine hidden feature similarity loss $\uparrow O_k$ from the start of training per Eq. 4.
     - Added periodic checkpointing and per-epoch CSV loss logging to `logs/gan_losses.csv`.

3. **`src/dataset.py`**
   - Clean implementation replacing upstream `src/datasets/` which hardcoded STL-10 downloads.
   - Loads the preprocessed CICIDS2017 training set (`x_train.csv`, `y_train.csv`) and manages fast class-conditional sampling.
