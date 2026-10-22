import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam

from src import config
from src.dataset import CICIDSTrainingData
from src.models import CDModel, GeneratorModel


class TMGGAN:
    """
    TMG-GAN implementation following Ding et al. (IEEE TIFS 2024).
    Contains:
      - 1 Tied Discriminator/Classifier (CDModel)
      - N Class-specific Generators (GeneratorModel)
    """

    def __init__(
        self,
        feature_num: int = config.FEATURE_NUM,
        label_num: int = config.LABEL_NUM,
        z_size: int = config.Z_SIZE,
        device: torch.device = config.DEVICE,
    ):
        self.device = device
        self.feature_num = feature_num
        self.label_num = label_num
        self.z_size = z_size

        self.cd = CDModel(in_features=feature_num, label_num=label_num).to(device)
        self.generators = [
            GeneratorModel(z_size=z_size, feature_num=feature_num).to(device)
            for _ in range(label_num)
        ]

    def fit(
        self,
        dataset: CICIDSTrainingData,
        epochs: int = config.EPOCHS,
        batch_size: int = config.BATCH_SIZE,
        g_lr: float = config.G_LR,
        cd_lr: float = config.CD_LR,
        cd_loops: int = config.CD_LOOPS,
        g_loops: int = config.G_LOOPS,
        checkpoint_dir: Path = None,
        loss_log_path: Path = None,
        save_freq: int = 100,
    ) -> None:
        """
        Train TMG-GAN following Algorithm 1.
        """
        self.cd.train()
        for g in self.generators:
            g.train()

        cd_optimizer = Adam(self.cd.parameters(), lr=cd_lr, betas=config.ADAM_BETAS)
        g_optimizers = [
            Adam(g.parameters(), lr=g_lr, betas=config.ADAM_BETAS)
            for g in self.generators
        ]

        # Setup loss logging
        if loss_log_path:
            loss_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(loss_log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'd_loss', 'c_loss', 'g_loss', 'cosine_real', 'diversity_loss'])

        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f'Starting TMG-GAN training for {epochs} epochs on {self.device}...')
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_d_losses = []
            epoch_c_losses = []
            epoch_g_losses = []
            epoch_cos_losses = []

            # Loop over all N classes
            for target_class in range(self.label_num):
                # -------------------------------------------------------------
                # 1. Train Discriminator and Classifier (td steps)
                # -------------------------------------------------------------
                for _ in range(cd_loops):
                    cd_optimizer.zero_grad()
                    real_samples = dataset.sample_class(target_class, batch_size)
                    fake_samples = self.generators[target_class].generate_samples(batch_size, self.device)

                    # D and C forward
                    score_real, logits_real = self.cd(real_samples)
                    score_fake, _ = self.cd(fake_samples.detach())

                    # WGAN loss on discriminator
                    d_loss = 0.5 * (score_fake.mean() - score_real.mean())

                    # Classifier cross-entropy on real positive samples (Eq. 6)
                    target_labels = torch.full((len(real_samples),), target_class, device=self.device, dtype=torch.long)
                    c_loss = F.cross_entropy(logits_real, target_labels)

                    loss_cd = d_loss + c_loss
                    loss_cd.backward()
                    cd_optimizer.step()

                    epoch_d_losses.append(d_loss.item())
                    epoch_c_losses.append(c_loss.item())

                # -------------------------------------------------------------
                # 2. Train Generator G_k (tg steps)
                # -------------------------------------------------------------
                for _ in range(g_loops):
                    g_optimizers[target_class].zero_grad()
                    fake_samples = self.generators[target_class].generate_samples(batch_size, self.device)
                    score_fake, logits_fake = self.cd(fake_samples)
                    feat_fake = self.cd.hidden_status

                    # Feature representation of real positive samples (with frozen backbone)
                    with torch.no_grad():
                        real_samples = dataset.sample_class(target_class, batch_size)
                        feat_real = self.cd.extract_features(real_samples)

                    # Cosine similarity to real positive samples (Eq. 2)
                    cos_real = F.cosine_similarity(feat_fake, feat_real, dim=1).mean()

                    # Classification objective on generated samples
                    target_labels = torch.full((len(fake_samples),), target_class, device=self.device, dtype=torch.long)
                    loss_cls_fake = F.cross_entropy(logits_fake, target_labels)

                    # Generator loss (Eq. 8): maximize D score (-score_fake), minimize classification error, maximize similarity to real
                    g_loss = -score_fake.mean() + loss_cls_fake - cos_real
                    g_loss.backward()
                    g_optimizers[target_class].step()

                    epoch_g_losses.append(g_loss.item())
                    epoch_cos_losses.append(cos_real.item())

            # -----------------------------------------------------------------
            # 3. Inter-Generator Diversity Update (Eq. 3, Eq. 4)
            # -----------------------------------------------------------------
            for opt in g_optimizers:
                opt.zero_grad()

            gen_feats = []
            for g in self.generators:
                gen_batch = g.generate_samples(16, self.device)
                feat = self.cd.extract_features(gen_batch)
                gen_feats.append(feat)

            diversity_losses = []
            for i in range(self.label_num):
                for j in range(self.label_num):
                    if i != j:
                        diversity_losses.append(
                            F.cosine_similarity(gen_feats[i], gen_feats[j], dim=1).mean()
                        )

            diversity_loss = torch.stack(diversity_losses).mean()
            diversity_loss.backward()
            for opt in g_optimizers:
                opt.step()

            # Averages for logging
            avg_d = sum(epoch_d_losses) / len(epoch_d_losses)
            avg_c = sum(epoch_c_losses) / len(epoch_c_losses)
            avg_g = sum(epoch_g_losses) / len(epoch_g_losses)
            avg_cos = sum(epoch_cos_losses) / len(epoch_cos_losses)
            div_val = diversity_loss.item()

            if loss_log_path:
                with open(loss_log_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, f'{avg_d:.5f}', f'{avg_c:.5f}', f'{avg_g:.5f}', f'{avg_cos:.5f}', f'{div_val:.5f}'])

            if epoch % 10 == 0 or epoch == 1:
                elapsed = time.time() - start_time
                print(
                    f'Epoch {epoch:4d}/{epochs:4d} [{elapsed:.1f}s] '
                    f'D_loss: {avg_d:+.4f} | C_loss: {avg_c:.4f} | G_loss: {avg_g:+.4f} | '
                    f'Cos_sim: {avg_cos:.4f} | Div: {div_val:.4f}'
                )

            # Save periodic checkpoints
            if checkpoint_dir and (epoch % save_freq == 0 or epoch == epochs):
                ckpt_path = checkpoint_dir / f'tmg_gan_epoch_{epoch}.pt'
                self.save_checkpoint(ckpt_path, epoch)
                print(f'  Saved checkpoint: {ckpt_path}')

        self.cd.eval()
        for g in self.generators:
            g.eval()
        print(f'TMG-GAN training finished in {time.time() - start_time:.1f}s.')

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'epoch': epoch,
                'cd_state_dict': self.cd.state_dict(),
                'generator_state_dicts': [g.state_dict() for g in self.generators],
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.cd.load_state_dict(ckpt['cd_state_dict'])
        for g, state in zip(self.generators, ckpt['generator_state_dicts']):
            g.load_state_dict(state)
        return ckpt.get('epoch', 0)

    def generate_raw_samples(self, target_label: int, num: int) -> torch.Tensor:
        """Generate raw synthetic samples from generator for `target_label`."""
        self.generators[target_label].eval()
        with torch.no_grad():
            samples = self.generators[target_label].generate_samples(num, self.device)
        return samples.cpu()

    def generate_qualified_samples(
        self,
        target_label: int,
        count: int,
        batch_size: int = 512,
        max_batches: int = 200,
        strict: bool = True,
    ) -> tuple[torch.Tensor, float]:
        """
        Algorithm 2: Noise sample filtering.
        Keeps samples generated by G_k only if the classifier C predicts class k.
        Returns: (accepted_samples_tensor, acceptance_rate)
        """
        self.cd.eval()
        self.generators[target_label].eval()

        accepted = []
        total_generated = 0
        batches = 0

        with torch.no_grad():
            while len(accepted) < count and batches < max_batches:
                batches += 1
                curr_batch_size = min(batch_size, (count - len(accepted)) * 4)
                curr_batch_size = max(curr_batch_size, 64)

                candidates = self.generators[target_label].generate_samples(curr_batch_size, self.device)
                total_generated += len(candidates)

                _, logits = self.cd(candidates)
                preds = logits.argmax(dim=1)

                mask = (preds == target_label)
                valid = candidates[mask]

                if len(valid) > 0:
                    accepted.append(valid.cpu())

        if len(accepted) == 0:
            if strict:
                raise RuntimeError(
                    f'Algorithm 2 filtering failed: 0 samples accepted for class {target_label} after {total_generated} generated!'
                )
            # If not strict (e.g. during smoke test on untrained model), fallback to raw generation
            raw = self.generate_raw_samples(target_label, count)
            return raw, 0.0

        combined = torch.cat(accepted, dim=0)
        final_samples = combined[:count]
        acceptance_rate = len(final_samples) / total_generated

        return final_samples, acceptance_rate
