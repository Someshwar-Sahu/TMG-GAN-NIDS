import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


def init_weights(m: nn.Module) -> None:
    """Weight initialization following standard GAN best practices."""
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight, mean=1.0, std=0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class CDModel(nn.Module):
    """
    Discriminator and Classifier with tied shared layers following Figure 6(a).
    Input: (B, 78) sequence features
    Reshape: (B, 1, 13, 6)
    Conv2d: 32 filters, 3x3 kernel, padding=1 (Spectral Norm)
    MaxPool2d: pool_size=(1, 1)
    Flatten: (B, 2496)
    Dropout: 0.2
    Dense: 256 units (Spectral Norm) -> High-dimensional feature extractor F(x)
    Split into:
      - Discriminator Head: Dense(1) (Spectral Norm) -> scalar WGAN score
      - Classifier Head: Dense(6) -> logits
    """

    def __init__(self, in_features: int = 78, label_num: int = 6):
        super().__init__()
        assert in_features == 78, f'CDModel expects 78 features for 13x6 reshape, got {in_features}'
        self.in_features = in_features
        self.label_num = label_num

        # Shared feature extraction backbone
        self.conv = spectral_norm(nn.Conv2d(1, 32, kernel_size=3, padding=1))
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=(1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc_shared = spectral_norm(nn.Linear(32 * 13 * 6, 256))
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        # Output heads
        self.d_head = spectral_norm(nn.Linear(256, 1))
        self.c_head = nn.Linear(256, label_num)

        # Store last extracted feature representation for cosine similarity
        self.hidden_status: torch.Tensor = None

        self.apply(init_weights)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts the 256-dimensional hidden state F(x)."""
        # Ensure 2D tensor (B, 78)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # Reshape to 2D spatial feature map (B, 1, 13, 6)
        h = x.view(-1, 1, 13, 6)
        h = self.conv(h)
        h = self.act1(h)
        h = self.pool(h)
        h = torch.flatten(h, 1)  # (B, 2496) - safe on batch_size=1
        h = self.dropout(h)
        h = self.fc_shared(h)
        h = self.act2(h)  # (B, 256)
        return h

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning (d_score, c_logits).
        d_score shape: (B, 1)
        c_logits shape: (B, label_num)
        """
        features = self.extract_features(x)
        self.hidden_status = features
        d_score = self.d_head(features)
        c_logits = self.c_head(features)
        return d_score, c_logits


class GeneratorModel(nn.Module):
    """
    Generator following Figure 6(a).
    Input: Gaussian noise z ~ N(0, 1) of shape (B, z_size)
    Dense: 512 -> BatchNorm1d -> LeakyReLU(0.2)
    Dense: 128 -> BatchNorm1d -> LeakyReLU(0.2)
    Dense: 32  -> BatchNorm1d -> LeakyReLU(0.2)
    Dense: 78  -> Sigmoid (outputs in [0, 1] matching min-max scaling)
    """

    def __init__(self, z_size: int = 128, feature_num: int = 78):
        super().__init__()
        self.z_size = z_size
        self.feature_num = feature_num

        self.block1 = nn.Sequential(
            nn.Linear(z_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_block = nn.Sequential(
            nn.Linear(32, feature_num),
            nn.Sigmoid(),
        )

        self.hidden_status: torch.Tensor = None
        self.apply(init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h1 = self.block1(z)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        self.hidden_status = h3  # 32-dim penultimate feature
        out = self.out_block(h3)
        return out

    def generate_samples(self, num: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num, self.z_size, device=device, dtype=torch.float32)
        return self.forward(z)
