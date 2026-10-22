import torch
from torch import nn


class DNNClassifier(nn.Module):
    """
    DNN Classifier from Figure 6(a), third column:
    Input: (B, 78) sequence features
    Reshape: (B, 1, 13, 6)
    Conv2d: 32 filters, 3x3 kernel, padding=1
    ReLU
    MaxPool2d: pool_size=(1, 1)
    Flatten: (B, 2496)
    Dropout: 0.2
    Dense: 256 units
    ReLU
    Dense: 6 units (outputting logits for numerical stability with cross_entropy)
    """

    def __init__(self, in_features: int = 78, num_classes: int = 6):
        super().__init__()
        assert in_features == 78, f'DNNClassifier expects 78 features for 13x6 reshape, got {in_features}'
        self.in_features = in_features
        self.num_classes = num_classes

        self.conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.act1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=(1, 1))

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32 * 13 * 6, 256)
        self.act2 = nn.ReLU(inplace=True)

        self.fc_out = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits of shape (B, num_classes)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = x.view(-1, 1, 13, 6)
        h = self.conv(h)
        h = self.act1(h)
        h = self.pool(h)
        h = torch.flatten(h, 1)  # (B, 2496) - safe on batch_size=1
        h = self.dropout(h)
        h = self.fc1(h)
        h = self.act2(h)
        logits = self.fc_out(h)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Inference helper returning normalized probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference helper returning integer class predictions."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            preds = logits.argmax(dim=1)
        return preds
