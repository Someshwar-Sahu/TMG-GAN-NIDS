import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CICIDSTrainingData:
    """
    In-memory dataset container for CICIDS2017 training flows.
    Loads x_train.csv and y_train.csv into memory as float32 tensors and
    groups them by class for class-conditional sampling.
    """

    def __init__(self, data_dir: Path, device: torch.device):
        self.device = device
        x_path = data_dir / 'x_train.csv'
        y_path = data_dir / 'y_train.csv'

        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f'Training files missing in {data_dir}. Run scripts/preprocess_paper.py first.'
            )

        print(f'Loading training data from {data_dir}...')
        x_df = pd.read_csv(x_path, dtype=np.float32)
        y_df = pd.read_csv(y_path, dtype=np.int8)

        self.features = torch.tensor(x_df.values, dtype=torch.float32)
        # y_df is one-hot; convert to integer class labels 0..5
        self.labels = torch.tensor(y_df.values.argmax(axis=1), dtype=torch.long)
        self.feature_names = x_df.columns.tolist()
        self.feature_count = self.features.shape[1]
        self.total_samples = len(self.labels)

        # Group sample indices by class
        self.samples_by_class = {}
        for c in range(y_df.shape[1]):
            indices = (self.labels == c).nonzero(as_tuple=True)[0]
            self.samples_by_class[c] = self.features[indices]
            print(f'  Class {c} samples: {len(indices):,}')

    def sample_class(self, label: int, batch_size: int) -> torch.Tensor:
        """Sample random batch of real flows for class `label`."""
        class_tensor = self.samples_by_class[label]
        num_avail = len(class_tensor)
        # Random choice with replacement if batch_size > available
        idx = torch.randint(0, num_avail, (batch_size,))
        return class_tensor[idx].to(self.device)


class FlowDataset(Dataset):
    """Standard PyTorch Dataset for classifier training."""

    def __init__(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
