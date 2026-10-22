import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.dnn import DNNClassifier

EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-3
SEEDS = [0, 1, 2]
METHODS = ['original', 'smote', 'tmg_gan']


def load_dataset_for_method(method: str):
    """Loads feature matrix X and integer labels y for the specified training set."""
    if method == 'original':
        data_dir = PROJECT_ROOT / 'data' / 'datasets' / 'CICIDS2017'
        x_path = data_dir / 'x_train.csv'
        y_path = data_dir / 'y_train.csv'
    elif method == 'smote':
        data_dir = PROJECT_ROOT / 'data' / 'balanced'
        x_path = data_dir / 'x_train_smote.csv'
        y_path = data_dir / 'y_train_smote.csv'
    elif method == 'tmg_gan':
        data_dir = PROJECT_ROOT / 'data' / 'balanced'
        x_path = data_dir / 'x_train_gan.csv'
        y_path = data_dir / 'y_train_gan.csv'
    else:
        raise ValueError(f'Unknown method: {method}')

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f'Missing training data: {x_path} or {y_path}')

    print(f'Loading {method} training set from {x_path}...')
    x_df = pd.read_csv(x_path, dtype=np.float32)
    y_df = pd.read_csv(y_path, dtype=np.int8)

    X = torch.tensor(x_df.values, dtype=torch.float32)
    y = torch.tensor(y_df.values.argmax(axis=1), dtype=torch.long)
    print(f'  Loaded {len(y):,} samples with {X.shape[1]} features.')
    return X, y


def train_single_model(method: str, seed: int, X: torch.Tensor, y: torch.Tensor, device: torch.device):
    print(f'\n{"-"*60}')
    print(f'Training DNN on [{method.upper()}] with Seed={seed} on {device}...')
    print(f'{"-"*60}')

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = DNNClassifier(in_features=78, num_classes=6).to(device)
    optimizer = Adam(model.parameters(), lr=LR)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True if device.type == 'cuda' else False)

    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

        train_loss = total_loss / total
        train_acc = correct / total

        if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
            elapsed = time.time() - start_time
            print(f'  Epoch {epoch:2d}/{EPOCHS} [{elapsed:4.1f}s] Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}')

    total_time = time.time() - start_time
    print(f'Finished training {method} (seed={seed}) in {total_time:.1f}s.')

    models_dir = PROJECT_ROOT / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = models_dir / f'dnn_{method}_seed_{seed}.pt'
    torch.save(model.state_dict(), model_save_path)
    print(f'Saved model weights to: {model_save_path}')

    return model_save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', default=METHODS, help='Methods to train: original smote tmg_gan')
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS, help='Seeds to train: 0 1 2')
    args = parser.parse_args()

    print('=' * 75)
    print('STEP 4B: Train DNN Classifiers on Original, SMOTE, and TMG-GAN Datasets')
    print('=' * 75)

    config.print_gpu_status()
    device = config.DEVICE

    print(f'Target configuration: {len(args.methods)} methods x {len(args.seeds)} seeds = {len(args.methods) * len(args.seeds)} models')
    print(f'Epochs per model: {EPOCHS} (fixed once and documented in docs/DECISIONS.md)')
    print(f'Batch size: {BATCH_SIZE} | Learning rate: {LR}')

    overall_start = time.time()
    trained_models = []

    for method in args.methods:
        X, y = load_dataset_for_method(method)
        for seed in args.seeds:
            path = train_single_model(method, seed, X, y, device)
            trained_models.append(str(path))

    overall_time = time.time() - overall_start
    print('\n' + '=' * 75)
    print(f'ALL {len(trained_models)} MODELS TRAINED SUCCESSFULLY in {overall_time / 60:.2f} minutes ({overall_time:.1f}s)!')
    print('Saved models:')
    for p in trained_models:
        print(f'  {p}')
    print('=' * 75)


if __name__ == '__main__':
    main()
