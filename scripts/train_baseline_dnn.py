from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = Path(__file__).absolute().parent.parent / 'data' / 'datasets' / 'CICIDS2017'
PLOT_DIR = DATA_DIR.parent.parent / 'plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ['Benign', 'DoS', 'PortScan', 'BruteForce', 'WebAttack', 'Bot']
EPOCHS = 10
BATCH_SIZE = 1024
LR = 1e-3
SEED = 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load(x_name: str, y_name: str) -> tuple[np.ndarray, np.ndarray]:
    x = pd.read_csv(DATA_DIR / x_name, dtype=np.float32).values
    y = pd.read_csv(DATA_DIR / y_name).values.argmax(axis=1)
    return x, y


def make_model(in_features: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, len(CLASSES)),
    ).to(DEVICE)


def train(x: np.ndarray, y: np.ndarray) -> nn.Module:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = make_model(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    dl = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y).long()),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    model.train()
    for e in range(EPOCHS):
        total = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        print(f'\repoch {e + 1}/{EPOCHS}  loss {total / len(dl.dataset):.4f}', end='')
    print()
    return model


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray) -> dict:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), 65536):
            batch = torch.from_numpy(x[i:i + 65536]).to(DEVICE)
            preds.append(model(batch).argmax(dim=1).cpu().numpy())
    pred = np.concatenate(preds)
    labels = list(range(len(CLASSES)))
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)
    benign_total = cm[0].sum()
    fpr = (benign_total - cm[0, 0]) / benign_total  # benign traffic flagged as attack
    return {
        'precision': p,
        'recall': r,
        'f1': f,
        'macro_f1': f.mean(),
        'accuracy': (pred == y).mean(),
        'fpr': fpr,
    }


def report(title: str, m: dict) -> None:
    print(f'\n{title}')
    print(f'{"class":<12}{"precision":>10}{"recall":>9}{"f1":>8}')
    for i, c in enumerate(CLASSES):
        print(f'{c:<12}{m["precision"][i]:>10.4f}{m["recall"][i]:>9.4f}{m["f1"][i]:>8.4f}')
    print(f'macro F1 {m["macro_f1"]:.4f} | accuracy {m["accuracy"]:.4f} | benign FPR {m["fpr"]:.4f}')


def main() -> None:
    x_te, y_te = load('x_test.csv', 'y_test.csv')

    print('== training on ORIGINAL (imbalanced) data ==')
    x_tr, y_tr = load('x_train.csv', 'y_train.csv')
    m_before = evaluate(train(x_tr, y_tr), x_te, y_te)
    del x_tr, y_tr

    print('\n== training on BALANCED data ==')
    x_bal, y_bal = load('x_train_balanced.csv', 'y_train_balanced.csv')
    m_after = evaluate(train(x_bal, y_bal), x_te, y_te)

    report('Original', m_before)
    report('Balanced', m_after)

    pd.DataFrame(
        {
            'class': CLASSES,
            'recall_before': m_before['recall'],
            'recall_after': m_after['recall'],
            'f1_before': m_before['f1'],
            'f1_after': m_after['f1'],
        }
    ).to_csv(PLOT_DIR / 'baseline_results.csv', index=False)

    idx = np.arange(len(CLASSES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.8))
    b1 = ax.bar(idx - w / 2, m_before['recall'], w, label='Original', color='#9AA5B1')
    b2 = ax.bar(idx + w / 2, m_after['recall'], w, label='Balanced', color='#4C78A8')
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f'{b.get_height():.2f}',
                    ha='center', va='bottom', fontsize=8)
    ax.set_xticks(idx)
    ax.set_xticklabels(CLASSES)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Recall')
    ax.set_title('Per-class recall on test set: original vs balanced')
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    out = PLOT_DIR / 'recall_before_after.png'
    plt.savefig(out, dpi=150)
    print(f'\nsaved plot: {out}')


if __name__ == '__main__':
    main()