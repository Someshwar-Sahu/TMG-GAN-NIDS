from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE

DATA_DIR = Path(__file__).absolute().parent.parent / 'data' / 'datasets' / 'CICIDS2017'
PLOT_DIR = DATA_DIR.parent.parent / 'plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ['Benign', 'DoS', 'PortScan', 'BruteForce', 'WebAttack', 'Bot']
COLORS = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#B279A2', '#72B7B2']
METHOD = 'SMOTE'   # 'SMOTE' or 'ADASYN'
TARGET = 15_000    # minority classes below this are oversampled up to it; nothing is removed
SEED = 0


def main() -> None:
    x = pd.read_csv(DATA_DIR / 'x_train.csv', dtype=np.float32)
    y = pd.read_csv(DATA_DIR / 'y_train.csv').values.argmax(axis=1)
    columns = x.columns
    x = x.values

    before = np.bincount(y, minlength=len(CLASSES))

    over = {i: TARGET for i, n in enumerate(before) if n < TARGET}
    if METHOD == 'SMOTE':
        sampler = SMOTE(sampling_strategy=over, k_neighbors=5, random_state=SEED)
    else:
        sampler = ADASYN(sampling_strategy=over, n_neighbors=5, random_state=SEED)
    x, y = sampler.fit_resample(x, y)

    after = np.bincount(y, minlength=len(CLASSES))

    pd.DataFrame(x, columns=columns).to_csv(
        DATA_DIR / 'x_train_balanced.csv', index=False, float_format='%.6g'
    )
    pd.DataFrame(np.eye(len(CLASSES), dtype=np.int8)[y], columns=CLASSES).to_csv(
        DATA_DIR / 'y_train_balanced.csv', index=False
    )

    summary = pd.DataFrame(
        {
            'class': CLASSES,
            'before': before,
            'after': after,
            'synthetic_added': after - before,
            'synthetic_share': (after - before) / after,
        }
    )
    summary.to_csv(PLOT_DIR / 'balance_summary.csv', index=False)

    print(f'{"class":<12}{"before":>10}{"after":>10}{"synthetic":>11}{"syn %":>8}')
    for i, c in enumerate(CLASSES):
        syn = after[i] - before[i]
        print(f'{c:<12}{before[i]:>10,}{after[i]:>10,}{syn:>11,}{syn / after[i]:>8.1%}')
    print(f'{"total":<12}{before.sum():>10,}{after.sum():>10,}')

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, counts, title in (
        (axs[0], before, 'Before balancing'),
        (axs[1], after, f'After {METHOD} (minority classes raised to {TARGET:,})'),
    ):
        bars = ax.bar(CLASSES, counts, color=COLORS)
        ax.set_title(title)
        ax.set_yscale('log')
        ax.tick_params(axis='x', rotation=20)
        ax.spines[['top', 'right']].set_visible(False)
        for b, v in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2, v, f'{v:,}', ha='center', va='bottom', fontsize=8)
    axs[0].set_ylabel('Samples (log scale)')
    plt.tight_layout()
    out = PLOT_DIR / f'balance_{METHOD.lower()}.png'
    plt.savefig(out, dpi=150)
    print(f'\nsaved plot: {out}')


if __name__ == '__main__':
    main()