from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).absolute().parent.parent / 'data' / 'datasets' / 'CICIDS2017'
PLOT_DIR = DATA_DIR.parent.parent / 'plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ['Benign', 'DoS', 'PortScan', 'BruteForce', 'WebAttack', 'Bot']
COLORS = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#B279A2', '#72B7B2']


def counts(name: str) -> pd.Series:
    y = pd.read_csv(DATA_DIR / name)
    return y.sum().astype(int).reindex(CLASSES)


def main() -> None:
    train, test = counts('y_train.csv'), counts('y_test.csv')

    print(f'{"class":<12}{"train":>10}{"train %":>9}{"test":>9}{"test %":>9}{"IR":>9}')
    for c in CLASSES:
        ir = train.max() / train[c]
        print(f'{c:<12}{train[c]:>10}{train[c] / train.sum():>9.2%}{test[c]:>9}{test[c] / test.sum():>9.2%}{ir:>9.1f}')

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))

    bars = axs[0].bar(CLASSES, train, color=COLORS)
    axs[0].set_yscale('log')
    axs[0].set_title('Train class distribution (log scale)')
    axs[0].set_ylabel('Samples')
    axs[0].tick_params(axis='x', rotation=20)
    axs[0].spines[['top', 'right']].set_visible(False)
    for b, v in zip(bars, train):
        axs[0].text(b.get_x() + b.get_width() / 2, v, f'{v:,}', ha='center', va='bottom', fontsize=8)

    axs[1].pie(
        train,
        labels=CLASSES,
        colors=COLORS,
        autopct=lambda p: f'{p:.1f}%' if p >= 1 else '',
        startangle=90,
        pctdistance=0.8,
        wedgeprops={'width': 0.45, 'edgecolor': 'white'},
    )
    axs[1].set_title('Train class share')

    plt.tight_layout()
    out = PLOT_DIR / 'class_imbalance.png'
    plt.savefig(out, dpi=150)
    print(f'\nsaved plot: {out}')


if __name__ == '__main__':
    main()