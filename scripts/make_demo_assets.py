import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parent.parent
DATA_DIR = ROOT / 'data' / 'datasets' / 'CICIDS2017'
RAW_DIR = DATA_DIR / 'raw'
PLOT_DIR = ROOT / 'data' / 'plots'
SAMPLE_DIR = ROOT / 'data' / 'samples'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ['Benign', 'DoS', 'PortScan', 'BruteForce', 'WebAttack', 'Bot']
GROUP = {
    'BENIGN': 'Benign',
    'DoS Hulk': 'DoS',
    'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'PortScan': 'PortScan',
    'FTP-Patator': 'BruteForce',
    'SSH-Patator': 'BruteForce',
    'Bot': 'Bot',
}
MAX_PER_CLASS = 200_000
PER_CLASS = 50
HIST_FEATURES = ['Flow Duration', 'Flow Bytes/s', 'Total Fwd Packets']
SEED = 0


def clean_label(label: str) -> str:
    label = label.strip()
    return re.sub(r'^Web Attack\s+\S+\s+', 'Web Attack - ', label)  # fix broken dash


def to_group(label: str):
    if label.startswith('Web Attack'):
        return 'WebAttack'
    return GROUP.get(label.strip())


def read_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='latin1', skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    label = df['Label']
    feats = df.drop(columns=['Label']).apply(pd.to_numeric, errors='coerce')
    feats = feats.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    feats['Label'] = label
    return feats


def plot_funnel(stages: list[tuple[str, int]]) -> None:
    names = [s[0] for s in stages][::-1]
    values = [s[1] for s in stages][::-1]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    bars = ax.barh(names, values, color='#4C78A8', height=0.6)
    bars[0].set_color('#72B7B2')  # final stage
    for b, v in zip(bars, values):
        ax.text(v, b.get_y() + b.get_height() / 2, f'  {v:,}', va='center', fontsize=9)
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_xlabel('Rows')
    ax.set_title('Data cleaning funnel (CICIDS2017)')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'cleaning_funnel.png', dpi=150)
    plt.close()


def plot_label_table(tbl: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 0.42 * (len(tbl) + 2)))
    ax.axis('off')
    cells = [[r.Label, r['class'], f'{r.rows:,}'] for _, r in tbl.iterrows()]
    t = ax.table(
        cellText=cells,
        colLabels=['Original label', 'Our class', 'Rows (after cleaning)'],
        loc='center',
        cellLoc='left',
    )
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.4)
    for (row, _), cell in t.get_celld().items():
        cell.set_edgecolor('#D0D5DB')
        if row == 0:
            cell.set_facecolor('#4C78A8')
            cell.set_text_props(color='white', weight='bold')
        elif cells[row - 1][1] == '(dropped)':
            cell.set_facecolor('#EEEEEE')
    ax.set_title('Label merging: 15 original labels to 6 classes', pad=12)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'label_mapping.png', dpi=150)
    plt.close()


def plot_scaling(kept: pd.DataFrame) -> None:
    feats = [f for f in HIST_FEATURES if f in kept.columns]
    sample = kept.sample(n=min(len(kept), 100_000), random_state=SEED)
    fig, axs = plt.subplots(2, len(feats), figsize=(4.4 * len(feats), 6.4))
    axs = np.atleast_2d(axs).reshape(2, len(feats))
    for j, f in enumerate(feats):
        x = sample[f].astype(np.float64)
        scaled = (x - x.min()) / (x.max() - x.min())
        axs[0, j].hist(x, bins=60, color='#9AA5B1')
        axs[0, j].set_title(f'{f}\nraw (min {x.min():.3g}, max {x.max():.3g})', fontsize=10)
        axs[1, j].hist(scaled, bins=60, color='#4C78A8')
        axs[1, j].set_title('after min-max scaling (0 to 1)', fontsize=10)
        for i in range(2):
            axs[i, j].set_yscale('log')
            axs[i, j].spines[['top', 'right']].set_visible(False)
    axs[0, 0].set_ylabel('Count (log)')
    axs[1, 0].set_ylabel('Count (log)')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'scaling_before_after.png', dpi=150)
    plt.close()


def main() -> None:
    files = sorted(RAW_DIR.glob('*.csv'))
    if not files:
        raise FileNotFoundError(f'No csv files in {RAW_DIR}')

    parts, file_rows = [], []
    for f in files:
        part = read_file(f)
        file_rows.append((f.name, len(part)))
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    del parts

    n_raw = len(df)
    df = df.dropna()
    n_nan = len(df)
    df = df.drop_duplicates()
    n_dedup = len(df)
    df['Label'] = df['Label'].map(clean_label)
    df['group'] = df['Label'].map(to_group)
    kept = df.dropna(subset=['group'])
    n_classes = len(kept)
    n_cap = int(sum(min(c, MAX_PER_CLASS) for c in kept['group'].value_counts()))

    stages = [
        ('Raw (8 files)', n_raw),
        ('After removing NaN/inf', n_nan),
        ('After removing duplicates', n_dedup),
        ('After dropping DDoS/Infiltration/Heartbleed', n_classes),
        ('Final (class cap 200k)', n_cap),
    ]
    print('rows per raw file:')
    for name, n in file_rows:
        print(f'  {name:<62}{n:>9,}')
    print('\ncleaning funnel:')
    prev = None
    for name, n in stages:
        diff = '' if prev is None else f'  (-{prev - n:,})'
        print(f'  {name:<46}{n:>10,}{diff}')
        prev = n
    plot_funnel(stages)

    tbl = df.groupby('Label').size().rename('rows').reset_index()
    tbl['class'] = tbl['Label'].map(to_group).fillna('(dropped)')
    order = {c: i for i, c in enumerate(CLASSES + ['(dropped)'])}
    tbl = tbl.sort_values(by=['class', 'rows'], key=lambda s: s.map(order) if s.name == 'class' else -s)
    tbl.to_csv(SAMPLE_DIR / 'label_mapping.csv', index=False)
    plot_label_table(tbl)

    plot_scaling(kept)

    # sample files small enough for Excel
    cols = list(pd.read_csv(DATA_DIR / 'x_train.csv', nrows=0).columns)
    raw = kept.groupby('group', group_keys=False).sample(n=PER_CLASS, random_state=SEED)
    raw = raw.rename(columns={'group': 'class'})[['class'] + cols]
    raw['class'] = pd.Categorical(raw['class'], categories=CLASSES, ordered=True)
    raw.sort_values('class').to_csv(SAMPLE_DIR / 'raw_cleaned_sample.csv', index=False)

    x = pd.read_csv(DATA_DIR / 'x_train.csv', dtype=np.float32)
    y = pd.read_csv(DATA_DIR / 'y_train.csv').values.argmax(axis=1)
    rng = np.random.default_rng(SEED)
    idx = np.concatenate(
        [rng.choice(np.where(y == i)[0], PER_CLASS, replace=False) for i in range(len(CLASSES))]
    )
    scaled = x.iloc[idx].copy()
    scaled.insert(0, 'class', [CLASSES[i] for i in y[idx]])
    scaled.to_csv(SAMPLE_DIR / 'scaled_sample.csv', index=False, float_format='%.5f')

    print(f'\nplots   -> {PLOT_DIR}: cleaning_funnel.png, label_mapping.png, scaling_before_after.png')
    print(f'samples -> {SAMPLE_DIR}: raw_cleaned_sample.csv, scaled_sample.csv, label_mapping.csv')
    print(f'each sample file has {PER_CLASS * len(CLASSES)} rows, opens in Excel')


if __name__ == '__main__':
    main()