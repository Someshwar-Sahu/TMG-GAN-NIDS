import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

PROJECT_ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config

# Target oversampling counts matching Table II from Ding et al. (IEEE TIFS 2024)
TARGET_OS_COUNTS = {
    'Benign': 105_222,
    'DoS': 34_609,
    'PortScan': 23_868,
    'BruteForce': 18_294,
    'WebAttack': 14_535,
    'Bot': 13_916,
}

COLORS = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#B279A2', '#72B7B2']


def main():
    print('=' * 75)
    print('STEP 4A: SMOTE Baseline Generation (Matched Table II Target Sizes)')
    print('=' * 75)

    data_dir = PROJECT_ROOT / 'data' / 'datasets' / 'CICIDS2017'
    x_path = data_dir / 'x_train.csv'
    y_path = data_dir / 'y_train.csv'

    print(f'Loading original training data from {data_dir}...')
    x_df = pd.read_csv(x_path, dtype=np.float32)
    y_df = pd.read_csv(y_path, dtype=np.int8)

    feature_names = x_df.columns.tolist()
    X = x_df.values
    y = y_df.values.argmax(axis=1)

    before_counts = np.bincount(y, minlength=len(config.CLASSES))
    print('\nOriginal training set counts:')
    for c, name in enumerate(config.CLASSES):
        print(f'  {name:<12}: {before_counts[c]:>10,}')

    # Construct sampling strategy dictionary matching exact Table II counts
    strategy = {}
    for c, name in enumerate(config.CLASSES):
        target = TARGET_OS_COUNTS[name]
        if target > before_counts[c]:
            strategy[c] = target

    print('\nApplying SMOTE with matched Table II targets:')
    for c, target in strategy.items():
        name = config.CLASSES[c]
        print(f'  {name:<12}: {before_counts[c]:>7,} -> {target:>7,} (+{target - before_counts[c]:>6,} synthetic)')

    start_time = time.time()
    smote = SMOTE(sampling_strategy=strategy, k_neighbors=5, random_state=config.SEED)
    X_res, y_res = smote.fit_resample(X, y)
    smote_time = time.time() - start_time
    print(f'SMOTE resampling completed in {smote_time:.2f}s.')

    after_counts = np.bincount(y_res, minlength=len(config.CLASSES))
    total_res = len(y_res)
    print(f'\nTotal SMOTE balanced rows: {total_res:,} (Expected: 210,444)')
    assert total_res == 210_444, f'Expected 210,444 rows, got {total_res}'
    assert X_res.shape[1] == 78, f'Expected 78 features, got {X_res.shape[1]}'

    # Check bounds
    assert not np.isnan(X_res).any(), 'NaNs detected in SMOTE data!'
    assert not np.isinf(X_res).any(), 'Infs detected in SMOTE data!'

    # Save to data/balanced/
    balanced_dir = PROJECT_ROOT / 'data' / 'balanced'
    balanced_dir.mkdir(parents=True, exist_ok=True)

    smote_train_csv = balanced_dir / 'smote_train.csv'
    x_smote_csv = balanced_dir / 'x_train_smote.csv'
    y_smote_csv = balanced_dir / 'y_train_smote.csv'

    print(f'\nSaving SMOTE balanced datasets to {balanced_dir}...')
    df_smote = pd.DataFrame(X_res, columns=feature_names)
    df_smote['Label'] = [config.CLASSES[i] for i in y_res]
    df_smote.to_csv(smote_train_csv, index=False, float_format='%.6g')
    print(f'Saved: {smote_train_csv} ({total_res:,} rows, {df_smote.shape[1]} cols)')

    pd.DataFrame(X_res, columns=feature_names).to_csv(x_smote_csv, index=False, float_format='%.6g')
    y_onehot = np.eye(len(config.CLASSES), dtype=np.int8)[y_res]
    pd.DataFrame(y_onehot, columns=config.CLASSES).to_csv(y_smote_csv, index=False)
    print(f'Saved: {x_smote_csv}')
    print(f'Saved: {y_smote_csv}')

    # Save summary and plot
    plot_dir = PROJECT_ROOT / 'data' / 'plots'
    summary_df = pd.DataFrame({
        'class': config.CLASSES,
        'before': before_counts,
        'after': after_counts,
        'synthetic_added': after_counts - before_counts,
        'synthetic_share': (after_counts - before_counts) / after_counts,
    })
    summary_path = plot_dir / 'smote_generation_summary.csv'
    summary_df.to_csv(summary_path, index=False)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, counts, title in (
        (axs[0], before_counts, 'Before Augmentation (Training Set)'),
        (axs[1], after_counts, 'After SMOTE Augmentation (Table II Target)'),
    ):
        bars = ax.bar(config.CLASSES, counts, color=COLORS)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.tick_params(axis='x', rotation=20)
        ax.spines[['top', 'right']].set_visible(False)
        for b, v in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2, v, f'{v:,}', ha='center', va='bottom', fontsize=9)
    axs[0].set_ylabel('Sample Count (log scale)', fontsize=11)
    plt.tight_layout()
    plot_path = plot_dir / 'balance_smote_matched.png'
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f'Saved comparison plot: {plot_path}')

    print('\n' + '=' * 75)
    print('SMOTE BASELINE GENERATION COMPLETE')
    print('=' * 75)


if __name__ == '__main__':
    main()
