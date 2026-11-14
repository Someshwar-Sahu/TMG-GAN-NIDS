"""
STEP 4C: ADASYN Baseline Generation (Matched Table II Target Sizes)
Requested in Section B.2 of the IEEE revision plan.
Applies Adaptive Synthetic Sampling (ADASYN) strictly to the training partition (x_train.csv, y_train.csv).
The held-out test partition is kept 100% untouched.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN

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
    print('STEP 4C: ADASYN Baseline Generation (Matched Table II Target Sizes)')
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

    print('\nApplying ADASYN with matched Table II targets:')
    for c, target in strategy.items():
        name = config.CLASSES[c]
        print(f'  {name:<12}: {before_counts[c]:>7,} -> {target:>7,} (+{target - before_counts[c]:>6,} synthetic)')

    start_time = time.time()
    adasyn = ADASYN(sampling_strategy=strategy, n_neighbors=5, random_state=config.SEED)
    X_res, y_res = adasyn.fit_resample(X, y)
    adasyn_time = time.time() - start_time
    print(f'ADASYN resampling completed in {adasyn_time:.2f}s.')

    after_counts = np.bincount(y_res, minlength=len(config.CLASSES))
    total_res = len(y_res)
    print(f'\nTotal ADASYN balanced rows: {total_res:,}')
    assert X_res.shape[1] == 78, f'Expected 78 features, got {X_res.shape[1]}'

    # Min-max clamp to [0, 1] bounds
    X_res = np.clip(X_res, 0.0, 1.0)

    # Check bounds
    assert not np.isnan(X_res).any(), 'NaNs detected in ADASYN data!'
    assert not np.isinf(X_res).any(), 'Infs detected in ADASYN data!'

    # Save to data/balanced/
    balanced_dir = PROJECT_ROOT / 'data' / 'balanced'
    balanced_dir.mkdir(parents=True, exist_ok=True)

    adasyn_train_csv = balanced_dir / 'adasyn_train.csv'
    x_adasyn_csv = balanced_dir / 'x_train_adasyn.csv'
    y_adasyn_csv = balanced_dir / 'y_train_adasyn.csv'

    print(f'\nSaving ADASYN balanced datasets to {balanced_dir}...')
    df_adasyn = pd.DataFrame(X_res, columns=feature_names)
    df_adasyn['Label'] = [config.CLASSES[i] for i in y_res]
    df_adasyn.to_csv(adasyn_train_csv, index=False, float_format='%.6g')
    print(f'Saved: {adasyn_train_csv} ({total_res:,} rows, {df_adasyn.shape[1]} cols)')

    pd.DataFrame(X_res, columns=feature_names).to_csv(x_adasyn_csv, index=False, float_format='%.6g')
    y_onehot = np.eye(len(config.CLASSES), dtype=np.int8)[y_res]
    pd.DataFrame(y_onehot, columns=config.CLASSES).to_csv(y_adasyn_csv, index=False)
    print(f'Saved: {x_adasyn_csv}')
    print(f'Saved: {y_adasyn_csv}')

    # Save summary stats
    plot_dir = PROJECT_ROOT / 'data' / 'plots'
    summary_records = []
    for c, name in enumerate(config.CLASSES):
        summary_records.append({
            'class': name,
            'original': before_counts[c],
            'target': TARGET_OS_COUNTS[name],
            'after': after_counts[c],
            'synthetic_added': after_counts[c] - before_counts[c],
        })
    pd.DataFrame(summary_records).to_csv(plot_dir / 'adasyn_generation_summary.csv', index=False)
    print(f'Saved ADASYN generation statistics: {plot_dir / "adasyn_generation_summary.csv"}')

    print('\n' + '=' * 75)
    print('STEP 4C ADASYN DATASET GENERATION COMPLETE')
    print('=' * 75)


if __name__ == '__main__':
    main()

