import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).absolute().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'datasets' / 'CICIDS2017'
RAW_DIR = DATA_DIR / 'raw'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
LOGS_DIR = PROJECT_ROOT / 'logs'

CLASSES = ['Benign', 'DoS', 'PortScan', 'BruteForce', 'WebAttack', 'Bot']
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

GROUP_MAP = {
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

# Exact training set sample counts from Table II of Ding et al. (IEEE TIFS 2024)
TABLE_II_TRAIN_COUNTS = {
    'Benign': 105_222,
    'DoS': 21_550,
    'PortScan': 10_809,
    'BruteForce': 5_235,
    'WebAttack': 1_476,
    'Bot': 857,
}

TEST_SIZE = 0.30
SEED = 0


def to_group(label: str):
    label = str(label).strip()
    if label.startswith('Web Attack'):
        # Matches prefix to handle encoding anomalies in raw CSVs (e.g. non-ASCII hyphens)
        return 'WebAttack'
    return GROUP_MAP.get(label, None)


def read_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='latin1', skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    label_col = df['Label']
    feat_cols = df.drop(columns=['Label'])

    # Efficient numeric conversion
    for col in feat_cols.columns:
        if feat_cols[col].dtype == object:
            feat_cols[col] = pd.to_numeric(feat_cols[col], errors='coerce')
    feat_cols = feat_cols.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    feat_cols['Label'] = label_col
    return feat_cols


class TeeLogger:
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tee = TeeLogger(LOGS_DIR / 'preprocess.log')
    sys.stdout = tee
    sys.stderr = tee

    print('=' * 70)
    print('STEP 1: CICIDS2017 Preprocessing for TMG-GAN Pipeline')
    print('=' * 70)

    files = sorted(RAW_DIR.glob('*.csv'))
    if not files:
        raise FileNotFoundError(f'No csv files found in {RAW_DIR}')

    print(f'\nReading {len(files)} raw files from {RAW_DIR}:')
    parts = []
    total_raw_rows = 0
    for f in files:
        part = read_file(f)
        count = len(part)
        total_raw_rows += count
        print(f'  {f.name:<55} : {count:>9,} rows')
        parts.append(part)

    df = pd.concat(parts, ignore_index=True)
    del parts
    print(f'\n[Funnel Step 1] Combined raw rows: {len(df):,} | Columns: {df.shape[1]}')

    # Drop NaNs and Infs
    n_before_nan = len(df)
    df = df.dropna()
    dropped_nan = n_before_nan - len(df)
    print(f'[Funnel Step 2] Dropped NaN/Inf rows: {dropped_nan:,} (Remaining: {len(df):,})')

    # Drop Duplicates
    n_before_dup = len(df)
    df = df.drop_duplicates()
    dropped_dup = n_before_dup - len(df)
    print(f'[Funnel Step 3] Dropped duplicate rows: {dropped_dup:,} (Remaining: {len(df):,})')

    # Group and filter classes
    df['group'] = df['Label'].map(to_group)
    dropped_classes_counts = df[df['group'].isna()]['Label'].value_counts()
    print('\n[Funnel Step 4] Dropped classes not evaluated in paper (DDoS, Infiltration, Heartbleed):')
    for lbl, cnt in dropped_classes_counts.items():
        print(f'  {lbl:<35} : {cnt:>9,} rows')

    df = df.dropna(subset=['group']).reset_index(drop=True)
    print(f'Remaining rows across 6 target classes: {len(df):,}')

    print('\nClass counts before split:')
    for c in CLASSES:
        cnt = (df['group'] == c).sum()
        print(f'  {c:<12}: {cnt:>9,} ({cnt / len(df):.2%})')

    # Separate features and labels
    feature_df = df.drop(columns=['Label', 'group'])
    num_features = feature_df.shape[1]
    print(f'\nTotal feature columns kept: {num_features}')
    assert num_features == 78, f'Expected 78 features for 13x6 spatial reshape, but found {num_features}!'

    labels_series = df['group']

    # Stratified Train/Test split
    print(f'\nPerforming stratified train/test split (Train: {1 - TEST_SIZE:.0%}, Test: {TEST_SIZE:.0%}, seed={SEED})...')
    x_tr_pool, x_te, y_tr_pool, y_te = train_test_split(
        feature_df,
        labels_series,
        test_size=TEST_SIZE,
        stratify=labels_series,
        random_state=SEED,
    )
    print(f'Training pool size: {len(x_tr_pool):,} rows | Test set size: {len(x_te):,} rows')

    # Subsample training split to exact Table II counts
    print('\nSubsampling training set to match Table II sizes:')
    train_subsampled_indices = []
    for c in CLASSES:
        target_count = TABLE_II_TRAIN_COUNTS[c]
        class_indices = y_tr_pool[y_tr_pool == c].index
        available_count = len(class_indices)
        if available_count < target_count:
            raise ValueError(
                f'Insufficient samples for class {c}: requested {target_count}, but pool only has {available_count}!'
            )
        # Sample exactly target_count without replacement
        sampled = y_tr_pool[y_tr_pool == c].sample(n=target_count, random_state=SEED)
        train_subsampled_indices.extend(sampled.index)
        print(f'  {c:<12}: Subsampled {target_count:>7,} from pool of {available_count:>9,} ({target_count / available_count:.1%})')

    x_tr = x_tr_pool.loc[train_subsampled_indices].reset_index(drop=True)
    y_tr = y_tr_pool.loc[train_subsampled_indices].reset_index(drop=True)

    print(f'\nFinal Subsampled Training Set Size: {len(x_tr):,} rows')
    print(f'Final Untouched Test Set Size:       {len(x_te):,} rows')

    # Min-Max Normalization: fit strictly on training set
    print('\nFitting Min-Max scaler strictly on training set...')
    lo = x_tr.min()
    hi = x_tr.max()
    span = hi - lo
    # Constant columns check: where span == 0, replace with 1.0 to avoid divide-by-zero
    constant_cols = span[span == 0].index.tolist()
    print(f'Detected {len(constant_cols)} constant columns (setting span=1.0): {constant_cols}')
    span = span.replace(0, 1.0)

    x_tr_scaled = (x_tr - lo) / span
    x_te_scaled = ((x_te - lo) / span).clip(0.0, 1.0)

    # Sanity checks
    assert not x_tr_scaled.isna().any().any(), 'x_train contains NaNs!'
    assert not x_te_scaled.isna().any().any(), 'x_test contains NaNs!'
    assert not np.isinf(x_tr_scaled.values).any(), 'x_train contains Infs!'
    assert not np.isinf(x_te_scaled.values).any(), 'x_test contains Infs!'
    assert x_tr_scaled.shape[1] == 78, f'Expected 78 features in x_train, got {x_tr_scaled.shape[1]}'
    assert x_te_scaled.shape[1] == 78, f'Expected 78 features in x_test, got {x_te_scaled.shape[1]}'

    # Save Scaler to artifacts/scaler.json
    scaler_payload = {
        'feature_names': x_tr.columns.tolist(),
        'feature_count': len(x_tr.columns),
        'min': lo.to_dict(),
        'max': hi.to_dict(),
        'span': span.to_dict(),
        'constant_columns': constant_cols,
        'classes': CLASSES,
        'class_to_id': CLASS_TO_ID,
        'train_counts': TABLE_II_TRAIN_COUNTS,
    }
    with open(ARTIFACTS_DIR / 'scaler.json', 'w', encoding='utf-8') as f:
        json.dump(scaler_payload, f, indent=2)
    print(f'\nSaved scaler parameters to: {ARTIFACTS_DIR / "scaler.json"}')

    # One-hot encode labels
    def onehot(labels_series: pd.Series) -> pd.DataFrame:
        ids = labels_series.map(CLASS_TO_ID).values
        eye = np.eye(len(CLASSES), dtype=np.int8)
        return pd.DataFrame(eye[ids], columns=CLASSES)

    print('\nSaving processed CSV datasets to disk...')
    x_tr_scaled.to_csv(DATA_DIR / 'x_train.csv', index=False, float_format='%.6g')
    x_te_scaled.to_csv(DATA_DIR / 'x_test.csv', index=False, float_format='%.6g')
    onehot(y_tr).to_csv(DATA_DIR / 'y_train.csv', index=False)
    onehot(y_te).to_csv(DATA_DIR / 'y_test.csv', index=False)

    print('Saved files:')
    print(f'  {DATA_DIR / "x_train.csv"} : {len(x_tr_scaled):,} rows, {x_tr_scaled.shape[1]} cols')
    print(f'  {DATA_DIR / "x_test.csv"}  : {len(x_te_scaled):,} rows, {x_te_scaled.shape[1]} cols')
    print(f'  {DATA_DIR / "y_train.csv"} : {len(y_tr):,} rows')
    print(f'  {DATA_DIR / "y_test.csv"}  : {len(y_te):,} rows')

    # Print final summary tables
    print('\n' + '=' * 50)
    print(f'TRAINING SET SUMMARY (Total: {len(y_tr):,} flows):')
    print(f'{"Label":<6} {"Class":<14} {"Count":>10} {"Share":>8}')
    print('-' * 42)
    for c in CLASSES:
        cnt = (y_tr == c).sum()
        print(f'{CLASS_TO_ID[c]:<6} {c:<14} {cnt:>10,} {cnt / len(y_tr):>7.2%}')

    print('\n' + '=' * 50)
    print(f'TEST SET SUMMARY (Untouched, Total: {len(y_te):,} flows):')
    print(f'{"Label":<6} {"Class":<14} {"Count":>10} {"Share":>8}')
    print('-' * 42)
    for c in CLASSES:
        cnt = (y_te == c).sum()
        print(f'{CLASS_TO_ID[c]:<6} {c:<14} {cnt:>10,} {cnt / len(y_te):>7.2%}')

    print('\n' + '=' * 70)
    print('STEP 1 PREPROCESSING COMPLETE')
    print('=' * 70)


if __name__ == '__main__':
    main()
