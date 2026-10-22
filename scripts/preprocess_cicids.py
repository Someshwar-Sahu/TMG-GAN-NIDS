from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).absolute().parent.parent / 'data' / 'datasets' / 'CICIDS2017'
RAW_DIR = DATA_DIR / 'raw'

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
MAX_PER_CLASS = 200_000  # cap huge classes so everything fits in memory
TEST_SIZE = 0.3
SEED = 0


def to_group(label: str):
    label = label.strip()
    if label.startswith('Web Attack'):  # raw files have a broken dash in these names
        return 'WebAttack'
    return GROUP.get(label)


def read_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='latin1', skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    label = df['Label']
    feats = df.drop(columns=['Label']).apply(pd.to_numeric, errors='coerce')
    feats = feats.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    feats['Label'] = label
    return feats


def main() -> None:
    files = sorted(RAW_DIR.glob('*.csv'))
    if not files:
        raise FileNotFoundError(f'No csv files in {RAW_DIR}')

    parts = []
    for f in files:
        part = read_file(f)
        print(f'{f.name}: {len(part):,} rows')
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    del parts
    print(f'\nraw rows: {len(df):,} | columns: {df.shape[1]}')

    n0 = len(df)
    df = df.dropna()
    print(f'dropped NaN/inf rows: {n0 - len(df):,}')
    n0 = len(df)
    df = df.drop_duplicates()
    print(f'dropped duplicate rows: {n0 - len(df):,}')

    df['group'] = df['Label'].map(to_group)
    dropped = df[df['group'].isna()]['Label'].value_counts()
    print(f'\ndropped classes:\n{dropped.to_string()}')
    df = df.dropna(subset=['group'])

    print('\nclass counts before cap:')
    print(df['group'].value_counts().reindex(CLASSES).to_string())

    capped = [
        g.sample(n=min(len(g), MAX_PER_CLASS), random_state=SEED)
        for _, g in df.groupby('group')
    ]
    df = pd.concat(capped, ignore_index=True)
    del capped

    features = df.drop(columns=['Label', 'group'])
    features = features.loc[:, features.nunique() > 1]  # drop constant columns
    y = df['group'].map({c: i for i, c in enumerate(CLASSES)}).values

    x_tr, x_te, y_tr, y_te = train_test_split(
        features, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    lo, hi = x_tr.min(), x_tr.max()
    span = (hi - lo).replace(0, 1)
    x_tr = (x_tr - lo) / span
    x_te = ((x_te - lo) / span).clip(0, 1)

    def onehot(y_idx: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(np.eye(len(CLASSES), dtype=np.int8)[y_idx], columns=CLASSES)

    x_tr.to_csv(DATA_DIR / 'x_train.csv', index=False, float_format='%.6g')
    x_te.to_csv(DATA_DIR / 'x_test.csv', index=False, float_format='%.6g')
    onehot(y_tr).to_csv(DATA_DIR / 'y_train.csv', index=False)
    onehot(y_te).to_csv(DATA_DIR / 'y_test.csv', index=False)

    print(f'\nfeatures kept: {x_tr.shape[1]}')
    for title, yy in (('train', y_tr), ('test', y_te)):
        cnt = np.bincount(yy, minlength=len(CLASSES))
        print(f'\n{title} ({len(yy):,} rows)')
        for i, c in enumerate(CLASSES):
            print(f'  {c:<11}{cnt[i]:>9,}  {cnt[i] / len(yy):.2%}')


if __name__ == '__main__':
    main()