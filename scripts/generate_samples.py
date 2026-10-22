import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.dataset import CICIDSTrainingData
from src.models import CDModel, GeneratorModel

# Table II Targets from Ding et al. (IEEE TIFS 2024)
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
    print('STEP 3: TMG-GAN Sample Generation and Filtering (Algorithm 2)')
    print('=' * 75)

    config.print_gpu_status()
    device = config.DEVICE
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    # 1. Load Preprocessed Original Training Data
    data_dir = PROJECT_ROOT / 'data' / 'datasets' / 'CICIDS2017'
    dataset = CICIDSTrainingData(data_dir=data_dir, device=device)
    feature_names = dataset.feature_names
    num_features = dataset.feature_count
    assert num_features == 78, f'Expected 78 features, got {num_features}'

    # 2. Load Trained TMG-GAN Models
    models_dir = PROJECT_ROOT / 'models'
    cd_path = models_dir / 'tmg_gan_cd.pt'
    gen_path = models_dir / 'tmg_gan_generators.pt'

    if not cd_path.exists() or not gen_path.exists():
        raise FileNotFoundError(
            f'Trained model weights missing in {models_dir}. Complete Step 2 first.'
        )

    print(f'Loading CDModel from {cd_path}...')
    cd = CDModel(in_features=78, label_num=6).to(device)
    cd.load_state_dict(torch.load(cd_path, map_location=device))
    cd.eval()

    print(f'Loading Generator weights from {gen_path}...')
    gen_weights = torch.load(gen_path, map_location=device)
    generators = []
    for state in gen_weights:
        g = GeneratorModel(z_size=config.Z_SIZE, feature_num=78).to(device)
        g.load_state_dict(state)
        g.eval()
        generators.append(g)

    # 3. Algorithm 2 Generation & Filtering per Class
    print('\nExecuting Algorithm 2 generation and noise filtering:')
    print(f'{"Class":<12} {"Original":>10} {"Target OS":>10} {"Needed":>10} {"Raw Drawn":>12} {"Filter Acc %":>14}')
    print('-' * 75)

    synthetic_samples_dict = {}
    generation_stats = []
    start_time = time.time()

    for c in range(config.LABEL_NUM):
        class_name = config.CLASSES[c]
        orig_count = len(dataset.samples_by_class[c])
        target_count = TARGET_OS_COUNTS[class_name]
        needed_count = target_count - orig_count

        if needed_count <= 0:
            print(f'{class_name:<12} {orig_count:>10,} {target_count:>10,} {0:>10} {"(No Augmentation Needed)":>28}')
            synthetic_samples_dict[c] = None
            generation_stats.append({
                'class': class_name,
                'original': orig_count,
                'target': target_count,
                'needed': 0,
                'raw_drawn': 0,
                'accepted': 0,
                'acceptance_rate': 1.0,
            })
            continue

        # Algorithm 2 Loop
        accepted_tensors = []
        total_raw_drawn = 0
        batch_size = 1024
        max_batches = 300  # Guard against infinite loops
        batches_run = 0

        with torch.no_grad():
            while sum(len(t) for t in accepted_tensors) < needed_count and batches_run < max_batches:
                batches_run += 1
                candidates = generators[c].generate_samples(batch_size, device)
                total_raw_drawn += len(candidates)

                _, logits = cd(candidates)
                preds = logits.argmax(dim=1)

                # Noise filter: keep sample only if classifier agrees with target generator label
                mask = (preds == c)
                valid = candidates[mask]

                if len(valid) > 0:
                    accepted_tensors.append(valid.cpu())

        if batches_run >= max_batches and sum(len(t) for t in accepted_tensors) < needed_count:
            raise RuntimeError(
                f'Infinite loop guard triggered for class {class_name}! Only gathered {sum(len(t) for t in accepted_tensors)} / {needed_count} after {total_raw_drawn} candidates.'
            )

        combined_class_samples = torch.cat(accepted_tensors, dim=0)[:needed_count]
        acc_rate = needed_count / total_raw_drawn
        synthetic_samples_dict[c] = combined_class_samples

        print(
            f'{class_name:<12} {orig_count:>10,} {target_count:>10,} {needed_count:>10,} '
            f'{total_raw_drawn:>12,} {acc_rate:>13.2%}'
        )

        generation_stats.append({
            'class': class_name,
            'original': orig_count,
            'target': target_count,
            'needed': needed_count,
            'raw_drawn': total_raw_drawn,
            'accepted': needed_count,
            'acceptance_rate': acc_rate,
        })

    gen_duration = time.time() - start_time
    print(f'\nGeneration and filtering completed in {gen_duration:.2f}s.')

    # 4. Construct Full Augmented Dataset
    print('\nAssembling augmented dataset...')
    all_x = []
    all_y = []

    for c in range(config.LABEL_NUM):
        # 1. Real original samples
        real_x = dataset.samples_by_class[c].cpu().numpy()
        real_y = np.full(len(real_x), c, dtype=np.int64)
        all_x.append(real_x)
        all_y.append(real_y)

        # 2. Filtered synthetic samples
        if synthetic_samples_dict[c] is not None:
            syn_x = synthetic_samples_dict[c].numpy()
            syn_y = np.full(len(syn_x), c, dtype=np.int64)
            all_x.append(syn_x)
            all_y.append(syn_y)

    final_x = np.vstack(all_x)
    final_y = np.concatenate(all_y)

    total_rows = len(final_y)
    print(f'Total augmented dataset rows: {total_rows:,} (Expected: 210,444)')
    assert total_rows == 210_444, f'Expected exactly 210,444 rows, got {total_rows:,}'
    assert final_x.shape[1] == 78, f'Expected 78 feature columns, got {final_x.shape[1]}'

    # Check bounds
    assert not np.isnan(final_x).any(), 'NaNs detected in final augmented dataset!'
    assert not np.isinf(final_x).any(), 'Infs detected in final augmented dataset!'
    assert final_x.min() >= 0.0 and final_x.max() <= 1.0, f'Values outside [0, 1] range: [{final_x.min()}, {final_x.max()}]'

    # 5. Save Datasets to data/balanced/
    balanced_dir = PROJECT_ROOT / 'data' / 'balanced'
    balanced_dir.mkdir(parents=True, exist_ok=True)

    # Save gan_train.csv (78 features + Label name)
    print(f'\nSaving gan_train.csv to {balanced_dir}...')
    df_augmented = pd.DataFrame(final_x, columns=feature_names)
    df_augmented['Label'] = [config.CLASSES[i] for i in final_y]
    gan_train_csv = balanced_dir / 'gan_train.csv'
    df_augmented.to_csv(gan_train_csv, index=False, float_format='%.6g')
    print(f'Saved: {gan_train_csv} ({total_rows:,} rows, {df_augmented.shape[1]} columns)')

    # Also save x_train_gan.csv and y_train_gan.csv for standard model training
    x_gan_csv = balanced_dir / 'x_train_gan.csv'
    y_gan_csv = balanced_dir / 'y_train_gan.csv'
    pd.DataFrame(final_x, columns=feature_names).to_csv(x_gan_csv, index=False, float_format='%.6g')
    y_onehot = np.eye(len(config.CLASSES), dtype=np.int8)[final_y]
    pd.DataFrame(y_onehot, columns=config.CLASSES).to_csv(y_gan_csv, index=False)
    print(f'Saved: {x_gan_csv}')
    print(f'Saved: {y_gan_csv}')

    # 6. Save Summary and Plot
    plot_dir = PROJECT_ROOT / 'data' / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(generation_stats)
    summary_csv = plot_dir / 'tmg_gan_generation_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f'Saved generation statistics: {summary_csv}')

    # Plot Before vs After TMG-GAN Augmentation
    before_counts = [len(dataset.samples_by_class[c]) for c in range(config.LABEL_NUM)]
    after_counts = [TARGET_OS_COUNTS[config.CLASSES[c]] for c in range(config.LABEL_NUM)]

    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, counts, title in (
        (axs[0], before_counts, 'Before Augmentation (Training Set)'),
        (axs[1], after_counts, 'After TMG-GAN Augmentation (Table II Target)'),
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
    plot_path = plot_dir / 'balance_tmg_gan.png'
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f'Saved comparison plot: {plot_path}')

    print('\n' + '=' * 75)
    print('STEP 3 GENERATION & FILTERING COMPLETE')
    print('=' * 75)


if __name__ == '__main__':
    main()
