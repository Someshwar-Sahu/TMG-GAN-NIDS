import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.dataset import CICIDSTrainingData
from src.tmg_gan import TMGGAN


def parse_args():
    parser = argparse.ArgumentParser(description='Train TMG-GAN on CICIDS2017')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='Number of epochs (default: 2000)')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE, help='Batch size (default: 128)')
    parser.add_argument('--save-freq', type=int, default=100, help='Checkpoint save frequency in epochs (default: 100)')
    parser.add_argument('--seed', type=int, default=config.SEED, help='Random seed (default: 0)')
    parser.add_argument('--skip-tsne', action='store_true', help='Skip t-SNE visualization at end')
    return parser.parse_args()


def plot_tsne(tmg_gan: TMGGAN, dataset: CICIDSTrainingData, save_path: Path, samples_per_class: int = 500):
    """Generates t-SNE plot comparing real and TMG-GAN generated samples per class."""
    print(f'\nGenerating t-SNE comparison plot ({samples_per_class} real + {samples_per_class} generated per class)...')
    save_path.parent.mkdir(parents=True, exist_ok=True)

    x_list = []
    labels_list = []
    origin_list = []

    for c in range(config.LABEL_NUM):
        class_name = config.CLASSES[c]
        # Real samples
        real_all = dataset.samples_by_class[c]
        num_real = min(samples_per_class, len(real_all))
        idx = torch.randperm(len(real_all))[:num_real]
        real_sub = real_all[idx].numpy()

        x_list.append(real_sub)
        labels_list.extend([class_name] * num_real)
        origin_list.extend(['Real'] * num_real)

        # Generated samples
        gen_sub = tmg_gan.generate_raw_samples(c, num_real).numpy()
        x_list.append(gen_sub)
        labels_list.extend([class_name] * num_real)
        origin_list.extend(['TMG-GAN Generated'] * num_real)

    X = np.vstack(x_list)
    print(f'Running TSNE on {len(X):,} total samples with 78 dimensions...')
    tsne = TSNE(n_components=2, random_state=config.SEED, init='pca', learning_rate='auto')
    embedded = tsne.fit_transform(X)

    import pandas as pd
    plot_df = pd.DataFrame({
        'Dim 1': embedded[:, 0],
        'Dim 2': embedded[:, 1],
        'Class': labels_list,
        'Type': origin_list,
    })

    plt.figure(figsize=(14, 10))
    sns.scatterplot(
        data=plot_df,
        x='Dim 1',
        y='Dim 2',
        hue='Class',
        style='Type',
        alpha=0.6,
        palette='tab10',
        s=30,
    )
    plt.title('t-SNE Projection: CICIDS2017 Real Flows vs TMG-GAN Generated Flows', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f't-SNE plot saved to: {save_path}')


def main():
    args = parse_args()

    print('=' * 70)
    print('STEP 2: TMG-GAN Full Training on CICIDS2017')
    print('=' * 70)

    config.print_gpu_status()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = PROJECT_ROOT / 'data' / 'datasets' / 'CICIDS2017'
    dataset = CICIDSTrainingData(data_dir=data_dir, device=config.DEVICE)

    tmg_gan = TMGGAN(
        feature_num=config.FEATURE_NUM,
        label_num=config.LABEL_NUM,
        z_size=config.Z_SIZE,
        device=config.DEVICE,
    )

    checkpoint_dir = PROJECT_ROOT / 'models' / 'checkpoints'
    loss_log_path = PROJECT_ROOT / 'logs' / 'gan_losses.csv'
    models_dir = PROJECT_ROOT / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    start_train = time.time()
    tmg_gan.fit(
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        g_lr=config.G_LR,
        cd_lr=config.CD_LR,
        cd_loops=config.CD_LOOPS,
        g_loops=config.G_LOOPS,
        checkpoint_dir=checkpoint_dir,
        loss_log_path=loss_log_path,
        save_freq=args.save_freq,
    )
    total_time = time.time() - start_train

    # Save final model weights
    cd_final_path = models_dir / 'tmg_gan_cd.pt'
    gen_final_path = models_dir / 'tmg_gan_generators.pt'

    torch.save(tmg_gan.cd.state_dict(), cd_final_path)
    torch.save([g.state_dict() for g in tmg_gan.generators], gen_final_path)
    print(f'\nSaved final CDModel weights to: {cd_final_path}')
    print(f'Saved final Generator weights to: {gen_final_path}')

    # Optional t-SNE plot
    if not args.skip_tsne:
        tsne_path = PROJECT_ROOT / 'data' / 'plots' / 'tsne_real_vs_generated.png'
        plot_tsne(tmg_gan, dataset, tsne_path, samples_per_class=400)

    print('\n' + '=' * 70)
    print(f'TRAINING COMPLETE in {total_time / 60:.2f} minutes ({total_time:.1f}s)')
    print(f'Checkpoints: {checkpoint_dir}')
    print(f'Losses Log : {loss_log_path}')
    print('=' * 70)


if __name__ == '__main__':
    main()
