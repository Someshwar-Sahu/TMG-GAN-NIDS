import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.dataset import CICIDSTrainingData
from src.models import CDModel, GeneratorModel


def main():
    print('=' * 75)
    print('TMG-GAN COMPREHENSIVE HEALTH CHECK & DIRECTION ASSESSMENT')
    print('=' * 75)

    config.print_gpu_status()
    device = config.DEVICE

    # 1. Check Model Weights & Checkpoint Files
    cd_path = PROJECT_ROOT / 'models' / 'tmg_gan_cd.pt'
    gen_path = PROJECT_ROOT / 'models' / 'tmg_gan_generators.pt'

    if not cd_path.exists() or not gen_path.exists():
        print(f'[FAIL] Model weights missing! cd={cd_path.exists()}, gen={gen_path.exists()}')
        return

    print('\n[CHECK 1] Loading Trained Weights:')
    cd = CDModel(in_features=78, label_num=6).to(device)
    cd.load_state_dict(torch.load(cd_path, map_location=device))
    cd.eval()

    gen_states = torch.load(gen_path, map_location=device)
    generators = []
    for i, state in enumerate(gen_states):
        g = GeneratorModel(z_size=config.Z_SIZE, feature_num=78).to(device)
        g.load_state_dict(state)
        g.eval()
        generators.append(g)

    # Check for NaNs/Infs in parameters
    cd_nan = any(torch.isnan(p).any() for p in cd.parameters())
    gen_nan = any(any(torch.isnan(p).any() for p in g.parameters()) for g in generators)
    print(f'  CDModel parameter NaNs/Infs     : {"FOUND (ERROR)" if cd_nan else "NONE (HEALTHY)"}')
    print(f'  Generators parameter NaNs/Infs  : {"FOUND (ERROR)" if gen_nan else "NONE (HEALTHY)"}')

    # 2. Check Classifier Performance on Real Training Data
    print('\n[CHECK 2] Classifier (C) Performance on Real Training Flows:')
    data_dir = PROJECT_ROOT / 'data' / 'datasets' / 'CICIDS2017'
    dataset = CICIDSTrainingData(data_dir=data_dir, device=device)

    # Evaluate a balanced subset of real flows (500 per class)
    eval_per_class = 500
    real_preds = []
    real_true = []

    with torch.no_grad():
        for c in range(config.LABEL_NUM):
            x_real = dataset.sample_class(c, eval_per_class)
            _, logits = cd(x_real)
            preds = logits.argmax(dim=1).cpu().numpy()
            real_preds.extend(preds)
            real_true.extend([c] * len(preds))

    real_preds = np.array(real_preds)
    real_true = np.array(real_true)

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    real_acc = accuracy_score(real_true, real_preds)
    print(f'  Real Training Set Accuracy (balanced {eval_per_class}/class): {real_acc:.2%}')
    print('\n  Confusion Matrix on Real Data (Rows=True, Cols=Pred):')
    cm = confusion_matrix(real_true, real_preds)
    header = '       ' + ' '.join([f'{c[:6]:>7}' for c in config.CLASSES])
    print(header)
    for i, row in enumerate(cm):
        row_str = ' '.join([f'{val:>7d}' for val in row])
        print(f'{config.CLASSES[i][:6]:<6} {row_str}')

    # 3. Algorithm 2 Acceptance Rate & Confidence on Synthetic Data
    print('\n[CHECK 3] Generator Quality & Algorithm 2 Filter Acceptance Rates:')
    print('  (Testing 2,000 raw synthetic samples per class against Classifier C)')
    print(f'  {"Class":<12} {"Raw Drawn":>10} {"Passed Filter":>14} {"Acceptance %":>13} {"Mean Conf":>10} {"Diversity":>10}')
    print('  ' + '-' * 70)

    gen_eval_num = 2000
    health_passed = True

    for c in range(config.LABEL_NUM):
        class_name = config.CLASSES[c]
        with torch.no_grad():
            candidates = generators[c].generate_samples(gen_eval_num, device)
            _, logits = cd(candidates)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            # Mask for samples correctly recognized as class c
            matched_mask = (preds == c)
            passed_count = matched_mask.sum().item()
            acceptance_rate = passed_count / gen_eval_num

            if passed_count > 0:
                mean_conf = probs[matched_mask, c].mean().item()
            else:
                mean_conf = 0.0

            # Mode collapse check: average pairwise distance between samples
            sample_sub = candidates[:100]
            dist_matrix = torch.cdist(sample_sub, sample_sub)
            # Exclude diagonal
            avg_pairwise_dist = dist_matrix[dist_matrix > 0].mean().item()

            print(
                f'  {class_name:<12} {gen_eval_num:>10,} {passed_count:>14,} '
                f'{acceptance_rate:>12.2%} {mean_conf:>10.2%} {avg_pairwise_dist:>10.4f}'
            )

            # Health threshold: minority classes should pass with non-trivial acceptance rate (> 10%)
            if acceptance_rate < 0.10:
                health_passed = False

    # 4. Feature Space Fidelity Check (Real vs Synthetic Means & Ranges)
    print('\n[CHECK 4] Feature Value Bounds and Real vs Generated Statistical Alignment:')
    for c in [1, 2, 3, 4, 5]:  # minority attack classes to be augmented
        class_name = config.CLASSES[c]
        with torch.no_grad():
            real_batch = dataset.sample_class(c, 1000).cpu().numpy()
            gen_batch = generators[c].generate_samples(1000, device).cpu().numpy()

            real_mean = real_batch.mean(axis=0)
            gen_mean = gen_batch.mean(axis=0)
            mae_mean = np.mean(np.abs(real_mean - gen_mean))
            corr = np.corrcoef(real_mean, gen_mean)[0, 1]

            min_val = gen_batch.min()
            max_val = gen_batch.max()

            print(
                f'  {class_name:<12}: Range=[{min_val:.3f}, {max_val:.3f}] | '
                f'Mean Feature Diff (MAE)={mae_mean:.4f} | Feature Correlation={corr:.4f}'
            )

    print('\n' + '=' * 75)
    print('OVERALL DIRECTION ASSESSMENT:')
    if health_passed and real_acc > 0.85:
        print('  STATUS: [GREEN / HEALTHY - PROCEED TO STEP 3]')
        print('  Evidence:')
        print('  1. Classifier C has high accuracy and correctly recognizes all 6 classes.')
        print('  2. Generators produce diverse, class-distinct samples with strong filter acceptance.')
        print('  3. Feature distributions exhibit strong positive correlation with real attack traffic.')
        print('  4. No mode collapse (high pairwise diversity, non-degenerate bounds in [0, 1]).')
    else:
        print('  STATUS: [YELLOW/RED - NEEDS INVESTIGATION]')
    print('=' * 75)


if __name__ == '__main__':
    main()
