import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.dnn import DNNClassifier

CLASSES = config.CLASSES
METHODS = ['original', 'smote', 'tmg_gan']
SEEDS = [0, 1, 2]
BATCH_SIZE = 1024


def load_test_data(device: torch.device):
    data_dir = PROJECT_ROOT / 'data' / 'datasets' / 'CICIDS2017'
    x_path = data_dir / 'x_test.csv'
    y_path = data_dir / 'y_test.csv'

    print(f'Loading untouched test set from {data_dir}...')
    x_df = pd.read_csv(x_path, dtype=np.float32)
    y_df = pd.read_csv(y_path, dtype=np.int8)

    X_test = torch.tensor(x_df.values, dtype=torch.float32)
    y_test = y_df.values.argmax(axis=1)

    print(f'  Test set flows: {len(y_test):,} | Features: {X_test.shape[1]}')
    return X_test, y_test


def evaluate_single_model(model_path: Path, X_test: torch.Tensor, y_test: np.ndarray, device: torch.device):
    model = DNNClassifier(in_features=78, num_classes=6).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), BATCH_SIZE):
            batch_x = X_test[i : i + BATCH_SIZE].to(device)
            logits = model(batch_x)
            batch_preds = logits.argmax(dim=1).cpu().numpy()
            preds.extend(batch_preds)

    preds = np.array(preds)

    # Per-class metrics
    p, r, f1, _ = precision_recall_fscore_support(y_test, preds, labels=list(range(6)), zero_division=0)
    acc = accuracy_score(y_test, preds)

    # Macro averages
    macro_p = float(np.mean(p))
    macro_r = float(np.mean(r))
    macro_f1 = float(np.mean(f1))

    # Confusion Matrix (6x6)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, preds, labels=list(range(6)))

    # Benign False-Positive Rate: Fraction of true Benign (class 0) predicted as any attack (classes 1..5)
    benign_total = cm[0].sum()
    benign_false_alarms = cm[0, 1:].sum()
    benign_fpr = float(benign_false_alarms / benign_total) if benign_total > 0 else 0.0

    return {
        'accuracy': acc,
        'macro_precision': macro_p,
        'macro_recall': macro_r,
        'macro_f1': macro_f1,
        'benign_fpr': benign_fpr,
        'per_class_precision': p,
        'per_class_recall': r,
        'per_class_f1': f1,
        'confusion_matrix': cm,
    }


def main():
    print('=' * 75)
    print('STEP 5: Comprehensive Model Evaluation on Untouched CICIDS2017 Test Set')
    print('=' * 75)

    config.print_gpu_status()
    device = config.DEVICE

    results_dir = PROJECT_ROOT / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = PROJECT_ROOT / 'models'

    X_test, y_test = load_test_data(device)

    # 1. Run inference across all 9 models
    per_class_records = []
    model_results = {m: [] for m in METHODS}
    cms_by_method = {m: np.zeros((6, 6), dtype=np.float64) for m in METHODS}

    print('\nEvaluating models on 710,976 test flows:')
    for method in METHODS:
        for seed in SEEDS:
            model_path = models_dir / f'dnn_{method}_seed_{seed}.pt'
            if not model_path.exists():
                raise FileNotFoundError(f'Missing model: {model_path}')

            t0 = time.time()
            res = evaluate_single_model(model_path, X_test, y_test, device)
            dur = time.time() - t0

            model_results[method].append(res)
            cms_by_method[method] += res['confusion_matrix']

            # Log per-class records
            for c_idx, c_name in enumerate(CLASSES):
                per_class_records.append({
                    'seed': seed,
                    'method': method,
                    'class': c_name,
                    'precision': res['per_class_precision'][c_idx],
                    'recall': res['per_class_recall'][c_idx],
                    'f1': res['per_class_f1'][c_idx],
                })

            print(
                f'  [{method.upper():<8} seed={seed}] ({dur:.1f}s) '
                f'Macro P: {res["macro_precision"]:.4f} | '
                f'Macro R: {res["macro_recall"]:.4f} | '
                f'Macro F1: {res["macro_f1"]:.4f} | '
                f'Benign FPR: {res["benign_fpr"]:.4%} | '
                f'Acc: {res["accuracy"]:.4f}'
            )

    # 2. Output: results/per_class.csv
    per_class_df = pd.DataFrame(per_class_records)
    per_class_path = results_dir / 'per_class.csv'
    per_class_df.to_csv(per_class_path, index=False)
    print(f'\nSaved per-class results: {per_class_path}')

    # 3. Output: results/summary.csv (Mean and Std over seeds)
    summary_records = []
    for method in METHODS:
        runs = model_results[method]
        accs = [r['accuracy'] for r in runs]
        precs = [r['macro_precision'] for r in runs]
        recs = [r['macro_recall'] for r in runs]
        f1s = [r['macro_f1'] for r in runs]
        fprs = [r['benign_fpr'] for r in runs]

        summary_records.append({
            'method': method,
            'accuracy_mean': np.mean(accs),
            'accuracy_std': np.std(accs),
            'macro_precision_mean': np.mean(precs),
            'macro_precision_std': np.std(precs),
            'macro_recall_mean': np.mean(recs),
            'macro_recall_std': np.std(recs),
            'macro_f1_mean': np.mean(f1s),
            'macro_f1_std': np.std(f1s),
            'benign_fpr_mean': np.mean(fprs),
            'benign_fpr_std': np.std(fprs),
        })

    summary_df = pd.DataFrame(summary_records)
    summary_path = results_dir / 'summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f'Saved summary metrics: {summary_path}')

    # 4. Output: results/paper_vs_ours.csv
    # Paper numbers transcribed directly from Table VI of Ding et al. (IEEE TIFS 2024)
    paper_comparison = [
        {
            'Method': 'DNN (Baseline)',
            'Source': 'Paper (Table VI)',
            'Macro Precision': 0.9125,
            'Macro Recall': 0.6501,
            'Macro F1': 0.6900,
            'Benign FPR': 'Not Reported',
        },
        {
            'Method': 'TMG-IDS',
            'Source': 'Paper (Table VI)',
            'Macro Precision': 0.9963,
            'Macro Recall': 0.9981,
            'Macro F1': 0.9972,
            'Benign FPR': 'Not Reported',
        },
        {
            'Method': 'DNN (Original)',
            'Source': 'Ours (Mean +/- Std)',
            'Macro Precision': f"{summary_df.loc[summary_df['method']=='original', 'macro_precision_mean'].values[0]:.4f} +/- {summary_df.loc[summary_df['method']=='original', 'macro_precision_std'].values[0]:.4f}",
            'Macro Recall': f"{summary_df.loc[summary_df['method']=='original', 'macro_recall_mean'].values[0]:.4f} +/- {summary_df.loc[summary_df['method']=='original', 'macro_recall_std'].values[0]:.4f}",
            'Macro F1': f"{summary_df.loc[summary_df['method']=='original', 'macro_f1_mean'].values[0]:.4f} +/- {summary_df.loc[summary_df['method']=='original', 'macro_f1_std'].values[0]:.4f}",
            'Benign FPR': f"{summary_df.loc[summary_df['method']=='original', 'benign_fpr_mean'].values[0]:.4%} +/- {summary_df.loc[summary_df['method']=='original', 'benign_fpr_std'].values[0]:.4%}",
        },
        {
            'Method': 'SMOTE (Matched Targets)',
            'Source': 'Ours (Mean +/- Std)',
            'Macro Precision': f"{summary_df.loc[summary_df['method']=='smote', 'macro_precision_mean'].values[0]:.4f} +/- {summary_df.loc[summary_df['method']=='smote', 'macro_precision_std'].values[0]:.4f}",
            'Macro Recall': f"{summary_df.loc[summary_df['method']=='smote', 'macro_recall_mean'].values[0]:.4f} +/- {summary_df.loc[summary_df['method']=='smote', 'macro_recall_std'].values[0]:.4f}",
            'Macro F1': f"{summary_df.loc[summary_df['method']=='smote', 'macro_f1_mean'].values[0]:.4f} +/- {summary_df.loc[summary_df['method']=='smote', 'macro_f1_std'].values[0]:.4f}",
            'Benign FPR': f"{summary_df.loc[summary_df['method']=='smote', 'benign_fpr_mean'].values[0]:.4%} +/- {summary_df.loc[summary_df['method']=='smote', 'benign_fpr_std'].values[0]:.4%}",
        },
        {
            'Method': 'TMG-GAN',
            'Source': 'Ours (Mean +/- Std)',
            'Macro Precision': f"{summary_df.loc[summary_df['method']=='tmg_gan', 'macro_precision_mean'].values[0]:.4f} +/- {summary_df.loc[summary_df['method']=='tmg_gan', 'macro_precision_std'].values[0]:.4f}",
            'Macro Recall': f"{summary_df.loc[summary_df['method']=='tmg_gan', 'macro_recall_mean'].values[0]:.4f} +/- {summary_df.loc[summary_df['method']=='tmg_gan', 'macro_recall_std'].values[0]:.4f}",
            'Macro F1': f"{summary_df.loc[summary_df['method']=='tmg_gan', 'macro_f1_mean'].values[0]:.4f} +/- {summary_df.loc[summary_df['method']=='tmg_gan', 'macro_f1_std'].values[0]:.4f}",
            'Benign FPR': f"{summary_df.loc[summary_df['method']=='tmg_gan', 'benign_fpr_mean'].values[0]:.4%} +/- {summary_df.loc[summary_df['method']=='tmg_gan', 'benign_fpr_std'].values[0]:.4%}",
        },
    ]
    paper_vs_ours_df = pd.DataFrame(paper_comparison)
    paper_vs_ours_path = results_dir / 'paper_vs_ours.csv'
    paper_vs_ours_df.to_csv(paper_vs_ours_path, index=False)
    print(f'Saved paper comparison: {paper_vs_ours_path}')

    # 5. Visualizations
    print('\nGenerating publication plots:')

    # Figure 1: Recall by Class (with Error Bars)
    plt.figure(figsize=(12, 6))
    recall_agg = per_class_df.groupby(['method', 'class'])['recall'].agg(['mean', 'std']).reset_index()

    # Reorder methods and classes
    method_order = ['original', 'smote', 'tmg_gan']
    label_map = {'original': 'DNN (Original)', 'smote': 'SMOTE (Matched)', 'tmg_gan': 'TMG-GAN (Ours)'}

    palette = {'original': '#4C78A8', 'smote': '#F58518', 'tmg_gan': '#54A24B'}

    sns.barplot(
        data=per_class_df,
        x='class',
        y='recall',
        hue='method',
        hue_order=method_order,
        palette=palette,
        errorbar='sd',
        capsize=0.1,
    )
    plt.title('Test Set Recall by Attack Class (Mean +/- Std over 3 Seeds)', fontsize=14, fontweight='bold')
    plt.xlabel('Traffic Class', fontsize=12)
    plt.ylabel('Recall Rate', fontsize=12)
    plt.ylim(0.0, 1.05)
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles, [label_map[m] for m in method_order], title='Method', loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig1_path = results_dir / 'recall_by_class.png'
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f'  Saved: {fig1_path}')

    # Figure 2: Benign False Positive Rate (FPR)
    fpr_records = []
    for m in METHODS:
        for r in model_results[m]:
            fpr_records.append({'Method': label_map[m], 'Benign FPR (%)': r['benign_fpr'] * 100})
    fpr_df = pd.DataFrame(fpr_records)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=fpr_df,
        x='Method',
        y='Benign FPR (%)',
        palette=['#4C78A8', '#F58518', '#54A24B'],
        errorbar='sd',
        capsize=0.15,
        width=0.5,
    )
    plt.title('Benign Traffic False Positive Rate (FPR) on 621,677 Benign Test Flows', fontsize=12, fontweight='bold')
    plt.ylabel('False Positive Rate (%) [Lower is Better]', fontsize=11)
    plt.xlabel('Method', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig2_path = results_dir / 'benign_fpr.png'
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f'  Saved: {fig2_path}')

    # Figure 3: Confusion Matrices per Method (Averaged over 3 seeds)
    for method in METHODS:
        cm_norm = (cms_by_method[method] / 3.0)  # average over 3 runs
        # Normalize by true row totals
        cm_row_norm = cm_norm / cm_norm.sum(axis=1, keepdims=True)

        plt.figure(figsize=(8, 6.5))
        sns.heatmap(
            cm_row_norm,
            annot=True,
            fmt='.3f',
            cmap='Blues',
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            cbar=True,
        )
        plt.title(f'Normalized Confusion Matrix: {label_map[method]} (Averaged over 3 Seeds)', fontsize=12, fontweight='bold')
        plt.ylabel('True Class', fontsize=11)
        plt.xlabel('Predicted Class', fontsize=11)
        plt.tight_layout()
        cm_path = results_dir / f'confusion_{method}.png'
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f'  Saved: {cm_path}')

    print('\n' + '=' * 75)
    print('STEP 5 EVALUATION COMPLETE')
    print('=' * 75)

    # Print Summary Table
    print('\n' + '=' * 80)
    print('FINAL RESEARCH BENCHMARK SUMMARY (Table VI & Phase 2 Addition):')
    print('=' * 80)
    print(paper_vs_ours_df.to_string(index=False))
    print('=' * 80)


if __name__ == '__main__':
    main()
