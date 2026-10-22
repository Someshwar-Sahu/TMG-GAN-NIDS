"""Build a rich pool of 300 diverse test flows for live random sampling.

Extracts 50 random test samples per class (total 300 flows) from the untouched test set:
- Benign, DoS, PortScan, BruteForce, WebAttack, Bot
- Includes raw and scaled features, ground truth labels, and pre-computed model predictions.
- Saved to data/samples/random_pool.json.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).absolute().parent.parent
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.dnn import DNNClassifier

DATA_DIR = ROOT / 'data' / 'datasets' / 'CICIDS2017'
SAMPLE_DIR = ROOT / 'data' / 'samples'
ARTIFACT_DIR = ROOT / 'artifacts'
MODEL_DIR = ROOT / 'models'
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ['Benign', 'DoS', 'PortScan', 'BruteForce', 'WebAttack', 'Bot']

DESCRIPTIONS = {
    'Benign': 'Legitimate network communication (web browsing, DNS, normal TLS traffic).',
    'DoS': 'High-volume Denial of Service attack attempting to overwhelm network buffers.',
    'PortScan': 'Reconnaissance attack probing multiple destination ports to discover services.',
    'BruteForce': 'Authentication attack repeatedly attempting credentials (FTP/SSH Patator).',
    'WebAttack': 'Application-layer exploit targeting web vulnerabilities (XSS / SQLi).',
    'Bot': 'Automated malware command-and-control communication beaconing.'
}


def load_models():
    models = {}
    for name, path in [
        ('original', MODEL_DIR / 'dnn_original_seed_0.pt'),
        ('smote', MODEL_DIR / 'dnn_smote_seed_0.pt'),
        ('tmg_gan', MODEL_DIR / 'dnn_tmg_gan_seed_0.pt')
    ]:
        if path.exists():
            m = DNNClassifier(in_features=78, num_classes=6)
            m.load_state_dict(torch.load(path, map_location='cpu'))
            m.eval()
            models[name] = m
    return models


def main():
    print("Loading scaler from artifacts/scaler.json...")
    with open(ARTIFACT_DIR / 'scaler.json', 'r') as f:
        scaler = json.load(f)

    feature_names = scaler['feature_names']
    mins = np.array([scaler['min'][k] for k in feature_names], dtype=np.float32)
    spans = np.array([scaler['span'][k] for k in feature_names], dtype=np.float32)

    print("Loading test dataset (first 100,000 flows for rich random sampling pool)...")
    x_test = pd.read_csv(DATA_DIR / 'x_test.csv', nrows=100000, dtype=np.float32)
    y_test = pd.read_csv(DATA_DIR / 'y_test.csv', nrows=100000)
    y_labels = y_test.values.argmax(axis=1)

    models = load_models()
    print(f"Loaded {len(models)} models for verification.")

    rng = np.random.default_rng(42)
    pool = []

    for class_idx, class_name in enumerate(CLASSES):
        indices = np.where(y_labels == class_idx)[0]
        if len(indices) == 0:
            continue
        
        sample_size = min(50, len(indices))
        selected_indices = rng.choice(indices, size=sample_size, replace=False)

        for k, idx in enumerate(selected_indices):
            scaled_vec = x_test.iloc[idx].values.astype(np.float32)
            raw_vec = scaled_vec * spans + mins

            # Run predictions across all 3 models
            x_tensor = torch.tensor(scaled_vec, dtype=torch.float32).unsqueeze(0)
            preds = {}
            for m_name, model in models.items():
                with torch.no_grad():
                    logits = model(x_tensor)
                    probs = torch.softmax(logits, dim=1).numpy()[0]
                    pred_idx = int(np.argmax(probs))
                    preds[m_name] = {
                        'predicted_class': CLASSES[pred_idx],
                        'confidence': float(probs[pred_idx]),
                        'probabilities': {CLASSES[i]: float(probs[i]) for i in range(6)},
                        'is_attack': bool(pred_idx != 0),
                        'is_correct': bool(pred_idx == class_idx)
                    }

            flow_id = f"test_{class_name.lower()}_{k+1}"

            summary_features = {
                'Destination Port': int(raw_vec[feature_names.index('Destination Port')]),
                'Flow Duration (us)': float(raw_vec[feature_names.index('Flow Duration')]),
                'Total Fwd Packets': int(raw_vec[feature_names.index('Total Fwd Packets')]),
                'Total Backward Packets': int(raw_vec[feature_names.index('Total Backward Packets')]),
                'Flow Bytes/s': float(raw_vec[feature_names.index('Flow Bytes/s')]),
                'Flow Packets/s': float(raw_vec[feature_names.index('Flow Packets/s')]),
                'Packet Length Mean': float(raw_vec[feature_names.index('Packet Length Mean')]),
                'Average Packet Size': float(raw_vec[feature_names.index('Average Packet Size')])
            }

            pool.append({
                'id': flow_id,
                'class_name': class_name,
                'class_index': class_idx,
                'description': f"{DESCRIPTIONS[class_name]} (Random Test #{k+1})",
                'summary_features': summary_features,
                'raw_features': {feature_names[i]: float(raw_vec[i]) for i in range(len(feature_names))},
                'scaled_features': [float(v) for v in scaled_vec],
                'model_predictions': preds
            })

    output_path = SAMPLE_DIR / 'random_pool.json'
    with open(output_path, 'w') as f:
        json.dump(pool, f, indent=2)

    print(f"Extracted {len(pool)} diverse test flows across all {len(CLASSES)} classes.")
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    main()
