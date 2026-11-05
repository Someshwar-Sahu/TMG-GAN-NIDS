"""Model Inspector Utility for PyTorch Checkpoints (.pt files).

Allows direct terminal inspection of model architecture, layer tensors,
parameter counts, and weight statistics to show evaluators and TAs.
"""

import sys
from pathlib import Path
import torch

ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(ROOT))

from src.dnn import DNNClassifier
from src.models import CDModel, GeneratorModel

MODELS_DIR = ROOT / 'models'


def inspect_checkpoint(filename: str):
    path = MODELS_DIR / filename
    if not path.exists():
        print(f"Error: {filename} not found in {MODELS_DIR}")
        return

    size_mb = path.stat().st_size / (1024 * 1024)
    print("=" * 80)
    print(f" PYTORCH MODEL CHECKPOINT: {filename}")
    print(f" Path: {path}")
    print(f" File Size: {size_mb:.2f} MB")
    print("=" * 80)

    data = torch.load(path, map_location='cpu')

    if isinstance(data, list):
        print(f"Detected: List of {len(data)} Generator State Dictionaries (Dedicated Multi-Generator Bank)")
        for idx, state in enumerate(data):
            total_gen_params = sum(p.numel() for p in state.values() if p.dtype.is_floating_point)
            print(f"\n--- Generator G_{idx} (Minority Class {idx}) • {total_gen_params:,} parameters ---")
            for k, v in state.items():
                shape_str = str(list(v.shape))
                if v.numel() == 1:
                    print(f"  {k:<35} Value: {v.item():<20}")
                else:
                    vf = v.float()
                    print(f"  {k:<35} Shape: {shape_str:<20} Mean: {vf.mean().item():+.4f} Std: {vf.std().item():.4f}")
        print("=" * 80)
        return

    if isinstance(data, dict):
        total_params = sum(p.numel() for p in data.values())
        print(f"Total Serialized Tensors: {len(data)}")
        print(f"Total Parameters: {total_params:,}")
        print("-" * 80)
        print(f"{'Layer Key':<35} {'Tensor Shape':<22} {'Mean':<10} {'Std':<10}")
        print("-" * 80)
        for k, v in data.items():
            shape_str = str(list(v.shape))
            mean_val = f"{v.float().mean().item():+.4f}" if v.numel() > 0 else "N/A"
            std_val = f"{v.float().std().item():.4f}" if v.numel() > 1 else "N/A"
            print(f"{k:<35} {shape_str:<22} {mean_val:<10} {std_val:<10}")

        print("\n--- SAMPLE RAW WEIGHT VALUES (First 5 weights of output layer) ---")
        out_key = 'fc_out.weight' if 'fc_out.weight' in data else list(data.keys())[-2]
        sample_weights = data[out_key][0][:5].tolist()
        print(f"Layer '{out_key}' [Row 0, Cols 0-4]:")
        print(" ", [round(w, 5) for w in sample_weights])
        print("=" * 80)


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else 'dnn_tmg_gan_seed_0.pt'
    inspect_checkpoint(target)

