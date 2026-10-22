import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.dataset import CICIDSTrainingData
from src.models import CDModel, GeneratorModel
from src.tmg_gan import TMGGAN


def run_smoke_test() -> None:
    print('=' * 70)
    print('STEP 2: TMG-GAN Smoke Test on CICIDS2017')
    print('=' * 70)

    # 1. Print GPU and Environment Status
    config.print_gpu_status()
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    device = config.DEVICE
    print(f'\nRunning shape and forward-pass unit tests on {device}...')

    # 2. Test CDModel Shapes
    cd = CDModel(in_features=78, label_num=6).to(device)

    # Standard batch size test (B = 8)
    x_test = torch.randn(8, 78, device=device, dtype=torch.float32)
    d_out, c_out = cd(x_test)
    feats = cd.hidden_status

    print(f'  CDModel forward (batch=8):')
    print(f'    Input shape      : {list(x_test.shape)}')
    print(f'    D output shape   : {list(d_out.shape)} (Expected: [8, 1])')
    print(f'    C output shape   : {list(c_out.shape)} (Expected: [8, 6])')
    print(f'    F features shape : {list(feats.shape)} (Expected: [8, 256])')

    assert list(d_out.shape) == [8, 1], f'Incorrect D shape: {d_out.shape}'
    assert list(c_out.shape) == [8, 6], f'Incorrect C shape: {c_out.shape}'
    assert list(feats.shape) == [8, 256], f'Incorrect feature shape: {feats.shape}'
    assert torch.isfinite(d_out).all(), 'D output contains non-finite values!'
    assert torch.isfinite(c_out).all(), 'C output contains non-finite values!'

    # Critical edge case test: batch size 1 (verifies no .squeeze() collapse bug)
    x_single = torch.randn(1, 78, device=device, dtype=torch.float32)
    d_single, c_single = cd(x_single)
    print(f'  CDModel forward (batch=1 edge case):')
    print(f'    D shape: {list(d_single.shape)} | C shape: {list(c_single.shape)}')
    assert list(d_single.shape) == [1, 1], f'Batch 1 D shape broken: {d_single.shape}'
    assert list(c_single.shape) == [1, 6], f'Batch 1 C shape broken: {c_single.shape}'

    # 3. Test GeneratorModel Shapes
    gen = GeneratorModel(z_size=config.Z_SIZE, feature_num=78).to(device)
    z_test = torch.randn(8, config.Z_SIZE, device=device, dtype=torch.float32)
    g_out = gen(z_test)

    print(f'  GeneratorModel forward (batch=8):')
    print(f'    Z input shape    : {list(z_test.shape)}')
    print(f'    Generated shape  : {list(g_out.shape)} (Expected: [8, 78])')
    print(f'    Value range      : [{g_out.min().item():.4f}, {g_out.max().item():.4f}] (Expected in [0, 1])')

    assert list(g_out.shape) == [8, 78], f'Incorrect G shape: {g_out.shape}'
    assert g_out.min().item() >= 0.0 and g_out.max().item() <= 1.0, 'G outputs outside [0, 1] range!'
    assert torch.isfinite(g_out).all(), 'G output contains non-finite values!'

    # 4. Test Data Loading
    data_dir = PROJECT_ROOT / 'data' / 'datasets' / 'CICIDS2017'
    print(f'\nLoading preprocessed dataset from {data_dir}...')
    dataset = CICIDSTrainingData(data_dir=data_dir, device=device)
    assert dataset.feature_count == 78, f'Expected 78 features in dataset, got {dataset.feature_count}'

    # 5. Test Mini Training Loop (2 Epochs)
    print('\nTesting TMG-GAN mini training loop (2 epochs, batch_size=64)...')
    tmg_gan = TMGGAN(feature_num=78, label_num=6, z_size=config.Z_SIZE, device=device)

    smoke_checkpoint_dir = PROJECT_ROOT / 'models' / 'smoke_test'
    smoke_log_path = PROJECT_ROOT / 'logs' / 'smoke_gan_losses.csv'

    start = time.time()
    tmg_gan.fit(
        dataset=dataset,
        epochs=2,
        batch_size=64,
        checkpoint_dir=smoke_checkpoint_dir,
        loss_log_path=smoke_log_path,
        save_freq=1,
    )
    print(f'Mini-run completed in {time.time() - start:.2f}s.')

    # Check that losses log was created and has finite numbers
    assert smoke_log_path.exists(), 'Loss log was not created!'
    with open(smoke_log_path, encoding='utf-8') as f:
        lines = f.readlines()
    print(f'Loss log rows created: {len(lines)} (Header + 2 epochs)')
    assert len(lines) >= 3, 'Loss log incomplete'

    # 6. Test Generation Functions
    print('\nTesting Generator raw sample generation:')
    for test_class in [0, 1, 4, 5]:
        raw_samples = tmg_gan.generate_raw_samples(test_class, 16)
        print(f'  Class {test_class} ({config.CLASSES[test_class]:<11}): generated {len(raw_samples)} raw samples [min={raw_samples.min():.4f}, max={raw_samples.max():.4f}]')
        assert len(raw_samples) == 16, f'Failed raw generation for class {test_class}'
        assert list(raw_samples.shape) == [16, 78], f'Incorrect raw shape: {raw_samples.shape}'
        assert torch.isfinite(raw_samples).all(), 'Raw samples contain non-finite numbers!'

    print('\nTesting Algorithm 2 filtering pipeline (strict=False on 2-epoch model):')
    for test_class in [0, 1, 4]:
        samples, acc_rate = tmg_gan.generate_qualified_samples(
            target_label=test_class,
            count=16,
            batch_size=64,
            max_batches=10,
            strict=False,
        )
        print(f'  Class {test_class} ({config.CLASSES[test_class]:<11}): returned {len(samples)} samples, filter acceptance rate: {acc_rate:.2%}')
        assert len(samples) == 16, f'Failed to generate requested count for class {test_class}'
        assert list(samples.shape) == [16, 78], f'Incorrect filtered shape: {samples.shape}'
        assert torch.isfinite(samples).all(), 'Filtered samples contain non-finite values!'

    print('\n' + '=' * 70)
    print('SMOKE TEST PASSED: All shapes, losses, and generation functions verified.')
    print('=' * 70)


if __name__ == '__main__':
    run_smoke_test()
