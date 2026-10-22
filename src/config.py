import torch

# Global random seed for reproducibility
SEED = 0

# Precision
DTYPE = torch.float32

# Hyperparameters based on Ding et al. (IEEE TIFS 2024)
EPOCHS = 2000
BATCH_SIZE = 128
Z_SIZE = 128  # Latent noise dimension

# Learning rates and loop counts
G_LR = 2e-4
CD_LR = 2e-4
CD_LOOPS = 5
G_LOOPS = 1
ADAM_BETAS = (0.5, 0.999)

# Feature and class dimensions for CICIDS2017
FEATURE_NUM = 78
LABEL_NUM = 6
CLASSES = ['Benign', 'DoS', 'PortScan', 'BruteForce', 'WebAttack', 'Bot']
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# Device selection
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_gpu_status():
    print('=' * 60)
    print(f'PyTorch Version : {torch.__version__}')
    print(f'CUDA Available  : {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'Device Count    : {torch.cuda.device_count()}')
        print(f'Active Device   : {torch.cuda.get_device_name(0)}')
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f'Total GPU Memory: {total_mem:.2f} GB')
    else:
        print('Running on CPU (CUDA unavailable)')
    print(f'Execution Device: {DEVICE}')
    print('=' * 60)
