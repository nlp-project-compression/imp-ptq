
# src/utils.py

import os
import random
import numpy as np
import torch

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make deterministic (may slow down computation slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_size(path):
    """Compute the size of a file or the total size of files in a directory, in bytes."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    elif os.path.isdir(path):
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                total += os.path.getsize(fp)
        return total
    else:
        return 0

def pretty_print_metrics(metrics: dict):
    """Pretty-print metrics dictionary to console."""
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
