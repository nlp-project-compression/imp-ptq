"""
Iterative Magnitude Pruning (IMP) utilities.

- apply_global_pruning(model, sparsity)
- calculate_sparsity(model)
"""

"""
- Pick out prunable weight tensors
- Compute a global threshold for a target sparsity
- Zero out the smallest weights in-place
- Report resulting sparsity

"""

from typing import Dict, Tuple
import torch
from torch import nn

def measure_sparsity(model) -> float:
    total = 0
    zeros = 0
    for name, p in model.named_parameters():
        if not is_prunable_param(name, p):
            continue
        t = p.data
        total += t.numel()
        zeros += (t == 0).sum().item()
    return zeros / total if total > 0 else 0.0

def is_prunable_param(name: str, param: torch.Tensor) -> bool:
    """
    Decide whether a parameter should be included in pruning.
      - prune tensors with dim > 1 (e.g. Linear weights)
      - skip biases / LayerNorm / embeddings (dim <= 1)
    """
    if not param.requires_grad:
        return False
    if param.dim() <= 1:
        return False
    return True


def collect_prunable_weights(model: nn.Module) -> torch.Tensor:
    """
    Flatten and concatenate all prunable weights into a single 1D tensor.
    Used only to compute the global magnitude threshold.
    """
    all_w = []
    for name, p in model.named_parameters():
        if is_prunable_param(name, p):
            all_w.append(p.data.view(-1).abs())
    if not all_w:
        return torch.tensor([])
    return torch.cat(all_w)


def prune_model_global_magnitude(
    model: nn.Module,
    target_sparsity: float,
) -> float:
    """
    One-shot global unstructured magnitude pruning.

    Args:
        model: the nn.Module to prune in-place
        target_sparsity: desired fraction of weights to set to zero
                         across all prunable tensors (0.0–1.0)

    Returns:
        achieved_sparsity: actual sparsity after pruning
    """
    assert 0.0 <= target_sparsity < 1.0, "target_sparsity must be in [0, 1)."

    # Collect all magnitudes
    all_weights = collect_prunable_weights(model)
    if all_weights.numel() == 0:
        print("No prunable parameters found.")
        return 0.0

    num_total = all_weights.numel()
    num_prune = int(target_sparsity * num_total)
    if num_prune <= 0:
        print("target_sparsity too small; nothing pruned.")
        return 0.0

    # Find global threshold: the |w| below which we prune
    # kthvalue is 1-indexed; clamp k to valid range
    k = min(max(num_prune, 1), num_total)
    threshold, _ = torch.kthvalue(all_weights, k)
    threshold = threshold.item()

    # Apply pruning mask in-place
    total = 0
    zeros = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if is_prunable_param(name, p):
                mask = (p.data.abs() > threshold).to(p.data.dtype)
                p.data.mul_(mask)
                total += mask.numel()
                zeros += (mask == 0).sum().item()

    achieved_sparsity = zeros / total if total > 0 else 0.0
    print(
        f"Pruned to target_sparsity={target_sparsity:.2f}, "
        f"achieved_sparsity={achieved_sparsity:.4f}"
    )
    return achieved_sparsity
