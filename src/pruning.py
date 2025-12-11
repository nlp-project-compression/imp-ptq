from typing import List
import torch
from torch import nn
from torch.utils.data import DataLoader

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
    all_w = []
    for name, p in model.named_parameters():
        if is_prunable_param(name, p):
            all_w.append(p.data.view(-1).abs())
    if not all_w:
        return torch.tensor([])
    return torch.cat(all_w)


def measure_sparsity(model: nn.Module) -> float:
    total = 0
    zeros = 0
    for name, p in model.named_parameters():
        if not is_prunable_param(name, p):
            continue
        t = p.data
        total += t.numel()
        zeros += (t == 0).sum().item()
    return zeros / total if total > 0 else 0.0


def calculate_sparsity(model: nn.Module) -> float:
    return measure_sparsity(model)

def get_masks_from_model(model: nn.Module):
    masks = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if is_prunable_param(name, p):
                masks[name] = (p.data != 0).to(p.data.dtype)
    return masks


def apply_masks(model: nn.Module, masks):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in masks:
                p.data.mul_(masks[name])

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

    all_weights = collect_prunable_weights(model)
    if all_weights.numel() == 0:
        print("No prunable parameters found.")
        return 0.0

    num_total = all_weights.numel()
    num_prune = int(target_sparsity * num_total)
    if num_prune <= 0:
        print("target_sparsity too small; nothing pruned.")
        return 0.0

    k = min(max(num_prune, 1), num_total)
    threshold, _ = torch.kthvalue(all_weights, k)
    threshold = threshold.item()

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


def apply_global_pruning(model: nn.Module, sparsity: float) -> float:
    """
    Convenience wrapper to match the docstring:
    - Pick out prunable tensors
    - Compute global threshold
    - Zero out smallest weights
    - Return achieved sparsity
    """
    return prune_model_global_magnitude(model, sparsity)

def geometric_schedule(
    final_sparsity: float,
    num_rounds: int,
) -> List[float]:
    """
    Geometric IMP: prune a fixed fraction of the remaining weights each round.

    If r = fraction of remaining weights kept after each round,
    then (1 - final_sparsity) = r**num_rounds  =>  r = (1 - final_sparsity)**(1/num_rounds).

    At round t (1-based), cumulative sparsity = 1 - r**t.
    """
    assert 0.0 < final_sparsity < 1.0
    assert num_rounds >= 1
    r = (1.0 - final_sparsity) ** (1.0 / num_rounds)
    sparsities = []
    for t in range(1, num_rounds + 1):
        s_t = 1.0 - (r ** t)
        sparsities.append(s_t)
    return sparsities

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    masks=None
):
    """
    Minimal training loop over one epoch.

    Assumes batches look like:
      {"input_ids", "attention_mask", (optional) "token_type_ids", "labels"}

    which matches the formatted GLUE datasets from src.data.load_glue_dataset.
    """
    model.train()
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")

        optimizer.zero_grad()
        outputs = model(**batch)
        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        if masks is not None:
            apply_masks(model, masks)


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")

        outputs = model(**batch)
        logits = outputs.logits
        preds = logits.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.numel()

    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# IMP driver
# ---------------------------------------------------------------------------

def iterative_magnitude_pruning(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    final_sparsity: float,
    num_rounds: int,
    ft_epochs_per_round: int = 1,
    rewind_to_initial: bool = False,
):
    model.to(device)
    sparsities = geometric_schedule(final_sparsity, num_rounds)
    initial_state = None
    if rewind_to_initial:
        initial_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

    history = []
    current_masks = None

    for round_idx, target_s in enumerate(sparsities, start=1):
        print(f"\n=== IMP round {round_idx}/{len(sparsities)}: "
              f"target_sparsity={target_s:.3f} ===")

        if rewind_to_initial and initial_state is not None:
            model.load_state_dict(initial_state)
            model.to(device)
            optimizer.state.clear()

        if current_masks is not None:
            apply_masks(model, current_masks)

        achieved_s = apply_global_pruning(model, target_s)

        current_masks = get_masks_from_model(model)
        print(f"Global sparsity after pruning: {achieved_s:.4f}")

        for epoch in range(ft_epochs_per_round):
            print(f"  Fine-tuning epoch {epoch+1}/{ft_epochs_per_round}")
            train_one_epoch(
                model,
                train_dataloader,
                optimizer,
                device=device,
                masks=current_masks,
            )

        acc = evaluate_accuracy(model, val_dataloader, device=device)
        print(f"Validation accuracy after round {round_idx}: {acc * 100:.2f}%")

        history.append({
            "round": round_idx,
            "target_sparsity": float(target_s),
            "achieved_sparsity": float(measure_sparsity(model)),
            "val_accuracy": float(acc),
        })

    return history
