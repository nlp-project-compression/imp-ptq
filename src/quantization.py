"""
Post-Training Quantization helpers.

- prepare_calibration_dataloader(...)
- quantize_static_w8a8(model, calibration_loader, per_channel=True)
"""

import torch
import torch.nn as nn
from torch.ao.quantization import (
    get_default_qconfig_mapping,
    QConfigMapping,
    default_qconfig,
    QConfig,
    default_per_channel_qconfig,
)
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.utils.data import DataLoader
from typing import Optional


def prepare_calibration_dataloader(
    dataset,
    batch_size: int = 32,
    num_samples: Optional[int] = None,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> DataLoader:
    """
    Prepare a DataLoader for calibration data.

    Args:
        dataset: HuggingFace dataset (or any dataset with __len__ and __getitem__)
        batch_size: Batch size for calibration
        num_samples: Number of samples to use for calibration. If None, uses full dataset.
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling

    Returns:
        DataLoader for calibration
    """
    if num_samples is not None and num_samples < len(dataset):
        # Create a subset
        if shuffle and seed is not None:
            torch.manual_seed(seed)
        indices = torch.randperm(len(dataset))[:num_samples] if shuffle else torch.arange(num_samples)
        subset = dataset.select(indices.tolist())
    else:
        subset = dataset

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle during calibration
        collate_fn=lambda x: {
            k: torch.stack([item[k] for item in x]) if isinstance(x[0][k], torch.Tensor) else [item[k] for item in x]
            for k in x[0].keys()
        },
    )


def get_qconfig_mapping(per_channel: bool = True) -> QConfigMapping:
    """
    Get quantization config mapping for W8A8 (weights 8-bit, activations 8-bit).

    Args:
        per_channel: If True, use per-channel quantization for weights (recommended).
                     If False, use per-tensor quantization for weights.

    Returns:
        QConfigMapping for static quantization
    """
    if per_channel:
        # Per-channel weight quantization, per-tensor activation quantization
        qconfig = QConfig(
            activation=default_qconfig.activation,  # per-tensor activation
            weight=default_per_channel_qconfig.weight,  # per-channel weight
        )
    else:
        qconfig = default_qconfig  # per-tensor for both

    # Create mapping: apply to all Linear layers
    qconfig_mapping = QConfigMapping().set_global(qconfig)

    return qconfig_mapping


def quantize_static_w8a8(
    model: nn.Module,
    calibration_loader: DataLoader,
    per_channel: bool = True,
    device: str = "cpu",
) -> nn.Module:
    """
    Apply static W8A8 post-training quantization to a BERT model.

    This function:
    1. Prepares the model for quantization using FX Graph Mode
    2. Calibrates the model using the provided calibration data
    3. Converts the model to a quantized model

    Args:
        model: The model to quantize (should be in eval mode)
        calibration_loader: DataLoader with calibration data
        per_channel: If True, use per-channel weight quantization (recommended)
        device: Device to run calibration on ("cpu" or "cuda")

    Returns:
        Quantized model (in eval mode)
    """
    model.eval()
    model = model.to(device)

    # Get quantization config
    qconfig_mapping = get_qconfig_mapping(per_channel=per_channel)

    # Prepare example input for FX tracing
    # Get one batch from calibration loader to determine input structure
    example_batch = next(iter(calibration_loader))
    input_ids = example_batch["input_ids"].to(device)
    attention_mask = example_batch.get("attention_mask", None)
    
    # For HuggingFace models, we need to create a wrapper that FX can trace
    # FX works best with positional args, so we'll create a wrapper class
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, input_ids, attention_mask=None):
            return self.model(input_ids=input_ids, attention_mask=attention_mask)
    
    wrapped_model = ModelWrapper(model)
    
    # Prepare example inputs as tuple for FX
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        example_inputs = (input_ids, attention_mask)
    else:
        example_inputs = (input_ids,)

    # Prepare model for quantization (insert observers)
    print("Preparing model for quantization (FX Graph Mode)...")
    prepared_model = prepare_fx(
        wrapped_model,
        qconfig_mapping,
        example_inputs,
    )

    # Calibrate the model
    print(f"Calibrating model with {len(calibration_loader)} batches...")
    prepared_model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(calibration_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                prepared_model(input_ids, attention_mask)
            else:
                prepared_model(input_ids)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Calibrated {batch_idx + 1}/{len(calibration_loader)} batches")

    # Convert to quantized model
    print("Converting to quantized model...")
    quantized_wrapped = convert_fx(prepared_model)
    
    # The quantized model is still wrapped, which is fine for our use case
    # The wrapper preserves the forward signature we need
    print("Quantization complete!")
    return quantized_wrapped


def apply_dynamic_w8a32(model: nn.Module) -> nn.Module:
    """
    Apply dynamic W8A32 quantization (weights quantized, activations in FP32).

    This is a simple baseline for comparison.

    Args:
        model: The model to quantize

    Returns:
        Dynamically quantized model
    """
    # Dynamic quantization only quantizes weights, not activations
    # It's simpler and doesn't require calibration
    return torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},  # Only quantize Linear layers
        dtype=torch.qint8,
    )
