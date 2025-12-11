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
from torch.fx import symbolic_trace
from torch.utils.data import DataLoader
from typing import Optional

# Try importing Optimum for transformer quantization
try:
    from optimum.onnxruntime import ORTQuantizer, ORTModelForSequenceClassification
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False


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


def get_qconfig_mapping(per_channel: bool = True, quantize_embeddings: bool = False) -> QConfigMapping:
    """
    Get quantization config mapping for W8A8 (weights 8-bit, activations 8-bit).

    Args:
        per_channel: If True, use per-channel quantization for weights (recommended).
                     If False, use per-tensor quantization for weights.
        quantize_embeddings: If False, skip quantizing embeddings (common practice).

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

    # Create mapping: apply to Linear layers
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    
    # If not quantizing embeddings, set embeddings to None (FP32)
    if not quantize_embeddings:
        # Set embeddings to None qconfig (no quantization)
        qconfig_mapping.set_object_type(nn.Embedding, None)

    return qconfig_mapping


def quantize_static_w8a8(
    model: nn.Module,
    calibration_loader: DataLoader,
    per_channel: bool = True,
    device: str = "cpu",
    quantize_embeddings: bool = False,
    use_optimum: bool = True,
    calibration_method: str = "minmax",
) -> nn.Module:
    """
    Apply static W8A8 post-training quantization to a BERT model.

    This function:
    1. Prepares the model for quantization (using Optimum or FX Graph Mode)
    2. Calibrates the model using the provided calibration data
    3. Converts the model to a quantized model

    Args:
        model: The model to quantize (should be in eval mode)
        calibration_loader: DataLoader with calibration data
        per_channel: If True, use per-channel weight quantization (recommended)
        device: Device to run calibration on ("cpu" or "cuda")
        quantize_embeddings: If False, skip quantizing embeddings (default: False)
        use_optimum: If True, use HuggingFace Optimum (recommended for transformers).
                     If False, try PyTorch FX (has known issues with BERT)
        calibration_method: Calibration method to use. Options: "minmax" (default), "entropy", "percentile"

    Returns:
        Quantized model (in eval mode)
    """
    # Use Optimum if available and requested (recommended for transformers)
    if use_optimum and OPTIMUM_AVAILABLE:
        return _quantize_static_w8a8_optimum(model, calibration_loader, per_channel, device, save_path=None, calibration_method=calibration_method)
    else:
        if use_optimum and not OPTIMUM_AVAILABLE:
            print("Warning: Optimum not available, falling back to PyTorch FX (may have issues)")
        return _quantize_static_w8a8_fx(model, calibration_loader, per_channel, device, quantize_embeddings)


def _quantize_static_w8a8_optimum(
    model: nn.Module,
    calibration_loader: DataLoader,
    per_channel: bool = True,
    device: str = "cpu",
    save_path: Optional[str] = None,
    calibration_method: str = "minmax",
) -> nn.Module:
    """
    Apply static W8A8 quantization using HuggingFace Optimum (ONNX Runtime backend).
    This is the recommended approach for transformer models.
    """
    print("Using HuggingFace Optimum for static W8A8 quantization...")
    print("Note: This converts the model to ONNX format first")
    
    import tempfile
    import os
    import shutil
    
    if not hasattr(model, 'save_pretrained'):
        raise ValueError("Model must be a HuggingFace model with save_pretrained method")
    
    # Use provided save_path or create temporary directory
    use_temp = save_path is None
    if use_temp:
        tmpdir = tempfile.mkdtemp()
        model_path = os.path.join(tmpdir, "model")
    else:
        model_path = save_path
        os.makedirs(model_path, exist_ok=True)
    
    try:
        
        # Save model
        print("Saving model for ONNX conversion...")
        model.save_pretrained(model_path)
        
        # Load as ORTModel and export to ONNX
        print("Converting to ONNX format...")
        ort_model = ORTModelForSequenceClassification.from_pretrained(
            model_path,
            export=True,
        )
        
        # Configure quantization - use static quantization with per-channel weights
        print("Configuring static W8A8 quantization...")
        # Use avx512_vnni config for static quantization (W8A8)
        # Note: For CPU, we might want to use a different config
        try:
            qconfig = AutoQuantizationConfig.avx512_vnni(
                is_static=True,
                per_channel=per_channel,
            )
        except:
            # Fallback to default static config if avx512_vnni is not available
            qconfig = AutoQuantizationConfig.with_static_quantization(
                per_channel=per_channel,
            )
        
        # Create quantizer from ORTModel object (this is the recommended way)
        print("Creating quantizer...")
        quantizer = ORTQuantizer.from_pretrained(ort_model)
        
        # For static quantization, we need to compute calibration ranges using fit()
        # Convert calibration data to HuggingFace Dataset format
        print(f"Preparing calibration dataset from {len(calibration_loader)} batches...")
        from datasets import Dataset
        
        calibration_samples = []
        for batch_idx, batch in enumerate(calibration_loader):
            batch_size = batch['input_ids'].shape[0] if isinstance(batch['input_ids'], torch.Tensor) else len(batch['input_ids'])
            for i in range(batch_size):
                sample = {
                    k: (v[i].cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v[i] if isinstance(v, list) else v)
                    for k, v in batch.items()
                    if k != 'labels'  # Remove labels - not a model input
                }
                calibration_samples.append(sample)
            if len(calibration_samples) >= 200:  # Limit calibration samples
                break
        
        calibration_dataset = Dataset.from_list(calibration_samples)
        print(f"Using {len(calibration_samples)} calibration samples")
        
        # Create calibration config
        from optimum.onnxruntime.configuration import CalibrationConfig, CalibrationMethod
        # CalibrationConfig requires dataset info, but we provide dataset directly to fit()
        # So we can use dummy values for dataset_name, etc.
        calibration_config = CalibrationConfig(
            dataset_name="dummy",  # Not used since we provide dataset directly
            dataset_config_name=None,
            dataset_split="train",
            dataset_num_samples=len(calibration_samples),
            method=CalibrationMethod.MinMax,  # Use MinMax calibration method
        )
        
        # Compute calibration ranges using fit()
        print("Computing calibration ranges...")
        calibration_tensors_range = quantizer.fit(
            dataset=calibration_dataset,
            calibration_config=calibration_config,
            batch_size=1,
        )
        
        # Quantize with the computed ranges
        print("Applying static quantization...")
        quantizer.quantize(
            quantization_config=qconfig,
            save_dir=model_path,
            calibration_tensors_range=calibration_tensors_range,
        )
        
        # Load quantized model
        print("Loading quantized model...")
        quantized_model = ORTModelForSequenceClassification.from_pretrained(model_path)
        print("Quantization complete!")
        
        # If using temp directory, we need to keep it alive
        # Store the temp directory path in the model so it doesn't get garbage collected
        # Also store a reference to prevent garbage collection
        if use_temp:
            quantized_model._temp_dir = tmpdir  # Keep reference to prevent deletion
            # Also store the model path so save_pretrained can find the files
            quantized_model._model_path = model_path
        
        return quantized_model
    
    except Exception as e:
        # Clean up temp directory on error
        if use_temp and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        raise


def _quantize_static_w8a8_fx(
    model: nn.Module,
    calibration_loader: DataLoader,
    per_channel: bool = True,
    device: str = "cpu",
    quantize_embeddings: bool = False,
) -> nn.Module:
    # FX quantization works best on CPU, so we'll use CPU for quantization
    # even if the original model was on CUDA
    model.eval()
    model = model.cpu()  # Force CPU for quantization
    device = "cpu"  # Override device to CPU

    # Fix FX tracing issue: The BERT model accesses embeddings.token_type_ids buffer directly
    # which fails during FX tracing. We need to patch BERT's forward method.
    if hasattr(model, 'bert'):
        bert_model = model.bert
        original_bert_forward = bert_model.forward
        
        # Patch BERT forward to avoid the problematic buffer access
        def patched_bert_forward(self_bert, input_ids=None, attention_mask=None, token_type_ids=None,
                                position_ids=None, head_mask=None, inputs_embeds=None,
                                encoder_hidden_states=None, encoder_attention_mask=None,
                                past_key_values=None, use_cache=None, output_attentions=None,
                                output_hidden_states=None, return_dict=None):
            # The problematic code checks: if self.embeddings.token_type_ids is not None
            # Then does: buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
            # We'll force token_type_ids to be computed as zeros if None
            if token_type_ids is None:
                # Compute token_type_ids as zeros to avoid buffer access
                if input_ids is not None:
                    input_shape = input_ids.size()
                else:
                    input_shape = inputs_embeds.size()[:-1]
                seq_length = input_shape[1]
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            
            # Call original forward with computed token_type_ids
            return original_bert_forward(
                input_ids, attention_mask, token_type_ids, position_ids, head_mask,
                inputs_embeds, encoder_hidden_states, encoder_attention_mask,
                past_key_values, use_cache, output_attentions, output_hidden_states, return_dict
            )
        
        import types
        bert_model.forward = types.MethodType(patched_bert_forward, bert_model)
        print("Patched BERT forward to avoid token_type_ids buffer access during FX tracing")
        
        # Also patch embeddings forward to handle position_ids slicing issue
        if hasattr(bert_model, 'embeddings'):
            embeddings = bert_model.embeddings
            original_embeddings_forward = embeddings.forward
            
            def patched_embeddings_forward(self_emb, input_ids=None, token_type_ids=None, position_ids=None,
                                          inputs_embeds=None, past_key_values_length=0):
                # The problematic code: position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
                # During FX tracing, past_key_values_length and seq_length are Proxy objects
                # We need to compute position_ids explicitly
                if position_ids is None:
                    if input_ids is not None:
                        input_shape = input_ids.size()
                    else:
                        input_shape = inputs_embeds.size()[:-1]
                    seq_length = input_shape[1]
                    # Create position_ids explicitly to avoid slicing Proxy objects
                    position_ids = torch.arange(
                        past_key_values_length, 
                        seq_length + past_key_values_length, 
                        dtype=torch.long, 
                        device=input_ids.device if input_ids is not None else inputs_embeds.device
                    )
                    position_ids = position_ids.unsqueeze(0).expand(input_shape)
                
                # Also handle token_type_ids if None
                if token_type_ids is None:
                    if input_ids is not None:
                        input_shape = input_ids.size()
                    else:
                        input_shape = inputs_embeds.size()[:-1]
                    device = input_ids.device if input_ids is not None else inputs_embeds.device
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
                
                return original_embeddings_forward(input_ids, token_type_ids, position_ids, inputs_embeds, past_key_values_length)
            
            embeddings.forward = types.MethodType(patched_embeddings_forward, embeddings)
            print("Patched embeddings forward to avoid position_ids slicing issues during FX tracing")

    # Get quantization config (skip embeddings if requested to avoid FX tracing issues)
    qconfig_mapping = get_qconfig_mapping(per_channel=per_channel, quantize_embeddings=quantize_embeddings)

    # Prepare example input for FX tracing
    # Get one batch from calibration loader to determine input structure
    example_batch = next(iter(calibration_loader))
    input_ids = example_batch["input_ids"].to(device)
    attention_mask = example_batch.get("attention_mask", None)
    
    # Create a simple wrapper for the model
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, input_ids, attention_mask=None):
            return self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                token_type_ids=None  # SST-2 doesn't use token_type_ids
            )
    
    wrapped_model = ModelWrapper(model)
    
    # Prepare example inputs as tuple for FX
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        example_inputs = (input_ids, attention_mask)
    else:
        example_inputs = (input_ids,)

    # Prepare model for quantization (insert observers)
    # Note: FX quantization with HuggingFace models can be tricky due to dynamic operations
    print("Preparing model for quantization (FX Graph Mode)...")
    if not quantize_embeddings:
        print("Note: Embeddings will remain in FP32 (workaround for FX tracing issues)")
    print("Note: Using BERT forward patching to avoid token_type_ids buffer access...")
    
    # Try prepare_fx - it handles tracing internally
    try:
        prepared_model = prepare_fx(
            wrapped_model,
            qconfig_mapping,
            example_inputs,
        )
    except Exception as e:
        error_msg = str(e)
        # Provide helpful error message for common FX tracing issues
        if "slice indices" in error_msg or "token_type_ids" in error_msg.lower() or "__index__" in error_msg:
            raise RuntimeError(
                f"\n{'='*70}\n"
                f"FX Quantization Error: HuggingFace BERT models have dynamic operations\n"
                f"that FX tracing cannot handle (e.g., token_type_ids buffer slicing).\n"
                f"{'='*70}\n"
                f"Original error: {error_msg}\n\n"
                f"Possible solutions:\n"
                f"1. Use dynamic quantization (W8A32) instead - it works with BERT\n"
                f"2. Use HuggingFace Optimum library for transformer quantization\n"
                f"3. Manually quantize only the Linear layers (more complex)\n"
                f"{'='*70}\n"
            ) from e
        else:
            raise

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
        Dynamically quantized model (on CPU, as dynamic quantization only supports CPU)
    """
    # Dynamic quantization only works on CPU, not CUDA
    # Move model to CPU, quantize, then return
    original_device = next(model.parameters()).device
    model_cpu = model.cpu()
    model_cpu.eval()
    
    # Dynamic quantization only quantizes weights, not activations
    # It's simpler and doesn't require calibration
    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},  # Only quantize Linear layers
        dtype=torch.qint8,
    )
    
    # Note: Quantized model stays on CPU (dynamic quantization doesn't support CUDA)
    return quantized_model
