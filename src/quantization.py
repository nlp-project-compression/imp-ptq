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
    if num_samples is not None and num_samples < len(dataset):
        if shuffle and seed is not None:
            torch.manual_seed(seed)
        indices = torch.randperm(len(dataset))[:num_samples] if shuffle else torch.arange(num_samples)
        subset = dataset.select(indices.tolist())
    else:
        subset = dataset

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: {
            k: torch.stack([item[k] for item in x]) if isinstance(x[0][k], torch.Tensor) else [item[k] for item in x]
            for k in x[0].keys()
        },
    )


def get_qconfig_mapping(per_channel: bool = True, quantize_embeddings: bool = False) -> QConfigMapping:
    if per_channel:
        qconfig = QConfig(
            activation=default_qconfig.activation,
            weight=default_per_channel_qconfig.weight,
        )
    else:
        qconfig = default_qconfig

    qconfig_mapping = QConfigMapping().set_global(qconfig)
    
    if not quantize_embeddings:
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

    """
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
    print("Using HuggingFace Optimum for static W8A8 quantization...")
    
    import tempfile
    import os
    import shutil
    
    if not hasattr(model, 'save_pretrained'):
        raise ValueError("Model must be a HuggingFace model with save_pretrained method")
    
    use_temp = save_path is None
    if use_temp:
        tmpdir = tempfile.mkdtemp()
        model_path = os.path.join(tmpdir, "model")
    else:
        model_path = save_path
        os.makedirs(model_path, exist_ok=True)
    
    try:
        print("Saving model for ONNX conversion...")
        model.save_pretrained(model_path)
        
        print("Converting to ONNX format...")
        ort_model = ORTModelForSequenceClassification.from_pretrained(
            model_path,
            export=True,
        )
        
        print("Configuring static W8A8 quantization...")
        try:
            qconfig = AutoQuantizationConfig.avx512_vnni(
                is_static=True,
                per_channel=per_channel,
            )
        except:
            qconfig = AutoQuantizationConfig.with_static_quantization(
                per_channel=per_channel,
            )

        print("Creating quantizer...")
        quantizer = ORTQuantizer.from_pretrained(ort_model)
        
        print(f"Preparing calibration dataset from {len(calibration_loader)} batches...")
        from datasets import Dataset
        
        calibration_samples = []
        for batch_idx, batch in enumerate(calibration_loader):
            batch_size = batch['input_ids'].shape[0] if isinstance(batch['input_ids'], torch.Tensor) else len(batch['input_ids'])
            for i in range(batch_size):
                sample = {
                    k: (v[i].cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v[i] if isinstance(v, list) else v)
                    for k, v in batch.items()
                    if k != 'labels'
                }
                calibration_samples.append(sample)
            if len(calibration_samples) >= 200:
                break
        
        calibration_dataset = Dataset.from_list(calibration_samples)
        print(f"Using {len(calibration_samples)} calibration samples")
        
        from optimum.onnxruntime.configuration import CalibrationConfig, CalibrationMethod
        calibration_config = CalibrationConfig(
            dataset_name="dummy", 
            dataset_config_name=None,
            dataset_split="train",
            dataset_num_samples=len(calibration_samples),
            method=CalibrationMethod.MinMax,
        )
        
        print("Computing calibration ranges...")
        calibration_tensors_range = quantizer.fit(
            dataset=calibration_dataset,
            calibration_config=calibration_config,
            batch_size=1,
        )
        
        print("Applying static quantization...")
        quantizer.quantize(
            quantization_config=qconfig,
            save_dir=model_path,
            calibration_tensors_range=calibration_tensors_range,
        )
        
        print("Loading quantized model...")
        quantized_model = ORTModelForSequenceClassification.from_pretrained(model_path)
        print("Quantization complete!")

        if use_temp:
            quantized_model._temp_dir = tmpdir 
            quantized_model._model_path = model_path
        
        return quantized_model
    
    except Exception as e:
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
    model.eval()
    model = model.cpu() 
    device = "cpu" 

    if hasattr(model, 'bert'):
        bert_model = model.bert
        original_bert_forward = bert_model.forward
        
        def patched_bert_forward(self_bert, input_ids=None, attention_mask=None, token_type_ids=None,
                                position_ids=None, head_mask=None, inputs_embeds=None,
                                encoder_hidden_states=None, encoder_attention_mask=None,
                                past_key_values=None, use_cache=None, output_attentions=None,
                                output_hidden_states=None, return_dict=None):

            if token_type_ids is None:
                if input_ids is not None:
                    input_shape = input_ids.size()
                else:
                    input_shape = inputs_embeds.size()[:-1]
                seq_length = input_shape[1]
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            
            return original_bert_forward(
                input_ids, attention_mask, token_type_ids, position_ids, head_mask,
                inputs_embeds, encoder_hidden_states, encoder_attention_mask,
                past_key_values, use_cache, output_attentions, output_hidden_states, return_dict
            )
        
        import types
        bert_model.forward = types.MethodType(patched_bert_forward, bert_model)
        print("Patched BERT forward to avoid token_type_ids buffer access during FX tracing")
        
        if hasattr(bert_model, 'embeddings'):
            embeddings = bert_model.embeddings
            original_embeddings_forward = embeddings.forward
            
            def patched_embeddings_forward(self_emb, input_ids=None, token_type_ids=None, position_ids=None,
                                          inputs_embeds=None, past_key_values_length=0):
                if position_ids is None:
                    if input_ids is not None:
                        input_shape = input_ids.size()
                    else:
                        input_shape = inputs_embeds.size()[:-1]
                    seq_length = input_shape[1]
                    position_ids = torch.arange(
                        past_key_values_length, 
                        seq_length + past_key_values_length, 
                        dtype=torch.long, 
                        device=input_ids.device if input_ids is not None else inputs_embeds.device
                    )
                    position_ids = position_ids.unsqueeze(0).expand(input_shape)
                
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

    qconfig_mapping = get_qconfig_mapping(per_channel=per_channel, quantize_embeddings=quantize_embeddings)

    example_batch = next(iter(calibration_loader))
    input_ids = example_batch["input_ids"].to(device)
    attention_mask = example_batch.get("attention_mask", None)
    
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, input_ids, attention_mask=None):
            return self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                token_type_ids=None
            )
    
    wrapped_model = ModelWrapper(model)
    
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        example_inputs = (input_ids, attention_mask)
    else:
        example_inputs = (input_ids,)

    print("Preparing model for quantization (FX Graph Mode)...")
    if not quantize_embeddings:
        print("Note: Embeddings will remain in FP32 (workaround for FX tracing issues)")
    print("Note: Using BERT forward patching to avoid token_type_ids buffer access...")
    
    try:
        prepared_model = prepare_fx(
            wrapped_model,
            qconfig_mapping,
            example_inputs,
        )
    except Exception as e:
        error_msg = str(e)
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

    print("Converting to quantized model...")
    quantized_wrapped = convert_fx(prepared_model)
    
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
    original_device = next(model.parameters()).device
    model_cpu = model.cpu()
    model_cpu.eval()

    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},
        dtype=torch.qint8,
    )
    
    return quantized_model
