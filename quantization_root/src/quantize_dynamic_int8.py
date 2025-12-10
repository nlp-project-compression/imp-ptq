#!/usr/bin/env python
import argparse
import json
import os
from typing import Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from data_utils import load_sst2_dataset, get_dataloader
from eval_utils import evaluate
from utils import set_seed


def find_model_file(model_dir: str) -> Optional[str]:
    """
    Return the first existing model file in `model_dir`, trying common filenames.
    Supports both HF .bin and safetensors formats.
    """
    candidate_names = ["pytorch_model.bin", "model.safetensors"]
    for name in candidate_names:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path
    return None


def compute_sizes(fp32_model_dir: str, quant_path: str) -> Optional[Dict[str, Any]]:
    """
    Compute size (in bytes) of the FP32 checkpoint and the quantized .pt file.
    Returns a dict with sizes and compression factor, or None if the FP32 file
    could not be found.
    """
    fp32_model_file = find_model_file(fp32_model_dir)
    if fp32_model_file is None:
        print("FP32 model file not found; skipping size comparison.")
        return None

    if not os.path.exists(quant_path):
        print(f"Quantized model file {quant_path} not found; skipping size comparison.")
        return None

    fp32_size = os.path.getsize(fp32_model_file)
    quant_size = os.path.getsize(quant_path)
    compression = fp32_size / float(quant_size) if quant_size > 0 else float("inf")

    print(f"FP32 model file:      {fp32_model_file}")
    print(f"FP32 model size:      {fp32_size / 1e6:.2f} MB")
    print(f"Quantized model file: {quant_path}")
    print(f"Quantized model size: {quant_size / 1e6:.2f} MB")
    print(f"Compression factor:   {compression:.2f}×")

    return {
        "fp32_model_file": fp32_model_file,
        "fp32_size_bytes": fp32_size,
        "quant_model_file": quant_path,
        "quant_size_bytes": quant_size,
        "compression_factor": compression,
    }


def main():
    parser = argparse.ArgumentParser(description="Dynamic INT8 PTQ (W8A32) for BERT on SST-2")
    parser.add_argument(
        "--fp32_model_dir",
        type=str,
        required=True,
        help="Directory containing the fine-tuned FP32 BERT checkpoint (HF format).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for the quantized INT8 model (.pt).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for SST-2 validation evaluation.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Max sequence length for tokenization.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    device_cpu = torch.device("cpu")

    # -------------------------------------------------------------------------
    # Load tokenizer, data, and FP32 model
    # -------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.fp32_model_dir)
    train_dataset, val_dataset = load_sst2_dataset(tokenizer, max_length=args.max_length)

    val_loader = get_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(f"Loading FP32 model from {args.fp32_model_dir}")
    fp32_model = AutoModelForSequenceClassification.from_pretrained(args.fp32_model_dir)
    fp32_model.to(device_cpu)
    fp32_model.eval()

    # -------------------------------------------------------------------------
    # Evaluate FP32 baseline on SST-2 dev set
    # -------------------------------------------------------------------------
    fp32_metrics = evaluate(fp32_model, val_loader, device_cpu)
    fp32_acc = fp32_metrics.get("accuracy", None)
    if fp32_acc is None:
        raise ValueError("`evaluate` did not return an 'accuracy' field.")

    print(f"FP32 model accuracy: {fp32_acc * 100:.2f}%")

    # -------------------------------------------------------------------------
    # Dynamic INT8 quantization (W8A32)
    # -------------------------------------------------------------------------
    print("Applying dynamic INT8 quantization (W8A32) ...")
    # Ensure model is on CPU for dynamic quantization
    fp32_model.cpu()

    quantized_model = torch.quantization.quantize_dynamic(
        fp32_model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )

    quantized_model.eval()

    # -------------------------------------------------------------------------
    # Evaluate quantized model on SST-2 dev set (CPU)
    # -------------------------------------------------------------------------
    quant_metrics = evaluate(quantized_model, val_loader, device_cpu)
    quant_acc = quant_metrics.get("accuracy", None)
    if quant_acc is None:
        raise ValueError("`evaluate` did not return an 'accuracy' field for quantized model.")

    print(f"Dynamic INT8 model accuracy: {quant_acc * 100:.2f}%")
    acc_drop = (fp32_acc - quant_acc) * 100.0
    print(f"Accuracy drop: {acc_drop:.2f} percentage points")

    # -------------------------------------------------------------------------
    # Save quantized model
    # -------------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(quantized_model.state_dict(), args.output_path)
    print(f"Quantized model saved to {args.output_path}")

    # -------------------------------------------------------------------------
    # Size + compression factor
    # -------------------------------------------------------------------------
    size_info = compute_sizes(args.fp32_model_dir, args.output_path)

    # -------------------------------------------------------------------------
    # Save metrics to JSON for later analysis
    # -------------------------------------------------------------------------
    metrics = {
        "fp32_accuracy": float(fp32_acc),
        "int8_dynamic_accuracy": float(quant_acc),
        "accuracy_drop_points": float(acc_drop),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
    }
    if size_info is not None:
        metrics.update(size_info)

    metrics_path = args.output_path + ".metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
