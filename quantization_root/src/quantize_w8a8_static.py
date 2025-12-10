#!/usr/bin/env python
import argparse
import json
import os
from typing import Optional, Dict, Any, Tuple, Union

import torch
import torch.ao.quantization as tq
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from data_utils import load_sst2_dataset, get_dataloader
from eval_utils import evaluate
from utils import set_seed


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

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


def compute_sizes(fp32_model_dir: str, quant_dir: str) -> Optional[Dict[str, Any]]:
    """
    Compute size of the FP32 checkpoint and the quantized model file.
    Assumes quantized model is saved as 'model_int8.pt' in quant_dir.
    """
    fp32_model_file = find_model_file(fp32_model_dir)
    if fp32_model_file is None:
        print("FP32 model file not found; skipping size comparison.")
        return None

    quant_model_file = os.path.join(quant_dir, "model_int8.pt")
    if not os.path.exists(quant_model_file):
        print(f"Quantized model file {quant_model_file} not found; skipping size comparison.")
        return None

    fp32_size = os.path.getsize(fp32_model_file)
    quant_size = os.path.getsize(quant_model_file)
    compression = fp32_size / float(quant_size) if quant_size > 0 else float("inf")

    print(f"FP32 model file:      {fp32_model_file}")
    print(f"FP32 model size:      {fp32_size / 1e6:.2f} MB")
    print(f"Quantized model file: {quant_model_file}")
    print(f"Quantized model size: {quant_size / 1e6:.2f} MB")
    print(f"Compression factor:   {compression:.2f}×")

    return {
        "fp32_model_file": fp32_model_file,
        "fp32_size_bytes": fp32_size,
        "quant_model_file": quant_model_file,
        "quant_size_bytes": quant_size,
        "compression_factor": compression,
    }


def normalize_eval_output(
    result: Union[Dict[str, Any], Tuple[Any, Any]]
) -> Dict[str, float]:
    """
    Normalize the output of eval_utils.evaluate into a dict
    with at least {"loss": ..., "accuracy": ...}.
    Supports dicts or (loss, accuracy) tuples.
    """
    if isinstance(result, dict):
        loss = result.get("loss") or result.get("eval_loss")
        acc = result.get("accuracy") or result.get("acc") or result.get("eval_accuracy")
        if loss is None and acc is None:
            raise ValueError(
                "evaluate() returned a dict but no recognizable 'loss'/'accuracy' keys were found."
            )
        return {
            "loss": float(loss) if loss is not None else float("nan"),
            "accuracy": float(acc) if acc is not None else float("nan"),
        }

    if isinstance(result, tuple):
        if len(result) < 2:
            raise ValueError("evaluate() tuple must be (loss, accuracy).")
        loss, acc = result[0], result[1]
        return {"loss": float(loss), "accuracy": float(acc)}

    raise TypeError(f"Unsupported evaluate() return type {type(result)}")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Static W8A8 PTQ for BERT on SST-2")
    parser.add_argument(
        "--fp32_model_dir",
        type=str,
        required=True,
        help="Directory containing the fine-tuned FP32 BERT checkpoint (HF format).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the static W8A8 quantized model and metrics.",
    )
    parser.add_argument(
        "--calib_samples",
        type=int,
        default=500,
        help="Number of calibration samples from SST-2 train split.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for calibration and evaluation.",
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

    # *** IMPORTANT: force CPU + quantization engine for static W8A8 ***
    device = torch.device("cpu")
    torch.backends.quantized.engine = "fbgemm"

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------------------------------------------------
    # Load tokenizer + datasets
    # ---------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.fp32_model_dir)
    train_dataset, val_dataset = load_sst2_dataset(tokenizer, max_length=args.max_length)

    # calibration subset
    calib_indices = list(range(min(args.calib_samples, len(train_dataset))))
    calib_subset = train_dataset.select(calib_indices)

    calib_loader = get_dataloader(
        calib_subset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    val_loader = get_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # ---------------------------------------------------------------------
    # Load FP32 model, evaluate baseline
    # ---------------------------------------------------------------------
    print(f"Loading FP32 model from {args.fp32_model_dir}")
    model_fp32 = AutoModelForSequenceClassification.from_pretrained(args.fp32_model_dir)
    model_fp32.to(device)
    model_fp32.eval()

    raw_fp32_eval = evaluate(model_fp32, val_loader, device=device)
    fp32_metrics = normalize_eval_output(raw_fp32_eval)
    fp32_acc = fp32_metrics["accuracy"]
    fp32_loss = fp32_metrics["loss"]
    print(f"FP32 model accuracy: {fp32_acc * 100:.2f}%")
    print(f"FP32 model loss:     {fp32_loss:.4f}")

    # ---------------------------------------------------------------------
    # Configure model for static W8A8 quantization (CPU only)
    # ---------------------------------------------------------------------
    model_fp32.cpu()  # ensure on CPU for quantization

    # Per-channel weight, per-tensor activation quantization (default fbgemm qconfig)
    qconfig = tq.get_default_qconfig("fbgemm")
    model_fp32.qconfig = qconfig

    print("Preparing model for static W8A8 quantization (CPU)...")
    model_prepared = tq.prepare(model_fp32, inplace=False)

    # ---------------------------------------------------------------------
    # Calibration loop (no gradients)
    # ---------------------------------------------------------------------
    model_prepared.eval()
    with torch.inference_mode():
        num_seen = 0
        for batch in calib_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model_prepared(**batch)
            num_seen += batch["input_ids"].size(0)
    print(f"Calibration done on {num_seen} samples.")

    # ---------------------------------------------------------------------
    # Convert to quantized model
    # ---------------------------------------------------------------------
    print("Converting to quantized (W8A8) model...")
    model_int8 = tq.convert(model_prepared, inplace=False)
    model_int8.eval()
    model_int8.to(device)  # device is CPU

    # ---------------------------------------------------------------------
    # Evaluate quantized model on SST-2 dev (CPU)
    # ---------------------------------------------------------------------
    raw_int8_eval = evaluate(model_int8, val_loader, device=device)
    int8_metrics = normalize_eval_output(raw_int8_eval)
    int8_acc = int8_metrics["accuracy"]
    int8_loss = int8_metrics["loss"]
    print(f"W8A8 model accuracy: {int8_acc * 100:.2f}%")
    print(f"W8A8 model loss:     {int8_loss:.4f}")

    acc_drop = (fp32_acc - int8_acc) * 100.0
    print(f"Accuracy drop:       {acc_drop:.2f} percentage points")

    # ---------------------------------------------------------------------
    # Save quantized model
    # ---------------------------------------------------------------------
    quant_model_path = os.path.join(args.output_dir, "model_int8.pt")
    torch.save(model_int8.state_dict(), quant_model_path)
    print(f"Quantized W8A8 model saved to {quant_model_path}")

    # ---------------------------------------------------------------------
    # Size + compression factor
    # ---------------------------------------------------------------------
    size_info = compute_sizes(args.fp32_model_dir, args.output_dir)

    # ---------------------------------------------------------------------
    # Save metrics JSON
    # ---------------------------------------------------------------------
    metrics = {
        "fp32_accuracy": float(fp32_acc),
        "fp32_loss": float(fp32_loss),
        "w8a8_accuracy": float(int8_acc),
        "w8a8_loss": float(int8_loss),
        "accuracy_drop_points": float(acc_drop),
        "calib_samples": args.calib_samples,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "seed": args.seed,
    }
    if size_info is not None:
        metrics.update(size_info)

    metrics_path = os.path.join(args.output_dir, "w8a8_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
