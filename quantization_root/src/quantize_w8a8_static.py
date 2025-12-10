
# src/quantize_w8a8_static.py

import os
import torch
import copy
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from data_utils import load_sst2_dataset, get_dataloader, get_calibration_dataset
from eval_utils import evaluate
from utils import compute_size

import torch.quantization as tq

import argparse

def main():
    parser = argparse.ArgumentParser(description="Static INT8 PTQ for BERT (W8A8) with calibration")
    parser.add_argument("--fp32_model_dir", type=str, required=True,
                        help="Path to the fine-tuned FP32 BERT model directory")
    parser.add_argument("--output_dir", type=str, default="outputs/quantized/bert_sst2_w8a8",
                        help="Directory to save the static quantized model and logs")
    parser.add_argument("--calib_samples", type=int, default=500,
                        help="Number of samples from training set to use for calibration")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for calibration and evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling calibration data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load FP32 model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(args.fp32_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.fp32_model_dir)
    model.eval()
    model.cpu()

    # Save a copy of the FP32 model for baseline evaluation (to avoid quantizing it)
    fp32_model = copy.deepcopy(model).eval().cpu()

    # Set quantization backend and configuration
    torch.backends.quantized.engine = 'fbgemm'  # use fbgemm for x86 CPUs
    qconfig = tq.get_default_qconfig('fbgemm')
    model.qconfig = qconfig

    # Exclude embeddings, LayerNorm, etc., from quantization
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding) or isinstance(module, torch.nn.LayerNorm):
            module.qconfig = None
    # (We don't explicitly find nn.Softmax or activation functions because BERT uses them via functional calls.)
    # By leaving those as-is, they will be executed in FP32. Observers will be placed around them as needed.

    # Prepare the model for static quantization (insert observers)
    tq.prepare(model, inplace=True)
    
    # Calibration: run a few hundred samples through the model to collect activation stats
    train_dataset, _ = load_sst2_dataset(tokenizer)
    calib_dataset = get_calibration_dataset(train_dataset, num_samples=args.calib_samples, seed=args.seed)
    calib_loader = get_dataloader(calib_dataset, batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for batch in calib_loader:
            # Move batch to CPU (already CPU) and forward pass
            batch = {k: v for k, v in batch.items()}  # all tensors are on CPU by default after set_format
            _ = model(**batch)
    print("Calibration done on {} samples.".format(args.calib_samples))
    
    # Convert to quantized model (actual int8 weights and int8 activations with scale/zero-point)
    tq.convert(model, inplace=True)
    # After conversion, `model` is now quantized. Linear layers are now int8, and quant/dequant nodes inserted as needed.
    
    # Evaluate the quantized model on validation set
    _, val_dataset = load_sst2_dataset(tokenizer)
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    model.eval()
    fp32_model.eval()
    # Ensure model is on CPU (it is quantized, so CPU-only), and evaluate both
    device = torch.device('cpu')
    fp32_acc, _ = evaluate(fp32_model.to(device), val_loader, device=device)
    int8_acc, _ = evaluate(model, val_loader, device=device)
    
    print(f"FP32 model accuracy: {fp32_acc:.2f}%")
    print(f"Static INT8 model accuracy: {int8_acc:.2f}%")
    print(f"Accuracy drop: {(fp32_acc - int8_acc):.2f} pts")
    
    # Save quantized model weights (state_dict)
    quant_model_path = os.path.join(args.output_dir, "quantized_model.pt")
    torch.save(model.state_dict(), quant_model_path)
    print(f"Quantized model saved to {quant_model_path}")
    
    # Report model sizes and compression
    fp32_bin = os.path.join(args.fp32_model_dir, "pytorch_model.bin")
    if os.path.exists(fp32_bin):
        fp32_size = compute_size(fp32_bin)
        int8_size = compute_size(quant_model_path)
        print(f"FP32 model size: {fp32_size/1024**2:.2f} MB")
        print(f"INT8 model size: {int8_size/1024**2:.2f} MB")
        if int8_size > 0:
            print(f"Compression factor (FP32/INT8): {fp32_size/int8_size:.2f}x")
    
    # (Optional) Save accuracy results to a text or JSON file
    results = {
        "fp32_accuracy": float(f"{fp32_acc:.2f}"),
        "int8_accuracy": float(f"{int8_acc:.2f}"),
        "accuracy_drop": float(f"{(fp32_acc - int8_acc):.2f}"),
        "fp32_model_size_bytes": int(fp32_size) if 'fp32_size' in locals() else None,
        "int8_model_size_bytes": int(int8_size) if 'int8_size' in locals() else None
    }
    results_path = os.path.join(args.output_dir, "quantization_metrics.json")
    try:
        import json
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        pass

if __name__ == "__main__":
    main()
