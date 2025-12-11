"""
Runs calibration-size ablation (A.2):
- Test calibration with ~100, ~500, and ~2000 examples
- Evaluate each on full dev set
- Record accuracy and runtime
"""

import argparse
import json
import os
import time
import torch
import numpy as np
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from src.data import load_glue_dataset
from src.quantization import quantize_static_w8a8, prepare_calibration_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Run calibration-size ablation study")
    parser.add_argument("--task", type=str, default="sst2", choices=["sst2", "mrpc"])
    parser.add_argument("--model_dir", required=True, help="Path to fine-tuned model checkpoint")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./calib_ablation_results")
    parser.add_argument("--calib_sizes", type=int, nargs="+", default=[100, 500, 2000],
                        help="Calibration sizes to test")
    parser.add_argument("--calib_batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--per_channel", action="store_true", default=True, help="Use per-channel weight quantization")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate_model(model, eval_dataset, tokenizer, device="cpu", task="sst2"):
    """Evaluate a model and return metrics."""
    from torch.utils.data import DataLoader
    
    metric = evaluate.load("glue", task)
    model.eval()
    
    def collate_fn(batch):
        return {
            k: torch.stack([item[k] for item in batch]) if isinstance(batch[0][k], torch.Tensor) 
            else [item[k] for item in batch]
            for k in batch[0].keys()
        }
    
    eval_loader = DataLoader(eval_dataset, batch_size=32, collate_fn=collate_fn)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            labels = batch["labels"]
            
            try:
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                    outputs = model(input_ids, attention_mask)
                else:
                    outputs = model(input_ids)
            except TypeError:
                if attention_mask is not None:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    outputs = model(input_ids=input_ids)
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            preds = np.argmax(logits.cpu().numpy(), axis=-1)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
    
    metrics = metric.compute(predictions=all_preds, references=all_labels)
    return {f"eval_{k}": v for k, v in metrics.items()}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"\nLoading model from: {args.model_dir}")
    model_fp32 = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model_fp32.to(device)
    model_fp32.eval()
    print("Model loaded.")
    
    print(f"\nLoading {args.task} dataset...")
    train_ds, eval_ds, _ = load_glue_dataset(args.task, args.model_name, args.max_length)
    print(f"Train size: {len(train_ds)}, Eval size: {len(eval_ds)}")
    
    print("\n" + "="*60)
    print("Evaluating FP32 baseline...")
    print("="*60)
    fp32_metrics = evaluate_model(model_fp32, eval_ds, tokenizer, device, task=args.task)
    fp32_acc = fp32_metrics.get("eval_accuracy", fp32_metrics.get("eval_f1", 0.0))
    print(f"FP32 Accuracy: {fp32_acc:.4f}")
    
    results = {
        "task": args.task,
        "model_dir": args.model_dir,
        "fp32_accuracy": fp32_acc,
        "fp32_metrics": fp32_metrics,
        "calibration_results": [],
    }
    
    print("\n" + "="*60)
    print("Running calibration-size ablation...")
    print("="*60)
    
    for calib_size in args.calib_sizes:
        print(f"\n{'='*60}")
        print(f"Calibration size: {calib_size} examples")
        print(f"{'='*60}")
        
        print(f"Preparing calibration data ({calib_size} examples)...")
        calib_loader = prepare_calibration_dataloader(
            train_ds,
            batch_size=args.calib_batch_size,
            num_samples=calib_size,
            shuffle=True,
            seed=args.seed,
        )
        print(f"Calibration loader: {len(calib_loader)} batches")
        
        print("Applying static W8A8 quantization...")
        start_time = time.time()
        
        model_copy = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
        model_copy.to(device)
        model_copy.eval()
        
        try:
            model_quantized = quantize_static_w8a8(
                model_copy,
                calib_loader,
                per_channel=args.per_channel,
                device=device,
            )
            quant_time = time.time() - start_time
            
            print("Evaluating quantized model...")
            w8a8_metrics = evaluate_model(model_quantized, eval_ds, tokenizer, device, task=args.task)
            w8a8_acc = w8a8_metrics.get("eval_accuracy", w8a8_metrics.get("eval_f1", 0.0))
            delta = w8a8_acc - fp32_acc
            
            print(f"Results:")
            print(f"  Accuracy: {w8a8_acc:.4f}")
            print(f"  Δ vs FP32: {delta:+.4f}")
            print(f"  Quantization time: {quant_time:.2f} seconds")
            
            results["calibration_results"].append({
                "calib_size": calib_size,
                "accuracy": w8a8_acc,
                "delta_vs_fp32": delta,
                "quantization_time_seconds": quant_time,
                "metrics": w8a8_metrics,
            })
            
        except Exception as e:
            print(f"ERROR during quantization: {e}")
            results["calibration_results"].append({
                "calib_size": calib_size,
                "error": str(e),
            })
    
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f"{args.task}_calib_ablation.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("Summary Table")
    print("="*60)
    print(f"{'Calib size':<12} {'Acc (W8A8)':<15} {'Δ vs FP32':<15} {'Time (s)':<12}")
    print("-" * 60)
    print(f"{'FP32 (baseline)':<12} {fp32_acc:<15.4f} {'0.0000':<15} {'-':<12}")
    
    for res in results["calibration_results"]:
        if "error" not in res:
            print(f"{res['calib_size']:<12} {res['accuracy']:<15.4f} {res['delta_vs_fp32']:+.4f}        {res['quantization_time_seconds']:<12.2f}")
        else:
            print(f"{res['calib_size']:<12} {'ERROR':<15} {'-':<15} {'-':<12}")
    
    print("="*60)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()

