import argparse
import json
import os
import time
import torch
import numpy as np
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from src.data import load_glue_dataset
from src.quantization import quantize_static_w8a8, prepare_calibration_dataloader, apply_dynamic_w8a32


def parse_args():
    parser = argparse.ArgumentParser(description="Run static W8A8 PTQ on dense SST-2 model")
    parser.add_argument("--task", type=str, default="sst2", choices=["sst2", "mrpc"])
    parser.add_argument("--model_dir", required=True, help="Path to fine-tuned model checkpoint")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./quantized_models")
    parser.add_argument("--calib_size", type=int, default=500, help="Number of calibration examples")
    parser.add_argument("--calib_batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--per_channel", action="store_true", default=True, help="Use per-channel weight quantization")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate_model(model, eval_dataset, tokenizer, device="cpu", task="sst2"):
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
    
    print("Evaluating FP32 baseline...")
    fp32_metrics = evaluate_model(model_fp32, eval_ds, tokenizer, device, task=args.task)
    fp32_acc = fp32_metrics.get("eval_accuracy", fp32_metrics.get("eval_f1", 0.0))
    print(f"FP32 Accuracy: {fp32_acc:.4f}")
    
    print("Evaluating dynamic W8A32...")
    model_dynamic = apply_dynamic_w8a32(model_fp32)
    dynamic_metrics = evaluate_model(model_dynamic, eval_ds, tokenizer, device, task=args.task)
    dynamic_acc = dynamic_metrics.get("eval_accuracy", dynamic_metrics.get("eval_f1", 0.0))
    print(f"Dynamic W8A32 Accuracy: {dynamic_acc:.4f}")
    print(f"Δ vs FP32: {dynamic_acc - fp32_acc:.4f}")
    
    print(f"Preparing calibration data ({args.calib_size} examples)...")
    calib_loader = prepare_calibration_dataloader(
        train_ds,
        batch_size=args.calib_batch_size,
        num_samples=args.calib_size,
        shuffle=True,
        seed=args.seed,
    )
    print(f"Calibration loader prepared: {len(calib_loader)} batches")
    
    print("Applying static W8A8 quantization...")
    start_time = time.time()
    model_quantized = quantize_static_w8a8(
        model_fp32,
        calib_loader,
        per_channel=args.per_channel,
        device=device,
    )
    quant_time = time.time() - start_time
    print(f"Quantization took {quant_time:.2f} seconds")
    
    print("Evaluating static W8A8...")
    w8a8_metrics = evaluate_model(model_quantized, eval_ds, tokenizer, device, task=args.task)
    w8a8_acc = w8a8_metrics.get("eval_accuracy", w8a8_metrics.get("eval_f1", 0.0))
    print(f"Static W8A8 Accuracy: {w8a8_acc:.4f}")
    print(f"Δ vs FP32: {w8a8_acc - fp32_acc:.4f}")
    print(f"Δ vs Dynamic W8A32: {w8a8_acc - dynamic_acc:.4f}")
    
    results = {
        "task": args.task,
        "model_dir": args.model_dir,
        "calib_size": args.calib_size,
        "per_channel": args.per_channel,
        "fp32_accuracy": fp32_acc,
        "dynamic_w8a32_accuracy": dynamic_acc,
        "static_w8a8_accuracy": w8a8_acc,
        "delta_vs_fp32": w8a8_acc - fp32_acc,
        "delta_vs_dynamic": w8a8_acc - dynamic_acc,
        "quantization_time_seconds": quant_time,
        "fp32_metrics": fp32_metrics,
        "dynamic_metrics": dynamic_metrics,
        "w8a8_metrics": w8a8_metrics,
    }
    
    output_name = f"{args.task}_static_w8a8_calib{args.calib_size}"
    output_path = os.path.join(args.output_dir, output_name)
    os.makedirs(output_path, exist_ok=True)
    
    torch.save(model_quantized.state_dict(), os.path.join(output_path, "quantized_state_dict.pt"))
    
    results_path = os.path.join(output_path, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("Summary")
    print(f"FP32 Accuracy:        {fp32_acc:.4f}")
    print(f"Dynamic W8A32:         {dynamic_acc:.4f} (Δ: {dynamic_acc - fp32_acc:+.4f})")
    print(f"Static W8A8:           {w8a8_acc:.4f} (Δ: {w8a8_acc - fp32_acc:+.4f})")
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()

