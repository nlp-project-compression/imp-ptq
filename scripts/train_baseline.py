"""
Fine-tunes a model on a GLUE task and saves:
- Best checkpoint
- Evaluation metrics
"""
import argparse
import json
import os
import numpy as np
import torch
import evaluate

from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from src.data import load_glue_dataset

PRIMARY_METRIC = {
    "sst2": "accuracy",
    "mnli": "accuracy",
    "mrpc": "f1",
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="sst2",
                        choices=["sst2", "mrpc", "mnli"])
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_model_size_mb(model, bytes_per_param: int = 2):
    """
    Estimate model size assuming `bytes_per_param` bytes per parameter.
    For bf16 baseline, bytes_per_param = 2.
    (If you want FP32 reference, use 4.)
    """
    params = sum(p.numel() for p in model.parameters())
    return params * bytes_per_param / (1024 * 1024)

def main():
    args = parse_args()
    set_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    if use_bf16:
        print(">> Using bf16 mixed precision for training")
    elif use_cuda:
        print(">> bf16 not supported on this GPU, falling back to fp16")
    else:
        print(">> No GPU detected, training in pure fp32")

    print(f"Loading data for task {args.task}...")
    train_ds, eval_ds, tokenizer = load_glue_dataset(
        args.task, args.model_name, args.max_length
    )

    num_labels = len(set(train_ds["labels"]))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels
    )

    metric = evaluate.load("glue", args.task)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    exp_dir = os.path.join(
        args.output_dir,
        f"{args.task}_{args.model_name.replace('/', '_')}_seed{args.seed}"
    )
    os.makedirs(exp_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=exp_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        logging_dir=os.path.join(exp_dir, "logs"),
        report_to=["none"],
        # 🔴 CHANGE HERE: use bf16 when supported, otherwise fp16 on GPU
        fp16=use_cuda and not use_bf16,
        bf16=use_bf16,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    print("Evaluating best checkpoint...")
    eval_metrics = trainer.evaluate()
    eval_metrics["model_size_mb_bf16_estimate"] = compute_model_size_mb(model, bytes_per_param=2)

    with open(os.path.join(exp_dir, "eval_metrics.json"), "w") as f:
        json.dump(eval_metrics, f, indent=2)

    trainer.save_model(exp_dir)
    tokenizer.save_pretrained(exp_dir)

    print("Done. Saved to:", exp_dir)
    print("Metrics:", eval_metrics)

if __name__ == "__main__":
    main()
