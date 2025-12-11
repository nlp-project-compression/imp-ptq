"""
Evaluate all baseline + pruned checkpoints for SST-2 and MRPC with readable output.

Results saved to: checkpoints/all_eval_results.txt
Each model's metrics are printed in a clean multi-line block.
"""

import os
import json
import torch
import numpy as np
import evaluate

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from src.data import load_glue_dataset


RESULTS_PATH = os.path.join("checkpoints", "all_eval_results.txt")


def eval_single_model(task: str, model_dir: str, model_name: str = "bert-base-uncased", max_length: int = 128):
    """Evaluate a single checkpoint on the given GLUE task and return metrics dict."""
    print(f"\n=== Evaluating [{task}] model: {model_dir} ===")

    # Load model + tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f" Loaded model on {device}")

    # Load eval dataset
    _, eval_ds, _ = load_glue_dataset(task, model_name, max_length)
    metric = evaluate.load("glue", task)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    eval_args = TrainingArguments(
        output_dir="tmp_eval",
        per_device_eval_batch_size=32,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    return trainer.evaluate()


def collect_model_dirs():
    """Collect all (task, model_dir) pairs for evaluation."""
    pairs = []
    tasks = ["sst2", "mrpc"]

    for task in tasks:
        # Baseline
        baseline_dir = os.path.join("checkpoints", f"{task}_bert-base-uncased_seed42")
        if os.path.isdir(baseline_dir):
            pairs.append((task, baseline_dir))

        # Experiment folder: checkpoints/<task>/
        exp_root = os.path.join("checkpoints", task)
        if os.path.isdir(exp_root):
            for name in sorted(os.listdir(exp_root)):
                full_path = os.path.join(exp_root, name)
                if os.path.isdir(full_path):
                    pairs.append((task, full_path))

    return pairs


def main():
    pairs = collect_model_dirs()

    print("\nFound models to evaluate:")
    for t, p in pairs:
        print(f"  • {t}: {p}")

    os.makedirs("checkpoints", exist_ok=True)

    with open(RESULTS_PATH, "w") as f:
        f.write("=========== ALL MODEL EVALUATION RESULTS ===========\n\n")

        for task, model_dir in pairs:
            print(f"\n--- Running eval for: {model_dir} ---")
            metrics = eval_single_model(task, model_dir)

            # Pretty-write
            f.write(f"TASK: {task}\n")
            f.write(f"MODEL: {model_dir}\n")
            f.write("METRICS:\n")
            f.write(json.dumps(metrics, indent=4))  # <--- pretty JSON
            f.write("\n--------------------------------------------------\n\n")

    print(f"\n=== All evaluations complete. Results saved to: {RESULTS_PATH} ===")


if __name__ == "__main__":
    main()
