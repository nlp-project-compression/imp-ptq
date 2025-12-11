"""
Evaluate all baseline + pruned checkpoints for SST-2 and MRPC.

- Writes a human-readable report to checkpoints/all_eval_results.txt
- Computes average GLUE scores (SST-2 accuracy + MRPC F1) / 2
  for each matching (SST-2, MRPC) model family.
"""

import os
import json
import torch
import numpy as np
import evaluate

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from src.data import load_glue_dataset


RESULTS_PATH = os.path.join("checkpoints", "all_eval_results.txt")
GLUE_JSON_PATH = os.path.join("checkpoints", "glue_scores.json")


def eval_single_model(task: str, model_dir: str, model_name: str = "bert-base-uncased", max_length: int = 128):
    """Evaluate a single checkpoint on the given GLUE task and return metrics dict."""
    print(f"\n=== Evaluating [{task}] model: {model_dir} ===")

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f" Loaded model on {device}")

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
        baseline_dir = os.path.join("checkpoints", f"{task}_bert-base-uncased_seed42")
        if os.path.isdir(baseline_dir):
            pairs.append((task, baseline_dir))

        exp_root = os.path.join("checkpoints", task)
        if os.path.isdir(exp_root):
            for name in sorted(os.listdir(exp_root)):
                full_path = os.path.join(exp_root, name)
                if os.path.isdir(full_path):
                    pairs.append((task, full_path))

    return pairs


def canonical_tag(task: str, model_dir: str) -> str:
    """
    Produce a tag that matches SST-2 and MRPC versions of the same model.

    Assumes dirs are named like:
      checkpoints/sst2_bert-base-uncased_seed42
      checkpoints/mrpc_bert-base-uncased_seed42
      checkpoints/sst2/sst2_bert-base-uncased_seed42_IMP_...
      checkpoints/mrpc/mrpc_bert-base-uncased_seed42_IMP_...

    We strip the leading '{task}_' prefix so the rest matches across tasks.
    """
    basename = os.path.basename(model_dir.rstrip("/"))
    prefix = f"{task}_"
    if basename.startswith(prefix):
        return basename[len(prefix):]
    return basename


def compute_glue_scores(results_by_tag):
    """
    Given a dict:
        tag -> { 'sst2': metrics_dict, 'mrpc': metrics_dict }
    compute GLUE = (SST-2 accuracy + MRPC F1) / 2 for tags that have both tasks.
    """
    glue_scores = {}

    for tag, task_metrics in results_by_tag.items():
        if "sst2" in task_metrics and "mrpc" in task_metrics:
            sst2_acc = task_metrics["sst2"].get("eval_accuracy", 0.0)
            mrpc_metrics = task_metrics["mrpc"]
            mrpc_f1 = mrpc_metrics.get("eval_f1", mrpc_metrics.get("eval_accuracy", 0.0))

            avg_glue = (sst2_acc + mrpc_f1) / 2.0

            glue_scores[tag] = {
                "sst2_accuracy": sst2_acc,
                "mrpc_f1": mrpc_f1,
                "average_glue": avg_glue,
            }

    return glue_scores


def print_glue_summary(glue_scores):
    """Print a formatted table of GLUE scores."""
    if not glue_scores:
        print("\nNo matching SST-2 + MRPC model pairs found for GLUE computation.")
        return ""

    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("Average GLUE Scores (paired SST-2 + MRPC models)")
    lines.append("=" * 60)
    lines.append(f"\n{'Model tag':<40} {'SST-2 Acc':>10} {'MRPC F1':>10} {'GLUE':>10}")
    lines.append("-" * 60)

    for tag, vals in sorted(glue_scores.items()):
        sst2_acc = vals["sst2_accuracy"] * 100
        mrpc_f1 = vals["mrpc_f1"] * 100
        avg_glue = vals["average_glue"] * 100
        lines.append(f"{tag:<40} {sst2_acc:>9.2f}% {mrpc_f1:>9.2f}% {avg_glue:>9.2f}%")

    lines.append("=" * 60)

    summary_text = "\n".join(lines)
    print(summary_text)
    return summary_text + "\n"


def main():
    pairs = collect_model_dirs()

    print("\nFound models to evaluate:")
    for t, p in pairs:
        print(f"  • {t}: {p}")

    os.makedirs("checkpoints", exist_ok=True)

    # For GLUE score aggregation:
    # tag -> { 'sst2': metrics_dict, 'mrpc': metrics_dict }
    results_by_tag = {}

    with open(RESULTS_PATH, "w") as f:
        f.write("=========== ALL MODEL EVALUATION RESULTS ===========\n\n")

        for task, model_dir in pairs:
            print(f"\n--- Running eval for: {model_dir} ---")
            metrics = eval_single_model(task, model_dir)

            f.write(f"TASK: {task}\n")
            f.write(f"MODEL: {model_dir}\n")
            f.write("METRICS:\n")
            f.write(json.dumps(metrics, indent=4))
            f.write("\n--------------------------------------------------\n\n")

            tag = canonical_tag(task, model_dir)
            task_dict = results_by_tag.setdefault(tag, {})
            task_dict[task] = metrics

        glue_scores = compute_glue_scores(results_by_tag)
        glue_summary = print_glue_summary(glue_scores)

        f.write(glue_summary)

    try:
        with open(GLUE_JSON_PATH, "w") as jf:
            json.dump(glue_scores, jf, indent=2)
        print(f"\nGLUE scores saved to: {GLUE_JSON_PATH}")
    except Exception as e:
        print(f"Could not save GLUE scores JSON: {e}")

    print(f"\n=== All evaluations complete. Results saved to: {RESULTS_PATH} ===")


if __name__ == "__main__":
    main()
