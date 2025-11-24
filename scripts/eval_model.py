"""
Evaluates any saved model checkpoint on a GLUE task.
"""

import argparse
import torch
import numpy as np
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.data import load_glue_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["sst2", "mrpc", "mnli"])
    parser.add_argument("--model_dir", required=True,
                        help="Path to checkpoint to evaluate.")
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n Loading model from: {args.model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f" Loaded model on {device}")

    # Load eval dataset (train not needed)
    _, eval_ds, _ = load_glue_dataset(args.task, args.model_name, args.max_length)

    metric = evaluate.load("glue", args.task)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    from transformers import Trainer, TrainingArguments

    # Dummy training args just for eval
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

    print("\n Running evaluation...")
    metrics = trainer.evaluate()
    print("\n Results:", metrics)


if __name__ == "__main__":
    main()
