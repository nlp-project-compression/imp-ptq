"""
Runs a simple one-shot global magnitude pruning step.

Workflow:
1. Load fine-tuned baseline model
2. Apply global magnitude pruning (one-shot)
3. Save pruned model to new checkpoint
"""

import argparse
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


from src.pruning import prune_model_global_magnitude


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["sst2", "mrpc", "mnli"],
        help="GLUE task name to load baseline checkpoint."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Base model name (used for folder naming)."
    )

    parser.add_argument(
        "--sparsity",
        type=float,
        required=True,
        help="Target global sparsity (e.g., 0.7 for 70%)."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed to match baseline folder naming."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="checkpoints",
        help="Where baseline checkpoints are stored."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Where to save pruned checkpoints."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    basemodel_folder = f"{args.task}_{args.model_name.replace('/', '_')}_seed{args.seed}"
    baseline_ckpt = os.path.join(args.input_dir, basemodel_folder)

    print(" Running one-shot magnitude pruning")
    print(f"Task:            {args.task}")
    print(f"Model:           {args.model_name}")
    print(f"Baseline folder: {baseline_ckpt}")
    print(f"Target sparsity: {args.sparsity:.2f}")
    print("\n")

    if not os.path.isdir(baseline_ckpt):
        raise ValueError(f"Baseline checkpoint not found: {baseline_ckpt}")

    print(" Loading fine-tuned model...")
    model = AutoModelForSequenceClassification.from_pretrained(baseline_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(baseline_ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f" Loaded model to {device}")


    # 2) Prune model
    print("\n Applying global magnitude pruning...")
    achieved = prune_model_global_magnitude(model, args.sparsity)
    print(f" Target sparsity:    {args.sparsity:.2f}")
    print(f" Achieved sparsity:  {achieved:.4f}")

    # 3) Save pruned model
    pruned_name = (
        f"{args.task}_{args.model_name.replace('/', '_')}_seed{args.seed}_pruned{int(args.sparsity*100)}"
    )
    save_path = os.path.join(args.output_dir, pruned_name)

    print(f"\n Saving pruned model to:\n   {save_path}")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print("\n Done.")


if __name__ == "__main__":
    main()
