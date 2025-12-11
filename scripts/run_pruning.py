"""
Pruning script.

Modes:
  - oneshot: one-shot global magnitude pruning (original behavior)
  - imp:     Iterative Magnitude Pruning (IMP) with fine-tuning between rounds

Workflow (oneshot):
1. Load fine-tuned baseline model
2. Apply global magnitude pruning (one-shot)
3. Save pruned model to new checkpoint

Workflow (imp):
1. Load fine-tuned baseline model
2. Build GLUE train/val dataloaders
3. Run IMP to reach final global sparsity
4. Save final IMP-pruned model to new checkpoint
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import load_glue_dataset
from src.pruning import (
    prune_model_global_magnitude,
    iterative_magnitude_pruning,
    calculate_sparsity,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="oneshot",
        choices=["oneshot", "imp"],
        help="Pruning mode: 'oneshot' (original) or 'imp' (iterative magnitude pruning).",
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["sst2", "mrpc"],
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
        help="Target global sparsity (e.g., 0.7 for 70%%). "
             "For IMP, this is the final sparsity."
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

    # IMP-specific arguments (ignored in oneshot mode)
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=5,
        help="[IMP] Number of pruning rounds (ignored in oneshot mode).",
    )

    parser.add_argument(
        "--ft_epochs_per_round",
        type=int,
        default=1,
        help="[IMP] Fine-tuning epochs per round.",
    )

    parser.add_argument(
        "--schedule_type",
        type=str,
        default="geometric",
        choices=["geometric", "fixed"],
        help="[IMP] Sparsity schedule type.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="[IMP] Batch size for train/val loaders.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="[IMP] Learning rate for fine-tuning.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="[IMP] Max sequence length for tokenization.",
    )

    parser.add_argument(
        "--rewind_to_initial",
        action="store_true",
        help="[IMP] Rewind to initial dense weights each round (classic IMP/LTH).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    basemodel_folder = f"{args.task}_{args.model_name.replace('/', '_')}_seed{args.seed}"
    baseline_ckpt = os.path.join(args.input_dir, basemodel_folder)

    print(f"\n Mode:            {args.mode}")
    print(f" Task:            {args.task}")
    print(f" Model:           {args.model_name}")
    print(f" Baseline folder: {baseline_ckpt}")
    print(f" Target sparsity: {args.sparsity:.2f}\n")

    if not os.path.isdir(baseline_ckpt):
        raise ValueError(f"Baseline checkpoint not found: {baseline_ckpt}")

    print(" Loading fine-tuned model...")
    model = AutoModelForSequenceClassification.from_pretrained(baseline_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(baseline_ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f" Loaded model to {device}")

    if args.mode == "oneshot":
        # -------------------------------------------------------------------
        # One-shot global magnitude pruning (original behavior)
        # -------------------------------------------------------------------
        print("\n Running ONE-SHOT global magnitude pruning...")
        achieved = prune_model_global_magnitude(model, args.sparsity)
        print(f" Target sparsity:    {args.sparsity:.2f}")
        print(f" Achieved sparsity:  {achieved:.4f}")

        pruned_name = (
            f"{args.task}_{args.model_name.replace('/', '_')}_seed{args.seed}"
            f"_pruned{int(args.sparsity * 100)}"
        )
        save_path = os.path.join(args.output_dir, pruned_name)

        print(f"\n Saving pruned model to:\n   {save_path}")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        print("\n Done (oneshot).")
        return

    # -----------------------------------------------------------------------
    # IMP mode
    # -----------------------------------------------------------------------
    print("\n Running IMP (Iterative Magnitude Pruning)...")
    print(f"  Final sparsity:      {args.sparsity:.2f}")
    print(f"  Rounds:              {args.num_rounds}")
    print(f"  Schedule:            {args.schedule_type}")
    print(f"  FT epochs / round:   {args.ft_epochs_per_round}")
    print(f"  Batch size:          {args.batch_size}")
    print(f"  Learning rate:       {args.lr}")
    print(f"  Rewind to initial?:  {args.rewind_to_initial}")
    print("")

    # Load GLUE datasets as torch tensors
    train_ds, val_ds, _ = load_glue_dataset(
        args.task, args.model_name, args.max_length
    )

    # DataLoaders (padding is already applied in load_glue_dataset)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Initial sparsity: {calculate_sparsity(model):.4f}")

    history = iterative_magnitude_pruning(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        device=device,
        final_sparsity=args.sparsity,
        num_rounds=args.num_rounds,
        ft_epochs_per_round=args.ft_epochs_per_round,
        schedule_type=args.schedule_type,
        rewind_to_initial=args.rewind_to_initial,
    )

    print("\nIMP history:")
    for h in history:
        print(
            f"  Round {h['round']}: "
            f"target_s={h['target_sparsity']:.3f}, "
            f"achieved_s={h['achieved_sparsity']:.3f}, "
            f"val_acc={h['val_accuracy'] * 100:.2f}%"
        )

    # Save final IMP-pruned model
    final_s_percent = int(args.sparsity * 100)
    suffix = f"_IMP_{args.schedule_type}_S{final_s_percent}"

    # Distinguish rewinding runs
    if args.rewind_to_initial:
        suffix += "_rewind"

    imp_name = f"{args.task}_{args.model_name.replace('/', '_')}_seed{args.seed}{suffix}"

    save_path = os.path.join(args.output_dir, imp_name)

    print(f"\n Saving final IMP-pruned model to:\n   {save_path}")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Optionally, save history as JSON for plotting later
    try:
        import json
        with open(os.path.join(save_path, "imp_history.json"), "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Could not save history JSON: {e}")

    print("\n Done (IMP).")


if __name__ == "__main__":
    main()
