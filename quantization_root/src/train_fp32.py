
# src/train_fp32.py

import os
import json
import math
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from data_utils import load_sst2_dataset, get_dataloader
from eval_utils import evaluate
from utils import set_seed, compute_size

import argparse

def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT-base on SST-2 (FP32 baseline)")
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoints/fp32_sst2",
                        help="Directory to save the fine-tuned model and logs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=32, help="Validation batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # Load tokenizer and model (BERT-base for sequence classification)
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Load and preprocess SST-2 dataset
    train_dataset, val_dataset = load_sst2_dataset(tokenizer)
    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=args.val_batch_size, shuffle=False)

    # Setup optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps),
                                                  num_training_steps=total_steps)

    # Loss function (classification)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_model_state = None

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            # Move batch to device
            batch = {k: v.to(model.device) for k, v in batch.items()}
            labels = batch.pop("labels")  # separate labels for loss

            outputs = model(**batch, labels=labels)
            # outputs is a transformers.modeling_outputs.SequenceClassifierOutput
            loss = outputs.loss
            logits = outputs.logits

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            # (optional) Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

            # Accumulate training metrics
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total if total > 0 else 0.0
        train_acc = 100.0 * correct / total if total > 0 else 0.0

        # Evaluate on validation set
        model.eval()
        val_acc, val_loss = evaluate(model, val_loader, device=model.device, criterion=criterion)

        print(f"Epoch {epoch}: Train loss={train_loss:.4f}, Train acc={train_acc:.2f}%, "
              f"Val loss={val_loss:.4f}, Val acc={val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}  # save CPU state for safety

    # Save the best model to HuggingFace format
    if best_model_state is not None:
        model.cpu()
        model.load_state_dict(best_model_state)
        model.save_pretrained(args.output_dir)  # saves model weights, config, etc.
        tokenizer.save_pretrained(args.output_dir)
        print(f"Best model saved to {args.output_dir} (Val acc = {best_val_acc:.2f}%).")

    # Log metrics and hyperparameters to a JSON file
    metrics = {
        "train_loss": float(f"{train_loss:.4f}"),
        "train_accuracy": float(f"{train_acc:.2f}"),
        "val_loss": float(f"{val_loss:.4f}"),
        "val_accuracy": float(f"{val_acc:.2f}"),
        "best_val_accuracy": float(f"{best_val_acc:.2f}"),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed
    }
    metrics_path = os.path.join(args.output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Training metrics saved to {metrics_path}")

    # Optionally, report the final model size for reference
    model_bin_path = os.path.join(args.output_dir, "pytorch_model.bin")
    if os.path.exists(model_bin_path):
        size_mb = compute_size(model_bin_path) / (1024*1024)
        print(f"FP32 model size: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
