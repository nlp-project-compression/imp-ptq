
# src/eval_utils.py

import torch
import torch.nn.functional as F

def evaluate(model, dataloader, device=torch.device("cpu"), criterion=None):
    """
    Evaluate the model on the given DataLoader.
    Returns (accuracy_percentage, avg_loss).
    """
    model.to(device)
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    count_batches = 0
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels") if "labels" in batch else None

            outputs = model(**batch)
            # HuggingFace models return outputs (logits) either as .logits or as first element
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
            # If outputs are quantized tensors, convert to float for accuracy calculation
            if isinstance(logits, torch.Tensor) and logits.dtype != torch.float32:
                logits = logits.dequantize()  # convert quantized tensor to float, if necessary

            # Compute loss if criterion is provided and labels are available
            loss = None
            if criterion is not None and labels is not None:
                # Use FP32 logits for loss calculation for numerical stability
                loss = criterion(logits.float(), labels)
                total_loss += loss.item()

            # Compute accuracy
            if labels is not None:
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            count_batches += 1

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = total_loss / count_batches if (criterion is not None and count_batches > 0) else 0.0
    return accuracy, avg_loss
