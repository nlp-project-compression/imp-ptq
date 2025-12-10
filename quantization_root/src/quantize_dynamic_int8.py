
# src/quantize_dynamic_int8.py

import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from data_utils import load_sst2_dataset, get_dataloader
from eval_utils import evaluate
from utils import compute_size

import argparse

def main():
    parser = argparse.ArgumentParser(description="Dynamic INT8 PTQ for BERT (W8A32)")
    parser.add_argument("--fp32_model_dir", type=str, required=True,
                        help="Path to the fine-tuned FP32 BERT model directory (HuggingFace format)")
    parser.add_argument("--output_path", type=str, default="outputs/quantized/bert_sst2_dynamic_int8.pt",
                        help="File path to save the quantized model (INT8 weights)")
    args = parser.parse_args()

    # Load FP32 fine-tuned model (and tokenizer for data)
    model = AutoModelForSequenceClassification.from_pretrained(args.fp32_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.fp32_model_dir)  # uses the same tokenizer as during training

    # Ensure model is on CPU for quantization
    model.cpu()
    model.eval()  # set eval mode to disable dropout
    
    # Apply dynamic quantization on all Linear layers (weights 8-bit, activations dynamically quantized per batch)
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    # Note: quantize_dynamic will internally convert Linear layers to dynamically quantized versions.
    # LayerNorm, Embeddings, Softmax, etc., remain in FP32 (no change).
    
    # Prepare SST-2 validation data
    _, val_dataset = load_sst2_dataset(tokenizer)  # load tokenized datasets
    val_loader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    
    # Evaluate FP32 model and quantized model on validation set
    # (Note: For fairness, evaluate both on CPU)
    model.eval()
    quantized_model.eval()
    device = torch.device("cpu")
    val_acc_fp32, _ = evaluate(model.to(device), val_loader, device=device)
    val_acc_int8, _ = evaluate(quantized_model, val_loader, device=device)
    
    # Print accuracy results
    print(f"FP32 model accuracy: {val_acc_fp32:.2f}%")
    print(f"Dynamic INT8 model accuracy: {val_acc_int8:.2f}%")
    print(f"Accuracy drop: {(val_acc_fp32 - val_acc_int8):.2f} percentage points")
    
    # Save the quantized model to disk
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(quantized_model.state_dict(), args.output_path)
    print(f"Quantized model saved to {args.output_path}")
    
    # Model size and compression factor
    fp32_bin = os.path.join(args.fp32_model_dir, "pytorch_model.bin")
    if os.path.exists(fp32_bin):
        fp32_size = compute_size(fp32_bin)
        int8_size = compute_size(args.output_path)
        print(f"FP32 model size (fp32 weights): {fp32_size/1024**2:.2f} MB")
        print(f"INT8 model size (quantized): {int8_size/1024**2:.2f} MB")
        if int8_size > 0:
            compression_factor = fp32_size / int8_size
            print(f"Compression factor (FP32/INT8): {compression_factor:.2f}x")
    else:
        print("FP32 model file not found; skipping size comparison.")

if __name__ == "__main__":
    main()
