import argparse
import json
import os
import sys
import numpy as np
import torch
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser(description="Error analysis for quantized models")
    parser.add_argument("--task", type=str, required=True, choices=["sst2", "mrpc"],
                        help="GLUE task name")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to FP32 model (local or HuggingFace Hub)")
    parser.add_argument("--quantized_model_dir", type=str, required=True,
                        help="Path to quantized model directory")
    parser.add_argument("--output_dir", type=str, default="./error_analysis",
                        help="Output directory for analysis results")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def load_models_and_tokenizer(model_dir, quantized_model_dir, device="cpu"):
    print("Loading models...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model_fp32 = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model_fp32.to(device)
    model_fp32.eval()
    
    model_quantized = ORTModelForSequenceClassification.from_pretrained(quantized_model_dir)
    
    print("Models loaded successfully.")
    return model_fp32, model_quantized, tokenizer


def get_predictions_with_confidence(model, eval_dataset, tokenizer, device="cpu", task="sst2"):
    """Get predictions and confidence scores for all examples."""
    from torch.utils.data import DataLoader
    
    is_ort_model = 'ORTModel' in str(type(model))
    
    def collate_fn(batch):
        return {
            k: torch.stack([item[k] for item in batch]) if isinstance(batch[0][k], torch.Tensor) 
            else [item[k] for item in batch]
            for k in batch[0].keys()
        }
    
    eval_loader = DataLoader(eval_dataset, batch_size=32, collate_fn=collate_fn)
    
    all_preds = []
    all_labels = []
    all_logits = []
    all_confidences = []
    all_indices = []
    
    idx = 0
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)
            token_type_ids = batch.get("token_type_ids", None)
            labels = batch["labels"]
            
            if is_ort_model:
                inputs = {
                    "input_ids": input_ids.cpu().numpy() if isinstance(input_ids, torch.Tensor) else input_ids,
                }
                if attention_mask is not None:
                    inputs["attention_mask"] = attention_mask.cpu().numpy() if isinstance(attention_mask, torch.Tensor) else attention_mask
                if token_type_ids is not None:
                    inputs["token_type_ids"] = token_type_ids.cpu().numpy() if isinstance(token_type_ids, torch.Tensor) else token_type_ids
                
                outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                logits = torch.tensor(logits) if not isinstance(logits, torch.Tensor) else logits
            else:
                input_ids = input_ids.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                
                if token_type_ids is not None and attention_mask is not None:
                    outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                elif attention_mask is not None:
                    outputs = model(input_ids, attention_mask=attention_mask)
                elif token_type_ids is not None:
                    outputs = model(input_ids, token_type_ids=token_type_ids)
                else:
                    outputs = model(input_ids)
                
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                logits = logits.cpu()
            
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            confidences = torch.max(probs, dim=-1)[0]
            
            batch_size = input_ids.shape[0] if hasattr(input_ids, 'shape') else len(input_ids)
            for i in range(batch_size):
                all_preds.append(preds[i].item())
                all_labels.append(labels[i].item() if isinstance(labels, torch.Tensor) else labels[i])
                all_logits.append(logits[i].cpu().numpy())
                all_confidences.append(confidences[i].item())
                all_indices.append(idx)
                idx += 1
    
    return {
        'predictions': all_preds,
        'labels': all_labels,
        'logits': all_logits,
        'confidences': all_confidences,
        'indices': all_indices
    }


def load_raw_dataset(task):
    """Load raw dataset to get original text."""
    raw = load_dataset("glue", task)
    return raw["validation"]


def analyze_errors(fp32_results, quantized_results, raw_dataset, task, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    if task == "sst2":
        text_field = "sentence"
        text_field2 = None
    elif task == "mrpc":
        text_field = "sentence1"
        text_field2 = "sentence2"
    else: 
        print("Unknown task: ", task)
    
    all_lengths = []
    for idx in range(len(raw_dataset)):
        if text_field2:
            length1 = len(raw_dataset[idx][text_field].split())
            length2 = len(raw_dataset[idx][text_field2].split())
            all_lengths.append(length1 + length2)
        else:
            all_lengths.append(len(raw_dataset[idx][text_field].split()))
    
    fp32_correct = [p == l for p, l in zip(fp32_results['predictions'], fp32_results['labels'])]
    quantized_correct = [p == l for p, l in zip(quantized_results['predictions'], quantized_results['labels'])]
    
    both_correct = [fc and qc for fc, qc in zip(fp32_correct, quantized_correct)]
    both_wrong = [not fc and not qc for fc, qc in zip(fp32_correct, quantized_correct)]
    fp32_only_correct = [fc and not qc for fc, qc in zip(fp32_correct, quantized_correct)]
    quantized_only_correct = [not fc and qc for fc, qc in zip(fp32_correct, quantized_correct)]
    
    error_analysis = {
        'summary': {
            'total_examples': len(fp32_results['predictions']),
            'fp32_accuracy': sum(fp32_correct) / len(fp32_correct),
            'quantized_accuracy': sum(quantized_correct) / len(quantized_correct),
            'both_correct': sum(both_correct),
            'both_wrong': sum(both_wrong),
            'fp32_only_correct': sum(fp32_only_correct),
            'quantized_only_correct': sum(quantized_only_correct),
            'accuracy_drop': sum(quantized_correct) / len(quantized_correct) - sum(fp32_correct) / len(fp32_correct)
        },
        'fp32_only_correct_examples': [],
        'quantized_only_correct_examples': [],
        'both_wrong_examples': []
    }
    
    for idx in range(len(fp32_results['predictions'])):
        if fp32_only_correct[idx]:
            example = {
                'index': idx,
                'label': fp32_results['labels'][idx],
                'fp32_pred': fp32_results['predictions'][idx],
                'quantized_pred': quantized_results['predictions'][idx],
                'fp32_confidence': fp32_results['confidences'][idx],
                'quantized_confidence': quantized_results['confidences'][idx],
            }
            
            if text_field2:
                example['text1'] = raw_dataset[idx][text_field]
                example['text2'] = raw_dataset[idx][text_field2]
            else:
                example['text'] = raw_dataset[idx][text_field]
            
            if text_field2:
                example['length1'] = len(raw_dataset[idx][text_field].split())
                example['length2'] = len(raw_dataset[idx][text_field2].split())
                example['total_length'] = example['length1'] + example['length2']
            else:
                example['length'] = len(raw_dataset[idx][text_field].split())
            
            error_analysis['fp32_only_correct_examples'].append(example)
        
        elif quantized_only_correct[idx]:
            example = {
                'index': idx,
                'label': fp32_results['labels'][idx],
                'fp32_pred': fp32_results['predictions'][idx],
                'quantized_pred': quantized_results['predictions'][idx],
                'fp32_confidence': fp32_results['confidences'][idx],
                'quantized_confidence': quantized_results['confidences'][idx],
            }
            if text_field2:
                example['text1'] = raw_dataset[idx][text_field]
                example['text2'] = raw_dataset[idx][text_field2]
            else:
                example['text'] = raw_dataset[idx][text_field]
            error_analysis['quantized_only_correct_examples'].append(example)
        
        elif both_wrong[idx]:
            example = {
                'index': idx,
                'label': fp32_results['labels'][idx],
                'fp32_pred': fp32_results['predictions'][idx],
                'quantized_pred': quantized_results['predictions'][idx],
            }
            if text_field2:
                example['text1'] = raw_dataset[idx][text_field]
                example['text2'] = raw_dataset[idx][text_field2]
            else:
                example['text'] = raw_dataset[idx][text_field]
            error_analysis['both_wrong_examples'].append(example)
    
    error_analysis['fp32_only_correct_examples'].sort(
        key=lambda x: x['fp32_confidence'] - x['quantized_confidence'], 
        reverse=True
    )
    
    if error_analysis['fp32_only_correct_examples']:
        lengths = [ex.get('length', ex.get('total_length', 0)) 
                  for ex in error_analysis['fp32_only_correct_examples']]
        fp32_confs = [ex['fp32_confidence'] for ex in error_analysis['fp32_only_correct_examples']]
        quantized_confs = [ex['quantized_confidence'] for ex in error_analysis['fp32_only_correct_examples']]
        
        error_analysis['statistics'] = {
            'avg_length': np.mean(lengths),
            'avg_fp32_confidence': np.mean(fp32_confs),
            'avg_quantized_confidence': np.mean(quantized_confs),
            'confidence_drop': np.mean(fp32_confs) - np.mean(quantized_confs)
        }
    
    return error_analysis


def create_visualizations(error_analysis, fp32_results, quantized_results, task, raw_dataset, output_dir):
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(fp32_results['confidences'], bins=30, alpha=0.6, label='FP32', color='blue')
    axes[0].hist(quantized_results['confidences'], bins=30, alpha=0.6, label='Quantized', color='red')
    axes[0].set_xlabel('Confidence Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Confidence Distribution Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    correct_indices = [i for i in range(len(fp32_results['predictions'])) 
                       if fp32_results['predictions'][i] == fp32_results['labels'][i]]
    wrong_indices = [i for i in range(len(fp32_results['predictions'])) 
                    if fp32_results['predictions'][i] != fp32_results['labels'][i]]
    
    if correct_indices:
        axes[1].scatter([fp32_results['confidences'][i] for i in correct_indices],
                       [quantized_results['confidences'][i] for i in correct_indices],
                       alpha=0.5, label='Correct', s=20)
    if wrong_indices:
        axes[1].scatter([fp32_results['confidences'][i] for i in wrong_indices],
                       [quantized_results['confidences'][i] for i in wrong_indices],
                       alpha=0.5, label='Wrong', s=20, color='red')
    
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1].set_xlabel('FP32 Confidence')
    axes[1].set_ylabel('Quantized Confidence')
    axes[1].set_title('Confidence Correlation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    if error_analysis['fp32_only_correct_examples']:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        lengths = [ex.get('length', ex.get('total_length', 0)) 
                  for ex in error_analysis['fp32_only_correct_examples']]
        
        all_lengths_list = []
        if task == "sst2":
            text_field = "sentence"
            for i in range(len(raw_dataset)):
                all_lengths_list.append(len(raw_dataset[i][text_field].split()))
        elif task == "mrpc":
            text_field1 = "sentence1"
            text_field2 = "sentence2"
            for i in range(len(raw_dataset)):
                all_lengths_list.append(
                    len(raw_dataset[i][text_field1].split()) + 
                    len(raw_dataset[i][text_field2].split())
                )
        else:
            for i in range(len(raw_dataset)):
                all_lengths_list.append(50)
        
        axes[0].hist(all_lengths_list, bins=20, alpha=0.5, label='All Examples', color='gray')
        axes[0].hist(lengths, bins=20, alpha=0.7, label='Quantization Errors', color='red')
        axes[0].set_xlabel('Sentence Length (words)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Error Distribution by Length')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
        conf_drops = [ex['fp32_confidence'] - ex['quantized_confidence'] 
                     for ex in error_analysis['fp32_only_correct_examples']]
        axes[1].hist(conf_drops, bins=20, color='red', alpha=0.7)
        axes[1].set_xlabel('Confidence Drop (FP32 - Quantized)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Confidence Drop for Quantization Errors')
        axes[1].axvline(x=0, color='k', linestyle='--', alpha=0.5)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_patterns.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    num_classes = len(set(fp32_results['labels']))
    agreement_matrix = np.zeros((num_classes, num_classes))
    
    for i in range(len(fp32_results['predictions'])):
        fp32_pred = fp32_results['predictions'][i]
        quantized_pred = quantized_results['predictions'][i]
        agreement_matrix[fp32_pred][quantized_pred] += 1
    
    sns.heatmap(agreement_matrix, annot=True, fmt='.0f', cmap='Blues', ax=ax,
                xticklabels=[f'Q{i}' for i in range(num_classes)],
                yticklabels=[f'FP32_{i}' for i in range(num_classes)])
    ax.set_xlabel('Quantized Prediction')
    ax.set_ylabel('FP32 Prediction')
    ax.set_title('Prediction Agreement Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_agreement.png'), dpi=150, bbox_inches='tight')
    plt.close()


def print_summary(error_analysis, task):
    print(f"ERROR ANALYSIS SUMMARY - {task.upper()}")
    
    summary = error_analysis['summary']
    print(f"\nOverall Performance:")
    print(f"  FP32 Accuracy:        {summary['fp32_accuracy']:.4f} ({summary['fp32_accuracy']*100:.2f}%)")
    print(f"  Quantized Accuracy:   {summary['quantized_accuracy']:.4f} ({summary['quantized_accuracy']*100:.2f}%)")
    print(f"  Accuracy Drop:        {summary['accuracy_drop']:.4f} ({summary['accuracy_drop']*100:.2f}%)")
    
    print(f"\nError Breakdown:")
    print(f"  Both Correct:          {summary['both_correct']} ({summary['both_correct']/summary['total_examples']*100:.1f}%)")
    print(f"  Both Wrong:            {summary['both_wrong']} ({summary['both_wrong']/summary['total_examples']*100:.1f}%)")
    print(f"  FP32 Only Correct:     {summary['fp32_only_correct']} ({summary['fp32_only_correct']/summary['total_examples']*100:.1f}%)")
    print(f"  Quantized Only Correct: {summary['quantized_only_correct']} ({summary['quantized_only_correct']/summary['total_examples']*100:.1f}%)")
    
    if error_analysis['fp32_only_correct_examples']:
        stats = error_analysis['statistics']
        print(f"\nQuantization Error Characteristics:")
        print(f"  Average Sentence Length: {stats['avg_length']:.1f} words")
        print(f"  Avg FP32 Confidence:    {stats['avg_fp32_confidence']:.4f}")
        print(f"  Avg Quantized Confidence: {stats['avg_quantized_confidence']:.4f}")
        print(f"  Average Confidence Drop: {stats['confidence_drop']:.4f}")
        
        print(f"\nTop 5 Most Confident FP32 Predictions (that quantized got wrong):")
        for i, ex in enumerate(error_analysis['fp32_only_correct_examples'][:5]):
            print(f"\n  Example {i+1} (Index {ex['index']}):")
            if 'text' in ex:
                print(f"    Text: {ex['text'][:100]}...")
            else:
                print(f"    Text1: {ex['text1'][:80]}...")
                print(f"    Text2: {ex['text2'][:80]}...")
            print(f"    Label: {ex['label']}, FP32 Pred: {ex['fp32_pred']}, Quantized Pred: {ex['quantized_pred']}")
            print(f"    FP32 Conf: {ex['fp32_confidence']:.4f}, Quantized Conf: {ex['quantized_confidence']:.4f}")
    
    print("\n" + "="*70)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cpu" 
    
    print(f"Loading {args.task} dataset...")
    from src.data import load_glue_dataset
    train_ds, eval_ds, _ = load_glue_dataset(args.task, args.model_dir, args.max_length)
    raw_dataset = load_raw_dataset(args.task)
    
    model_fp32, model_quantized, tokenizer = load_models_and_tokenizer(
        args.model_dir, args.quantized_model_dir, device
    )
    
    print("\nGetting FP32 predictions...")
    fp32_results = get_predictions_with_confidence(model_fp32, eval_ds, tokenizer, device, args.task)
    
    print("Getting quantized predictions...")
    quantized_results = get_predictions_with_confidence(model_quantized, eval_ds, tokenizer, device, args.task)
    
    print("\nPerforming error analysis...")
    error_analysis = analyze_errors(fp32_results, quantized_results, raw_dataset, args.task, args.output_dir)
    
    print("Creating visualizations...")
    create_visualizations(error_analysis, fp32_results, quantized_results, args.task, raw_dataset, args.output_dir)
    
    output_file = os.path.join(args.output_dir, f"{args.task}_error_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(error_analysis, f, indent=2)

    print_summary(error_analysis, args.task)
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - Error analysis: {output_file}")
    print(f"  - Visualizations: {os.path.join(args.output_dir, '*.png')}")


if __name__ == "__main__":
    main()

