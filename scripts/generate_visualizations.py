"""
Generate visualizations for the final report.

Creates:
1. Accuracy comparison bar chart (FP32 vs Dynamic vs Static)
2. Calibration size ablation line plot
3. Task sensitivity heatmap
4. Speedup comparison
5. Model size comparison
6. Accuracy vs Speed trade-off scatter
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import argparse


def load_results(results_dir="./quantized_models"):
    """Load all quantization results."""
    results = {}
    tasks = ["sst2", "mrpc"]
    
    for task in tasks:
        result_file = os.path.join(results_dir, f"{task}_static_w8a8_calib500", "results.json")
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                results[task] = json.load(f)
    
    return results


def load_calib_ablation(ablation_dir="./calib_ablation_results"):
    """Load calibration ablation results."""
    ablation = {}
    tasks = ["sst2", "mrpc"]
    
    for task in tasks:
        ablation_file = os.path.join(ablation_dir, f"{task}_calib_ablation.json")
        if os.path.exists(ablation_file):
            with open(ablation_file, "r") as f:
                ablation[task] = json.load(f)
    
    return ablation


def plot_accuracy_comparison(results, output_dir="./figures"):
    """Plot 1: Accuracy comparison across models and tasks."""
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = list(results.keys())
    models = ["FP32", "Dynamic W8A32", "Static W8A8"]
    
    # Prepare data
    data = {model: [] for model in models}
    task_labels = []
    
    for task in tasks:
        task_labels.append(task.upper())
        r = results[task]
        
        # Get appropriate metric (F1 for MRPC, accuracy for others)
        if task == "mrpc":
            data["FP32"].append(r["fp32_metrics"].get("eval_f1", r["fp32_accuracy"]))
            data["Dynamic W8A32"].append(r["dynamic_metrics"].get("eval_f1", r["dynamic_w8a32_accuracy"]))
            data["Static W8A8"].append(r["w8a8_metrics"].get("eval_f1", r["static_w8a8_accuracy"]))
        else:
            data["FP32"].append(r["fp32_accuracy"])
            data["Dynamic W8A32"].append(r["dynamic_w8a32_accuracy"])
            data["Static W8A8"].append(r["static_w8a8_accuracy"])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(tasks))
    width = 0.25
    
    colors = ["#2ecc71", "#3498db", "#e74c3c"]  # Green, Blue, Red
    bars = []
    for i, model in enumerate(models):
        bars.append(ax.bar(x + i*width, [v*100 for v in data[model]], width, 
                          label=model, color=colors[i], alpha=0.8))
        # Add value labels on bars
        for j, v in enumerate(data[model]):
            ax.text(x[j] + i*width, v*100 + 0.5, f'{v*100:.2f}%', 
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy / F1 Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison Across Tasks', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(task_labels)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/accuracy_comparison.png")


def plot_calibration_ablation(ablation, output_dir="./figures"):
    """Plot 2: Calibration size vs accuracy."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {"sst2": "#3498db", "mrpc": "#e74c3c"}
    markers = {"sst2": "o", "mrpc": "s"}
    
    for task in ablation.keys():
        data = ablation[task]
        fp32_acc = data["fp32_accuracy"]
        
        # Get appropriate metric
        if task == "mrpc":
            fp32_metric = data["fp32_metrics"].get("eval_f1", fp32_acc)
        else:
            fp32_metric = fp32_acc
        
        calib_sizes = []
        accuracies = []
        
        for res in data["calibration_results"]:
            if "error" not in res:
                calib_sizes.append(res["calib_size"])
                if task == "mrpc":
                    acc = res["metrics"].get("eval_f1", res["accuracy"])
                else:
                    acc = res["accuracy"]
                accuracies.append(acc * 100)
        
        # Add FP32 baseline
        calib_sizes = [0] + calib_sizes
        accuracies = [fp32_metric * 100] + accuracies
        
        ax.plot(calib_sizes, accuracies, marker=markers[task], linewidth=2, 
               markersize=8, label=f"{task.upper()}", color=colors[task])
        ax.scatter(calib_sizes, accuracies, s=100, color=colors[task], zorder=5)
    
    ax.set_xlabel('Calibration Dataset Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy / F1 Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Calibration Size Ablation Study', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks([0, 100, 500, 2000])
    ax.set_xticklabels(['FP32\n(baseline)', '100', '500', '2000'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration_ablation.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/calibration_ablation.png")


def plot_accuracy_drop_heatmap(results, output_dir="./figures"):
    """Plot 3: Heatmap showing accuracy drop vs FP32."""
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = list(results.keys())
    models = ["Dynamic W8A32", "Static W8A8"]
    
    # Prepare data (accuracy drop in percentage points)
    data = []
    for model in models:
        row = []
        for task in tasks:
            r = results[task]
            if model == "Dynamic W8A32":
                drop = (r["dynamic_w8a32_accuracy"] - r["fp32_accuracy"]) * 100
            else:
                drop = (r["static_w8a8_accuracy"] - r["fp32_accuracy"]) * 100
            
            # For MRPC, use F1 if available
            if task == "mrpc" and model == "Dynamic W8A32":
                drop = (r["dynamic_metrics"].get("eval_f1", r["dynamic_w8a32_accuracy"]) - 
                       r["fp32_metrics"].get("eval_f1", r["fp32_accuracy"])) * 100
            elif task == "mrpc" and model == "Static W8A8":
                drop = (r["w8a8_metrics"].get("eval_f1", r["static_w8a8_accuracy"]) - 
                       r["fp32_metrics"].get("eval_f1", r["fp32_accuracy"])) * 100
            
            row.append(drop)
        data.append(row)
    
    data = np.array(data)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=-15, vmax=0)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(tasks)):
            text = ax.text(j, i, f'{data[i, j]:.2f}%',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([t.upper() for t in tasks])
    ax.set_yticklabels(models)
    ax.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax.set_ylabel('Quantization Method', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Drop vs FP32 Baseline (Percentage Points)', 
                fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy Drop (%)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_drop_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/accuracy_drop_heatmap.png")


def plot_speedup_comparison(results, output_dir="./figures"):
    """Plot 4: Inference speedup comparison."""
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = list(results.keys())
    models = ["Dynamic W8A32", "Static W8A8"]
    
    # Prepare data
    speedups = {model: [] for model in models}
    
    for task in tasks:
        r = results[task]
        fp32_stats = r.get("fp32_metrics", {}).get("inference_stats", {})
        fp32_speed = fp32_stats.get("samples_per_second", 0)
        
        if fp32_speed > 0:
            for model in models:
                if model == "Dynamic W8A32":
                    model_stats = r.get("dynamic_metrics", {}).get("inference_stats", {})
                else:
                    model_stats = r.get("w8a8_metrics", {}).get("inference_stats", {})
                
                model_speed = model_stats.get("samples_per_second", 0)
                if model_speed > 0:
                    speedup = model_speed / fp32_speed
                    speedups[model].append(speedup)
                else:
                    speedups[model].append(0)
        else:
            for model in models:
                speedups[model].append(0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(tasks))
    width = 0.35
    
    colors = ["#3498db", "#e74c3c"]
    bars = []
    for i, model in enumerate(models):
        bars.append(ax.bar(x + i*width, speedups[model], width, 
                          label=model, color=colors[i], alpha=0.8))
        # Add value labels
        for j, v in enumerate(speedups[model]):
            if v > 0:
                ax.text(x[j] + i*width, v + 0.05, f'{v:.2f}x', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='No Speedup (1.0x)')
    ax.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (×)', fontsize=12, fontweight='bold')
    ax.set_title('Inference Speedup vs FP32 Baseline', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels([t.upper() for t in tasks])
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, max([max(speedups[m]) for m in models] + [1.5]) * 1.2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "speedup_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/speedup_comparison.png")


def plot_model_size_comparison(results, output_dir="./figures"):
    """Plot 5: Model size comparison."""
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = list(results.keys())
    models = ["FP32", "Dynamic W8A32", "Static W8A8"]
    
    # Prepare data
    sizes = {model: [] for model in models}
    
    for task in tasks:
        r = results[task]
        model_sizes = r.get("model_sizes_mb", {})
        
        for model in models:
            if model == "FP32":
                size = model_sizes.get("fp32")
            elif model == "Dynamic W8A32":
                size = model_sizes.get("dynamic_w8a32")
            else:
                size = model_sizes.get("static_w8a8")
            
            sizes[model].append(size if size else 0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(tasks))
    width = 0.25
    
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    bars = []
    for i, model in enumerate(models):
        bars.append(ax.bar(x + i*width, sizes[model], width, 
                          label=model, color=colors[i], alpha=0.8))
        # Add value labels
        for j, v in enumerate(sizes[model]):
            if v > 0:
                ax.text(x[j] + i*width, v + 5, f'{v:.1f}MB', 
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model Size (MB)', fontsize=12, fontweight='bold')
    ax.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([t.upper() for t in tasks])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_size_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/model_size_comparison.png")


def plot_accuracy_speed_tradeoff(results, output_dir="./figures"):
    """Plot 6: Accuracy vs Speed trade-off scatter plot."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {"sst2": "#3498db", "mrpc": "#e74c3c"}
    markers = {"sst2": "o", "mrpc": "s"}
    models = ["FP32", "Dynamic W8A32", "Static W8A8"]
    
    for task in results.keys():
        r = results[task]
        
        # Get appropriate metric
        if task == "mrpc":
            fp32_acc = r["fp32_metrics"].get("eval_f1", r["fp32_accuracy"])
            dynamic_acc = r["dynamic_metrics"].get("eval_f1", r["dynamic_w8a32_accuracy"])
            static_acc = r["w8a8_metrics"].get("eval_f1", r["static_w8a8_accuracy"])
        else:
            fp32_acc = r["fp32_accuracy"]
            dynamic_acc = r["dynamic_w8a32_accuracy"]
            static_acc = r["static_w8a8_accuracy"]
        
        # Get speeds
        fp32_stats = r.get("fp32_metrics", {}).get("inference_stats", {})
        dynamic_stats = r.get("dynamic_metrics", {}).get("inference_stats", {})
        static_stats = r.get("w8a8_metrics", {}).get("inference_stats", {})
        
        fp32_speed = fp32_stats.get("samples_per_second", 0)
        dynamic_speed = dynamic_stats.get("samples_per_second", 0)
        static_speed = static_stats.get("samples_per_second", 0)
        
        # Plot points
        if fp32_speed > 0:
            ax.scatter(fp32_speed, fp32_acc * 100, s=200, marker=markers[task], 
                      color=colors[task], alpha=0.7, edgecolors='black', linewidth=2,
                      label=f'{task.upper()} FP32' if task == list(results.keys())[0] else '')
        if dynamic_speed > 0:
            ax.scatter(dynamic_speed, dynamic_acc * 100, s=200, marker=markers[task], 
                      color=colors[task], alpha=0.5, edgecolors='black', linewidth=1.5,
                      label=f'{task.upper()} Dynamic' if task == list(results.keys())[0] else '')
        if static_speed > 0:
            ax.scatter(static_speed, static_acc * 100, s=200, marker=markers[task], 
                      color=colors[task], alpha=0.3, edgecolors='black', linewidth=1,
                      label=f'{task.upper()} Static' if task == list(results.keys())[0] else '')
    
    ax.set_xlabel('Inference Speed (samples/sec)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy / F1 Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs Speed Trade-off', fontsize=14, fontweight='bold')
    ax.legend(loc='best', ncol=2)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_speed_tradeoff.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/accuracy_speed_tradeoff.png")


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations for final report")
    parser.add_argument("--results_dir", type=str, default="./quantized_models",
                        help="Directory containing quantization results")
    parser.add_argument("--ablation_dir", type=str, default="./calib_ablation_results",
                        help="Directory containing calibration ablation results")
    parser.add_argument("--output_dir", type=str, default="./figures",
                        help="Output directory for figures")
    
    args = parser.parse_args()
    
    print("Generating visualizations...")
    print("="*60)
    
    # Load data
    results = load_results(args.results_dir)
    ablation = load_calib_ablation(args.ablation_dir)
    
    if not results:
        print("Error: No results found. Run quantization experiments first.")
        return
    
    # Generate all plots
    plot_accuracy_comparison(results, args.output_dir)
    
    if ablation:
        plot_calibration_ablation(ablation, args.output_dir)
    
    plot_accuracy_drop_heatmap(results, args.output_dir)
    plot_speedup_comparison(results, args.output_dir)
    plot_model_size_comparison(results, args.output_dir)
    plot_accuracy_speed_tradeoff(results, args.output_dir)
    
    print("="*60)
    print(f"✓ All visualizations saved to: {args.output_dir}/")
    print("\nGenerated figures:")
    print("  1. accuracy_comparison.png - Bar chart comparing all models")
    print("  2. calibration_ablation.png - Line plot of calibration size effects")
    print("  3. accuracy_drop_heatmap.png - Heatmap of accuracy drops")
    print("  4. speedup_comparison.png - Inference speedup comparison")
    print("  5. model_size_comparison.png - Model size comparison")
    print("  6. accuracy_speed_tradeoff.png - Accuracy vs speed scatter plot")


if __name__ == "__main__":
    main()

