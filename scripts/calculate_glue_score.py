"""
Calculate Average GLUE Score from quantization results.

For GLUE benchmark:
- SST-2: Use accuracy
- MRPC: Use F1 score (primary metric)
- MNLI: Use accuracy (when available)
"""

import json
import os
import argparse
from pathlib import Path


def get_task_score(results, task):
    """
    Get the appropriate score for a GLUE task.
    - SST-2, MNLI: accuracy
    - MRPC: F1 score
    """
    if task == "mrpc":
        # MRPC uses F1 as primary metric
        metrics = results.get("w8a8_metrics", {})
        score = metrics.get("eval_f1", metrics.get("eval_accuracy", 0.0))
    else:
        # SST-2, MNLI use accuracy
        score = results.get("static_w8a8_accuracy", 0.0)
    
    return score


def calculate_glue_score(results_dir="./quantized_models", tasks=None):
    """
    Calculate average GLUE score from quantization results.
    
    Args:
        results_dir: Directory containing results JSON files
        tasks: List of tasks to include (default: all found)
    
    Returns:
        Dictionary with scores and average
    """
    if tasks is None:
        tasks = ["sst2", "mrpc", "mnli"]
    
    results = {}
    scores = {}
    
    # Load results for each task
    for task in tasks:
        result_file = os.path.join(results_dir, f"{task}_static_w8a8_calib500", "results.json")
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                task_results = json.load(f)
                results[task] = task_results
                
                # Get appropriate score
                score = get_task_score(task_results, task)
                scores[task] = score
        else:
            print(f"Warning: Results not found for {task} at {result_file}")
    
    # Calculate average GLUE score
    if scores:
        avg_glue_score = sum(scores.values()) / len(scores)
    else:
        avg_glue_score = 0.0
    
    return {
        "task_scores": scores,
        "average_glue_score": avg_glue_score,
        "num_tasks": len(scores),
        "results": results
    }


def print_glue_summary(glue_data):
    """Print formatted GLUE score summary."""
    print("\n" + "="*60)
    print("Average GLUE Score Summary")
    print("="*60)
    print(f"\n{'Task':<10} {'Metric':<10} {'Score':<10}")
    print("-" * 30)
    
    for task, score in glue_data["task_scores"].items():
        metric = "F1" if task == "mrpc" else "Accuracy"
        print(f"{task.upper():<10} {metric:<10} {score*100:>8.2f}%")
    
    print("-" * 30)
    print(f"{'Average':<10} {'GLUE':<10} {glue_data['average_glue_score']*100:>8.2f}%")
    print(f"\nBased on {glue_data['num_tasks']} task(s)")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Calculate Average GLUE Score from quantization results")
    parser.add_argument("--results_dir", type=str, default="./quantized_models",
                        help="Directory containing results JSON files")
    parser.add_argument("--tasks", type=str, nargs="+", default=["sst2", "mrpc"],
                        choices=["sst2", "mrpc", "mnli"],
                        help="Tasks to include in GLUE score calculation")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (optional)")
    
    args = parser.parse_args()
    
    # Calculate GLUE score
    glue_data = calculate_glue_score(args.results_dir, args.tasks)
    
    # Print summary
    print_glue_summary(glue_data)
    
    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(glue_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return glue_data


if __name__ == "__main__":
    main()

