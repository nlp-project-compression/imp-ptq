# IMP–PTQ

IMP–PTQ is a project codebase for comparing Iterative Magnitude Pruning (IMP) and Post-Training Quantization (PTQ) on GLUE tasks using BERT-style models. It provides reproducible pipelines for pruning, quantization, calibration ablations, and evaluation.

## Environment Setup

We use a Python virtual environment with pip for dependency management.

### Create and activate the virtual environment
``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### Upgrade pip
``` bash
pip install --upgrade pip
```

### Install dependencies
``` bash
pip install -r requirements.txt
```

## Repository Structure

### src/

- **data.py**  
Loads a GLUE task and tokenizes it with a selected pretrained model.

- **pruning.py**
Implements global magnitude pruning and the full Iterative Magnitude Pruning (IMP) pipeline with geometric schedules.  
Includes utilities for mask creation, sparsity measurement, pruning, rewinding, and fine-tuning across IMP rounds.

- **quantization.py**
Provides utilities for post-training quantization, including calibration loader setup and W8A8 static or W8A32 dynamic quantization.  
Supports both Optimum-based transformer quantization and PyTorch FX fallback paths.

### scripts/

- **run_static_w8a8**
Runs static W8A8 post-training quantization on a fine-tuned GLUE model and compares it to FP32 and dynamic W8A32 baselines.  
Loads the checkpoint, builds a calibration set, applies quantization, evaluates all variants, and saves metrics plus the quantized weights.

- **run_pruning.py**
Prunes a fine-tuned GLUE model using either one-shot magnitude pruning or full iterative magnitude pruning (IMP), then saves the resulting sparse checkpoint.

- **run_calib_ablation.py**
Runs a calibration-size ablation for static W8A8 PTQ by testing multiple calibration sizes and evaluating their impact on accuracy and quantization time.  
Loads a fine-tuned model, quantizes it with each calibration size, and saves a full results table.

- **eval_model.py** 
Loads any saved checkpoint and evaluates it on a chosen GLUE task using the HuggingFace Trainer.  
Prints the model’s accuracy/F1 and handles both dense and compressed checkpoints identically.

- **glue_quantization.py**  
Computes an average GLUE score from saved quantization results (SST-2 accuracy, MRPC F1), prints a summary table, and optionally writes the aggregate scores to JSON.

- **glue_pruning.py** 
Evaluates all baseline and pruned checkpoints for SST-2 and MRPC, then computes and reports per-model GLUE scores (SST-2 accuracy + MRPC F1).
