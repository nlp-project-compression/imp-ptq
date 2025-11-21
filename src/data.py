"""
Data loading utilities for GLUE tasks.

Functions:
- load_glue_dataset(task_name)
- tokenize_dataset(dataset, tokenizer, max_length)
"""

from datasets import load_dataset
from transformers import AutoTokenizer

GLUE_TEXT_FIELDS = {
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
}

def load_glue_dataset(task, model_name, max_length=128):
    raw = load_dataset("glue", task)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    text1, text2 = GLUE_TEXT_FIELDS[task]

    def preprocess(batch):
        if text2 is None:
            return tokenizer(
                batch[text1],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
        return tokenizer(
            batch[text1],
            batch[text2],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    enc = raw.map(preprocess, batched=True)
    enc = enc.rename_column("label", "labels")

    enc.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
        + (["token_type_ids"] if "token_type_ids" in enc["train"].column_names else [])
    )

    if task == "mnli":
        return enc["train"], enc["validation_matched"], tokenizer

    return enc["train"], enc["validation"], tokenizer
