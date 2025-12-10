
# src/data_utils.py

from datasets import load_dataset

def load_sst2_dataset(tokenizer, max_length=128):
    """
    Load the GLUE SST-2 dataset and tokenize using the provided tokenizer.
    Returns (train_dataset, validation_dataset) tokenized.
    """
    # Load the dataset from the GLUE benchmark
    raw_datasets = load_dataset("glue", "sst2")
    train_ds = raw_datasets["train"]
    val_ds = raw_datasets["validation"]

    # Tokenize function
    def tokenize_batch(example):
        return tokenizer(
            example["sentence"],
            padding="max_length", 
            truncation=True, 
            max_length=max_length
        )
    
    # Apply tokenization to each split
    train_ds = train_ds.map(tokenize_batch, batched=True, load_from_cache_file=False)
    val_ds = val_ds.map(tokenize_batch, batched=True, load_from_cache_file=False)
    # Rename label column to "labels" to match model expectations
    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")
    # Set format to PyTorch tensors
    columns = ["input_ids", "attention_mask", "labels"]  # token_type_ids not always needed for single-sentence
    if "token_type_ids" in train_ds.column_names:
        columns.append("token_type_ids")
    train_ds.set_format(type="torch", columns=columns)
    val_ds.set_format(type="torch", columns=columns)
    return train_ds, val_ds

def get_dataloader(dataset, batch_size=32, shuffle=False):
    """Create a PyTorch DataLoader from a Dataset."""
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_calibration_dataset(train_dataset, num_samples=500, seed=42):
    """Select a random subset of the training dataset for calibration."""
    # Shuffle the training dataset with the given seed and take the first num_samples
    calib_ds = train_dataset.shuffle(seed=seed).select(range(min(num_samples, len(train_dataset))))
    return calib_ds
