"""
Data loading utilities for NHA site prediction.
Loads data from NHAC.csv and prepares sequences and labels for training.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
import os


def load_nha_data(csv_path: str, sequence_column: str = 'seq_61') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load NHA site prediction data from CSV file and split into train/valid/test sets.
    
    @param csv_path: Path to NHAC.csv file
    @param sequence_column: Column name for sequences (default: 'seq_61' for longest window)
    @returns: Tuple of (train_df, valid_df, test_df) DataFrames
    """
    print(f"📖 Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required_cols = [sequence_column, 'label', 'set']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV file must contain columns: {missing_cols}")
    
    # Drop rows with NaN values in required columns
    df = df.dropna(subset=[sequence_column, 'label', 'set'])
    
    # Convert sequence to string type and filter out empty strings
    df[sequence_column] = df[sequence_column].astype(str)
    df = df[df[sequence_column].str.len() > 0]
    df = df[df[sequence_column] != 'nan']
    
    # Convert label to int (should be 0 or 1)
    df['label'] = df['label'].astype(int)
    
    # Filter out invalid labels (should be 0 or 1)
    df = df[df['label'].isin([0, 1])]
    
    # Split by 'set' column
    train_df = df[df['set'] == 'train'].copy()
    valid_df = df[df['set'] == 'val'].copy()
    test_df = df[df['set'] == 'test'].copy()
    
    print(f"✅ Loaded {len(df)} total samples")
    print(f"   Train: {len(train_df)} samples")
    print(f"   Valid: {len(valid_df)} samples")
    print(f"   Test: {len(test_df)} samples")
    print(f"   Positive samples: {df['label'].sum()}, Negative samples: {(df['label'] == 0).sum()}")
    
    return train_df, valid_df, test_df


def prepare_sequences_and_labels(df: pd.DataFrame, sequence_column: str = 'seq_61') -> Tuple[List[str], List[int]]:
    """
    Extract sequences and labels from DataFrame.
    
    @param df: DataFrame with sequence and label columns
    @param sequence_column: Column name for sequences
    @returns: Tuple of (sequences list, labels list)
    """
    sequences = df[sequence_column].tolist()
    labels = df['label'].tolist()
    return sequences, labels


def save_split_data(df: pd.DataFrame, sequence_column: str, output_dir: str, split_name: str):
    """
    Save split data to separate files for compatibility with existing pipeline.
    
    @param df: DataFrame for the split
    @param sequence_column: Column name for sequences
    @param output_dir: Output directory
    @param split_name: Name of the split (train/valid/test)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a simple format: seq and label columns
    output_df = pd.DataFrame({
        'seq': df[sequence_column],
        'label': df['label']
    })
    
    output_path = os.path.join(output_dir, f"{split_name}.txt")
    output_df.to_csv(output_path, sep='\t', index=False, header=False)
    print(f"💾 Saved {split_name} data to {output_path} ({len(output_df)} samples)")


if __name__ == "__main__":
    # Test loading
    csv_path = "/home/zz/zheng/ptm-mlm/downstream_tasks/NHA_site_prediction/NHAC.csv"
    train_df, valid_df, test_df = load_nha_data(csv_path)
    
    print("\n📊 Sample train data:")
    print(train_df[['unique_id', 'seq_61', 'label']].head(3))

