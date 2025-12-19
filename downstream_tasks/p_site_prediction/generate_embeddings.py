"""
Script to generate embeddings from pre-trained model for downstream task training.
This script processes training and test data to generate embeddings that can be reused.
"""
import torch
import pandas as pd
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import sys

# Add current directory to path for local imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from inference import ModelInference


def load_data(data_path: str):
    """
    Load data from CSV file.
    
    @param data_path: Path to CSV file with 'seq' and 'label' columns
    @returns: Tuple of (sequences list, labels list)
    """
    print(f"📖 Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Check required columns
    if 'seq' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV file must contain 'seq' and 'label' columns")
    
    sequences = df['seq'].tolist()
    labels = df['label'].tolist()
    
    print(f"✅ Loaded {len(sequences)} sequences")
    return sequences, labels


def save_embeddings_and_labels(embeddings_list, labels, output_dir: str, split_name: str):
    """
    Save embeddings and labels to disk.
    
    @param embeddings_list: List of per-position embeddings (each is a tensor)
    @param labels: List of label strings
    @param output_dir: Output directory
    @param split_name: Name of the split (train/test/valid)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save embeddings
    embeddings_path = os.path.join(output_dir, f"{split_name}_embeddings.pt")
    torch.save(embeddings_list, embeddings_path)
    print(f"💾 Saved embeddings to {embeddings_path}")
    
    # Save labels
    labels_path = os.path.join(output_dir, f"{split_name}_labels.pt")
    torch.save(labels, labels_path)
    print(f"💾 Saved labels to {labels_path}")
    
    # Also save sequences for reference
    sequences_path = os.path.join(output_dir, f"{split_name}_sequences.pt")
    # We need to get sequences from somewhere - let's modify the function signature
    # For now, we'll just save labels
    print(f"✅ Saved {split_name} data to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for downstream task")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="/home/zz/zheng/ptm-mlm/downstream_tasks/p_site_prediction/best.ckpt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="/home/zz/zheng/ptm-mlm/downstream_tasks/p_site_prediction/PhosphositePTM.train.txt",
        help="Path to training data CSV"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="/home/zz/zheng/ptm-mlm/downstream_tasks/p_site_prediction/PhosphositePTM.test.txt",
        help="Path to test data CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/zz/zheng/ptm-mlm/downstream_tasks/p_site_prediction/embeddings",
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    
    args = parser.parse_args()
    
    # Initialize inference model
    print("🚀 Initializing model inference...")
    inferencer = ModelInference(args.checkpoint)
    
    # Process training data
    if args.train_data and os.path.exists(args.train_data):
        print("\n" + "="*50)
        print("📊 Processing training data...")
        print("="*50)
        train_sequences, train_labels = load_data(args.train_data)
        train_embeddings, train_lengths = inferencer.generate_per_position_embeddings(
            train_sequences,
            batch_size=args.batch_size
        )
        
        # Save training embeddings
        os.makedirs(args.output_dir, exist_ok=True)
        train_emb_path = os.path.join(args.output_dir, "train_embeddings.pt")
        train_labels_path = os.path.join(args.output_dir, "train_labels.pt")
        train_seqs_path = os.path.join(args.output_dir, "train_sequences.pt")
        
        torch.save(train_embeddings, train_emb_path)
        torch.save(train_labels, train_labels_path)
        torch.save(train_sequences, train_seqs_path)
        print(f"✅ Saved training embeddings to {train_emb_path}")
        print(f"✅ Saved training labels to {train_labels_path}")
        print(f"✅ Saved training sequences to {train_seqs_path}")
    
    # Process test data
    if args.test_data and os.path.exists(args.test_data):
        print("\n" + "="*50)
        print("📊 Processing test data...")
        print("="*50)
        test_sequences, test_labels = load_data(args.test_data)
        test_embeddings, test_lengths = inferencer.generate_per_position_embeddings(
            test_sequences,
            batch_size=args.batch_size
        )
        
        # Save test embeddings
        os.makedirs(args.output_dir, exist_ok=True)
        test_emb_path = os.path.join(args.output_dir, "test_embeddings.pt")
        test_labels_path = os.path.join(args.output_dir, "test_labels.pt")
        test_seqs_path = os.path.join(args.output_dir, "test_sequences.pt")
        
        torch.save(test_embeddings, test_emb_path)
        torch.save(test_labels, test_labels_path)
        torch.save(test_sequences, test_seqs_path)
        print(f"✅ Saved test embeddings to {test_emb_path}")
        print(f"✅ Saved test labels to {test_labels_path}")
        print(f"✅ Saved test sequences to {test_seqs_path}")
    
    print("\n" + "="*50)
    print("🎉 Embedding generation completed!")
    print("="*50)


if __name__ == "__main__":
    main()

