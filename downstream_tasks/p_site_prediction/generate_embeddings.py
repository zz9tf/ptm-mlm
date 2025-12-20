"""
Script to generate embeddings from pre-trained model for downstream task training.
This script processes training, validation, and test data to generate embeddings that can be reused.
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

# Try to import both inference classes
try:
    from inference import ModelInference
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

try:
    from inference_esm2 import ESM2Inference
    ESM2_AVAILABLE = True
except ImportError:
    ESM2_AVAILABLE = False


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
    
    # Drop rows with NaN values in seq or label columns
    df = df.dropna(subset=['seq', 'label'])
    
    # Convert to string type and filter out empty strings and 'nan' strings
    df['seq'] = df['seq'].astype(str)
    df['label'] = df['label'].astype(str)
    
    # Filter out empty strings and 'nan' strings (from astype(str) conversion of NaN)
    df = df[df['seq'].str.len() > 0]
    df = df[df['seq'] != 'nan']
    df = df[df['label'].str.len() > 0]
    df = df[df['label'] != 'nan']
    
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
        "--model_type",
        type=str,
        default="mamba",
        choices=["mamba", "esm2"],
        help="Model type to use: 'mamba' or 'esm2' (default: mamba)"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="/home/zz/zheng/ptm-mlm/downstream_tasks/p_site_prediction/best.ckpt",
        help="Path to model checkpoint (for Mamba model)"
    )
    parser.add_argument(
        "--esm2_model_name",
        type=str,
        default="facebook/esm2_t33_650M_UR50D",
        help="ESM2 model name from HuggingFace (default: facebook/esm2_t33_650M_UR50D)"
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
        "--valid_data",
        type=str,
        default="/home/zz/zheng/ptm-mlm/downstream_tasks/p_site_prediction/PhosphositePTM.valid.txt",
        help="Path to validation data CSV"
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
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=None,
        help="Maximum sequence length for a single window. "
             "If None, uses default from model config (typically 512). "
             "For sequences longer than this, sliding window will be used."
    )
    parser.add_argument(
        "--use_sliding_window",
        action="store_true",
        default=True,
        help="Use sliding window for sequences longer than max_sequence_length. "
             "This ensures all positions are processed and labels are preserved. (Default: True)"
    )
    parser.add_argument(
        "--window_overlap",
        type=float,
        default=0.5,
        help="Overlap ratio between sliding windows (0.0 to 1.0). "
             "Default 0.5 means 50%% overlap. Higher overlap provides better context but requires more computation."
    )
    
    args = parser.parse_args()
    
    # Initialize inference model based on model_type
    print(f"\n🚀 Initializing {args.model_type.upper()} model inference...")
    if args.model_type == "mamba":
        if not MAMBA_AVAILABLE:
            raise ImportError("Mamba inference not available. Please ensure inference.py exists.")
        # Use Mamba+ESM2-15B combination (matching training setup)
        # This matches train.py where use_esm=True by default
        inferencer = ModelInference(
            args.checkpoint, 
            max_sequence_length=args.max_sequence_length,
            use_esm=True  # Use Mamba+ESM2-15B combination (matching training)
        )
    elif args.model_type == "esm2":
        if not ESM2_AVAILABLE:
            raise ImportError("ESM2 inference not available. Please install transformers: pip install transformers")
        inferencer = ESM2Inference(
            model_name=args.esm2_model_name,
            max_sequence_length=args.max_sequence_length
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}. Choose 'mamba' or 'esm2'.")
    
    # Process training data
    if args.train_data and os.path.exists(args.train_data):
        print("\n" + "="*50)
        print("📊 Processing training data...")
        print("="*50)
        train_sequences, train_labels = load_data(args.train_data)
        train_embeddings, train_lengths = inferencer.generate_per_position_embeddings(
            train_sequences,
            batch_size=args.batch_size,
            max_sequence_length=args.max_sequence_length,
            use_sliding_window=args.use_sliding_window,
            window_overlap=args.window_overlap
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
            batch_size=args.batch_size,
            max_sequence_length=args.max_sequence_length,
            use_sliding_window=args.use_sliding_window,
            window_overlap=args.window_overlap
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
    
    # Process validation data
    if args.valid_data and os.path.exists(args.valid_data):
        print("\n" + "="*50)
        print("📊 Processing validation data...")
        print("="*50)
        valid_sequences, valid_labels = load_data(args.valid_data)
        valid_embeddings, valid_lengths = inferencer.generate_per_position_embeddings(
            valid_sequences,
            batch_size=args.batch_size,
            max_sequence_length=args.max_sequence_length,
            use_sliding_window=args.use_sliding_window,
            window_overlap=args.window_overlap
        )
        
        # Save validation embeddings
        os.makedirs(args.output_dir, exist_ok=True)
        valid_emb_path = os.path.join(args.output_dir, "valid_embeddings.pt")
        valid_labels_path = os.path.join(args.output_dir, "valid_labels.pt")
        valid_seqs_path = os.path.join(args.output_dir, "valid_sequences.pt")
        
        torch.save(valid_embeddings, valid_emb_path)
        torch.save(valid_labels, valid_labels_path)
        torch.save(valid_sequences, valid_seqs_path)
        print(f"✅ Saved validation embeddings to {valid_emb_path}")
        print(f"✅ Saved validation labels to {valid_labels_path}")
        print(f"✅ Saved validation sequences to {valid_seqs_path}")
    
    print("\n" + "="*50)
    print("🎉 Embedding generation completed!")
    print("="*50)


if __name__ == "__main__":
    main()

