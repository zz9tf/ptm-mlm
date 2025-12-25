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

# Add main_pipeline to path (inference.py needs getters.tokenizer, utils.checkpoint, etc.)
# These modules are still in main_pipeline
main_pipeline_path = Path(__file__).parent.parent.parent / "main_pipeline"
sys.path.insert(0, str(main_pipeline_path))

# Add inference directory to path for shared inference classes
inference_dir = Path(__file__).parent.parent / "inference"
sys.path.insert(0, str(inference_dir))

# Try to import both inference classes from shared inference directory
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


def save_embeddings_data(embeddings_list, labels, sequences, output_dir: str, split_name: str):
    """
    Save embeddings, labels, and sequences to disk.
    
    @param embeddings_list: List of per-position embeddings (each is a tensor)
    @param labels: List of label strings
    @param sequences: List of sequence strings
    @param output_dir: Output directory
    @param split_name: Name of the split (train/test/valid)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    embeddings_path = os.path.join(output_dir, f"{split_name}_embeddings.pt")
    labels_path = os.path.join(output_dir, f"{split_name}_labels.pt")
    sequences_path = os.path.join(output_dir, f"{split_name}_sequences.pt")
    
    torch.save(embeddings_list, embeddings_path)
    torch.save(labels, labels_path)
    torch.save(sequences, sequences_path)
    
    print(f"✅ Saved {split_name} embeddings to {embeddings_path}")
    print(f"✅ Saved {split_name} labels to {labels_path}")
    print(f"✅ Saved {split_name} sequences to {sequences_path}")


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
        default="/home/zz/zheng/ptm-mlm/downstream_tasks/outputs/p_site_prediction",
        help="Output directory. Embeddings will be stored in output_dir/embeddings/"
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
    parser.add_argument(
        "--use_esm",
        action="store_true",
        default=False,
        help="If set, load ESM2-15B and use Mamba+ESM2 combination (matching training setup). "
             "If not set, use Mamba-only mode. (Default: False)"
    )
    
    args = parser.parse_args()
    
    # Automatically create embeddings subdirectory in output_dir
    embeddings_dir = os.path.join(args.output_dir, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    print(f"📁 Embeddings will be stored in: {embeddings_dir}")
    
    # Initialize inference model based on model_type
    print(f"\n🚀 Initializing {args.model_type.upper()} model inference...")
    if args.model_type == "mamba":
        if not MAMBA_AVAILABLE:
            raise ImportError("Mamba inference not available. Please ensure inference.py exists.")
        # Use Mamba+ESM2-15B combination if use_esm is True, otherwise use Mamba-only
        inferencer = ModelInference(
            args.checkpoint, 
            max_sequence_length=args.max_sequence_length,
            use_esm=args.use_esm  # Control whether to load ESM2-15B
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
    process_split(
        args.train_data,
        "train",
        inferencer,
        embeddings_dir,
        args.batch_size,
        args.max_sequence_length,
        args.use_sliding_window,
        args.window_overlap
    )
    
    # Process test data
    process_split(
        args.test_data,
        "test",
        inferencer,
        embeddings_dir,
        args.batch_size,
        args.max_sequence_length,
        args.use_sliding_window,
        args.window_overlap
    )
    
    # Process validation data
    process_split(
        args.valid_data,
        "valid",
        inferencer,
        embeddings_dir,
        args.batch_size,
        args.max_sequence_length,
        args.use_sliding_window,
        args.window_overlap
    )
    
    print("\n" + "="*50)
    print("🎉 Embedding generation completed!")
    print("="*50)


def process_split(
    data_path: str,
    split_name: str,
    inferencer,
    output_dir: str,
    batch_size: int,
    max_sequence_length: int = None,
    use_sliding_window: bool = True,
    window_overlap: float = 0.5
):
    """
    Process a single data split: load data, generate embeddings, and save results.
    
    @param data_path: Path to CSV file with 'seq' and 'label' columns
    @param split_name: Name of the split (train/test/valid)
    @param inferencer: Inference model instance
    @param output_dir: Output directory for saved files
    @param batch_size: Batch size for inference
    @param max_sequence_length: Maximum sequence length
    @param use_sliding_window: Whether to use sliding window
    @param window_overlap: Window overlap ratio
    """
    if not data_path or not os.path.exists(data_path):
        print(f"\n⚠️  Skipping {split_name}: data file not found at {data_path}")
        return
    
    print("\n" + "="*50)
    print(f"📊 Processing {split_name} data...")
    print("="*50)
    
    try:
        # Load data
        sequences, labels = load_data(data_path)
        
        if len(sequences) == 0:
            print(f"⚠️  No sequences found in {split_name} data, skipping...")
            return
        
        # Generate embeddings
        print(f"\n🔄 Generating embeddings for {len(sequences)} sequences...")
        embeddings, lengths = inferencer.generate_per_position_embeddings(
            sequences,
            batch_size=batch_size,
            max_sequence_length=max_sequence_length,
            use_sliding_window=use_sliding_window,
            window_overlap=window_overlap
        )
        
        # Verify embeddings match sequences
        if len(embeddings) != len(sequences):
            raise RuntimeError(f"Length mismatch: {len(embeddings)} embeddings vs {len(sequences)} sequences")
        
        # 🔧 Verify embeddings match sequences and labels
        # After fixing tokenizer max_length issue, all lengths should match
        aligned_embeddings = []
        aligned_labels = []
        aligned_sequences = []
        
        # Track statistics for debugging
        seq_lengths = [len(seq) for seq in sequences]
        max_seq_len = max(seq_lengths) if seq_lengths else 0
        min_seq_len = min(seq_lengths) if seq_lengths else 0
        print(f"📊 Sequence length stats: min={min_seq_len}, max={max_seq_len}, total={len(sequences)}")
        
        for idx, (emb, lab, seq) in enumerate(zip(embeddings, labels, sequences)):
            lab_len = len(lab)  # Label length is the ground truth
            emb_len = emb.shape[0]
            seq_len = len(seq)
            
            # Verify sequence length matches label length (they should always match)
            if seq_len != lab_len:
                raise RuntimeError(
                    f"Sequence length ({seq_len}) != label length ({lab_len}) at index {idx}/{len(sequences)}. "
                    f"This indicates a data problem. Sequence: {seq[:50]}..., Label: {lab[:50]}..."
                )
            
            # Verify embedding length matches label length
            # This should not happen after fixing tokenizer max_length, but check for debugging
            if emb_len != lab_len:
                # 🔍 Detailed error info to help locate the problem
                is_long_seq = seq_len > max_sequence_length if max_sequence_length else False
                raise RuntimeError(
                    f"❌ Embedding length mismatch at index {idx}/{len(sequences)}:\n"
                    f"   - Embedding length: {emb_len}\n"
                    f"   - Label length: {lab_len}\n"
                    f"   - Sequence length: {seq_len}\n"
                    f"   - Max sequence length setting: {max_sequence_length}\n"
                    f"   - Sequence > max_length: {is_long_seq}\n"
                    f"   - Using sliding window: {use_sliding_window}\n"
                    f"   - Sequence preview: {seq[:100]}...\n"
                    f"   - Label preview: {lab[:100]}...\n"
                    f"This may indicate a bug in the inference code. "
                    f"Please check tokenizer max_length settings (should be max_sequence_length + 2)."
                )
            
            # All lengths match ✅
            aligned_embeddings.append(emb)
            aligned_labels.append(lab)
            aligned_sequences.append(seq)
        
        # Save aligned data
        save_embeddings_data(aligned_embeddings, aligned_labels, aligned_sequences, output_dir, split_name)
        
        print(f"\n✅ Successfully processed {split_name} data: {len(sequences)} samples")
        
    except Exception as e:
        print(f"\n❌ Error processing {split_name} data: {e}")
        raise


if __name__ == "__main__":
    main()

