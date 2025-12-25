"""
Script to generate embeddings from pre-trained model for PPI prediction downstream task.
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
    try:
        from inference import ESM2Inference
        ESM2_AVAILABLE = True
    except ImportError:
        ESM2_AVAILABLE = False


def load_ppi_data(data_path: str):
    """
    Load PPI data from CSV file.
    
    @param data_path: Path to CSV file with PPI data
    @returns: Tuple of (sequences list, labels list, metadata list)
    """
    print(f"📖 Loading PPI data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Check required columns
    if 'Sequence window(-5,+5)' not in df.columns or 'Effect' not in df.columns:
        raise ValueError("CSV file must contain 'Sequence window(-5,+5)' and 'Effect' columns")
    
    # Drop rows with NaN values
    df = df.dropna(subset=['Sequence window(-5,+5)', 'Effect'])
    
    # Convert to string type and filter out empty strings
    df['Sequence window(-5,+5)'] = df['Sequence window(-5,+5)'].astype(str)
    df['Effect'] = df['Effect'].astype(str)
    
    # Filter out empty strings and 'nan' strings
    df = df[df['Sequence window(-5,+5)'].str.len() > 0]
    df = df[df['Sequence window(-5,+5)'] != 'nan']
    df = df[df['Effect'].str.len() > 0]
    df = df[df['Effect'] != 'nan']
    
    sequences = df['Sequence window(-5,+5)'].tolist()
    
    # Convert Effect to binary labels: "Enhance" -> 1, "Inhibit" -> 0
    # Or we can use: "Enhance" -> 1, "Inhibit" -> 0, or keep both as separate classes
    labels = []
    for effect in df['Effect']:
        if effect.lower() == 'enhance':
            labels.append(1)
        elif effect.lower() == 'inhibit':
            labels.append(0)
        else:
            # Unknown effect, skip or assign default
            labels.append(-1)  # Will be filtered out
    
    # Filter out unknown labels
    valid_indices = [i for i, label in enumerate(labels) if label != -1]
    sequences = [sequences[i] for i in valid_indices]
    labels = [labels[i] for i in valid_indices]
    
    # Store metadata (optional, for reference)
    metadata = []
    for idx in valid_indices:
        row = df.iloc[idx]
        metadata.append({
            'Organism': row.get('Organism', ''),
            'Gene': row.get('Gene', ''),
            'Uniprot': row.get('Uniprot', ''),
            'PTM': row.get('PTM', ''),
            'Site': row.get('Site', ''),
            'AA': row.get('AA', ''),
            'Int_uniprot': row.get('Int_uniprot', ''),
            'Int_gene': row.get('Int_gene', ''),
            'Method': row.get('Method', ''),
            'Disease': row.get('Disease', ''),
            'Co-localized': row.get('Co-localized', ''),
            'PMID': row.get('PMID', '')
        })
    
    print(f"✅ Loaded {len(sequences)} sequences")
    print(f"   Enhance (label=1): {sum(labels)}")
    print(f"   Inhibit (label=0): {len(labels) - sum(labels)}")
    
    return sequences, labels, metadata


def save_embeddings_data(embeddings_list, labels, sequences, metadata, output_dir: str, split_name: str):
    """
    Save embeddings, labels, sequences, and metadata to disk.
    
    @param embeddings_list: List of per-position embeddings (each is a tensor)
    @param labels: List of binary labels (0 or 1)
    @param sequences: List of sequence strings
    @param metadata: List of metadata dictionaries
    @param output_dir: Output directory
    @param split_name: Name of the split (train/test/valid)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    embeddings_path = os.path.join(output_dir, f"{split_name}_embeddings.pt")
    labels_path = os.path.join(output_dir, f"{split_name}_labels.pt")
    sequences_path = os.path.join(output_dir, f"{split_name}_sequences.pt")
    metadata_path = os.path.join(output_dir, f"{split_name}_metadata.pt")
    
    torch.save(embeddings_list, embeddings_path)
    torch.save(labels, labels_path)
    torch.save(sequences, sequences_path)
    torch.save(metadata, metadata_path)
    
    print(f"✅ Saved {split_name} embeddings to {embeddings_path}")
    print(f"✅ Saved {split_name} labels to {labels_path}")
    print(f"✅ Saved {split_name} sequences to {sequences_path}")
    print(f"✅ Saved {split_name} metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for PPI prediction downstream task")
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
        default="/home/zz/zheng/ptm-mlm/downstream_tasks/ppi_prediction/best.ckpt",
        help="Path to model checkpoint (for Mamba model)"
    )
    parser.add_argument(
        "--esm2_model_name",
        type=str,
        default="facebook/esm2_t33_650M_UR50D",
        help="ESM2 model name from HuggingFace (default: facebook/esm2_t33_650M_UR50D)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/home/zz/zheng/ptm-mlm/downstream_tasks/ppi_prediction/PTM experimental evidence.csv",
        help="Path to PPI data CSV file"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Ratio of training data (default: 0.7)"
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.15,
        help="Ratio of validation data (default: 0.15)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Ratio of test data (default: 0.15)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/zz/zheng/ptm-mlm/downstream_tasks/ppi_prediction/embeddings",
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
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for data splitting (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate split ratios
    if abs(args.train_ratio + args.valid_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError(f"train_ratio + valid_ratio + test_ratio must equal 1.0, got {args.train_ratio + args.valid_ratio + args.test_ratio}")
    
    # Initialize inference model based on model_type
    print(f"\n🚀 Initializing {args.model_type.upper()} model inference...")
    if args.model_type == "mamba":
        if not MAMBA_AVAILABLE:
            raise ImportError("Mamba inference not available. Please ensure inference.py exists.")
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
    
    # Load data
    print("\n" + "="*50)
    print("📊 Loading PPI data...")
    print("="*50)
    
    sequences, labels, metadata = load_ppi_data(args.data)
    
    # Split data into train/valid/test
    import random
    random.seed(args.random_seed)
    indices = list(range(len(sequences)))
    random.shuffle(indices)
    
    n_total = len(sequences)
    n_train = int(n_total * args.train_ratio)
    n_valid = int(n_total * args.valid_ratio)
    
    train_indices = indices[:n_train]
    valid_indices = indices[n_train:n_train + n_valid]
    test_indices = indices[n_train + n_valid:]
    
    train_sequences = [sequences[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    train_metadata = [metadata[i] for i in train_indices]
    
    valid_sequences = [sequences[i] for i in valid_indices]
    valid_labels = [labels[i] for i in valid_indices]
    valid_metadata = [metadata[i] for i in valid_indices]
    
    test_sequences = [sequences[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    test_metadata = [metadata[i] for i in test_indices]
    
    print(f"\n📊 Data split:")
    print(f"   Train: {len(train_sequences)} samples")
    print(f"   Valid: {len(valid_sequences)} samples")
    print(f"   Test: {len(test_sequences)} samples")
    
    # Process each split
    splits = [
        ("train", train_sequences, train_labels, train_metadata),
        ("valid", valid_sequences, valid_labels, valid_metadata),
        ("test", test_sequences, test_labels, test_metadata)
    ]
    
    for split_name, split_sequences, split_labels, split_metadata in splits:
        if len(split_sequences) == 0:
            print(f"\n⚠️  Skipping {split_name}: no data")
            continue
        
        process_split(
            split_sequences,
            split_labels,
            split_metadata,
            split_name,
            inferencer,
            args.output_dir,
            args.batch_size,
            args.max_sequence_length,
            args.use_sliding_window,
            args.window_overlap
        )
    
    print("\n" + "="*50)
    print("🎉 Embedding generation completed!")
    print("="*50)


def process_split(
    sequences,
    labels,
    metadata,
    split_name: str,
    inferencer,
    output_dir: str,
    batch_size: int,
    max_sequence_length: int = None,
    use_sliding_window: bool = True,
    window_overlap: float = 0.5
):
    """
    Process a single data split: generate embeddings and save results.
    
    @param sequences: List of sequence strings
    @param labels: List of binary labels
    @param metadata: List of metadata dictionaries
    @param split_name: Name of the split (train/test/valid)
    @param inferencer: Inference model instance
    @param output_dir: Output directory for saved files
    @param batch_size: Batch size for inference
    @param max_sequence_length: Maximum sequence length
    @param use_sliding_window: Whether to use sliding window
    @param window_overlap: Window overlap ratio
    """
    print("\n" + "="*50)
    print(f"📊 Processing {split_name} data...")
    print("="*50)
    
    try:
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
        
        # Verify embeddings match sequences and labels
        aligned_embeddings = []
        aligned_labels = []
        aligned_sequences = []
        aligned_metadata = []
        
        for idx, (emb, lab, seq, meta) in enumerate(zip(embeddings, labels, sequences, metadata)):
            emb_len = emb.shape[0]
            seq_len = len(seq)
            
            # Verify embedding length matches sequence length
            if emb_len != seq_len:
                raise RuntimeError(
                    f"❌ Embedding length mismatch at index {idx}/{len(sequences)}:\n"
                    f"   - Embedding length: {emb_len}\n"
                    f"   - Sequence length: {seq_len}\n"
                    f"   - Sequence: {seq}\n"
                )
            
            # All lengths match ✅
            aligned_embeddings.append(emb)
            aligned_labels.append(lab)
            aligned_sequences.append(seq)
            aligned_metadata.append(meta)
        
        # Save aligned data
        save_embeddings_data(
            aligned_embeddings, 
            aligned_labels, 
            aligned_sequences, 
            aligned_metadata,
            output_dir, 
            split_name
        )
        
        print(f"\n✅ Successfully processed {split_name} data: {len(sequences)} samples")
        
    except Exception as e:
        print(f"\n❌ Error processing {split_name} data: {e}")
        raise


if __name__ == "__main__":
    main()














