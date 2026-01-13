"""
Script to generate embeddings from pre-trained model for downstream task training.
This script processes training, validation, and test data to generate embeddings that can be reused.
"""
import torch
import pandas as pd
import argparse
import os
from pathlib import Path
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

# Import EmbeddingGeneratorInference
from embedding_generator_inference import EmbeddingGeneratorInference


def infer_model_type(pretrained_model_name: str) -> str:
    """
    根据预训练模型名称推断模型类型。

    @param pretrained_model_name: 预训练模型名称
    @return: 模型类型 ('esm2' 或 'esmc')
    """
    return EmbeddingGeneratorInference.infer_model_type(pretrained_model_name)


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


def save_embeddings_data(embeddings_tensor, metadata_list, labels, sequences, output_dir: str, split_name: str):
    """
    Save batch embeddings tensor, metadata, labels, and sequences to disk.

    @param embeddings_tensor: Batch tensor of shape (total_items, max_seq_len, embed_dim)
    @param metadata_list: List of metadata dicts
    @param labels: List of label strings
    @param sequences: List of sequence strings
    @param output_dir: Output directory (should be task-specific subdirectory)
    @param split_name: Name of the split (train/test/valid)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save batch tensor and metadata (for pipeline processing)
    embeddings_path = os.path.join(output_dir, f"{split_name}_embeddings.pt")
    metadata_path = os.path.join(output_dir, f"{split_name}_embeddings_metadata.json")
    labels_path = os.path.join(output_dir, f"{split_name}_labels.pt")
    sequences_path = os.path.join(output_dir, f"{split_name}_sequences.pt")

    torch.save(embeddings_tensor, embeddings_path)
    EmbeddingGeneratorInference.save_metadata(metadata_list, metadata_path)
    torch.save(labels, labels_path)
    torch.save(sequences, sequences_path)

    print(f"✅ Saved {split_name} embeddings tensor to {embeddings_path}")
    print(f"✅ Saved {split_name} metadata to {metadata_path}")
    print(f"✅ Saved {split_name} labels to {labels_path}")
    print(f"✅ Saved {split_name} sequences to {sequences_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for downstream task")
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        default=None,
        help="Pretrained model name from HuggingFace. If None, uses default ESM2 model."
    )
    parser.add_argument(
        "--layer_index",
        type=int,
        default=None,
        help="Layer index to extract (1-based for esmc, 0-based for esm2). If None, uses last layer (default: None)"
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
        default=0.3,
        help="Overlap ratio between sliding windows (0.0 to 1.0). "
             "Default 0.3 means 30%% overlap. Higher overlap provides better context but requires more computation."
    )
    
    args = parser.parse_args()

    # Set default pretrained model name
    if args.pretrained_model_name is None:
        args.pretrained_model_name = "facebook/esm2_t30_150M_UR50D"

    # Infer model type from pretrained model name
    model_type = infer_model_type(args.pretrained_model_name)
    print(f"🔍 Inferred model type: {model_type} from {args.pretrained_model_name}")

    # Create model-specific directory structure
    model_short_name = args.pretrained_model_name.replace('/', '_').replace('facebook_', '').replace('esm2_', 'esm2-').replace('esm_', 'esm-')
    layer_suffix = f"layer{args.layer_index}" if args.layer_index is not None else "last"
    model_layer_dir = f"{model_short_name}_{layer_suffix}"

    # Create task-specific subdirectory
    output_dir = os.path.join(os.getcwd(), "embeddings")
    task_dir = os.path.join(output_dir, model_layer_dir, "p_site")
    os.makedirs(task_dir, exist_ok=True)
    print(f"📁 Embeddings will be stored in: {task_dir}")

    # Fixed data paths for phosphorylation site prediction
    train_data_path = "/home/zz/zheng/ptm-mlm/downstream_tasks/tasks/p_site_prediction/PhosphositePTM.train.txt"
    test_data_path = "/home/zz/zheng/ptm-mlm/downstream_tasks/tasks/p_site_prediction/PhosphositePTM.test.txt"
    valid_data_path = "/home/zz/zheng/ptm-mlm/downstream_tasks/tasks/p_site_prediction/PhosphositePTM.valid.txt"

    # Initialize EmbeddingGeneratorInference
    print(f"\n🚀 Initializing {model_type.upper()} embedding generator...")
    inferencer = EmbeddingGeneratorInference(
        model_type=model_type,
        model_name=args.pretrained_model_name,
        layer_index=args.layer_index,
        max_sequence_length=args.max_sequence_length
    )
    
    # Process training data
    process_split(
        train_data_path,
        "train",
        inferencer,
        task_dir,
        args.batch_size,
        args.max_sequence_length,
        args.use_sliding_window,
        args.window_overlap
    )

    # Process test data
    process_split(
        test_data_path,
        "test",
        inferencer,
        task_dir,
        args.batch_size,
        args.max_sequence_length,
        args.use_sliding_window,
        args.window_overlap
    )

    # Process validation data
    process_split(
        valid_data_path,
        "valid",
        inferencer,
        task_dir,
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
        
        # Generate batch embeddings and metadata
        print(f"\n🔄 Generating batch embeddings for {len(sequences)} sequences...")
        embeddings_tensor, metadata_list, original_lengths = inferencer.generate_batch_embeddings(
            sequences,
            batch_size=batch_size,
            max_sequence_length=max_sequence_length if max_sequence_length else 512,
            use_sliding_window=use_sliding_window,
            window_overlap=window_overlap
        )
        
        # Verify metadata matches sequences
        num_sequences_in_metadata = max(meta['sequence_id'] for meta in metadata_list) + 1
        if num_sequences_in_metadata != len(sequences):
            raise RuntimeError(
                f"Metadata sequence count ({num_sequences_in_metadata}) != input sequences ({len(sequences)})"
            )
        
        # Track statistics for debugging
        seq_lengths = [len(seq) for seq in sequences]
        max_seq_len = max(seq_lengths) if seq_lengths else 0
        min_seq_len = min(seq_lengths) if seq_lengths else 0
        print(f"📊 Sequence length stats: min={min_seq_len}, max={max_seq_len}, total={len(sequences)}")
        print(f"📊 Embeddings tensor shape: {embeddings_tensor.shape}")
        print(f"📊 Metadata records: {len(metadata_list)}")
        
        # Save batch tensor, metadata, labels, and sequences
        save_embeddings_data(embeddings_tensor, metadata_list, labels, sequences, output_dir, split_name)
        
        print(f"\n✅ Successfully processed {split_name} data: {len(sequences)} samples")
        
    except Exception as e:
        print(f"\n❌ Error processing {split_name} data: {e}")
        raise


if __name__ == "__main__":
    main()

