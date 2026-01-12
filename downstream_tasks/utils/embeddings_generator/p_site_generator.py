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

# Import ESM inference classes
from inference_esm2 import ESM2Inference
from inference_esmc import ESMCInference


def infer_model_type(pretrained_model_name: str) -> str:
    """
    根据预训练模型名称推断模型类型。

    @param pretrained_model_name: 预训练模型名称
    @return: 模型类型 ('esm2' 或 'esmc')
    """
    if 'esm2' in pretrained_model_name.lower():
        return 'esm2'
    elif 'esmc' in pretrained_model_name.lower() or 'esm_c' in pretrained_model_name.lower():
        return 'esmc'
    else:
        raise ValueError(f"无法从模型名称 '{pretrained_model_name}' 推断模型类型。只支持ESM2和ESMC模型")


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
    @param output_dir: Output directory (should be task-specific subdirectory)
    @param split_name: Name of the split (train/test/valid)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Simple filenames since model info is in directory structure
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

    # Initialize inference model based on inferred model_type
    print(f"\n🚀 Initializing {model_type.upper()} model inference...")
    if model_type == "esm2":
        inferencer = ESM2Inference(
            model_name=args.pretrained_model_name,
            layer_index=args.layer_index
        )
    elif model_type == "esmc":
        inferencer = ESMCInference(
            layer_index=args.layer_index
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'esm2' or 'esmc'.")
    
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
        
        # Generate per-position embeddings for ESM2/ESMC
        print(f"\n🔄 Generating embeddings for {len(sequences)} sequences...")
        embeddings = inferencer.generate_embeddings(sequences, return_pooled=False)
            
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

