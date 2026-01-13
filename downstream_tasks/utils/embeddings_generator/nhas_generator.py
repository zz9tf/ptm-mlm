"""
Script to generate embeddings from pre-trained model for NHA site prediction.
This script processes training, validation, and test data to generate embeddings.
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

import sys
import os
import importlib.util

# Import load_data from NHA_site_prediction directory
nha_load_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'tasks', 'NHA_site_prediction', 'load_data.py')
spec = importlib.util.spec_from_file_location("nha_load_data", nha_load_data_path)
nha_load_data = importlib.util.module_from_spec(spec)
sys.modules["nha_load_data"] = nha_load_data
spec.loader.exec_module(nha_load_data)

load_nha_data = nha_load_data.load_nha_data
prepare_sequences_and_labels = nha_load_data.prepare_sequences_and_labels

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


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for NHA site prediction")
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
        "--sequence_column",
        type=str,
        default=None,
        help="Column name for sequences. If None, will process all seq_* columns. "
             "If specified, will only process that column (default: None, process all)"
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
        default=False,
        help="Use sliding window for sequences longer than max_sequence_length. "
             "For NHA prediction, sequences are short (61), so this is usually not needed."
    )
    parser.add_argument(
        "--window_overlap",
        type=float,
        default=0.3,
        help="Overlap ratio between sliding windows (0.0 to 1.0). "
             "Default 0.3 means 30%% overlap."
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
    task_dir = os.path.join(output_dir, model_layer_dir, "nhas")
    os.makedirs(task_dir, exist_ok=True)
    print(f"📁 Embeddings will be stored in: {task_dir}")

    # Fixed data path for NHA
    data_path = "/home/zz/zheng/ptm-mlm/downstream_tasks/tasks/NHA_site_prediction/NHAC.csv"

    # Load data (first load to detect sequence columns)
    print("📖 Loading NHA data...")
    # Load with a dummy column first to get the dataframe structure
    df_temp = pd.read_csv(data_path)
    
    # Detect all sequence columns (columns starting with 'seq_')
    if args.sequence_column is None:
        sequence_columns = [col for col in df_temp.columns if col.startswith('seq_')]
        sequence_columns.sort()  # Sort to process in order: seq_11, seq_15, ...
        print(f"🔍 Detected {len(sequence_columns)} sequence columns: {sequence_columns}")
    else:
        sequence_columns = [args.sequence_column]
        print(f"📌 Processing single sequence column: {args.sequence_column}")
    
    if len(sequence_columns) == 0:
        raise ValueError("No sequence columns found in the data!")
    
    # Initialize EmbeddingGeneratorInference
    print(f"\n🚀 Initializing {model_type.upper()} embedding generator...")
    inferencer = EmbeddingGeneratorInference(
        model_type=model_type,
        model_name=args.pretrained_model_name,
        layer_index=args.layer_index,
        max_sequence_length=args.max_sequence_length
    )
    
    # Initialize lists to collect all data from all sequence columns
    all_train_embeddings_tensors = []  # List of batch tensors
    all_train_metadata_list = []  # List of metadata lists
    all_train_labels = []
    all_train_sequences = []
    all_valid_embeddings_tensors = []
    all_valid_metadata_list = []
    all_valid_labels = []
    all_valid_sequences = []
    all_test_embeddings_tensors = []
    all_test_metadata_list = []
    all_test_labels = []
    all_test_sequences = []
    
    # Process each sequence column
    for seq_col in sequence_columns:
        print("\n" + "="*70)
        print(f"🔄 Processing sequence column: {seq_col}")
        print("="*70)
        
        # Load data for this sequence column
        train_df, valid_df, test_df = load_nha_data(data_path, seq_col)
        
        # Process training data
        print("\n" + "="*50)
        print(f"📊 Processing training data for {seq_col}...")
        print("="*50)
        train_sequences, train_labels = prepare_sequences_and_labels(train_df, seq_col)
        
        # Generate batch embeddings and metadata (for pipeline processing)
        print("Generating batch embeddings and metadata...")
        train_embeddings_tensor, train_metadata_list, _ = inferencer.generate_batch_embeddings(
            train_sequences,
            batch_size=args.batch_size,
            max_sequence_length=args.max_sequence_length if args.max_sequence_length else 512,
            use_sliding_window=args.use_sliding_window,
            window_overlap=args.window_overlap
        )
        
        # Store batch tensor and metadata (will be saved later)
        all_train_embeddings_tensors.append(train_embeddings_tensor)
        all_train_metadata_list.extend(train_metadata_list)
        all_train_labels.extend(train_labels)
        all_train_sequences.extend(train_sequences)
        
        # Process validation data
        print("\n" + "="*50)
        print(f"📊 Processing validation data for {seq_col}...")
        print("="*50)
        valid_sequences, valid_labels = prepare_sequences_and_labels(valid_df, seq_col)
        
        # Generate batch embeddings and metadata
        valid_embeddings_tensor, valid_metadata_list, _ = inferencer.generate_batch_embeddings(
            valid_sequences,
            batch_size=args.batch_size,
            max_sequence_length=args.max_sequence_length if args.max_sequence_length else 512,
            use_sliding_window=args.use_sliding_window,
            window_overlap=args.window_overlap
        )
        
        # Store batch tensor and metadata
        all_valid_embeddings_tensors.append(valid_embeddings_tensor)
        all_valid_metadata_list.extend(valid_metadata_list)
        all_valid_labels.extend(valid_labels)
        all_valid_sequences.extend(valid_sequences)
        
        # Process test data
        print("\n" + "="*50)
        print(f"📊 Processing test data for {seq_col}...")
        print("="*50)
        test_sequences, test_labels = prepare_sequences_and_labels(test_df, seq_col)
        
        # Generate batch embeddings and metadata
        test_embeddings_tensor, test_metadata_list, _ = inferencer.generate_batch_embeddings(
            test_sequences,
            batch_size=args.batch_size,
            max_sequence_length=args.max_sequence_length if args.max_sequence_length else 512,
            use_sliding_window=args.use_sliding_window,
            window_overlap=args.window_overlap
        )
        
        # Store batch tensor and metadata
        all_test_embeddings_tensors.append(test_embeddings_tensor)
        all_test_metadata_list.extend(test_metadata_list)
        all_test_labels.extend(test_labels)
        all_test_sequences.extend(test_sequences)
    
    # Save all combined data
    print("\n" + "="*70)
    print("💾 Saving combined data from all sequence columns...")
    print("="*70)
    
    # Concatenate all batch tensors from different sequence columns
    print("\n📊 Concatenating batch tensors from all sequence columns...")
    
    # Concatenate training embeddings tensors
    if len(all_train_embeddings_tensors) > 0:
        # Update metadata embedding_idx and sequence_id with offsets before concatenation
        train_offset = 0
        train_seq_offset = 0
        metadata_idx = 0
        
        # First pass: calculate sequence count for each tensor
        # We need to know how many sequences each tensor has before we can update sequence_id
        train_seq_counts = []
        current_metadata_idx = 0
        for tensor in all_train_embeddings_tensors:
            tensor_size = tensor.shape[0]
            # Get metadata for this tensor
            tensor_metadata = all_train_metadata_list[current_metadata_idx:current_metadata_idx + tensor_size]
            if tensor_metadata:
                # Get max sequence_id in this tensor's metadata
                max_seq_id = max(meta['sequence_id'] for meta in tensor_metadata)
                seq_count = max_seq_id + 1  # sequence_id is 0-based, so count is max + 1
                train_seq_counts.append(seq_count)
            else:
                train_seq_counts.append(0)
            current_metadata_idx += tensor_size
        
        # Second pass: update metadata with offsets
        metadata_idx = 0
        for i, tensor in enumerate(all_train_embeddings_tensors):
            tensor_size = tensor.shape[0]
            
            for j in range(tensor_size):
                if metadata_idx < len(all_train_metadata_list):
                    # Update embedding_idx
                    all_train_metadata_list[metadata_idx]['embedding_idx'] += train_offset
                    # Update sequence_id (shift by accumulated sequence count from previous columns)
                    all_train_metadata_list[metadata_idx]['sequence_id'] += train_seq_offset
                    metadata_idx += 1
            
            train_offset += tensor_size
            train_seq_offset += train_seq_counts[i]
        
        # Concatenate tensors
        all_train_embeddings_tensor = torch.cat(all_train_embeddings_tensors, dim=0)
    else:
        all_train_embeddings_tensor = None
    
    # Concatenate validation embeddings tensors
    if len(all_valid_embeddings_tensors) > 0:
        valid_offset = 0
        valid_seq_offset = 0
        metadata_idx = 0
        
        # Calculate sequence count for each tensor
        valid_seq_counts = []
        current_metadata_idx = 0
        for tensor in all_valid_embeddings_tensors:
            tensor_size = tensor.shape[0]
            tensor_metadata = all_valid_metadata_list[current_metadata_idx:current_metadata_idx + tensor_size]
            if tensor_metadata:
                max_seq_id = max(meta['sequence_id'] for meta in tensor_metadata)
                seq_count = max_seq_id + 1
                valid_seq_counts.append(seq_count)
            else:
                valid_seq_counts.append(0)
            current_metadata_idx += tensor_size
        
        # Update metadata with offsets
        metadata_idx = 0
        for i, tensor in enumerate(all_valid_embeddings_tensors):
            tensor_size = tensor.shape[0]
            
            for j in range(tensor_size):
                if metadata_idx < len(all_valid_metadata_list):
                    all_valid_metadata_list[metadata_idx]['embedding_idx'] += valid_offset
                    all_valid_metadata_list[metadata_idx]['sequence_id'] += valid_seq_offset
                    metadata_idx += 1
            
            valid_offset += tensor_size
            valid_seq_offset += valid_seq_counts[i]
        
        all_valid_embeddings_tensor = torch.cat(all_valid_embeddings_tensors, dim=0)
    else:
        all_valid_embeddings_tensor = None
    
    # Concatenate test embeddings tensors
    if len(all_test_embeddings_tensors) > 0:
        test_offset = 0
        test_seq_offset = 0
        metadata_idx = 0
        
        # Calculate sequence count for each tensor
        test_seq_counts = []
        current_metadata_idx = 0
        for tensor in all_test_embeddings_tensors:
            tensor_size = tensor.shape[0]
            tensor_metadata = all_test_metadata_list[current_metadata_idx:current_metadata_idx + tensor_size]
            if tensor_metadata:
                max_seq_id = max(meta['sequence_id'] for meta in tensor_metadata)
                seq_count = max_seq_id + 1
                test_seq_counts.append(seq_count)
            else:
                test_seq_counts.append(0)
            current_metadata_idx += tensor_size
        
        # Update metadata with offsets
        metadata_idx = 0
        for i, tensor in enumerate(all_test_embeddings_tensors):
            tensor_size = tensor.shape[0]
            
            for j in range(tensor_size):
                if metadata_idx < len(all_test_metadata_list):
                    all_test_metadata_list[metadata_idx]['embedding_idx'] += test_offset
                    all_test_metadata_list[metadata_idx]['sequence_id'] += test_seq_offset
                    metadata_idx += 1
            
            test_offset += tensor_size
            test_seq_offset += test_seq_counts[i]
        
        all_test_embeddings_tensor = torch.cat(all_test_embeddings_tensors, dim=0)
    else:
        all_test_embeddings_tensor = None
    
    # Verify final consistency
    # After updating sequence_id, the max sequence_id should match the total number of sequences
    train_num_seqs = max(meta['sequence_id'] for meta in all_train_metadata_list) + 1 if all_train_metadata_list else 0
    valid_num_seqs = max(meta['sequence_id'] for meta in all_valid_metadata_list) + 1 if all_valid_metadata_list else 0
    test_num_seqs = max(meta['sequence_id'] for meta in all_test_metadata_list) + 1 if all_test_metadata_list else 0
    
    assert train_num_seqs == len(all_train_labels) == len(all_train_sequences), \
        f"Final train mismatch: sequences={train_num_seqs}, labels={len(all_train_labels)}, sequences={len(all_train_sequences)}"
    assert valid_num_seqs == len(all_valid_labels) == len(all_valid_sequences), \
        f"Final valid mismatch: sequences={valid_num_seqs}, labels={len(all_valid_labels)}, sequences={len(all_valid_sequences)}"
    assert test_num_seqs == len(all_test_labels) == len(all_test_sequences), \
        f"Final test mismatch: sequences={test_num_seqs}, labels={len(all_test_labels)}, sequences={len(all_test_sequences)}"
    
    # Print statistics
    print(f"\n📊 Final statistics:")
    print(f"   Train: {train_num_seqs} sequences, {all_train_embeddings_tensor.shape[0] if all_train_embeddings_tensor is not None else 0} items in batch tensor")
    print(f"   Valid: {valid_num_seqs} sequences, {all_valid_embeddings_tensor.shape[0] if all_valid_embeddings_tensor is not None else 0} items in batch tensor")
    print(f"   Test: {test_num_seqs} sequences, {all_test_embeddings_tensor.shape[0] if all_test_embeddings_tensor is not None else 0} items in batch tensor")
    
    # Save training data
    if all_train_embeddings_tensor is not None:
        train_emb_path = os.path.join(task_dir, "train_embeddings.pt")
        train_metadata_path = os.path.join(task_dir, "train_embeddings_metadata.json")
        train_labels_path = os.path.join(task_dir, "train_labels.pt")
        train_seqs_path = os.path.join(task_dir, "train_sequences.pt")
        torch.save(all_train_embeddings_tensor, train_emb_path)
        EmbeddingGeneratorInference.save_metadata(all_train_metadata_list, train_metadata_path)
        torch.save(all_train_labels, train_labels_path)
        torch.save(all_train_sequences, train_seqs_path)
        print(f"✅ Saved combined training data: {train_num_seqs} samples")

    # Save validation data
    if all_valid_embeddings_tensor is not None:
        valid_emb_path = os.path.join(task_dir, "valid_embeddings.pt")
        valid_metadata_path = os.path.join(task_dir, "valid_embeddings_metadata.json")
        valid_labels_path = os.path.join(task_dir, "valid_labels.pt")
        valid_seqs_path = os.path.join(task_dir, "valid_sequences.pt")
        torch.save(all_valid_embeddings_tensor, valid_emb_path)
        EmbeddingGeneratorInference.save_metadata(all_valid_metadata_list, valid_metadata_path)
        torch.save(all_valid_labels, valid_labels_path)
        torch.save(all_valid_sequences, valid_seqs_path)
        print(f"✅ Saved combined validation data: {valid_num_seqs} samples")

    # Save test data
    if all_test_embeddings_tensor is not None:
        test_emb_path = os.path.join(task_dir, "test_embeddings.pt")
        test_metadata_path = os.path.join(task_dir, "test_embeddings_metadata.json")
        test_labels_path = os.path.join(task_dir, "test_labels.pt")
        test_seqs_path = os.path.join(task_dir, "test_sequences.pt")
        torch.save(all_test_embeddings_tensor, test_emb_path)
        EmbeddingGeneratorInference.save_metadata(all_test_metadata_list, test_metadata_path)
        torch.save(all_test_labels, test_labels_path)
        torch.save(all_test_sequences, test_seqs_path)
        print(f"✅ Saved combined test data: {test_num_seqs} samples")
    
    print("\n" + "="*70)
    print("🎉 Embedding generation completed for all sequence columns!")
    print(f"   Total training samples: {train_num_seqs}")
    print(f"   Total validation samples: {valid_num_seqs}")
    print(f"   Total test samples: {test_num_seqs}")
    print("="*70)


if __name__ == "__main__":
    main()

