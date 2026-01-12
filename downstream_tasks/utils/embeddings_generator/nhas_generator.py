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
    
    # Initialize lists to collect all data from all sequence columns
    all_train_embeddings = []
    all_train_labels = []
    all_train_sequences = []
    all_valid_embeddings = []
    all_valid_labels = []
    all_valid_sequences = []
    all_test_embeddings = []
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
        
        # For NHA prediction, we use per-position embeddings for GNN
        # Each amino acid position will be a node in the graph
        # Process sequences, labels, and embeddings together to ensure they match
        print("Generating per-position embeddings for GNN...")
        train_embeddings_list = []
        train_labels_processed = []
        train_sequences_processed = []
        
        for i in tqdm(range(0, len(train_sequences), args.batch_size), desc="Training embeddings"):
            batch_sequences = train_sequences[i:i + args.batch_size]
            batch_labels = train_labels[i:i + args.batch_size]
            
            # Generate per-position embeddings for ESM2/ESMC
            batch_embeddings = inferencer.generate_embeddings(batch_sequences, return_pooled=False)
            
            # Ensure embeddings, labels, and sequences are matched
            # Keep per-position embeddings: [seq_len, hidden_size]
            for j, emb in enumerate(batch_embeddings):
                train_embeddings_list.append(emb)  # Shape: [seq_len, hidden_size]
                train_labels_processed.append(batch_labels[j])
                train_sequences_processed.append(batch_sequences[j])
        
        # Use processed lists
        train_labels = train_labels_processed
        train_sequences = train_sequences_processed
        
        # Verify consistency
        assert len(train_embeddings_list) == len(train_labels) == len(train_sequences), \
            f"Mismatch: embeddings={len(train_embeddings_list)}, labels={len(train_labels)}, sequences={len(train_sequences)}"
        
        # Add to overall lists
        all_train_embeddings.extend(train_embeddings_list)
        all_train_labels.extend(train_labels)
        all_train_sequences.extend(train_sequences)
        
        # Process validation data
        print("\n" + "="*50)
        print(f"📊 Processing validation data for {seq_col}...")
        print("="*50)
        valid_sequences, valid_labels = prepare_sequences_and_labels(valid_df, seq_col)
        
        valid_embeddings_list = []
        valid_labels_processed = []
        valid_sequences_processed = []
        
        for i in tqdm(range(0, len(valid_sequences), args.batch_size), desc="Validation embeddings"):
            batch_sequences = valid_sequences[i:i + args.batch_size]
            batch_labels = valid_labels[i:i + args.batch_size]
            
            # Generate embeddings using the appropriate method for ESM2/ESMC
            batch_embeddings = inferencer.generate_embeddings(batch_sequences, return_pooled=False)

            # Ensure embeddings, labels, and sequences are matched
            # Keep per-position embeddings: [seq_len, hidden_size]
            for j, emb in enumerate(batch_embeddings):
                valid_embeddings_list.append(emb)  # Shape: [seq_len, hidden_size]
                valid_labels_processed.append(batch_labels[j])
                valid_sequences_processed.append(batch_sequences[j])
        
        # Use processed lists
        valid_labels = valid_labels_processed
        valid_sequences = valid_sequences_processed
        
        # Verify consistency
        assert len(valid_embeddings_list) == len(valid_labels) == len(valid_sequences), \
            f"Mismatch: embeddings={len(valid_embeddings_list)}, labels={len(valid_labels)}, sequences={len(valid_sequences)}"
        
        # Add to overall lists
        all_valid_embeddings.extend(valid_embeddings_list)
        all_valid_labels.extend(valid_labels)
        all_valid_sequences.extend(valid_sequences)
        
        # Process test data
        print("\n" + "="*50)
        print(f"📊 Processing test data for {seq_col}...")
        print("="*50)
        test_sequences, test_labels = prepare_sequences_and_labels(test_df, seq_col)
        
        test_embeddings_list = []
        test_labels_processed = []
        test_sequences_processed = []
        
        for i in tqdm(range(0, len(test_sequences), args.batch_size), desc="Test embeddings"):
            batch_sequences = test_sequences[i:i + args.batch_size]
            batch_labels = test_labels[i:i + args.batch_size]
            
            # Generate per-position embeddings for ESM2/ESMC
            batch_embeddings, _ = inferencer.generate_per_position_embeddings(
                batch_sequences,
                batch_size=len(batch_sequences),
                max_sequence_length=args.max_sequence_length,
                use_sliding_window=args.use_sliding_window,
                window_overlap=args.window_overlap
            )
            
            # Ensure embeddings, labels, and sequences are matched
            # Keep per-position embeddings: [seq_len, hidden_size]
            for j, emb in enumerate(batch_embeddings):
                test_embeddings_list.append(emb)  # Shape: [seq_len, hidden_size]
                test_labels_processed.append(batch_labels[j])
                test_sequences_processed.append(batch_sequences[j])
        
        # Use processed lists
        test_labels = test_labels_processed
        test_sequences = test_sequences_processed
        
        # Verify consistency
        assert len(test_embeddings_list) == len(test_labels) == len(test_sequences), \
            f"Mismatch: embeddings={len(test_embeddings_list)}, labels={len(test_labels)}, sequences={len(test_sequences)}"
        
        # Add to overall lists
        all_test_embeddings.extend(test_embeddings_list)
        all_test_labels.extend(test_labels)
        all_test_sequences.extend(test_sequences)
    
    # Save all combined data
    print("\n" + "="*70)
    print("💾 Saving combined data from all sequence columns...")
    print("="*70)
    
    # Verify final consistency
    assert len(all_train_embeddings) == len(all_train_labels) == len(all_train_sequences), \
        f"Final train mismatch: embeddings={len(all_train_embeddings)}, labels={len(all_train_labels)}, sequences={len(all_train_sequences)}"
    assert len(all_valid_embeddings) == len(all_valid_labels) == len(all_valid_sequences), \
        f"Final valid mismatch: embeddings={len(all_valid_embeddings)}, labels={len(all_valid_labels)}, sequences={len(all_valid_sequences)}"
    assert len(all_test_embeddings) == len(all_test_labels) == len(all_test_sequences), \
        f"Final test mismatch: embeddings={len(all_test_embeddings)}, labels={len(all_test_labels)}, sequences={len(all_test_sequences)}"
    
    # Print sequence length distribution for reference
    print("\n📊 Sequence length distribution:")
    train_lengths = [emb.shape[0] for emb in all_train_embeddings]
    valid_lengths = [emb.shape[0] for emb in all_valid_embeddings] if len(all_valid_embeddings) > 0 else []
    test_lengths = [emb.shape[0] for emb in all_test_embeddings] if len(all_test_embeddings) > 0 else []
    
    from collections import Counter
    train_length_dist = Counter(train_lengths)
    valid_length_dist = Counter(valid_lengths) if valid_lengths else {}
    test_length_dist = Counter(test_lengths) if test_lengths else {}
    
    all_unique_lengths = set(train_lengths + valid_lengths + test_lengths)
    print(f"   Found {len(all_unique_lengths)} different sequence lengths: {sorted(all_unique_lengths)}")
    print(f"   Train length distribution: {dict(sorted(train_length_dist.items()))}")
    if valid_length_dist:
        print(f"   Valid length distribution: {dict(sorted(valid_length_dist.items()))}")
    if test_length_dist:
        print(f"   Test length distribution: {dict(sorted(test_length_dist.items()))}")
    
    # Save training data
    train_emb_path = os.path.join(task_dir, "train_embeddings.pt")
    train_labels_path = os.path.join(task_dir, "train_labels.pt")
    train_seqs_path = os.path.join(task_dir, "train_sequences.pt")
    torch.save(all_train_embeddings, train_emb_path)
    torch.save(all_train_labels, train_labels_path)
    torch.save(all_train_sequences, train_seqs_path)
    print(f"✅ Saved combined training data: {len(all_train_embeddings)} samples")

    # Save validation data
    valid_emb_path = os.path.join(task_dir, "valid_embeddings.pt")
    valid_labels_path = os.path.join(task_dir, "valid_labels.pt")
    valid_seqs_path = os.path.join(task_dir, "valid_sequences.pt")
    torch.save(all_valid_embeddings, valid_emb_path)
    torch.save(all_valid_labels, valid_labels_path)
    torch.save(all_valid_sequences, valid_seqs_path)
    print(f"✅ Saved combined validation data: {len(all_valid_embeddings)} samples")

    # Save test data
    test_emb_path = os.path.join(task_dir, "test_embeddings.pt")
    test_labels_path = os.path.join(task_dir, "test_labels.pt")
    test_seqs_path = os.path.join(task_dir, "test_sequences.pt")
    torch.save(all_test_embeddings, test_emb_path)
    torch.save(all_test_labels, test_labels_path)
    torch.save(all_test_sequences, test_seqs_path)
    print(f"✅ Saved combined test data: {len(all_test_embeddings)} samples")
    
    print("\n" + "="*70)
    print("🎉 Embedding generation completed for all sequence columns!")
    print(f"   Total training samples: {len(all_train_embeddings)}")
    print(f"   Total validation samples: {len(all_valid_embeddings)}")
    print(f"   Total test samples: {len(all_test_embeddings)}")
    print("="*70)


if __name__ == "__main__":
    main()

