"""
Script to generate embeddings from pre-trained model for PPI prediction.
This script processes training, validation, and test data to generate embeddings for binder, wt, and ptm sequences.
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

# Import load_data from ppi_prediction directory
ppi_load_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'tasks', 'ppi_prediction', 'load_data.py')
spec = importlib.util.spec_from_file_location("ppi_load_data", ppi_load_data_path)
ppi_load_data = importlib.util.module_from_spec(spec)
sys.modules["ppi_load_data"] = ppi_load_data
spec.loader.exec_module(ppi_load_data)

load_ppi_data = ppi_load_data.load_ppi_data
prepare_sequences_and_labels = ppi_load_data.prepare_sequences_and_labels

# Add main_pipeline to path (inference.py needs getters.tokenizer, utils.checkpoint, etc.)
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
    parser = argparse.ArgumentParser(description="Generate embeddings for PPI prediction")
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
        default=False,
        help="Use sliding window for sequences longer than max_sequence_length."
    )
    parser.add_argument(
        "--window_overlap",
        type=float,
        default=0.3,
        help="Overlap ratio between sliding windows (0.0 to 1.0). "
             "Default 0.3 means 30%% overlap."
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Ratio for training set (default: 0.7)"
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.15,
        help="Ratio for validation set (default: 0.15)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Ratio for test set (default: 0.15)"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for data splitting (default: 42)"
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
    task_dir = os.path.join(output_dir, model_layer_dir, "ppi")
    os.makedirs(task_dir, exist_ok=True)
    print(f"📁 Embeddings will be stored in: {task_dir}")

    # Fixed data path for PPI
    data_path = "/home/zz/zheng/ptm-mlm/downstream_tasks/tasks/ppi_prediction/PTM experimental evidence.csv"

    # Load data
    print("📖 Loading PPI data...")
    train_df, valid_df, test_df = load_ppi_data(
        data_path,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )
    
    # Initialize inference model based on inferred model_type
    print(f"\n🚀 Initializing {model_type.upper()} model inference...")
    if model_type == "esm2":
        inferencer = ESM2Inference(
            model_name=args.pretrained_model_name,
            max_sequence_length=args.max_sequence_length,
            layer_index=args.layer_index
        )
    elif model_type == "esmc":
        inferencer = ESMCInference(
            layer_index=args.layer_index
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'esm2' or 'esmc'.")
    
    # Process each split
    for split_name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        print("\n" + "="*70)
        print(f"🔄 Processing {split_name} data...")
        print("="*70)
        
        # Prepare sequences and labels
        binder_sequences, wt_sequences, ptm_sequences, labels = prepare_sequences_and_labels(df)
        
        if len(binder_sequences) == 0:
            print(f"⚠️  No valid samples in {split_name} set, skipping...")
            continue
        
        print(f"📊 Processing {len(binder_sequences)} samples...")
        
        # Generate embeddings for binder sequences
        print(f"\n🔹 Generating binder embeddings...")
        binder_embeddings_list = []
        # Use pooled embeddings for PPI prediction
        generate_method = inferencer.generate_embeddings
        for i in tqdm(range(0, len(binder_sequences), args.batch_size), desc=f"Binder embeddings ({split_name})"):
            batch_sequences = binder_sequences[i:i + args.batch_size]
            batch_embeddings = generate_method(
                batch_sequences,
                return_pooled=True
            )
            # batch_embeddings shape: [batch_size, hidden_size]
            for j in range(len(batch_sequences)):
                binder_embeddings_list.append(batch_embeddings[j])
        
        # Generate embeddings for wt sequences
        print(f"\n🔹 Generating wild-type embeddings...")
        wt_embeddings_list = []
        for i in tqdm(range(0, len(wt_sequences), args.batch_size), desc=f"WT embeddings ({split_name})"):
            batch_sequences = wt_sequences[i:i + args.batch_size]
            batch_embeddings = generate_method(
                batch_sequences,
                return_pooled=True
            )
            for j in range(len(batch_sequences)):
                wt_embeddings_list.append(batch_embeddings[j])
        
        # Generate embeddings for ptm sequences
        print(f"\n🔹 Generating PTM-modified embeddings...")
        ptm_embeddings_list = []
        for i in tqdm(range(0, len(ptm_sequences), args.batch_size), desc=f"PTM embeddings ({split_name})"):
            batch_sequences = ptm_sequences[i:i + args.batch_size]
            batch_embeddings = generate_method(
                batch_sequences,
                return_pooled=True
            )
            for j in range(len(batch_sequences)):
                ptm_embeddings_list.append(batch_embeddings[j])
        
        # Verify consistency
        assert len(binder_embeddings_list) == len(wt_embeddings_list) == len(ptm_embeddings_list) == len(labels), \
            f"Mismatch in {split_name}: binder={len(binder_embeddings_list)}, wt={len(wt_embeddings_list)}, " \
            f"ptm={len(ptm_embeddings_list)}, labels={len(labels)}"
        
        # Save embeddings
        print(f"\n💾 Saving {split_name} embeddings...")

        binder_emb_path = os.path.join(task_dir, f"{split_name}_binder_embeddings.pt")
        wt_emb_path = os.path.join(task_dir, f"{split_name}_wt_embeddings.pt")
        ptm_emb_path = os.path.join(task_dir, f"{split_name}_ptm_embeddings.pt")
        labels_path = os.path.join(task_dir, f"{split_name}_labels.pt")
        
        torch.save(binder_embeddings_list, binder_emb_path)
        torch.save(wt_embeddings_list, wt_emb_path)
        torch.save(ptm_embeddings_list, ptm_emb_path)
        torch.save(labels, labels_path)
        
        print(f"✅ Saved {split_name} data: {len(labels)} samples")
    
    print("\n" + "="*70)
    print("🎉 Embedding generation completed!")
    print("="*70)


if __name__ == "__main__":
    main()

