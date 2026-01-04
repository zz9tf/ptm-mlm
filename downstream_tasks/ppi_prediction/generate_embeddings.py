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

from load_data import load_ppi_data, prepare_sequences_and_labels

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
    ESM2_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for PPI prediction")
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
        "--data",
        type=str,
        default="/home/zz/zheng/ptm-mlm/downstream_tasks/ppi_prediction_freeze/PTM experimental evidence.csv",
        help="Path to PPI CSV file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/zz/zheng/ptm-mlm/downstream_tasks/outputs/ppi_prediction_freeze",
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
        default=False,
        help="Use sliding window for sequences longer than max_sequence_length."
    )
    parser.add_argument(
        "--window_overlap",
        type=float,
        default=0.5,
        help="Overlap ratio between sliding windows (0.0 to 1.0). "
             "Default 0.5 means 50%% overlap."
    )
    parser.add_argument(
        "--use_esm",
        action="store_true",
        default=False,
        help="If set, load ESM2-15B and use Mamba+ESM2 combination (matching training setup). "
             "If not set, use Mamba-only mode. (Default: False, only valid when model_type='mamba')"
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
    
    # Automatically create embeddings subdirectory in output_dir
    embeddings_dir = os.path.join(args.output_dir, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    print(f"📁 Embeddings will be stored in: {embeddings_dir}")
    
    # Load data
    print("📖 Loading PPI data...")
    train_df, valid_df, test_df = load_ppi_data(
        args.data,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )
    
    # Initialize inference model based on model_type
    print(f"\n🚀 Initializing {args.model_type.upper()} model inference...")
    if args.model_type == "mamba":
        if not MAMBA_AVAILABLE:
            raise ImportError("Mamba inference not available. Please ensure inference.py exists.")
        inferencer = ModelInference(
            args.checkpoint, 
            max_sequence_length=args.max_sequence_length,
            use_esm=args.use_esm
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
        for i in tqdm(range(0, len(binder_sequences), args.batch_size), desc=f"Binder embeddings ({split_name})"):
            batch_sequences = binder_sequences[i:i + args.batch_size]
            batch_embeddings = inferencer.generate_embeddings(
                batch_sequences,
                batch_size=len(batch_sequences),
                return_pooled=True,
                max_sequence_length=args.max_sequence_length
            )
            # batch_embeddings shape: [batch_size, hidden_size]
            for j in range(len(batch_sequences)):
                binder_embeddings_list.append(batch_embeddings[j])
        
        # Generate embeddings for wt sequences
        print(f"\n🔹 Generating wild-type embeddings...")
        wt_embeddings_list = []
        for i in tqdm(range(0, len(wt_sequences), args.batch_size), desc=f"WT embeddings ({split_name})"):
            batch_sequences = wt_sequences[i:i + args.batch_size]
            batch_embeddings = inferencer.generate_embeddings(
                batch_sequences,
                batch_size=len(batch_sequences),
                return_pooled=True,
                max_sequence_length=args.max_sequence_length
            )
            for j in range(len(batch_sequences)):
                wt_embeddings_list.append(batch_embeddings[j])
        
        # Generate embeddings for ptm sequences
        print(f"\n🔹 Generating PTM-modified embeddings...")
        ptm_embeddings_list = []
        for i in tqdm(range(0, len(ptm_sequences), args.batch_size), desc=f"PTM embeddings ({split_name})"):
            batch_sequences = ptm_sequences[i:i + args.batch_size]
            batch_embeddings = inferencer.generate_embeddings(
                batch_sequences,
                batch_size=len(batch_sequences),
                return_pooled=True,
                max_sequence_length=args.max_sequence_length
            )
            for j in range(len(batch_sequences)):
                ptm_embeddings_list.append(batch_embeddings[j])
        
        # Verify consistency
        assert len(binder_embeddings_list) == len(wt_embeddings_list) == len(ptm_embeddings_list) == len(labels), \
            f"Mismatch in {split_name}: binder={len(binder_embeddings_list)}, wt={len(wt_embeddings_list)}, " \
            f"ptm={len(ptm_embeddings_list)}, labels={len(labels)}"
        
        # Save embeddings
        print(f"\n💾 Saving {split_name} embeddings...")
        binder_emb_path = os.path.join(embeddings_dir, f"{split_name}_binder_embeddings.pt")
        wt_emb_path = os.path.join(embeddings_dir, f"{split_name}_wt_embeddings.pt")
        ptm_emb_path = os.path.join(embeddings_dir, f"{split_name}_ptm_embeddings.pt")
        labels_path = os.path.join(embeddings_dir, f"{split_name}_labels.pt")
        
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

