"""
Script to generate embeddings from pre-trained model for PPI prediction.
This script processes training, validation, and test data to generate embeddings for binder and wt sequences.
"""
import torch
import pandas as pd
import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add paths to sys.path for imports
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_file.parent.parent / "inference"))

# Import from downstream_tasks
from downstream_tasks.tasks.ppi_prediction.load_data import (
    load_ppi_data,
    prepare_sequences_and_labels_for_embedding_generation
)
from downstream_tasks.utils.inference.embedding_generator_inference import EmbeddingGeneratorInference


def infer_model_type(pretrained_model_name: str) -> str:
    """
    根据预训练模型名称推断模型类型。

    @param pretrained_model_name: 预训练模型名称
    @return: 模型类型 ('esm2' 或 'esmc')
    """
    return EmbeddingGeneratorInference.infer_model_type(pretrained_model_name)


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
    
    # 🔧 PPI任务特殊处理：
    # - binder和wt：使用final layer（最后一层）生成embeddings（original）
    # - PTM：使用WT序列（original），但用specific layer（layer_index参数指定的层）生成embeddings，留给block处理
    
    # Initialize EmbeddingGeneratorInference for original sequences (final layer)
    print(f"\n🚀 Initializing {model_type.upper()} embedding generator for original sequences (final layer)...")
    original_inferencer = EmbeddingGeneratorInference(
        model_type=model_type,
        model_name=args.pretrained_model_name,
        layer_index=None,  # None表示使用最后一层（final layer）
        max_sequence_length=args.max_sequence_length
    )
    
    # Initialize EmbeddingGeneratorInference for PTM sequences (specific layer)
    # PTM使用WT序列，但用specific layer生成embeddings
    if args.layer_index is not None:
        print(f"\n🚀 Initializing {model_type.upper()} embedding generator for PTM sequences (layer {args.layer_index})...")
        ptm_inferencer = EmbeddingGeneratorInference(
            model_type=model_type,
            model_name=args.pretrained_model_name,
            layer_index=args.layer_index,  # 使用specific layer
            max_sequence_length=args.max_sequence_length
        )
    else:
        # 如果layer_index为None，PTM也使用final layer
        print(f"\n🚀 PTM will use final layer (same as original)...")
        ptm_inferencer = original_inferencer
    
    # Process each split
    for split_name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        print("\n" + "="*70)
        print(f"🔄 Processing {split_name} data...")
        print("="*70)
        
        # 准备序列和标签（只生成原始序列，不生成PTM序列）
        binder_sequences, wt_sequences, labels = prepare_sequences_and_labels_for_embedding_generation(df)
        
        if len(binder_sequences) == 0:
            print(f"⚠️  No valid samples in {split_name} set, skipping...")
            continue
        
        print(f"📊 Processing {len(binder_sequences)} samples...")
        print(f"ℹ️  Binder and WT embeddings: final layer (original)")
        print(f"ℹ️  PTM embeddings: layer {args.layer_index if args.layer_index is not None else 'final'} (using WT sequences)")
        
        # 🔧 优化：根据layer_index决定生成策略
        if args.layer_index is None:
            # layer_index为None：所有embeddings都使用final layer，分别生成
            print(f"\n🔹 Generating binder embeddings (final layer)...")
            binder_embeddings_tensor, binder_metadata_list, _ = original_inferencer.generate_batch_embeddings(
                binder_sequences,
                batch_size=args.batch_size,
                max_sequence_length=args.max_sequence_length if args.max_sequence_length else 512,
                use_sliding_window=args.use_sliding_window,
                window_overlap=args.window_overlap,
                layer_indices=[None]  # None表示final layer，使用新接口
            )
            
            print(f"\n🔹 Generating WT embeddings (final layer, will be reused for PTM)...")
            wt_embeddings_tensor, wt_metadata_list, _ = original_inferencer.generate_batch_embeddings(
                wt_sequences,
                batch_size=args.batch_size,
                max_sequence_length=args.max_sequence_length if args.max_sequence_length else 512,
                use_sliding_window=args.use_sliding_window,
                window_overlap=args.window_overlap,
                layer_indices=[None]  # None表示final layer，使用新接口
            )
            # PTM使用相同的embeddings和metadata
            ptm_embeddings_tensor = wt_embeddings_tensor
            ptm_metadata_list = wt_metadata_list
            print(f"   ✅ PTM embeddings reused from WT (same layer)")
        else:
            # layer_index不是None：一次性生成final layer和specific layer的embeddings
            print(f"\n🔹 Generating embeddings (multiple layers in one pass)...")
            print(f"   - Binder: final layer")
            print(f"   - WT: final layer")
            print(f"   - PTM: layer {args.layer_index}")
            
            # 一次性生成binder的final layer embeddings
            binder_embeddings_tensor, binder_metadata_list, _ = original_inferencer.generate_batch_embeddings(
                binder_sequences,
                batch_size=args.batch_size,
                max_sequence_length=args.max_sequence_length if args.max_sequence_length else 512,
                use_sliding_window=args.use_sliding_window,
                window_overlap=args.window_overlap,
                layer_indices=[None]  # None表示final layer
            )
            
            # 一次性生成WT的final layer和PTM的specific layer embeddings
            layer_indices = [None, args.layer_index]  # None表示final layer
            result_dict = original_inferencer.generate_batch_embeddings(
                wt_sequences,
                batch_size=args.batch_size,
                max_sequence_length=args.max_sequence_length if args.max_sequence_length else 512,
                use_sliding_window=args.use_sliding_window,
                window_overlap=args.window_overlap,
                layer_indices=layer_indices  # 一次性生成两层
            )
            
            # 提取WT embeddings (final layer, layer_index=None)
            wt_embeddings_tensor, wt_metadata_list, _ = result_dict[None]
            
            # 提取PTM embeddings (specific layer)
            ptm_embeddings_tensor, ptm_metadata_list, _ = result_dict[args.layer_index]
        
        # Verify consistency
        binder_num_seqs = max(meta['sequence_id'] for meta in binder_metadata_list) + 1
        wt_num_seqs = max(meta['sequence_id'] for meta in wt_metadata_list) + 1
        ptm_num_seqs = max(meta['sequence_id'] for meta in ptm_metadata_list) + 1
        
        assert binder_num_seqs == wt_num_seqs == ptm_num_seqs == len(labels), \
            f"Mismatch in {split_name}: binder={binder_num_seqs}, wt={wt_num_seqs}, " \
            f"ptm={ptm_num_seqs}, labels={len(labels)}"
        
        # Save batch embeddings and metadata
        print(f"\n💾 Saving {split_name} embeddings...")

        binder_emb_path = os.path.join(task_dir, f"{split_name}_binder_embeddings.pt")
        binder_metadata_path = os.path.join(task_dir, f"{split_name}_binder_embeddings_metadata.json")
        wt_emb_path = os.path.join(task_dir, f"{split_name}_wt_embeddings.pt")
        wt_metadata_path = os.path.join(task_dir, f"{split_name}_wt_embeddings_metadata.json")
        ptm_emb_path = os.path.join(task_dir, f"{split_name}_ptm_embeddings.pt")
        ptm_metadata_path = os.path.join(task_dir, f"{split_name}_ptm_embeddings_metadata.json")
        labels_path = os.path.join(task_dir, f"{split_name}_labels.pt")
        
        torch.save(binder_embeddings_tensor, binder_emb_path)
        EmbeddingGeneratorInference.save_metadata(binder_metadata_list, binder_metadata_path)
        torch.save(wt_embeddings_tensor, wt_emb_path)
        EmbeddingGeneratorInference.save_metadata(wt_metadata_list, wt_metadata_path)
        torch.save(ptm_embeddings_tensor, ptm_emb_path)
        EmbeddingGeneratorInference.save_metadata(ptm_metadata_list, ptm_metadata_path)
        torch.save(labels, labels_path)
        
        print(f"✅ Saved {split_name} data: {len(labels)} samples")
        print(f"   - Binder embeddings (final layer): {len(binder_sequences)} sequences")
        print(f"   - WT embeddings (final layer): {len(wt_sequences)} sequences")
        print(f"   - PTM embeddings (layer {args.layer_index if args.layer_index is not None else 'final'}, using WT sequences): {len(wt_sequences)} sequences")
    
    print("\n" + "="*70)
    print("🎉 Embedding generation completed!")
    print("="*70)


if __name__ == "__main__":
    main()

