"""
Embedding generator inference class for downstream tasks.
Automatically selects and uses the appropriate model inference class from main_pipeline.
Supports ESM2, ESM-C 600M, and ESM-C 6B models.

This class is used during the embedding generation phase (before training).
It handles windows splitting, metadata generation, and batch processing.
"""
import torch
import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm
from collections import defaultdict

# Add project root to sys.path for importing main_pipeline
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..', '..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from main_pipeline.utils.inference.inference_esm2 import ESM2Inference as BaseESM2Inference
from main_pipeline.utils.inference.inference_esmc import ESMCInference as BaseESMCInference
from main_pipeline.utils.inference.inference_esmc_6b import ESMC6BInference as BaseESMC6BInference
from main_pipeline.getters.tokenizer import PTMTokenizer
import json


class EmbeddingGeneratorInference:
    """
    Embedding generator inference class for downstream tasks.
    Automatically selects the appropriate model inference class from main_pipeline.
    Supports ESM2, ESM-C 600M, and ESM-C 6B models.
    
    This class is used during the embedding generation phase to generate embeddings
    from pretrained models. It wraps the downstream inference classes (which extend
    main_pipeline inference classes with batch processing capabilities).
    """
    
    def __init__(self, model_type: str = "esm2", model_name: Optional[str] = None,
                 device: Optional[str] = None, layer_index: Optional[int] = None,
                 max_sequence_length: Optional[int] = None):
        """
        Initialize embedding generator inference class.
        
        @param model_type: Model type ('esm2', 'esmc', or 'esmc_6b')
        @param model_name: Model name (required for ESM2, e.g., "facebook/esm2_t33_650M_UR50D")
        @param device: Device to run inference on (None for auto-detect)
        @param layer_index: Layer index to extract
                          - ESM2: 0-based (None for last layer)
                          - ESM-C 600M: 0-based (None for last layer)
                          - ESM-C 6B: 1-based (None for last layer)
        @param max_sequence_length: Maximum sequence length (token-level, includes special tokens)
        """
        self.model_type = model_type.lower()
        
        if self.model_type == "esm2":
            if model_name is None:
                model_name = "facebook/esm2_t33_650M_UR50D"
            self.inferencer = BaseESM2Inference(
                model_name=model_name,
                device=device,
                max_sequence_length=max_sequence_length,
                layer_index=layer_index
            )
        elif self.model_type == "esmc":
            self.inferencer = BaseESMCInference(
                device=device,
                layer_index=layer_index
            )
        elif self.model_type == "esmc_6b" or self.model_type == "esmc6b":
            self.inferencer = BaseESMC6BInference(
                device=device,
                layer_index=layer_index
            )
        else:
            raise ValueError(
                f"Unknown model_type: {model_type}. "
                f"Supported types: 'esm2', 'esmc', 'esmc_6b'"
            )
        
        self.hidden_size = self.inferencer.hidden_size
        
        # 🔧 初始化PTMTokenizer用于统计token长度（PTM token被当作1个token）
        self.ptm_tokenizer = PTMTokenizer()
    
    def _create_sliding_windows(self, sequence: str, token_ids: List[int], window_size: int, overlap: float = 0.5):
        """
        Create sliding windows for a long sequence based on token positions.
        
        @param sequence: Input sequence string
        @param token_ids: Token IDs of the sequence (from PTMTokenizer)
        @param window_size: Size of each window (in tokens)
        @param overlap: Overlap ratio between windows (0.0 to 1.0, default 0.5 means 50% overlap)
        @returns: List of (window_sequence, start_token_idx, end_token_idx) tuples
        """
        windows = []
        seq_token_len = len(token_ids)
        step_size = max(1, int(window_size * (1 - overlap)))
        
        start_token = 0
        while start_token < seq_token_len:
            end_token = min(start_token + window_size, seq_token_len)
            
            # 获取窗口对应的token IDs
            window_token_ids = token_ids[start_token:end_token]
            # 解码得到窗口字符串
            window_seq = self.ptm_tokenizer.decode(window_token_ids)
            
            windows.append((window_seq, start_token, end_token))
            start_token += step_size
            
            if start_token < seq_token_len and start_token + window_size > seq_token_len:
                final_start_token = max(0, seq_token_len - window_size)
                if final_start_token > start_token - step_size:
                    final_window_token_ids = token_ids[final_start_token:seq_token_len]
                    final_window_seq = self.ptm_tokenizer.decode(final_window_token_ids)
                    windows.append((final_window_seq, final_start_token, seq_token_len))
                break
        
        return windows
    
    @torch.no_grad()
    def generate_batch_embeddings(self, sequences: List[str], batch_size: int = 32,
                                  max_sequence_length: int = 512,
                                  use_sliding_window: bool = False,
                                  window_overlap: float = 0.5,
                                  layer_indices: List[Optional[int]] = None) -> Tuple[torch.Tensor, List[Dict], List[int]]:
        """
        Generate batch embeddings with unified interface.
        Handles windows splitting, metadata generation, and batch processing.
        
        @param sequences: List of protein sequences (strings)
        @param batch_size: Batch size for inference
        @param max_sequence_length: Maximum sequence length for a single window (default: 512)
        @param use_sliding_window: If True, use sliding window for long sequences
        @param window_overlap: Overlap ratio between windows (0.0 to 1.0)
        @param layer_indices: List of layer indices to extract. If None, uses self.inferencer.layer_index or last layer.
                             For ESM2/ESMC: 0-based. For ESM-C 6B: 1-based.
        @returns: Tuple of (embeddings_tensor, metadata_list, original_lengths)
                 - embeddings_tensor: Large batch tensor of shape (total_items, max_seq_len, embed_dim)
                 - metadata_list: List of metadata dicts
                 - original_lengths: List of original sequence lengths
                 If layer_indices has multiple layers, returns Dict mapping layer_index to (embeddings_tensor, metadata_list, original_lengths)
        """
        original_lengths = [len(seq) for seq in sequences]
        
        # Calculate max embedding length (includes special tokens)
        # ESM2: <cls> + seq + <eos> = seq_len + 2
        # ESM-C: BOS + seq + EOS = seq_len + 2
        max_embedding_len = max_sequence_length + 2
        
        # Step 1: Prepare all items (windows + full sequences) and collect metadata
        all_items = []  # Collect all sequences to process (windows + full sequences)
        metadata_list = []
        
        with tqdm(total=len(sequences), desc="Preparing sequences and windows", unit="seq") as pbar:
            for seq_idx, sequence in enumerate(sequences):
                # 🔧 使用PTMTokenizer统计token长度（PTM token被当作1个token）
                # 这样WT和PTM序列的token长度会一致
                token_ids = self.ptm_tokenizer.encode(sequence, add_special_tokens=False)
                seq_token_len = len(token_ids)  # Token长度（不包含特殊token）
                
                # 🔧 窗口分割基于token长度
                if seq_token_len > max_sequence_length:
                    if use_sliding_window:
                        # Use sliding window - split into windows based on tokens
                        windows = self._create_sliding_windows(sequence, token_ids, max_sequence_length, window_overlap)
                        
                        for window_id, (window_seq, start_token, end_token) in enumerate(windows):
                            all_items.append(window_seq)
                            window_token_len = end_token - start_token  # 窗口的token长度
                            
                            # Prepare metadata (embedding_idx will be set later)
                            metadata_list.append({
                                'sequence_id': seq_idx,
                                'window_id': window_id,
                                'start_idx': start_token,  # Token起始位置
                                'end_idx': end_token,  # Token结束位置
                                'window_len': window_token_len,  # 窗口的token长度
                                'seq_len': seq_token_len,  # 完整序列的token长度
                                'is_window': True,
                                'merge_info': {
                                    'overlap_ratio': window_overlap,
                                    'window_size': max_sequence_length,
                                    'coverage_range': [start_token, end_token]  # Token位置范围
                                }
                            })
                    else:
                        # Truncate sequence based on tokens
                        truncated_token_ids = token_ids[:max_sequence_length]
                        truncated_seq = self.ptm_tokenizer.decode(truncated_token_ids)
                        all_items.append(truncated_seq)
                        
                        truncated_token_len = len(truncated_token_ids)
                        
                        metadata_list.append({
                            'sequence_id': seq_idx,
                            'window_id': -1,
                            'start_idx': 0,
                            'end_idx': truncated_token_len,  # Token结束位置
                            'window_len': truncated_token_len,  # Token长度
                            'seq_len': truncated_token_len,  # Token长度
                            'is_window': False,
                            'merge_info': None
                        })
                else:
                    # Process as single sequence
                    all_items.append(sequence)
                    metadata_list.append({
                        'sequence_id': seq_idx,
                        'window_id': -1,
                        'start_idx': 0,
                        'end_idx': seq_token_len,  # Token结束位置
                        'window_len': seq_token_len,  # Token长度
                        'seq_len': seq_token_len,  # Token长度（PTM token被当作1个token）
                        'is_window': False,
                        'merge_info': None
                    })
                
                pbar.update(1)
        
        # Step 2: Generate embeddings for all items (windows + full sequences) in batches
        # Determine layer indices to extract
        if layer_indices is None:
            layer_indices = [self.inferencer.layer_index]
        
        # Check if we need multiple layers
        multiple_layers = len(layer_indices) > 1
        
        if multiple_layers:
            # Multiple layers: collect embeddings for each layer
            all_embeddings_dict = {layer_idx: [] for layer_idx in layer_indices}
            
            with tqdm(total=len(all_items), desc="Generating embeddings (multiple layers)", unit="item") as pbar:
                for i in range(0, len(all_items), batch_size):
                    batch_seqs = all_items[i:i + batch_size]
                    
                    # Call base inference class method to generate embeddings for multiple layers
                    if self.model_type == "esm2":
                        batch_embeddings_dict = self.inferencer._compute_esm2_embedding(batch_seqs, layer_indices)
                    elif self.model_type == "esmc":
                        batch_embeddings_dict = self.inferencer._compute_esmc_embedding(batch_seqs, layer_indices)
                    elif self.model_type == "esmc_6b" or self.model_type == "esmc6b":
                        # ESM-C 6B supports multiple layers, but requires separate API calls for each layer
                        batch_embeddings_dict = self.inferencer._compute_esmc_embedding(batch_seqs, layer_indices)
                    else:
                        raise ValueError(f"Unknown model_type: {self.model_type}")
                    
                    # Extend embeddings for each layer
                    for layer_idx in layer_indices:
                        all_embeddings_dict[layer_idx].extend(batch_embeddings_dict[layer_idx])
                    
                    pbar.update(len(batch_seqs))
            
            # Step 3: Process each layer
            result_dict = {}
            for layer_idx in layer_indices:
                all_embeddings = all_embeddings_dict[layer_idx]
                embed_dim = all_embeddings[0].shape[1]
                total_items = len(all_embeddings)
                
                # Create large batch tensor with fixed max_embedding_len
                embeddings_tensor = torch.zeros(total_items, max_embedding_len, embed_dim)
                
                # Fill embeddings and update metadata with embedding_idx and valid_length
                for idx, emb in enumerate(all_embeddings):
                    valid_len = emb.shape[0]
                    embeddings_tensor[idx, :valid_len, :] = emb
                    
                    # Update corresponding metadata
                    if idx < len(metadata_list):
                        metadata_list[idx]['embedding_idx'] = idx
                        metadata_list[idx]['valid_length'] = valid_len
                
                result_dict[layer_idx] = (embeddings_tensor, metadata_list.copy(), original_lengths)
            
            return result_dict
        else:
            # Single layer: original behavior
            all_embeddings = []
            
            with tqdm(total=len(all_items), desc="Generating embeddings", unit="item") as pbar:
                for i in range(0, len(all_items), batch_size):
                    batch_seqs = all_items[i:i + batch_size]
                    
                    # Call base inference class method to generate embeddings
                    if self.model_type == "esm2":
                        batch_embeddings = self.inferencer._compute_esm2_embedding(batch_seqs, layer_indices)
                    elif self.model_type == "esmc":
                        batch_embeddings = self.inferencer._compute_esmc_embedding(batch_seqs, layer_indices)
                    elif self.model_type == "esmc_6b" or self.model_type == "esmc6b":
                        batch_embeddings = self.inferencer._compute_esmc_embedding(batch_seqs)
                    else:
                        raise ValueError(f"Unknown model_type: {self.model_type}")
                    
                    all_embeddings.extend(batch_embeddings)
                    pbar.update(len(batch_seqs))
            
            # Step 3: Pad all embeddings to fixed max_embedding_len and create batch tensor
            embed_dim = all_embeddings[0].shape[1]
            total_items = len(all_embeddings)
            
            # Create large batch tensor with fixed max_embedding_len
            embeddings_tensor = torch.zeros(total_items, max_embedding_len, embed_dim)
            
            # Fill embeddings and update metadata with embedding_idx and valid_length
            for idx, emb in enumerate(all_embeddings):
                valid_len = emb.shape[0]
                embeddings_tensor[idx, :valid_len, :] = emb
                
                # Update corresponding metadata
                if idx < len(metadata_list):
                    metadata_list[idx]['embedding_idx'] = idx
                    metadata_list[idx]['valid_length'] = valid_len
            
            return embeddings_tensor, metadata_list, original_lengths
    
    @staticmethod
    def save_metadata(metadata_list: list, filepath: str):
        """
        保存 metadata 到 JSON 文件。
        
        @param metadata_list: List of metadata dicts
        @param filepath: 保存路径
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        
        print(f"✅ Metadata 已保存到 {filepath}")
    
    @staticmethod
    def load_metadata(filepath: str) -> list:
        """
        从 JSON 文件加载 metadata。
        
        @param filepath: 文件路径
        @returns: List of metadata dicts
        """
        with open(filepath, 'r') as f:
            metadata_list = json.load(f)
        
        print(f"✅ Metadata 已从 {filepath} 加载，共 {len(metadata_list)} 条记录")
        return metadata_list
    
    @staticmethod
    def infer_model_type(pretrained_model_name: str) -> str:
        """
        Infer model type from pretrained model name.
        
        @param pretrained_model_name: Pretrained model name
        @return: Model type ('esm2', 'esmc', or 'esmc_6b')
        """
        name_lower = pretrained_model_name.lower()
        if 'esm2' in name_lower:
            return 'esm2'
        elif 'esmc_6b' in name_lower or 'esmc6b' in name_lower or 'esm-c-6b' in name_lower:
            return 'esmc_6b'
        elif 'esmc' in name_lower or 'esm_c' in name_lower or 'esm-c' in name_lower:
            return 'esmc'
        else:
            raise ValueError(
                f"Cannot infer model type from '{pretrained_model_name}'. "
                f"Supported models: ESM2, ESM-C 600M, ESM-C 6B"
            )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Embedding generator inference for ESM models")
    parser.add_argument("--model_type", type=str, default="esm2",
                       choices=["esm2", "esmc", "esmc_6b"],
                       help="Model type")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Model name (for ESM2)")
    parser.add_argument("--sequences", type=str, nargs="+", help="Input sequences")
    parser.add_argument("--output", type=str, help="Output path to save embeddings")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--layer_index", type=int, default=None, help="Layer index")
    
    args = parser.parse_args()
    
    # Initialize embedding generator inference
    inferencer = EmbeddingGeneratorInference(
        model_type=args.model_type,
        model_name=args.model_name,
        layer_index=args.layer_index
    )
    
    # Generate embeddings
    embeddings_tensor, metadata_list, lengths = inferencer.generate_batch_embeddings(
        args.sequences,
        batch_size=args.batch_size
    )
    
    if args.output:
        torch.save(embeddings_tensor, args.output)
        print(f"✅ Embeddings saved to {args.output}")
        
        # Save metadata
        metadata_path = str(args.output).replace('.pt', '_metadata.json')
        EmbeddingGeneratorInference.save_metadata(metadata_list, metadata_path)
    else:
        print(f"📊 Generated embeddings tensor shape: {embeddings_tensor.shape}")
        print(f"📋 Metadata records: {len(metadata_list)}")
        print(f"📏 Original sequences: {len(lengths)}")
