"""
Script to pre-generate ESM embeddings for all sequences in the dataset.
This allows faster training by loading pre-computed embeddings instead of computing them on-the-fly.
Supports multi-GPU generation using accelerate.
Supports both ESM2 15B and ESM3 7B models.
"""
import torch
import argparse
import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional, Any

from getters.tokenizer import PTMTokenizer
from utils.config import load_config
from utils.inference.inference_esmc import ESMCInference
from utils.inference.inference_esm2 import ESM2Inference


class EmbeddingBuffer:
    """
    Memory buffer for accumulating embeddings before saving to disk.
    Manages memory usage and flushes when threshold is reached.
    
    @param max_memory_gb: Maximum memory usage in GB before flushing (default: 10GB)
    @param embed_dim: Embedding dimension (will be auto-detected if None)
    """
    
    def __init__(self, max_memory_gb: float = 10.0, embed_dim: Optional[int] = None):
        self.max_memory_gb = max_memory_gb
        self.embed_dim = embed_dim
        # Structure: {AC_ID: {'embeddings': [emb1, emb2, ...], 'ranges': [...], 'ori_seqs': [...], 'ptm_seqs': [...], 'functional_roles': [...]}}
        self.data: Dict[str, Dict[str, List[Any]]] = {}
        # Track window counts for each AC_ID
        self.ac_id_window_counts: Dict[str, int] = {}
        self.ac_id_processed_counts: Dict[str, int] = {}
    
    def add_ac_id(self, ac_id: str, total_windows: int):
        """
        Register a new AC_ID with expected number of windows.
        
        @param ac_id: AC_ID string
        @param total_windows: Total number of windows for this AC_ID
        """
        if ac_id not in self.data:
            self.data[ac_id] = {
                'embeddings': [],
                'ranges': [],
                'ori_seqs': [],
                'ptm_seqs': [],
                'ori_token_ids': [],
                'ptm_token_ids': [],
                'functional_roles': [],
                'functional_role_positions': []
            }
            self.ac_id_window_counts[ac_id] = total_windows
            self.ac_id_processed_counts[ac_id] = 0
    
    def add_window(self, ac_id: str, embedding: np.ndarray, window_range: Tuple[int, int], 
                   ori_seq: str, ptm_seq: str, ori_token_ids: List[int], ptm_token_ids: List[int],
                   functional_role: int = -1, functional_role_position: int = -1):
        """
        Add a window's data to the buffer.
        
        @param ac_id: AC_ID string
        @param embedding: Window embedding (numpy array)
        @param window_range: (start, end) tuple
        @param ori_seq: Original sequence window
        @param ptm_seq: PTM sequence window
        @param ori_token_ids: Original sequence token IDs (pre-tokenized)
        @param ptm_token_ids: PTM sequence token IDs (pre-tokenized)
        @param functional_role: Functional role label index (-1 if not available)
        @param functional_role_position: PTM position (1-based) for this functional_role (-1 if not available)
        """
        if ac_id not in self.data:
            raise ValueError(f"AC_ID {ac_id} not registered. Call add_ac_id() first.")
        
        if self.embed_dim is None:
            self.embed_dim = embedding.shape[1]
        
        self.data[ac_id]['embeddings'].append(embedding)
        self.data[ac_id]['ranges'].append(window_range)
        self.data[ac_id]['ori_seqs'].append(ori_seq)
        self.data[ac_id]['ptm_seqs'].append(ptm_seq)
        self.data[ac_id]['ori_token_ids'].append(ori_token_ids)
        self.data[ac_id]['ptm_token_ids'].append(ptm_token_ids)
        self.data[ac_id]['functional_roles'].append(functional_role)
        self.data[ac_id]['functional_role_positions'].append(functional_role_position)
        self.ac_id_processed_counts[ac_id] += 1
    
    def get_complete_ac_ids(self) -> List[str]:
        """
        Get list of AC_IDs that have all windows processed (complete).
        
        @return: List of complete AC_IDs
        """
        complete = []
        for ac_id in self.data.keys():
            total = self.ac_id_window_counts.get(ac_id, 0)
            processed = self.ac_id_processed_counts.get(ac_id, 0)
            if processed >= total and total > 0:
                complete.append(ac_id)
        return complete
    
    def estimate_memory_gb(self) -> float:
        """
        Estimate current memory usage in GB (all data in buffer).
        
        @return: Estimated memory usage in GB
        """
        total_bytes = 0
        for ac_id, data in self.data.items():
            # Embeddings (实际写入时转换为float16 = 2 bytes)
            for emb in data['embeddings']:
                # 假设emb是float32，但最终写入float16，所以用2字节估算
                total_bytes += emb.shape[0] * emb.shape[1] * 2
            # Ranges (int32 = 4 bytes, 2 per range)
            total_bytes += len(data['ranges']) * 2 * 4
            # Sequences (estimate: average 100 bytes per sequence)
            total_bytes += len(data['ori_seqs']) * 100
            total_bytes += len(data['ptm_seqs']) * 100
            # Dict overhead (rough estimate)
            total_bytes += 1000
        
        return total_bytes / (1024 ** 3)
    
    def estimate_complete_data_size_gb(self) -> float:
        """
        Estimate compressed data size (in GB) for all complete AC_IDs.
        This is used to determine if we've accumulated enough data to flush.
        Uses a compression ratio estimate (gzip typically achieves 3-5x compression for embeddings).
        
        @return: Estimated compressed data size in GB for complete AC_IDs
        """
        total_bytes = 0
        complete_ac_ids = self.get_complete_ac_ids()
        
        for ac_id in complete_ac_ids:
            if ac_id not in self.data:
                continue
            data = self.data[ac_id]
            # Embeddings (float32 = 4 bytes, compressed with gzip ~3x)
            for emb in data['embeddings']:
                total_bytes += emb.nbytes / 3.0  # Estimate compressed size
            # Ranges (int32 = 4 bytes, 2 per range, compressed ~2x)
            total_bytes += len(data['ranges']) * 2 * 4 / 2.0
            # Sequences (strings, compressed ~2x)
            total_bytes += len(data['ori_seqs']) * 100 / 2.0
            total_bytes += len(data['ptm_seqs']) * 100 / 2.0
            # HDF5 overhead (group structure, metadata, etc.)
            total_bytes += 2000
        
        return total_bytes / (1024 ** 3)
    
    def should_flush(self) -> bool:
        """
        Check if buffer should be flushed (memory threshold reached).
        
        @return: True if should flush
        """
        return self.estimate_memory_gb() >= self.max_memory_gb
    
    def pop_complete_data(self) -> Dict[str, Dict[str, List[Any]]]:
        """
        Extract and remove complete AC_IDs from buffer.
        
        @return: Dictionary of complete AC_ID data
        """
        complete_ac_ids = self.get_complete_ac_ids()
        if not complete_ac_ids:
            return {}
        
        complete_data = {}
        for ac_id in complete_ac_ids:
            complete_data[ac_id] = self.data.pop(ac_id)
            self.ac_id_window_counts.pop(ac_id, None)
            self.ac_id_processed_counts.pop(ac_id, None)
        
        return complete_data
    
    def clear(self):
        """Clear all data from buffer."""
        self.data.clear()
        self.ac_id_window_counts.clear()
        self.ac_id_processed_counts.clear()
    
    def size(self) -> int:
        """Get number of AC_IDs in buffer."""
        return len(self.data)


def compute_extraction_windows(sequence_length, max_length):
    """
    Compute extraction windows for sequences of different lengths.
    
    Strategy:
    - sequence < max_length: Extract once (full sequence)
    - max_length <= sequence < 10 * max_length: Uniform extraction with overlap
    - sequence >= 10 * max_length: Uniform extraction without overlap
    
    @param sequence_length: Length of the sequence
    @param max_length: Maximum sequence length for each window
    @return: List of (start_idx, end_idx) tuples for each window
    """
    if max_length is None or sequence_length <= max_length:
        # Case 1: Sequence is shorter than max_length, extract once
        return [(0, sequence_length)]
    
    if sequence_length < 10 * max_length:
        # Case 2: Medium length sequence, uniform extraction with overlap
        # Calculate number of windows to ensure good coverage
        # Target: ~30% overlap between adjacent windows
        num_windows = int(np.ceil(sequence_length / (max_length * 0.7)))
        num_windows = max(2, num_windows)  # At least 2 windows
        
        # Calculate step size to ensure uniform distribution with overlap
        # Formula: step_size = (sequence_length - max_length) / (num_windows - 1)
        # This ensures first window starts at 0 and last window ends at sequence_length
        if num_windows == 1:
            step_size = 0
        else:
            step_size = (sequence_length - max_length) / (num_windows - 1)
        
        windows = []
        for i in range(num_windows):
            start_idx = int(i * step_size)
            end_idx = min(start_idx + max_length, sequence_length)
            if end_idx > start_idx:  # Ensure valid window
                windows.append((start_idx, end_idx))
        
        # Ensure the last window covers the end of the sequence
        if windows and windows[-1][1] < sequence_length:
            windows[-1] = (max(0, sequence_length - max_length), sequence_length)
        
        return windows
    else:
        # Case 3: Long sequence, uniform extraction without overlap
        num_windows = int(np.ceil(sequence_length / max_length))
        windows = []
        for i in range(num_windows):
            start_idx = i * max_length
            end_idx = min(start_idx + max_length, sequence_length)
            if end_idx > start_idx:  # Ensure valid window
                windows.append((start_idx, end_idx))
        return windows


def append_chunk_to_memmap(
    chunk_data_by_ac_id: Dict[str, Dict[str, List[Any]]],
    memmap_arrays: Dict[str, np.memmap],
    tokenizer: PTMTokenizer,
    embed_dim: int,
    max_seq_len: int,
    protein_id_to_idx: Dict[str, int],
    global_idx_counter: List[int]
) -> int:
    """
    Append a chunk of embeddings to memmap files.
    
    @param chunk_data_by_ac_id: Dict mapping AC_ID to window data (with pre-tokenized token_ids)
    @param memmap_arrays: Dict with keys 'embeddings', 'orig_tokens', 'ptm_tokens', 'range_data', 'meta_id'
    @param tokenizer: Tokenizer instance (only used for pad_token_id)
    @param embed_dim: Embedding dimension
    @param max_seq_len: Maximum sequence length (512)
    @param protein_id_to_idx: Mapping from protein ID to index
    @param global_idx_counter: List with single int element for in-memory counter [counter_value]
    @return: Number of samples written
    """
    if not chunk_data_by_ac_id:
        return 0
    
    pad_token_id = tokenizer.pad_token_id
    num_written = 0
    
    for ac_id, data in chunk_data_by_ac_id.items():
        num_windows = len(data['embeddings'])
        protein_idx = protein_id_to_idx.get(ac_id, -1)
        
        for window_idx in range(num_windows):
            global_idx = global_idx_counter[0]
            global_idx_counter[0] += 1

            # 🚀 Performance: Use pre-tokenized token_ids (no tokenization overhead)
            # Convert Python list to numpy array for efficient assignment
            ori_token_ids = data['ori_token_ids'][window_idx]
            ptm_token_ids = data['ptm_token_ids'][window_idx]
            
            # Convert to numpy array (int32) for efficient memmap assignment
            ori_ids = np.asarray(ori_token_ids, dtype=np.int32)
            ptm_ids = np.asarray(ptm_token_ids, dtype=np.int32)
            
            # Check lengths - raise error if exceeds max_seq_len (no truncation)
            ori_seq_len = len(ori_token_ids)
            ptm_seq_len = len(ptm_token_ids)
            
            if ori_seq_len > max_seq_len:
                raise ValueError(
                    f"Original token sequence length {ori_seq_len} exceeds max_seq_len {max_seq_len} "
                    f"for AC_ID {ac_id}, window {window_idx}. Sequence should be truncated before tokenization."
                )
            if ptm_seq_len > max_seq_len:
                raise ValueError(
                    f"PTM token sequence length {ptm_seq_len} exceeds max_seq_len {max_seq_len} "
                    f"for AC_ID {ac_id}, window {window_idx}. Sequence should be truncated before tokenization."
                )

            # 🚀 Performance: Directly write to memmap slice (safer and faster than buffer copy)
            # 2. Original sequence tokens
            orig_tokens_dst = memmap_arrays['orig_tokens'][global_idx]
            orig_tokens_dst[:] = pad_token_id  # Fill with pad token
            orig_tokens_dst[:ori_seq_len] = ori_ids[:ori_seq_len]

            # 3. PTM sequence tokens
            ptm_tokens_dst = memmap_arrays['ptm_tokens'][global_idx]
            ptm_tokens_dst[:] = pad_token_id  # Fill with pad token
            ptm_tokens_dst[:ptm_seq_len] = ptm_ids[:ptm_seq_len]

            # 🚀 Performance: Embeddings - check length and raise error if exceeds max_seq_len
            # Embeddings are already converted to float16 in torch and on CPU
            emb = data['embeddings'][window_idx]  # Shape: (emb_len, embed_dim), already float16
            emb_len = emb.shape[0]
            
            if emb_len > max_seq_len:
                raise ValueError(
                    f"Embedding length {emb_len} exceeds max_seq_len {max_seq_len} "
                    f"for AC_ID {ac_id}, window {window_idx}. Sequence should be truncated before inference."
                )
            
            embeddings_dst = memmap_arrays['embeddings'][global_idx]
            embeddings_dst[:] = 0  # Fill with zeros
            embeddings_dst[:emb_len] = emb[:emb_len]
            
            # 4. Range data: [start, end, valid_len]
            # valid_len is the actual embedding length (includes special tokens)
            start, end = data['ranges'][window_idx]
            valid_len = emb_len  # Use actual embedding length
            memmap_arrays['range_data'][global_idx] = [start, end, valid_len]
            
            # 5. Meta ID: protein index
            memmap_arrays['meta_id'][global_idx] = protein_idx
            
            # 6. Functional role label
            functional_role = data.get('functional_roles', [-1])[window_idx] if 'functional_roles' in data else -1
            if 'functional_role' in memmap_arrays:
                memmap_arrays['functional_role'][global_idx] = functional_role
            
            # 7. Functional role position (PTM position, 1-based)
            functional_role_position = data.get('functional_role_positions', [-1])[window_idx] if 'functional_role_positions' in data else -1
            if 'functional_role_position' in memmap_arrays:
                memmap_arrays['functional_role_position'][global_idx] = functional_role_position
            
            num_written += 1
    
    # 🚀 Performance: Don't flush here - flush will be done in flush_buffer_to_disk() after all writes
    return num_written


def extract_windows_for_sequence(
    ac_id: str,
    ori_seq_full: str,
    ptm_seq_full: str,
    max_sequence_length: Optional[int]
) -> List[Dict[str, Any]]:
    """
    Extract all windows for a single sequence.

    IMPORTANT: All ranges are based on character positions in the original sequence,
    not token positions. This ensures consistency with the new dataset loading method.
    ESM embeddings are always generated from original sequence (without PTM annotations).

    @param ac_id: AC_ID string
    @param ori_seq_full: Full original sequence (without PTM annotations)
    @param ptm_seq_full: Full PTM sequence (with PTM annotations)
    @param max_sequence_length: Maximum sequence length (in characters, not tokens)
    @return: List of window info dictionaries with:
        - 'char_start', 'char_end': Character positions in original sequence (for range)
        - 'window_seq': Sequence for ESM input (always original sequence)
        - 'ori_seq_window': Original sequence window
        - 'ptm_seq_window': PTM sequence window
    """
    windows = []
    
    # Always compute windows based on character positions in original sequence
    # This ensures range consistency with the new dataset loading method
    ori_seq_len = len(ori_seq_full)

    # Convert token-level limit to character-level limit for window extraction
    max_char_length = None
    if max_sequence_length:
        max_char_length = max_sequence_length - 2  # Subtract <cls> and <eos> tokens

    window_ranges = compute_extraction_windows(ori_seq_len, max_char_length)
    
    for win_idx, (char_start, char_end) in enumerate(window_ranges):
        ori_seq_window = ori_seq_full[char_start:char_end]
        ptm_seq_window = ptm_seq_full[char_start:char_end] if len(ptm_seq_full) >= char_end else ptm_seq_full[char_start:]
        window_seq = ori_seq_window  # Always use original sequence for ESM input
        window_len = len(window_seq)

        windows.append({
            'ac_id': ac_id,
            'window_idx': win_idx,
            'char_start': char_start,  # Character position in original sequence
            'char_end': char_end,      # Character position in original sequence
            'window_start': char_start,  # Alias for compatibility
            'window_end': char_end,      # Alias for compatibility
            'window_len': window_len,    # Length of window_seq (for ESM)
            'window_seq': window_seq,   # Sequence for ESM input (always original)
            'ori_seq_window': ori_seq_window,  # Original sequence window
            'ptm_seq_window': ptm_seq_window,  # PTM sequence window
        })
    
    return windows



def process_window_batch(
    window_batch: List[Dict[str, Any]],
    esm_inference: Any,
    model_type: str
) -> List[np.ndarray]:
    """
    Process a batch of windows and extract embeddings using inference classes.
    Uses the new unified interface with layer_indices parameter (single layer).
    
    @param window_batch: List of window info dictionaries
    @param esm_inference: Inference class instance (ESMCInference or ESM2Inference) - required
    @param model_type: Model type ('esm2', 'esmc', or 'esmc_6b')
    @return: List of embeddings (numpy arrays), each with shape (seq_len, embed_dim)
    """
    sequences = [w['window_seq'] for w in window_batch]
    
    # Use new unified interface with layer_indices parameter (single layer)
    # Pass single layer index to use the new interface
    layer_indices = [esm_inference.layer_index]
    
    if model_type == "esmc":
        batch_embeddings = esm_inference._compute_esmc_embedding(sequences, layer_indices)
    elif model_type == "esm2":
        batch_embeddings = esm_inference._compute_esm2_embedding(sequences, layer_indices)
    elif model_type == "esmc_6b" or model_type == "esmc6b":
        batch_embeddings = esm_inference._compute_esmc_embedding(sequences, layer_indices)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Supported: 'esmc', 'esm2', 'esmc_6b'")
    
    # Convert to numpy arrays (embeddings are already on CPU from inference classes)
    embeddings = []
    for emb in batch_embeddings:
        # Convert to float16 in torch before numpy conversion (faster)
        emb_np = emb.to(torch.float16).cpu().numpy()
        embeddings.append(emb_np)
    
    return embeddings


def flush_buffer_to_disk(
    buffer: EmbeddingBuffer,
    memmap_arrays: Dict[str, np.memmap],
    tokenizer: PTMTokenizer,
    embed_dim: int,
    max_seq_len: int,
    protein_id_to_idx: Dict[str, int],
    global_idx_counter: List[int],
    output_path: Path,
    chunk_idx: int
) -> int:
    """
    Flush complete AC_IDs from buffer to memmap files.
    
    @param buffer: EmbeddingBuffer instance
    @param memmap_arrays: Dict with memmap arrays
    @param tokenizer: Tokenizer instance
    @param embed_dim: Embedding dimension
    @param max_seq_len: Maximum sequence length (512)
    @param protein_id_to_idx: Mapping from protein ID to index
    @param global_idx_counter: List with single int element for in-memory counter [counter_value]
    @param output_path: Output directory path
    @param chunk_idx: Current chunk index
    @return: Number of samples written
    """
    complete_data = buffer.pop_complete_data()
    
    if not complete_data:
        return 0
    
    # Save to memmap files
    num_written = append_chunk_to_memmap(
        complete_data, memmap_arrays, tokenizer, embed_dim, max_seq_len,
        protein_id_to_idx, global_idx_counter
    )

    # 🚀 Performance: Flush all arrays once after all writes (much faster than flushing in append_chunk_to_memmap)
    for arr in memmap_arrays.values():
        arr.flush()

    num_saved_ac_ids = len(complete_data)
    print(f"💾 Saved chunk {chunk_idx} ({num_saved_ac_ids} complete AC_IDs, "
          f"{num_written} samples, buffer memory: {buffer.estimate_memory_gb():.2f}GB)")

    return num_written


def generate_embeddings_for_dataset(
    dataset_data: List[Dict[str, str]],
    tokenizer: PTMTokenizer,
    esm_inference=None,
    output_dir=None,
    embed_dim: int = None,
    batch_size=8,
    max_sequence_length=512,
    chunk_size_gb=100,
    repr_layer=33,
    model_type="esm2",
    ac_id_to_ptm_info: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    extract_functional_role: bool = False,
):
    """
    Generate ESM embeddings for all sequences in the dataset.
    Directly outputs memmap format compatible with reorganize_h5_to_memmap.py.

    Output format:
    - orig_tokens.dat [N, 512] int32 - 原始序列 token IDs (包含<cls>和<eos>特殊tokens)
    - ptm_tokens.dat [N, 512] int32 - PTM 序列 token IDs (包含<cls>和<eos>特殊tokens)
    - embeddings.dat [N, 512, embed_dim] float16 - 完整序列 embeddings (包含<cls>和<eos>tokens)
    - range.dat [N, 3] int32 - 范围信息 [start, end, valid_len]
        - start/end: 字符位置索引 (0-based, 基于原始序列)
        - valid_len: token数量 (包含<cls>和<eos>特殊tokens)
    - meta_id.dat [N] int64 - 蛋白质 ID 索引
    - functional_role.dat [N] int8 - 功能角色标签索引 (-1=无标签, 0=Enhancing, 1=Impairing, 2=Associated)
    - functional_role_position.dat [N] int32 - 功能角色对应的PTM位置 (1-based, -1=无位置)
    - meta_mapping.json - 蛋白质ID到索引的映射和功能角色映射

    @param dataset_data: List of dicts with:
        - 'ac_id': AC_ID string
        - 'ptm_sequence': PTM sequence (with PTM annotations)
        - 'original_sequence': Original sequence (without PTM annotations, optional)
    @param tokenizer: Tokenizer instance.
    @param esm_inference: Inference class instance (ESMCInference or ESM2Inference) - required.
    @param output_dir: Directory to save embeddings (will create memmap files).
    @param embed_dim: Embedding dimension.
    @param batch_size: Batch size for embedding generation (default: 8).
    @param max_sequence_length: Maximum character length for window extraction (training token limit - 2). If None, no limit.
    @param chunk_size_gb: Target chunk size in GB before flushing to disk (default: 100GB).
    @param repr_layer: Representation layer index to extract embeddings from (default: 33).
    @param model_type: Model type ('esm2', 'esm3', or 'esmc') (default: 'esm2').
    @param ac_id_to_ptm_info: Optional dictionary mapping AC_ID to list of PTM info dicts with 'position' (1-based) and 'functional_role' keys.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process entire dataset
    print(f"\n📊 Processing dataset ({len(dataset_data)} samples)...")
    
    output_path = Path(output_dir)
    
    # Functional role mapping
    functional_role_to_idx = {"Enhancing": 0, "Impairing": 1, "Associated": 2}
    idx_to_functional_role = ["Enhancing", "Impairing", "Associated"]
    
    # Use provided ac_id_to_ptm_info or initialize empty
    if ac_id_to_ptm_info is None:
        ac_id_to_ptm_info = {}

    if ac_id_to_ptm_info:
        print(f"✅ Using functional roles for {len(ac_id_to_ptm_info)} proteins")
    
    # Step 1: Count total samples and create mappings
    print(f"\n📊 Step 1: Counting total samples...")

    total_samples = len(dataset_data)
    all_ac_ids = []

    for i in range(total_samples):
        sample = dataset_data[i]
        ac_id = sample.get("ac_id", f"sample_{i}")
        all_ac_ids.append(ac_id)

    # Get unique AC_IDs and create mapping
    unique_ac_ids = sorted(list(set(all_ac_ids)))
    protein_id_to_idx = {ac_id: idx for idx, ac_id in enumerate(unique_ac_ids)}
    protein_ids_list = unique_ac_ids

    # Count total windows
    max_char_length = None
    if max_sequence_length:
        max_char_length = max_sequence_length - 2  # Subtract <cls> and <eos> tokens (same as extract_windows_for_sequence)
    
    total_windows = 0
    for sample in dataset_data:
        ori_seq = sample.get("original_sequence", None)
        if ori_seq is None:
            ori_seq = sample.get("ptm_sequence", "")
        ori_seq_len = len(ori_seq)
        window_ranges = compute_extraction_windows(ori_seq_len, max_char_length)
        total_windows += len(window_ranges)

    print(f"✅ Total proteins: {len(unique_ac_ids):,}")
    print(f"✅ Total windows (samples): {total_windows:,}")
    print(f"✅ Embedding dimension: {embed_dim}")
    print(f"✅ Sequence length: {max_sequence_length}")

    # Step 2: Create memmap files
    print(f"\n📝 Step 2: Creating memmap files...")

    orig_tokens_path = output_path / 'orig_tokens.dat'
    ptm_tokens_path = output_path / 'ptm_tokens.dat'
    embeddings_path = output_path / 'embeddings.dat'
    range_path = output_path / 'range.dat'
    meta_id_path = output_path / 'meta_id.dat'
    functional_role_path = output_path / 'functional_role.dat'
    functional_role_position_path = output_path / 'functional_role_position.dat'
    counter_path = output_path / 'counter.txt'
    
    # Ensure max_sequence_length has a value for memmap shape
    if max_sequence_length is None:
        max_sequence_length = 512  # Default to 512 if not specified
        print(f"⚠️  max_sequence_length was None, using default: {max_sequence_length}")
    
    memmap_arrays = {
        'orig_tokens': np.memmap(orig_tokens_path, dtype=np.int32, mode='w+', shape=(total_windows, max_sequence_length)),
        'ptm_tokens': np.memmap(ptm_tokens_path, dtype=np.int32, mode='w+', shape=(total_windows, max_sequence_length)),
        'embeddings': np.memmap(embeddings_path, dtype=np.float16, mode='w+', shape=(total_windows, max_sequence_length, embed_dim)),
        'range_data': np.memmap(range_path, dtype=np.int32, mode='w+', shape=(total_windows, 3)),
        'meta_id': np.memmap(meta_id_path, dtype=np.int64, mode='w+', shape=(total_windows,)),
        'functional_role': np.memmap(functional_role_path, dtype=np.int8, mode='w+', shape=(total_windows,)),
        'functional_role_position': np.memmap(functional_role_position_path, dtype=np.int32, mode='w+', shape=(total_windows,))
    }

    # 🔧 Fix: Use in-memory counter (much faster for single-process, no file I/O overhead)
    global_idx_counter = [0]  # Use list to allow modification in append_chunk_to_memmap

    print(f"✅ Created memmap files")
    
    # Process all samples
    total_samples = len(dataset_data)
    start_idx = 0
    end_idx = total_samples

    # Estimate chunk size in samples
    estimated_bytes_per_sample = 2 * 1024 * 1024  # 2MB per sample (compressed estimate)
    chunk_size_samples = int(chunk_size_gb * 1024 * 1024 * 1024 / estimated_bytes_per_sample)
    chunk_size_samples = max(1000, min(chunk_size_samples, 50000))  # Between 1K and 50K samples

    print(f"\n🔄 Step 3: Processing and writing embeddings...")
    print(f"📦 Using chunk size: ~{chunk_size_samples} samples per chunk (~{chunk_size_gb}GB)")
    
    # Initialize memory buffer
    # max_memory_gb: Safety threshold - flush if memory exceeds this (even if chunk_size not reached)
    max_memory_gb = chunk_size_gb * 1.1  # Allow buffer to grow up to 1.1x chunk_size for safety
    buffer = EmbeddingBuffer(max_memory_gb=max_memory_gb, embed_dim=embed_dim)
    chunk_idx = 0
    processed_sequences = 0
    
    # Process in batches
    pbar = tqdm(range(0, end_idx - start_idx, batch_size), desc="Generating embeddings")
    
    for batch_offset in pbar:
        batch_samples = []
        batch_indices = []
        
        batch_start = start_idx + batch_offset
        batch_end = min(batch_start + batch_size, end_idx)
        
        for j in range(batch_start, batch_end):
            sample = dataset_data[j]
            batch_samples.append(sample)
            batch_indices.append(j)
        
        # Get sequences and unique_ids (AC_IDs)
        ac_ids = [s.get("ac_id", f"sample_{j}") for j, s in zip(batch_indices, batch_samples)]
        
        # Get PTM sequences (always available)
        ptm_sequences = [s.get("ptm_sequence", "") for s in batch_samples]
        
        # Get original sequences (if available, otherwise use PTM sequences as fallback)
        ori_sequences = []
        for s in batch_samples:
            ori_seq = s.get("original_sequence", None)
            if ori_seq is None:
                # Fallback to PTM sequence if original_sequence not provided
                ori_seq = s.get("ptm_sequence", "")
            ori_sequences.append(ori_seq)
        
        # Always use original sequences for ESM input (PTM信息由下游模型处理)
        sequences_for_esm = ori_sequences
        if batch_offset == 0:
            print(f"📝 Using original sequences (without PTM) for ESM input")
        
        # Step 1: Extract windows for all sequences in batch
        all_windows = []
        for idx, ac_id in enumerate(ac_ids):
            # Extract windows for this sequence
            ori_seq_full = ori_sequences[idx]
            ptm_seq_full = ptm_sequences[idx]

            windows = extract_windows_for_sequence(
                ac_id=ac_id,
                ori_seq_full=ori_seq_full,
                ptm_seq_full=ptm_seq_full,
                max_sequence_length=max_sequence_length
            )

            # Filter windows based on functional role mode
            if extract_functional_role:
                # Only include windows that have functional role labels
                filtered_windows = []
                for window in windows:
                    char_start = window['char_start']
                    char_end = window['char_end']

                    # Check if this window has a functional role
                    has_functional_role = False
                    if ac_id in ac_id_to_ptm_info:
                        for ptm_info in ac_id_to_ptm_info[ac_id]:
                            # position is 1-based, convert to 0-based for comparison
                            pos_0based = ptm_info['position'] - 1
                            if char_start <= pos_0based < char_end:
                                has_functional_role = True
                                break

                    if has_functional_role:
                        filtered_windows.append(window)

                # Register AC_ID in buffer only if we have windows with functional roles
                if filtered_windows:
                    buffer.add_ac_id(ac_id, len(filtered_windows))
                    all_windows.extend(filtered_windows)
            else:
                # Include all windows when not in functional role mode
                if windows:
                    buffer.add_ac_id(ac_id, len(windows))
                    all_windows.extend(windows)
        
        if not all_windows:
            continue
        
        # Step 2: Sort windows by length for efficient batching
        # Sort by window_len (ESM input length) for efficient batching
        all_windows_sorted = sorted(all_windows, key=lambda w: w['window_len'])
        
        # Step 3: Process windows in batches and accumulate in buffer
        for window_batch_start in range(0, len(all_windows_sorted), batch_size):
            window_batch_end = min(window_batch_start + batch_size, len(all_windows_sorted))
            window_batch = all_windows_sorted[window_batch_start:window_batch_end]
            
            # Process batch and get embeddings
            embeddings = process_window_batch(
                window_batch=window_batch,
                esm_inference=esm_inference,
                model_type=model_type
            )
            
            # Verify embed_dim matches (should already be set from model)
            if embeddings:
                assert embeddings[0].ndim == 2, f"Expected 2D embedding (seq_len, embed_dim), got shape {embeddings[0].shape}"
                actual_dim = embeddings[0].shape[1]
                assert actual_dim == embed_dim, f"Embedding dimension mismatch. Expected {embed_dim}, got {actual_dim}. Shape: {embeddings[0].shape}"
                
                if embed_dim is None:
                    # Set embed_dim from first embedding
                    embed_dim = actual_dim
                    buffer.embed_dim = embed_dim
            
            # 🚀 Performance: Batch tokenize all windows in this batch
            ori_seq_batch = [w['ori_seq_window'] for w in window_batch]
            ptm_seq_batch = [w['ptm_seq_window'] for w in window_batch]
            
            # Batch encode if tokenizer supports it, otherwise encode individually
            if hasattr(tokenizer, 'encode_batch'):
                ori_token_ids_batch = tokenizer.encode_batch(ori_seq_batch, add_special_tokens=True)
                ptm_token_ids_batch = tokenizer.encode_batch(ptm_seq_batch, add_special_tokens=True)
            else:
                # Fallback: encode individually (still better than doing it in flush)
                ori_token_ids_batch = [tokenizer.encode(seq, add_special_tokens=True) for seq in ori_seq_batch]
                ptm_token_ids_batch = [tokenizer.encode(seq, add_special_tokens=True) for seq in ptm_seq_batch]
            
            # Add windows to buffer with tokenized IDs
            # Note: Some AC_IDs may have been flushed, so check if they still exist in buffer
            for window_info, embedding, ori_token_ids, ptm_token_ids in zip(window_batch, embeddings, ori_token_ids_batch, ptm_token_ids_batch):
                ac_id = window_info['ac_id']
                
                # Skip if AC_ID was already flushed (race condition: flush happened between window extraction and processing)
                if ac_id not in buffer.data:
                    continue
                
                # Use character positions for range (always based on original sequence)
                char_start = window_info.get('char_start', window_info['window_start'])
                char_end = window_info.get('char_end', window_info['window_end'])
                
                # Match functional role: check if any PTM position is within this window
                # position is 1-based, char_start/char_end are 0-based
                functional_role = -1  # -1 means no label
                functional_role_position = -1  # -1 means no position
                if ac_id in ac_id_to_ptm_info:
                    for ptm_info in ac_id_to_ptm_info[ac_id]:
                        # position is 1-based, convert to 0-based for comparison
                        pos_0based = ptm_info['position'] - 1
                        if char_start <= pos_0based < char_end:
                            functional_role = ptm_info['functional_role']
                            functional_role_position = ptm_info['position']  # Store 1-based position
                            break  # Use first matching PTM site

                # Only add window if it has a functional role label (when in functional role mode)
                should_add_window = True
                if extract_functional_role:
                    should_add_window = (functional_role != -1)

                if should_add_window:
                    buffer.add_window(
                        ac_id=ac_id,
                        embedding=embedding,
                        window_range=(char_start, char_end),  # Character positions in original sequence
                        ori_seq=window_info['ori_seq_window'],
                        ptm_seq=window_info['ptm_seq_window'],
                        ori_token_ids=ori_token_ids,  # Batch-tokenized token IDs
                        ptm_token_ids=ptm_token_ids,  # Batch-tokenized token IDs
                        functional_role=functional_role,
                        functional_role_position=functional_role_position
                    )
            
            # 🚀 Performance: Update progress file periodically (almost zero overhead)
            # Update counter.txt every 10000 samples for progress visualization
            if global_idx_counter[0] % 10 == 0:
                with open(counter_path, 'w') as f:
                    f.write(str(global_idx_counter[0]))
            
            # Step 4: Check if buffer should be flushed
            # Flush when:
            # 1. Complete AC_IDs data size reaches chunk_size_gb (target chunk size)
            # 2. Memory pressure too high (safety mechanism - flush even if chunk_size not reached)
            complete_ac_ids = buffer.get_complete_ac_ids()
            complete_data_size_gb = buffer.estimate_complete_data_size_gb()
            memory_pressure = buffer.should_flush()
            
            # Determine if we should flush
            should_flush_now = False
            flush_reason = ""
            
            if len(complete_ac_ids) > 0:
                # Check if we've accumulated enough data to reach chunk_size_gb
                if complete_data_size_gb >= chunk_size_gb:
                    should_flush_now = True
                    flush_reason = f"reached chunk size ({complete_data_size_gb:.2f}GB >= {chunk_size_gb}GB)"
                # Safety: flush if memory pressure is too high (even if chunk_size not reached)
                elif memory_pressure:
                    should_flush_now = True
                    flush_reason = f"memory pressure ({buffer.estimate_memory_gb():.2f}GB >= {buffer.max_memory_gb}GB)"
            
            if should_flush_now:
                print(f"💾 Flushing buffer: {flush_reason} ({len(complete_ac_ids)} complete AC_IDs, "
                      f"estimated size: {complete_data_size_gb:.2f}GB)")

                num_written = flush_buffer_to_disk(
                    buffer=buffer,
                    memmap_arrays=memmap_arrays,
                    tokenizer=tokenizer,
                    embed_dim=embed_dim,
                    max_seq_len=max_sequence_length,
                    protein_id_to_idx=protein_id_to_idx,
                    global_idx_counter=global_idx_counter,
                    output_path=output_path,
                    chunk_idx=chunk_idx
                )
                if num_written > 0:
                    chunk_idx += 1
                    processed_sequences += num_written

                # Log remaining buffer state
                remaining_complete = len(buffer.get_complete_ac_ids())
                remaining_size = buffer.estimate_complete_data_size_gb()
                if remaining_complete > 0:
                    print(f"   📊 Remaining in buffer: {remaining_complete} complete AC_IDs "
                          f"({remaining_size:.2f}GB), "
                          f"buffer memory: {buffer.estimate_memory_gb():.2f}GB")
        
    # Flush remaining complete AC_IDs from buffer
    if buffer.size() > 0:
        num_written = flush_buffer_to_disk(
            buffer=buffer,
            memmap_arrays=memmap_arrays,
            tokenizer=tokenizer,
            embed_dim=embed_dim,
            max_seq_len=max_sequence_length,
            protein_id_to_idx=protein_id_to_idx,
            global_idx_counter=global_idx_counter,
            output_path=output_path,
            chunk_idx=chunk_idx
        )
        if num_written > 0:
            processed_sequences += num_written
            print(f"💾 Saved final chunk ({num_written} samples)")

        # Warn if there are incomplete AC_IDs remaining
        remaining_ac_ids = buffer.get_complete_ac_ids()
        if not remaining_ac_ids and buffer.size() > 0:
            incomplete_ac_ids = list(buffer.data.keys())
            if incomplete_ac_ids:
                print(f"⚠️  Warning: {len(incomplete_ac_ids)} incomplete AC_IDs not saved: {incomplete_ac_ids[:5]}")

    # Step 4: Save meta_mapping.json
    print(f"\n💾 Step 4: Saving metadata...")

    # 🔧 Fix: Get total samples from in-memory counter and write back to file
    total_samples_written = global_idx_counter[0]
    with open(counter_path, 'w') as f:
        f.write(str(total_samples_written))
        
        meta_mapping = {
            'protein_id_to_idx': protein_id_to_idx,
            'idx_to_protein_id': protein_ids_list,
            'total_samples': total_samples_written,
            'embedding_dim': embed_dim,
            'sequence_length': max_sequence_length,
            'orig_tokens_dtype': 'int32',
            'ptm_tokens_dtype': 'int32',
            'embeddings_dtype': 'float16',
            'range_dtype': 'int32',
            'range_shape': [3],  # [start, end, length]
            'range_unit': 'char',
            'range_description': 'range[i] = [start, end, valid_len] 表示样本 i 在原始蛋白质中的位置范围和有效长度',
            'range_note': 'start/end are character indices in original sequence (0-based), NOT token indices. valid_len is token count (includes <cls> and <eos> special tokens).',
            'valid_len_description': 'valid_len 表示完整序列的token数量(包含<cls>和<eos>): mask = torch.arange(512)[None, :] < valid_len[:, None]',
            'meta_id_dtype': 'int64',
            'meta_id_description': 'meta_id[i] 表示样本 i 对应的蛋白质索引，可通过 idx_to_protein_id 获取蛋白质ID',
            'functional_role_dtype': 'int8',
            'functional_role_to_idx': functional_role_to_idx,
            'idx_to_functional_role': idx_to_functional_role,
            'functional_role_description': 'functional_role[i] 表示样本 i 的功能角色标签索引 (-1 表示无标签，0=Enhancing, 1=Impairing, 2=Associated)',
            'functional_role_position_dtype': 'int32',
            'functional_role_position_description': 'functional_role_position[i] 表示样本 i 的功能角色对应的PTM位置 (1-based, -1 表示无位置)'
        }
        
        with open(output_path / 'meta_mapping.json', 'w') as f:
            json.dump(meta_mapping, f, indent=2)

        print(f"✅ Meta mapping saved to: {output_path / 'meta_mapping.json'}")

        # Print file sizes
        print(f"\n📊 File sizes:")
        files = {
            'orig_tokens.dat': output_path / 'orig_tokens.dat',
            'ptm_tokens.dat': output_path / 'ptm_tokens.dat',
            'embeddings.dat': output_path / 'embeddings.dat',
            'range.dat': output_path / 'range.dat',
            'meta_id.dat': output_path / 'meta_id.dat',
            'functional_role.dat': output_path / 'functional_role.dat',
            'functional_role_position.dat': output_path / 'functional_role_position.dat',
        }

        total_size = 0
        for name, path in files.items():
            if path.exists():
                size = path.stat().st_size / (1024**3)  # GB
                total_size += size
                print(f"  {name}: {size:.2f} GB")

        print(f"  总计: {total_size:.2f} GB")
        print(f"✨ Processed {processed_sequences} samples")
        print(f"\n✅ All embeddings generated and saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Pre-generate ESM embeddings for dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="embeddings",
        help="Directory to save embeddings",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--chunk_size_gb",
        type=float,
        default=100,
        help="Target chunk size in GB for saving embeddings (default: 100GB)",
    )
    parser.add_argument(
        "--functional_role",
        action="store_true",
        help="Extract functional_role labels from dataset CSV (requires 'functional_role' and 'position' columns)",
    )
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    data_config = cfg["dataset"]
    train_config = cfg.get("training", {})
    preprocess_config = cfg.get("preprocess", {})
    
    # Get dataset location from preprocess config (required)
    dataset_location = preprocess_config.get("dataset_location")
    if not dataset_location:
        raise ValueError("❌ preprocess.dataset_location not found in config")
    
    # Get model and repr_layer from training config
    model_name = train_config.get("esm_model")
    if not model_name:
        raise ValueError("❌ training.esm_model not found in config")
    repr_layer = train_config.get("repr_layer")
    
    # Single GPU processing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load CSV data directly
    import pandas as pd

    # Load dataset CSV
    print(f"📖 Loading data from CSV: {dataset_location}")

    df = pd.read_csv(dataset_location)
    df.drop(df.filter(regex="^Unnamed"), axis=1, inplace=True)
    
    # Check if functional_role extraction is requested
    extract_functional_role = args.functional_role
    has_functional_role_columns = 'functional_role' in df.columns and 'position' in df.columns
    
    if extract_functional_role:
        if not has_functional_role_columns:
            raise ValueError("❌ --functional_role specified but dataset CSV does not contain 'functional_role' and 'position' columns")
        print(f"✅ Extracting functional_role labels from dataset CSV")
    elif has_functional_role_columns:
        print(f"ℹ️  Dataset CSV contains 'functional_role' and 'position' columns (use --functional_role to extract)")
    
    # Get sequence column name
    sequence_column_name = 'ptm_seq'
    
    # Apply subsample if specified
    subsample_size = data_config.get("subsample_size", None)
    if subsample_size is not None and subsample_size > 0:
        df = df.iloc[:subsample_size]
        print(f"📊 Subsampled to {len(df)} samples")
    
    # Get AC_ID column
    if "AC_ID" not in df.columns:
        df["AC_ID"] = df.index.astype(str)
    
    # Always use original sequence for ESM embeddings
    original_seq_col = 'ori_seq'
    if original_seq_col not in df.columns:
        raise ValueError(
            f"❌ Column '{original_seq_col}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )
    print(f"📝 Using original sequences from column '{original_seq_col}' for ESM input")
    
    # Extract functional_role info and prepare data list directly from DataFrame
    functional_role_to_idx = {"Enhancing": 0, "Impairing": 1, "Associated": 2}
    ac_id_to_ptm_info: Dict[str, List[Dict[str, Any]]] = {}  # {ac_id: [{'position': int, 'functional_role': int}, ...]}
    
    data_list = []
    for _, row in df.iterrows():
        # Prepare basic data dict
        data_dict = {
            "ac_id": str(row["AC_ID"]),
            "ptm_sequence": str(row[sequence_column_name]),
        }
        if original_seq_col is not None:
            data_dict["original_sequence"] = str(row[original_seq_col])
        data_list.append(data_dict)
        
        # Extract functional_role info if requested
        if extract_functional_role:
            ac_id = str(row["AC_ID"])
            position = int(row.get('position', 0))  # 1-based position
            functional_role_str = str(row.get('functional_role', '')).strip()
            
            if ac_id and functional_role_str and functional_role_str in functional_role_to_idx:
                if ac_id not in ac_id_to_ptm_info:
                    ac_id_to_ptm_info[ac_id] = []
                ac_id_to_ptm_info[ac_id].append({
                    'position': position,  # 1-based
                    'functional_role': functional_role_to_idx[functional_role_str]
                })
    
    if extract_functional_role:
        print(f"💾 Extracted functional_role info from dataset CSV ({sum(len(v) for v in ac_id_to_ptm_info.values())} PTM sites, {len(ac_id_to_ptm_info)} proteins)")
    
    print(f"📊 Dataset size: {len(data_list)} samples")
    
    # Load tokenizer (needed for ESM input processing)
    tokenizer = PTMTokenizer()

    # Load ESM model from config using inference classes
    print(f"📌 Using model from config: {model_name}")
    if repr_layer is not None:
        print(f"📌 Using representation layer from config: {repr_layer}")
    
    # Use inference classes for cleaner code and unified model loading
    esm_inference = None
    model_type = None
    embed_dim = None
    final_repr_layer = repr_layer
    
    if model_name == "esmc_600m":
        # Use ESMCInference for ESM-C 600M
        esm_inference = ESMCInference(device=str(device), layer_index=repr_layer)
        model_type = "esmc"
        embed_dim = esm_inference.hidden_size
        # ESM-C uses 1-based layer indexing
        final_repr_layer = repr_layer
    elif model_name in ["esm2_650m", "esm2_15b"]:
        # Use ESM2Inference for ESM2 models
        hf_model_name = "facebook/esm2_t33_650M_UR50D" if model_name == "esm2_650m" else "facebook/esm2_t48_15B_UR50D"
        esm_inference = ESM2Inference(model_name=hf_model_name, device=str(device), layer_index=repr_layer)
        model_type = "esm2"
        embed_dim = esm_inference.hidden_size
        # ESM2 uses 0-based layer indexing
        final_repr_layer = repr_layer
    else:
        # Other models (ESM3, etc.) not yet supported via inference classes
        raise ValueError(
            f"Model {model_name} is not supported. "
            f"Supported models: esmc_600m, esm2_650m, esm2_15b"
        )
    
    print(f"🚀 Using single GPU for embedding generation")
    print(f"📦 Model type: {model_type}, Representation layer: {final_repr_layer}")
    
    # Get max_sequence_length from training config (token-level limit)
    max_sequence_length = train_config.get("max_sequence_length", None)

    if max_sequence_length:
        print(f"📏 Using max_sequence_length: {max_sequence_length} tokens (sequences will be cropped)")
    else:
        print(f"📏 No sequence length limit (using full sequences)")
    print(f"📐 Embedding output format: (N, 512, {embed_dim}) - fixed shape for compatibility with reorganize_h5_to_memmap.py")
    
    # Generate embeddings using inference class
    generate_embeddings_for_dataset(
        dataset_data=data_list,
        tokenizer=tokenizer,
        esm_inference=esm_inference,
        output_dir=args.output_dir,
        embed_dim=embed_dim,
        batch_size=args.batch_size,
        max_sequence_length=max_sequence_length,
        chunk_size_gb=args.chunk_size_gb,
        repr_layer=final_repr_layer,
        model_type=model_type,
        ac_id_to_ptm_info=ac_id_to_ptm_info if extract_functional_role else None,
        extract_functional_role=extract_functional_role,
    )


if __name__ == "__main__":
    main()
