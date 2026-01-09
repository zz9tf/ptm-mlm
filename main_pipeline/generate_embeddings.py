"""
Script to pre-generate ESM embeddings for all sequences in the dataset.
This allows faster training by loading pre-computed embeddings instead of computing them on-the-fly.
Supports multi-GPU generation using accelerate.
Supports both ESM2 15B and ESM3 7B models.
"""
import torch
import esm
import argparse
import os
import yaml
from tqdm import tqdm
from pathlib import Path
import numpy as np
import fcntl
import time
import sys
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from getters.tokenizer import PTMTokenizer
from utils.config import load_config
from utils.esm_utils import make_esm_input_ids, load_esm_model


def _gather_object(accelerator, obj, device):
    """
    Gather objects from all processes using pickle serialization.
    Replacement for accelerator.gather_object() which doesn't exist in accelerate 1.12.0.
    
    @param accelerator: Accelerator instance
    @param obj: Object to gather (will be pickled)
    @param device: Device to use for tensor operations
    @return: List of objects from all processes (only meaningful on main process)
    """
    # Serialize object to bytes
    obj_bytes = pickle.dumps(obj)
    obj_len = torch.tensor([len(obj_bytes)], device=device, dtype=torch.int32)
    
    # Gather lengths first
    all_lengths = accelerator.gather(obj_len)
    
    # Create tensor for bytes data (pad to max length)
    max_len = all_lengths.max().item()
    obj_array = np.frombuffer(obj_bytes, dtype=np.uint8)
    obj_tensor = torch.zeros(max_len, device=device, dtype=torch.uint8)
    if len(obj_array) > 0:
        obj_tensor[:len(obj_array)] = torch.from_numpy(obj_array).to(device)
    
    # Gather all byte tensors
    all_obj_tensors = accelerator.gather(obj_tensor.unsqueeze(0))
    
    # Deserialize objects from each process
    result = []
    for i, length in enumerate(all_lengths):
        length_val = length.item()
        if length_val > 0:
            obj_bytes_recovered = all_obj_tensors[i][:length_val].cpu().numpy().tobytes()
            obj_recovered = pickle.loads(obj_bytes_recovered)
            result.append(obj_recovered)
        else:
            result.append(None)
    
    return result


def _broadcast_object(accelerator, obj, device):
    """
    Broadcast object from main process to all processes using pickle serialization.
    Replacement for accelerator.broadcast_object() which doesn't exist in accelerate 1.12.0.
    
    @param accelerator: Accelerator instance
    @param obj: Object to broadcast (from main process)
    @param device: Device to use for tensor operations
    @return: Broadcasted object (same on all processes)
    """
    if accelerator.is_main_process:
        # Main process: serialize and send
        obj_bytes = pickle.dumps(obj)
        obj_len = torch.tensor([len(obj_bytes)], device=device, dtype=torch.int32)
        obj_array = np.frombuffer(obj_bytes, dtype=np.uint8)
        obj_tensor = torch.zeros(obj_len.item(), device=device, dtype=torch.uint8)
        if len(obj_array) > 0:
            obj_tensor[:len(obj_array)] = torch.from_numpy(obj_array).to(device)
    else:
        # Other processes: prepare empty tensors
        obj_len = torch.tensor([0], device=device, dtype=torch.int32)
        obj_tensor = torch.zeros(1, device=device, dtype=torch.uint8)
    
    # Broadcast length first
    all_lengths = accelerator.gather(obj_len)
    max_len = all_lengths.max().item()
    
    # Pad tensor to max length
    if accelerator.is_main_process:
        padded_tensor = torch.zeros(max_len, device=device, dtype=torch.uint8)
        padded_tensor[:len(obj_array)] = obj_tensor
    else:
        padded_tensor = torch.zeros(max_len, device=device, dtype=torch.uint8)
    
    # Broadcast tensor
    all_tensors = accelerator.gather(padded_tensor.unsqueeze(0))
    
    # Main process tensor contains the data
    main_tensor = all_tensors[0]
    main_len = all_lengths[0].item()
    
    # Deserialize on all processes
    if main_len > 0:
        obj_bytes_recovered = main_tensor[:main_len].cpu().numpy().tobytes()
        obj_broadcasted = pickle.loads(obj_bytes_recovered)
    else:
        obj_broadcasted = None
    
    return obj_broadcasted


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
                'functional_roles': [],
                'functional_role_positions': []
            }
            self.ac_id_window_counts[ac_id] = total_windows
            self.ac_id_processed_counts[ac_id] = 0
    
    def add_window(self, ac_id: str, embedding: np.ndarray, window_range: Tuple[int, int], 
                   ori_seq: str, ptm_seq: str, functional_role: int = -1, functional_role_position: int = -1):
        """
        Add a window's data to the buffer.
        
        @param ac_id: AC_ID string
        @param embedding: Window embedding (numpy array)
        @param window_range: (start, end) tuple
        @param ori_seq: Original sequence window
        @param ptm_seq: PTM sequence window
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
            # Embeddings (float32 = 4 bytes)
            for emb in data['embeddings']:
                total_bytes += emb.nbytes
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
    
    def get_all_complete_data(self) -> Dict[str, Dict[str, List[Any]]]:
        """
        Get all complete AC_IDs data without removing them.
        
        @return: Dictionary of complete AC_ID data
        """
        complete_ac_ids = self.get_complete_ac_ids()
        return {ac_id: self.data[ac_id] for ac_id in complete_ac_ids if ac_id in self.data}
    
    def clear(self):
        """Clear all data from buffer."""
        self.data.clear()
        self.ac_id_window_counts.clear()
        self.ac_id_processed_counts.clear()
    
    def size(self) -> int:
        """Get number of AC_IDs in buffer."""
        return len(self.data)


def map_token_indices_to_char_positions(ptm_sequence, tokenizer, token_start, token_end, total_token_len):
    """
    Map token indices to character positions in PTM sequence.
    
    This function attempts to find the character positions in the original PTM sequence
    that correspond to the given token range. Since PTM tokens may have variable
    character lengths, this uses proportional mapping as an approximation.
    
    @param ptm_sequence: Original PTM sequence string
    @param tokenizer: Tokenizer instance (not used in current implementation, kept for future improvements)
    @param token_start: Start token index
    @param token_end: End token index
    @param total_token_len: Total length of tokenized sequence (for proportional mapping)
    @return: Tuple of (char_start, char_end) character positions
    """
    # Use proportional mapping: assume tokens are roughly evenly distributed
    # This is an approximation, but should work reasonably well for most cases
    if total_token_len > 0:
        ratio = len(ptm_sequence) / total_token_len
        char_start = int(token_start * ratio)
        char_end = int(token_end * ratio)
    else:
        char_start = 0
        char_end = len(ptm_sequence)
    
    # Clamp to sequence length
    char_start = min(char_start, len(ptm_sequence))
    char_end = min(char_end, len(ptm_sequence))
    
    return char_start, char_end


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


def get_atomic_counter(counter_path: str) -> int:
    """
    Get and increment atomic counter for multi-process index allocation.
    
    @param counter_path: Path to counter file
    @return: Current counter value (before increment)
    """
    lock_path = counter_path + '.lock'
    counter_file = open(counter_path, 'a+')
    lock_file = open(lock_path, 'w')
    
    try:
        # Acquire exclusive lock
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        
        # Read current counter value
        counter_file.seek(0)
        content = counter_file.read().strip()
        current_value = int(content) if content else 0
        
        # Increment and write back
        counter_file.seek(0)
        counter_file.truncate()
        counter_file.write(str(current_value + 1))
        counter_file.flush()
        
        # Release lock
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        
        return current_value
    finally:
        counter_file.close()
        lock_file.close()


def append_chunk_to_memmap(
    chunk_data_by_ac_id: Dict[str, Dict[str, List[Any]]],
    memmap_arrays: Dict[str, np.memmap],
    tokenizer: PTMTokenizer,
    embed_dim: int,
    max_seq_len: int,
    protein_id_to_idx: Dict[str, int],
    protein_ids_list: List[str],
    counter_path: str,
    accelerator: Accelerator
) -> int:
    """
    Append a chunk of embeddings to memmap files.
    
    @param chunk_data_by_ac_id: Dict mapping AC_ID to window data
    @param memmap_arrays: Dict with keys 'embeddings', 'orig_tokens', 'ptm_tokens', 'range_data', 'meta_id'
    @param tokenizer: Tokenizer instance for encoding sequences
    @param embed_dim: Embedding dimension
    @param max_seq_len: Maximum sequence length (512)
    @param protein_id_to_idx: Mapping from protein ID to index
    @param protein_ids_list: List of all protein IDs (for reverse lookup)
    @param counter_path: Path to atomic counter file
    @param accelerator: Accelerator instance
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
            # Get atomic index
            global_idx = get_atomic_counter(counter_path)
            
            # 1. Embeddings: convert to float16 and pad/truncate to (512, embed_dim)
            emb = data['embeddings'][window_idx]  # Shape: (seq_len, embed_dim)
            actual_len = min(emb.shape[0], max_seq_len)
            
            # Truncate or pad to max_seq_len
            if emb.shape[0] > max_seq_len:
                emb_padded = emb[:max_seq_len, :].astype(np.float16)
            else:
                emb_padded = np.zeros((max_seq_len, embed_dim), dtype=np.float16)
                emb_padded[:emb.shape[0], :] = emb.astype(np.float16)
            
            memmap_arrays['embeddings'][global_idx] = emb_padded
            
            # 2. Original sequence tokens
            ori_seq_str = data['ori_seqs'][window_idx]
            ori_token_ids = tokenizer.encode(ori_seq_str, add_special_tokens=False)
            ori_seq_len = min(len(ori_token_ids), max_seq_len)
            
            if len(ori_token_ids) > max_seq_len:
                ori_token_ids = ori_token_ids[:max_seq_len]
            
            memmap_arrays['orig_tokens'][global_idx, :ori_seq_len] = ori_token_ids
            memmap_arrays['orig_tokens'][global_idx, ori_seq_len:] = pad_token_id
            
            # 3. PTM sequence tokens
            ptm_seq_str = data['ptm_seqs'][window_idx]
            ptm_token_ids = tokenizer.encode(ptm_seq_str, add_special_tokens=False)
            ptm_seq_len = min(len(ptm_token_ids), max_seq_len)
            
            if len(ptm_token_ids) > max_seq_len:
                ptm_token_ids = ptm_token_ids[:max_seq_len]
            
            memmap_arrays['ptm_tokens'][global_idx, :ptm_seq_len] = ptm_token_ids
            memmap_arrays['ptm_tokens'][global_idx, ptm_seq_len:] = pad_token_id
            
            # 4. Range data: [start, end, actual_len]
            start, end = data['ranges'][window_idx]
            # Use embedding length as actual_len (before truncation)
            embedding_actual_len = min(emb.shape[0], max_seq_len)
            memmap_arrays['range_data'][global_idx] = [start, end, embedding_actual_len]
            
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
    
    # Flush all arrays
    for arr in memmap_arrays.values():
        arr.flush()
    
    return num_written


def extract_windows_for_sequence(
    ac_id: str,
    seq_for_esm: str,
    ori_seq_full: str,
    ptm_seq_full: str,
    tokenizer: PTMTokenizer,
    max_sequence_length: Optional[int],
    use_original_seq: bool,
    device: torch.device
) -> List[Dict[str, Any]]:
    """
    Extract all windows for a single sequence.
    
    IMPORTANT: All ranges are based on character positions in the original sequence,
    not token positions. This ensures consistency with the new dataset loading method.
    
    @param ac_id: AC_ID string
    @param seq_for_esm: Sequence to use for ESM input (original sequence if use_original_seq=True)
    @param ori_seq_full: Full original sequence (without PTM annotations)
    @param ptm_seq_full: Full PTM sequence (with PTM annotations)
    @param tokenizer: Tokenizer instance
    @param max_sequence_length: Maximum sequence length (in characters, not tokens)
    @param use_original_seq: Whether to use original sequence for ESM
    @param device: Device for tensor operations
    @return: List of window info dictionaries with:
        - 'char_start', 'char_end': Character positions in original sequence (for range)
        - 'window_seq': Sequence for ESM input
        - 'ori_seq_window': Original sequence window
        - 'ptm_seq_window': PTM sequence window
    """
    windows = []
    
    # Always compute windows based on character positions in original sequence
    # This ensures range consistency with the new dataset loading method
    ori_seq_len = len(ori_seq_full)
    window_ranges = compute_extraction_windows(ori_seq_len, max_sequence_length)
    
    for win_idx, (char_start, char_end) in enumerate(window_ranges):
        # Extract windows from original and PTM sequences based on character positions
        ori_seq_window = ori_seq_full[char_start:char_end]
        ptm_seq_window = ptm_seq_full[char_start:char_end] if len(ptm_seq_full) >= char_end else ptm_seq_full[char_start:]
        
        # Prepare sequence for ESM input
        if use_original_seq:
            # Use original sequence directly for ESM
            window_seq = ori_seq_window
            window_len = len(window_seq)
        else:
            # Use PTM sequence for ESM (will be processed to mask PTM tokens)
            # Tokenize the PTM sequence window to get token length
            tokenized = tokenizer(
                [ptm_seq_window],
                add_special_tokens=True,
                max_sequence_length=None,
            )
            input_ids = torch.tensor(tokenized[0], device=device)
            window_len = len(input_ids)
            
            # Prepare ESM input: replace PTM tokens with mask
            window_ids = input_ids.unsqueeze(0)
            esm_input_ids = make_esm_input_ids(window_ids, tokenizer)
            decoded_str = tokenizer.decode(esm_input_ids[0].detach().cpu().tolist())
            decoded_str_clean = decoded_str.replace("<cls>", "").replace("<eos>", "")
            window_seq = decoded_str_clean
        
        windows.append({
            'ac_id': ac_id,
            'window_idx': win_idx,
            'char_start': char_start,  # Character position in original sequence
            'char_end': char_end,      # Character position in original sequence
            'window_start': char_start,  # Alias for compatibility
            'window_end': char_end,      # Alias for compatibility
            'window_len': window_len,    # Length of window_seq (for ESM)
            'window_seq': window_seq,   # Sequence for ESM input
            'ori_seq_window': ori_seq_window,  # Original sequence window
            'ptm_seq_window': ptm_seq_window,  # PTM sequence window
        })
    
    return windows


def remove_special_token_embeddings(embedding: np.ndarray, seq_len: int, num_start: int = 1, num_end: int = 1) -> np.ndarray:
    """
    Remove special token embeddings (<cls> at start, <eos> at end) from embedding.
    
    @param embedding: Embedding array with shape [embed_len, embed_dim]
    @param seq_len: Expected sequence length (after removing special tokens)
    @param num_start: Number of special tokens at start (default: 1 for <cls>)
    @param num_end: Number of special tokens at end (default: 1 for <eos>)
    @return: Embedding array with special tokens removed
    """
    embed_len = embedding.shape[0]
    expected_with_special = seq_len + num_start + num_end
    
    if embed_len == expected_with_special:
        # Remove special tokens: first num_start and last num_end
        return embedding[num_start:-num_end] if num_end > 0 else embedding[num_start:]
    elif embed_len == seq_len:
        # Already correct length, no special tokens to remove
        return embedding
    else:
        raise ValueError(
            f"Embedding length mismatch: expected {seq_len} or {expected_with_special}, got {embed_len}"
        )


def process_window_batch(
    window_batch: List[Dict[str, Any]],
    esm_model: torch.nn.Module,
    batch_converter: Any,
    repr_layer: int,
    model_type: str,
    device: torch.device,
    accelerator: Accelerator,
    embed_dim: Optional[int] = None
) -> List[np.ndarray]:
    """
    Process a batch of windows and extract embeddings.
    
    @param window_batch: List of window info dictionaries
    @param esm_model: ESM model
    @param batch_converter: ESM batch converter (None for ESM C)
    @param repr_layer: Representation layer index
    @param model_type: Model type ('esm2', 'esm3', or 'esmc')
    @param device: Device for computation
    @param accelerator: Accelerator instance
    @param embed_dim: Expected embedding dimension (for fallback when creating zero embeddings)
    @return: List of embeddings (numpy arrays)
    """
    if model_type == "esmc":
        # ESM C supports batch processing through _tokenize() and forward() methods
        # Reference: https://github.com/evolutionaryscale/esm/blob/main/esm/models/esmc.py
        
        sequences = [w['window_seq'] for w in window_batch]
        ori_seq_windows = [w['ori_seq_window'] for w in window_batch]
        
        # Calculate actual token length for each sequence (sequence + 2 special tokens: <cls> and <eos>)
        # ESM C tokenizer adds <cls> at start and <eos> at end
        actual_token_lengths = [len(seq) + 2 for seq in sequences]
        
        with torch.no_grad():
            # Batch tokenize and forward pass
            # Note: _tokenize() pads all sequences to the same length
            sequence_tokens = esm_model._tokenize(sequences)
            output = esm_model.forward(sequence_tokens=sequence_tokens)
            
            # Extract embeddings based on repr_layer
            if repr_layer is not None and repr_layer >= 0:
                if output.hidden_states is None:
                    raise ValueError("hidden_states is None")
                layer_idx = min(repr_layer, output.hidden_states.shape[0] - 1)
                batch_embeddings = output.hidden_states[layer_idx]  # [batch_size, padded_seq_len, embed_dim]
            else:
                if output.embeddings is None:
                    raise ValueError("embeddings is None")
                batch_embeddings = output.embeddings  # [batch_size, padded_seq_len, embed_dim]
            
            # Process each embedding: extract actual length, remove special tokens, convert to numpy
            # Convert bfloat16 to float32 before numpy conversion (numpy doesn't support bfloat16)
            embeddings = []
            for batch_idx, (ori_seq_window, actual_token_len) in enumerate(zip(ori_seq_windows, actual_token_lengths)):
                # Extract only the actual sequence part (excluding padding)
                # actual_token_len includes <cls> and <eos>, so we get: [<cls>, seq_tokens, <eos>]
                window_emb = batch_embeddings[batch_idx, :actual_token_len, :]  # [actual_token_len, embed_dim]
                window_emb = window_emb.float().cpu().numpy()
                
                # Remove special tokens: <cls> (first) and <eos> (last)
                window_emb = remove_special_token_embeddings(window_emb, len(ori_seq_window))
                embeddings.append(window_emb)
        
        return embeddings
    else:
        # ESM2/ESM3 models
        # Prepare ESM inputs
        esm_inputs = [(i, w['window_seq']) for i, w in enumerate(window_batch)]
        
        # Compute ESM embeddings
        # batch_converter automatically adds <cls> at the beginning and <eos> at the end
        # We let the model process tokens with special tokens (as designed), then remove them from embeddings
        batch_labels, batch_strs, batch_tokens = batch_converter(esm_inputs)
        batch_tokens = batch_tokens.to(device)
        
        # Extract embeddings (with special tokens included)
        batch_embeddings = extract_esm_embeddings(
            esm_model, batch_tokens, repr_layer, model_type, accelerator
        )
        
        # Remove special token embeddings at the end (unified approach)
        # ESM2/ESM3 add <cls> at the beginning and <eos> at the end
        num_special_tokens_start = 1  # <cls> at the beginning
        num_special_tokens_end = 1    # <eos> at the end
        
        embeddings = []
        for batch_idx, window_info in enumerate(window_batch):
            ori_seq_window = window_info['ori_seq_window']
            actual_seq_len = len(ori_seq_window)  # Character length of original sequence
            
            window_emb = batch_embeddings[batch_idx].cpu().numpy()  # Shape: (token_len, embed_dim)
            
            # Remove special tokens from embeddings (unified with ESM C approach)
            window_emb = remove_special_token_embeddings(
                window_emb, 
                actual_seq_len, 
                num_special_tokens_start, 
                num_special_tokens_end
            )
            
            embeddings.append(window_emb)
        
        return embeddings


def flush_buffer_to_disk(
    buffer: EmbeddingBuffer,
    memmap_arrays: Dict[str, np.memmap],
    tokenizer: PTMTokenizer,
    embed_dim: int,
    max_seq_len: int,
    protein_id_to_idx: Dict[str, int],
    protein_ids_list: List[str],
    counter_path: str,
    accelerator: Accelerator,
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
    @param protein_ids_list: List of all protein IDs
    @param counter_path: Path to atomic counter file
    @param accelerator: Accelerator instance
    @param chunk_idx: Current chunk index
    @return: Number of samples written
    """
    complete_data = buffer.pop_complete_data()
    
    if not complete_data:
        return 0
    
    # Save to memmap files
    num_written = append_chunk_to_memmap(
        complete_data, memmap_arrays, tokenizer, embed_dim, max_seq_len,
        protein_id_to_idx, protein_ids_list, counter_path, accelerator
    )
    
    num_saved_ac_ids = len(complete_data)
    if accelerator.is_local_main_process:
        accelerator.print(f"💾 Saved chunk {chunk_idx} ({num_saved_ac_ids} complete AC_IDs, "
                         f"{num_written} samples, buffer memory: {buffer.estimate_memory_gb():.2f}GB)")
    
    return num_written


def extract_esm_embeddings(esm_model, batch_tokens, repr_layer, model_type, accelerator):
    """
    Extract embeddings from ESM model (supports both ESM2 and ESM3).
    
    @param esm_model: ESM model instance
    @param batch_tokens: Tokenized input tokens
    @param repr_layer: Layer number to extract embeddings from
    @param model_type: Model type ('esm2' or 'esm3')
    @param accelerator: Accelerator instance for error handling
    @return: Embeddings tensor
    """
    with torch.no_grad():
        if model_type == "esm2":
            # ESM2 API
            out = esm_model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
            embeddings = out["representations"][repr_layer]
        elif model_type == "esm3":
            # ESM3 API - try different approaches
            try:
                # Try ESM3 standard API
                out = esm_model(batch_tokens, repr_layers=[repr_layer] if repr_layer >= 0 else None, return_contacts=False)
                if isinstance(out, dict) and "representations" in out:
                    if repr_layer >= 0:
                        embeddings = out["representations"][repr_layer]
                    else:
                        # Use the last layer
                        last_layer = max(out["representations"].keys())
                        embeddings = out["representations"][last_layer]
                else:
                    # ESM3 might return embeddings directly
                    embeddings = out if isinstance(out, torch.Tensor) else out[0]
            except Exception as e:
                # Fallback: try to get last hidden state
                try:
                    out = esm_model(batch_tokens)
                    if isinstance(out, tuple):
                        embeddings = out[0]  # First element is usually hidden states
                    elif isinstance(out, dict):
                        # Try common keys
                        if "hidden_states" in out:
                            embeddings = out["hidden_states"][-1]  # Last layer
                        elif "representations" in out:
                            last_layer = max(out["representations"].keys())
                            embeddings = out["representations"][last_layer]
                        else:
                            raise RuntimeError(f"Unknown ESM3 output format: {out.keys()}")
                    else:
                        embeddings = out
                except Exception as e2:
                    if accelerator.is_local_main_process:
                        accelerator.print(f"❌ Error extracting ESM3 embeddings: {e}")
                        accelerator.print(f"   Fallback error: {e2}")
                    raise RuntimeError(f"Failed to extract embeddings from ESM3 model: {e}")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    return embeddings


def generate_embeddings_for_dataset(
    dataset_data: List[Dict[str, str]],
    tokenizer: PTMTokenizer,
    esm_model,
    batch_converter,
    accelerator,
    output_dir,
    embed_dim: int,
    batch_size=8,
    max_sequence_length=None,
    chunk_size_gb=100,
    repr_layer=33,
    model_type="esm2",
    ac_id_to_ptm_info: Optional[Dict[str, List[Dict[str, Any]]]] = None,
):
    """
    Generate ESM embeddings for all sequences in the dataset.
    Directly outputs memmap format compatible with reorganize_h5_to_memmap.py.
    
    Output format:
    - orig_tokens.dat [N, 512] int32 - 原始序列 token IDs
    - ptm_tokens.dat [N, 512] int32 - PTM 序列 token IDs
    - embeddings.dat [N, 512, embed_dim] float16 - Embeddings
    - range.dat [N, 3] int32 - 范围信息 [start, end, length]
    - meta_id.dat [N] int64 - 蛋白质 ID 索引
    - functional_role.dat [N] int8 - 功能角色标签索引 (-1=无标签, 0=Enhancing, 1=Impairing, 2=Associated)
    - functional_role_position.dat [N] int32 - 功能角色对应的PTM位置 (1-based, -1=无位置)
    - meta_mapping.json - 蛋白质ID到索引的映射和功能角色映射
    
    @param dataset_data: List of dicts with:
        - 'ac_id': AC_ID string
        - 'ptm_sequence': PTM sequence (with PTM annotations)
        - 'original_sequence': Original sequence (without PTM annotations, optional)
    @param tokenizer: Tokenizer instance.
    @param esm_model: ESM model (prepared by accelerator).
    @param batch_converter: ESM alphabet batch converter (None for ESM C models).
    @param accelerator: Accelerator instance for distributed processing.
    @param output_dir: Directory to save embeddings (will create memmap files).
    @param embed_dim: Embedding dimension.
    @param batch_size: Batch size for embedding generation (default: 8).
    @param max_sequence_length: Maximum sequence length (same as training). If None, no limit.
    @param chunk_size_gb: Target chunk size in GB before flushing to disk (default: 100GB).
    @param repr_layer: Representation layer index to extract embeddings from (default: 33).
    @param model_type: Model type ('esm2', 'esm3', or 'esmc') (default: 'esm2').
    @param ac_id_to_ptm_info: Optional dictionary mapping AC_ID to list of PTM info dicts with 'position' (1-based) and 'functional_role' keys.
    """
    device = accelerator.device
    
    # Create output directory
    if accelerator.is_local_main_process:
        os.makedirs(output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    
    # Process entire dataset
    if accelerator.is_local_main_process:
        accelerator.print(f"\n📊 Processing dataset ({len(dataset_data)} samples)...")
    
    # Fixed parameters for memmap format
    max_seq_len = 512
    output_path = Path(output_dir)
    
    # Functional role mapping
    functional_role_to_idx = {"Enhancing": 0, "Impairing": 1, "Associated": 2}
    idx_to_functional_role = ["Enhancing", "Impairing", "Associated"]
    
    # Use provided ac_id_to_ptm_info or initialize empty
    if ac_id_to_ptm_info is None:
        ac_id_to_ptm_info = {}
    
    if ac_id_to_ptm_info and accelerator.is_local_main_process:
        accelerator.print(f"✅ Using functional roles for {len(ac_id_to_ptm_info)} proteins")
    
    # Broadcast functional role mapping to all processes
    if accelerator.num_processes > 1:
        ac_id_to_ptm_info = _broadcast_object(accelerator, ac_id_to_ptm_info, device)
    
    # Step 1: Count total samples (all processes participate)
    if accelerator.is_local_main_process:
        accelerator.print(f"\n📊 Step 1: Counting total samples...")
    
    # Distribute samples across processes for counting
    num_processes = accelerator.num_processes
    process_rank = accelerator.process_index
    total_samples = len(dataset_data)
    samples_per_process = total_samples // num_processes
    start_idx = process_rank * samples_per_process
    end_idx = start_idx + samples_per_process if process_rank < num_processes - 1 else total_samples
    
    # Count windows for this process's samples
    local_window_count = 0
    local_ac_ids = []
    
    for i in range(start_idx, end_idx):
        sample = dataset_data[i]
        ac_id = sample.get("ac_id", f"sample_{i}")
        local_ac_ids.append(ac_id)
        
        ori_seq = sample.get("original_sequence", None)
        if ori_seq is None:
            ori_seq = sample.get("ptm_sequence", "")
        
        ori_seq_len = len(ori_seq)
        window_ranges = compute_extraction_windows(ori_seq_len, max_sequence_length)
        local_window_count += len(window_ranges)
    
    # Gather counts from all processes
    all_counts = accelerator.gather(torch.tensor([local_window_count], device=device))
    all_ac_ids_list = _gather_object(accelerator, local_ac_ids, device)
    
    if accelerator.is_local_main_process:
        total_windows = all_counts.sum().item()
        all_ac_ids = []
        for ac_id_list in all_ac_ids_list:
            if ac_id_list is not None:
                all_ac_ids.extend(ac_id_list)
        
        # Get unique AC_IDs and create mapping
        unique_ac_ids = sorted(list(set(all_ac_ids)))
        protein_id_to_idx = {ac_id: idx for idx, ac_id in enumerate(unique_ac_ids)}
        protein_ids_list = unique_ac_ids
        
        accelerator.print(f"✅ Total proteins: {len(unique_ac_ids):,}")
        accelerator.print(f"✅ Total windows (samples): {total_windows:,}")
        accelerator.print(f"✅ Embedding dimension: {embed_dim}")
        accelerator.print(f"✅ Sequence length: {max_seq_len}")
        
        # Step 2: Create memmap files (only main process)
        accelerator.print(f"\n📝 Step 2: Creating memmap files...")
        
        orig_tokens_path = output_path / 'orig_tokens.dat'
        ptm_tokens_path = output_path / 'ptm_tokens.dat'
        embeddings_path = output_path / 'embeddings.dat'
        range_path = output_path / 'range.dat'
        meta_id_path = output_path / 'meta_id.dat'
        functional_role_path = output_path / 'functional_role.dat'
        functional_role_position_path = output_path / 'functional_role_position.dat'
        counter_path = output_path / 'counter.txt'
        
        # Create memmap arrays
        memmap_arrays = {
            'orig_tokens': np.memmap(orig_tokens_path, dtype=np.int32, mode='w+', shape=(total_windows, max_seq_len)),
            'ptm_tokens': np.memmap(ptm_tokens_path, dtype=np.int32, mode='w+', shape=(total_windows, max_seq_len)),
            'embeddings': np.memmap(embeddings_path, dtype=np.float16, mode='w+', shape=(total_windows, max_seq_len, embed_dim)),
            'range_data': np.memmap(range_path, dtype=np.int32, mode='w+', shape=(total_windows, 3)),
            'meta_id': np.memmap(meta_id_path, dtype=np.int64, mode='w+', shape=(total_windows,)),
            'functional_role': np.memmap(functional_role_path, dtype=np.int8, mode='w+', shape=(total_windows,)),
            'functional_role_position': np.memmap(functional_role_position_path, dtype=np.int32, mode='w+', shape=(total_windows,))
        }
        
        # Initialize counter
        with open(counter_path, 'w') as f:
            f.write('0')
        
        # Convert counter_path to string for consistency
        counter_path = str(counter_path)
        
        accelerator.print(f"✅ Created memmap files")
    else:
        memmap_arrays = None
        protein_id_to_idx = None
        protein_ids_list = None
        counter_path = None
    
    accelerator.wait_for_everyone()
    
    # Broadcast memmap arrays and mappings to all processes
    if accelerator.num_processes > 1:
        # Gather total_windows and protein mappings from main process
        if accelerator.is_local_main_process:
            total_windows = all_counts.sum().item()
            protein_id_to_idx = {ac_id: idx for idx, ac_id in enumerate(sorted(list(set(all_ac_ids))))}
            protein_ids_list = sorted(list(set(all_ac_ids)))
        else:
            total_windows = None
            protein_id_to_idx = None
            protein_ids_list = None
        
        # Broadcast total_windows
        total_windows = _broadcast_object(accelerator, total_windows, device)
        protein_id_to_idx = _broadcast_object(accelerator, protein_id_to_idx, device)
        protein_ids_list = _broadcast_object(accelerator, protein_ids_list, device)
        
        # All processes open memmap files in read-write mode
        output_path = Path(output_dir)
        orig_tokens_path = output_path / 'orig_tokens.dat'
        ptm_tokens_path = output_path / 'ptm_tokens.dat'
        embeddings_path = output_path / 'embeddings.dat'
        range_path = output_path / 'range.dat'
        meta_id_path = output_path / 'meta_id.dat'
        functional_role_path = output_path / 'functional_role.dat'
        functional_role_position_path = output_path / 'functional_role_position.dat'
        counter_path = str(output_path / 'counter.txt')
        
        memmap_arrays = {
            'orig_tokens': np.memmap(orig_tokens_path, dtype=np.int32, mode='r+', shape=(total_windows, max_seq_len)),
            'ptm_tokens': np.memmap(ptm_tokens_path, dtype=np.int32, mode='r+', shape=(total_windows, max_seq_len)),
            'embeddings': np.memmap(embeddings_path, dtype=np.float16, mode='r+', shape=(total_windows, max_seq_len, embed_dim)),
            'range_data': np.memmap(range_path, dtype=np.int32, mode='r+', shape=(total_windows, 3)),
            'meta_id': np.memmap(meta_id_path, dtype=np.int64, mode='r+', shape=(total_windows,)),
            'functional_role': np.memmap(functional_role_path, dtype=np.int8, mode='r+', shape=(total_windows,)),
            'functional_role_position': np.memmap(functional_role_position_path, dtype=np.int32, mode='r+', shape=(total_windows,))
        }
    
    # 🔧 Resume functionality: Load already processed AC_IDs (if meta_mapping.json exists)
    processed_unique_ids = set()
    meta_mapping_path = output_path / 'meta_mapping.json'
    if meta_mapping_path.exists() and accelerator.is_local_main_process:
        try:
            with open(meta_mapping_path, 'r') as f:
                meta_mapping = json.load(f)
                processed_unique_ids = set(meta_mapping.get('idx_to_protein_id', []))
                if accelerator.is_local_main_process:
                    accelerator.print(f"📋 Found existing embeddings with {len(processed_unique_ids)} AC_IDs")
                    accelerator.print(f"🔄 Resuming from checkpoint...")
        except Exception as e:
            if accelerator.is_local_main_process:
                accelerator.print(f"⚠️  Warning: Could not read existing file for resume: {e}")
    
    accelerator.wait_for_everyone()
    
    # Distribute samples across processes
    num_processes = accelerator.num_processes
    process_rank = accelerator.process_index
    
    # Calculate indices for this process
    total_samples = len(dataset_data)
    samples_per_process = total_samples // num_processes
    start_idx = process_rank * samples_per_process
    end_idx = start_idx + samples_per_process if process_rank < num_processes - 1 else total_samples
    
    # Estimate chunk size in samples
    estimated_bytes_per_sample = 2 * 1024 * 1024  # 2MB per sample (compressed estimate)
    chunk_size_samples = int(chunk_size_gb * 1024 * 1024 * 1024 / estimated_bytes_per_sample)
    chunk_size_samples = max(1000, min(chunk_size_samples, 50000))  # Between 1K and 50K samples
    
    if accelerator.is_local_main_process:
        accelerator.print(f"\n🔄 Step 3: Processing and writing embeddings...")
        accelerator.print(f"📦 Using chunk size: ~{chunk_size_samples} samples per chunk (~{chunk_size_gb}GB)")
    
    # Initialize memory buffer
    # max_memory_gb: Safety threshold - flush if memory exceeds this (even if chunk_size not reached)
    max_memory_gb = chunk_size_gb * 1.1  # Allow buffer to grow up to 1.1x chunk_size for safety
    buffer = EmbeddingBuffer(max_memory_gb=max_memory_gb, embed_dim=embed_dim)
    chunk_idx = 0
    # Fixed max_seq_len to 512 for compatibility with reorganize_h5_to_memmap.py
    # reorganize_h5_to_memmap.py expects embedding_seq shape to be (N, 512, embed_dim)
    max_seq_len = 512
    
    # 🔧 Resume statistics
    skipped_sequences = 0
    processed_sequences = 0
    
    # Process in batches
    process_indices = range(start_idx, end_idx)
    if accelerator.is_local_main_process:
        pbar = tqdm(range(0, len(process_indices), batch_size), desc="Generating embeddings")
    else:
        pbar = range(0, len(process_indices), batch_size)
    
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
        
        # 🔄 Use original sequences if available, otherwise use PTM sequences for ESM input
        use_original_seq = any(s.get("original_sequence") is not None for s in batch_samples)
        if use_original_seq:
            # Use original sequences (without PTM) for ESM input
            sequences_for_esm = ori_sequences
            if accelerator.is_local_main_process and batch_offset == 0:
                accelerator.print(f"📝 Using original sequences (without PTM) for ESM input")
        else:
            # Use PTM sequences (will mask PTM tokens later)
            sequences_for_esm = ptm_sequences
        
        # Step 1: Extract windows for all sequences in batch
        all_windows = []
        for idx, ac_id in enumerate(ac_ids):
            # Resume check: Skip if already processed
            if ac_id in processed_unique_ids:
                skipped_sequences += 1
                continue
            
            # Extract windows for this sequence
            seq_for_esm = sequences_for_esm[idx]
            ori_seq_full = ori_sequences[idx]
            ptm_seq_full = ptm_sequences[idx]
            
            windows = extract_windows_for_sequence(
                ac_id=ac_id,
                seq_for_esm=seq_for_esm,
                ori_seq_full=ori_seq_full,
                ptm_seq_full=ptm_seq_full,
                tokenizer=tokenizer,
                max_sequence_length=max_sequence_length,
                use_original_seq=use_original_seq,
                device=device
            )
            
            # Register AC_ID in buffer
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
                esm_model=esm_model,
                batch_converter=batch_converter,
                repr_layer=repr_layer,
                model_type=model_type,
                device=device,
                accelerator=accelerator,
                embed_dim=embed_dim
            )
            
            # Verify embed_dim matches (should already be set from model)
            if embeddings:
                actual_dim = embeddings[0].shape[1]
                if embed_dim is not None and actual_dim != embed_dim:
                    if accelerator.is_local_main_process:
                        accelerator.print(f"⚠️  Warning: Embedding dimension mismatch. Expected {embed_dim}, got {actual_dim}")
                    # Update to actual dimension
                    embed_dim = actual_dim
                    buffer.embed_dim = embed_dim
                elif embed_dim is None:
                    # Set embed_dim from first embedding
                    embed_dim = actual_dim
                    buffer.embed_dim = embed_dim
            
            # Add windows to buffer
            # Note: Some AC_IDs may have been flushed, so check if they still exist in buffer
            for window_info, embedding in zip(window_batch, embeddings):
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
                
                buffer.add_window(
                    ac_id=ac_id,
                    embedding=embedding,
                    window_range=(char_start, char_end),  # Character positions in original sequence
                    ori_seq=window_info['ori_seq_window'],
                    ptm_seq=window_info['ptm_seq_window'],
                    functional_role=functional_role,
                    functional_role_position=functional_role_position
                )
            
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
                if accelerator.is_local_main_process:
                    accelerator.print(f"💾 Flushing buffer: {flush_reason} ({len(complete_ac_ids)} complete AC_IDs, "
                                     f"estimated size: {complete_data_size_gb:.2f}GB)")
                
                num_written = flush_buffer_to_disk(
                    buffer=buffer,
                    memmap_arrays=memmap_arrays,
                    tokenizer=tokenizer,
                    embed_dim=embed_dim,
                    max_seq_len=max_seq_len,
                    protein_id_to_idx=protein_id_to_idx,
                    protein_ids_list=protein_ids_list,
                    counter_path=counter_path,
                    accelerator=accelerator,
                    chunk_idx=chunk_idx
                )
                if num_written > 0:
                    chunk_idx += 1
                    processed_sequences += num_written
                
                # Log remaining buffer state
                if accelerator.is_local_main_process:
                    remaining_complete = len(buffer.get_complete_ac_ids())
                    remaining_size = buffer.estimate_complete_data_size_gb()
                    if remaining_complete > 0:
                        accelerator.print(f"   📊 Remaining in buffer: {remaining_complete} complete AC_IDs "
                                         f"({remaining_size:.2f}GB), "
                                         f"buffer memory: {buffer.estimate_memory_gb():.2f}GB")
        
    # Flush remaining complete AC_IDs from buffer
    if buffer.size() > 0:
        num_written = flush_buffer_to_disk(
            buffer=buffer,
            memmap_arrays=memmap_arrays,
            tokenizer=tokenizer,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            protein_id_to_idx=protein_id_to_idx,
            protein_ids_list=protein_ids_list,
            counter_path=counter_path,
            accelerator=accelerator,
            chunk_idx=chunk_idx
        )
        if num_written > 0:
            processed_sequences += num_written
            if accelerator.is_local_main_process:
                accelerator.print(f"💾 Saved final chunk ({num_written} samples)")
        
        # Warn if there are incomplete AC_IDs remaining
        remaining_ac_ids = buffer.get_complete_ac_ids()
        if not remaining_ac_ids and buffer.size() > 0:
            incomplete_ac_ids = list(buffer.data.keys())
            if incomplete_ac_ids and accelerator.is_local_main_process:
                accelerator.print(f"⚠️  Warning: {len(incomplete_ac_ids)} incomplete AC_IDs not saved: {incomplete_ac_ids[:5]}")
    
    # Wait for all processes to finish
    accelerator.wait_for_everyone()
    
    # Step 4: Save meta_mapping.json (only main process)
    if accelerator.is_local_main_process:
        accelerator.print(f"\n💾 Step 4: Saving metadata...")
        
        # Read final counter value to get total samples written
        with open(counter_path, 'r') as f:
            total_samples_written = int(f.read().strip())
        
        meta_mapping = {
            'protein_id_to_idx': protein_id_to_idx,
            'idx_to_protein_id': protein_ids_list,
            'total_samples': total_samples_written,
            'embedding_dim': embed_dim,
            'sequence_length': max_seq_len,
            'orig_tokens_dtype': 'int32',
            'ptm_tokens_dtype': 'int32',
            'embeddings_dtype': 'float16',
            'range_dtype': 'int32',
            'range_shape': [3],  # [start, end, length]
            'range_description': 'range[i] = [start, end, length] 表示样本 i 在原始蛋白质中的位置范围',
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
        
        accelerator.print(f"✅ Meta mapping saved to: {output_path / 'meta_mapping.json'}")
        
        # Print file sizes
        accelerator.print(f"\n📊 File sizes:")
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
                accelerator.print(f"  {name}: {size:.2f} GB")
        
        accelerator.print(f"  总计: {total_size:.2f} GB")
        
        if skipped_sequences > 0:
            accelerator.print(f"\n🔄 Resume: Skipped {skipped_sequences} already processed sequences")
        accelerator.print(f"✨ Processed {processed_sequences} new samples")
        accelerator.print(f"\n✅ All embeddings generated and saved to {output_path}")


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
        "--use_original_sequence",
        action="store_true",
        help="Use original sequence (ori_seq) instead of PTM sequence for ESM input. "
             "Requires 'ori_seq' column in the dataset CSV file.",
    )
    parser.add_argument(
        "--original_sequence_column",
        type=str,
        default="ori_seq",
        help="Column name for original sequences in CSV (default: 'ori_seq')",
    )
    parser.add_argument(
        "--missing_only",
        action="store_true",
        help="Only process AC_IDs that are missing from the output embeddings file",
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
    
    # Initialize accelerator for distributed processing
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    
    # Load CSV data directly
    import pandas as pd
    
    # Load dataset CSV
    if accelerator.is_local_main_process:
        accelerator.print(f"📖 Loading data from CSV: {dataset_location}")
    
    df = pd.read_csv(dataset_location)
    df.drop(df.filter(regex="^Unnamed"), axis=1, inplace=True)
    
    # Check if functional_role extraction is requested
    extract_functional_role = args.functional_role
    has_functional_role_columns = 'functional_role' in df.columns and 'position' in df.columns
    
    if extract_functional_role:
        if not has_functional_role_columns:
            raise ValueError("❌ --functional_role specified but dataset CSV does not contain 'functional_role' and 'position' columns")
        if accelerator.is_local_main_process:
            accelerator.print(f"✅ Extracting functional_role labels from dataset CSV")
    elif has_functional_role_columns and accelerator.is_local_main_process:
        accelerator.print(f"ℹ️  Dataset CSV contains 'functional_role' and 'position' columns (use --functional_role to extract)")
    
    # Get sequence column name
    sequence_column_name = data_config["sequence_column_name"]
    if sequence_column_name not in df.columns:
        raise ValueError(
            f"❌ Column '{sequence_column_name}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Apply subsample if specified
    subsample_size = data_config.get("subsample_size", None)
    if subsample_size is not None and subsample_size > 0:
        df = df.iloc[:subsample_size]
        if accelerator.is_local_main_process:
            accelerator.print(f"📊 Subsampled to {len(df)} samples")
    
    # Get AC_ID column
    if "AC_ID" not in df.columns:
        df["AC_ID"] = df.index.astype(str)
    
    # Get original sequence column (if available)
    original_seq_col = None
    if args.use_original_sequence:
        if args.original_sequence_column not in df.columns:
            raise ValueError(
                f"❌ Column '{args.original_sequence_column}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )
        original_seq_col = args.original_sequence_column
        if accelerator.is_local_main_process:
            accelerator.print(f"📝 Using original sequences from column '{original_seq_col}' for ESM input")
    elif 'ori_seq' in df.columns:
        original_seq_col = 'ori_seq'
        if accelerator.is_local_main_process:
            accelerator.print(f"📝 Auto-detected 'ori_seq' column - using original sequences for ESM input")
    
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
    
    if extract_functional_role and accelerator.is_local_main_process:
        accelerator.print(f"💾 Extracted functional_role info from dataset CSV ({sum(len(v) for v in ac_id_to_ptm_info.values())} PTM sites, {len(ac_id_to_ptm_info)} proteins)")
    
    # Filter to only missing AC_IDs if requested
    if args.missing_only:
        meta_mapping_path = os.path.join(args.output_dir, "meta_mapping.json")
        existing_ac_ids = set()
        
        if os.path.exists(meta_mapping_path):
            if accelerator.is_local_main_process:
                accelerator.print(f"🔍 Checking for missing AC_IDs in {meta_mapping_path}...")
            try:
                with open(meta_mapping_path, 'r') as f:
                    meta_mapping = json.load(f)
                    existing_ac_ids = set(meta_mapping.get('idx_to_protein_id', []))
            except Exception as e:
                if accelerator.is_local_main_process:
                    accelerator.print(f"⚠️  Warning: Could not read embeddings file: {e}")
        
        if accelerator.is_local_main_process:
            accelerator.print(f"📊 Found {len(existing_ac_ids)} existing AC_IDs in embeddings file")
        
        # Filter data_list to only include missing AC_IDs
        original_size = len(data_list)
        data_list = [d for d in data_list if d["ac_id"] not in existing_ac_ids]
        missing_count = len(data_list)
        
        if accelerator.is_local_main_process:
            accelerator.print(f"📊 Filtered dataset: {original_size} -> {missing_count} (removed {original_size - missing_count} existing AC_IDs)")
        
        if missing_count == 0:
            if accelerator.is_local_main_process:
                accelerator.print("✅ All AC_IDs already exist in embeddings file. Nothing to do.")
            return
    
    if accelerator.is_local_main_process:
        accelerator.print(f"📊 Dataset size: {len(data_list)} samples")
    
    # Load tokenizer (needed for ESM input processing)
    tokenizer = PTMTokenizer()
    
    # Load ESM model from config
    if accelerator.is_local_main_process:
        accelerator.print(f"📌 Using model from config: {model_name}")
        if repr_layer is not None:
            accelerator.print(f"📌 Using representation layer from config: {repr_layer}")
    
    esm_model, alphabet, batch_converter, final_repr_layer, model_type, embed_dim = load_esm_model(
        model_name, accelerator, repr_layer_override=repr_layer
    )
    
    # Prepare ESM model with accelerator (ESM C models may need special handling)
    if model_type != "esmc":
        esm_model = accelerator.prepare(esm_model)
    else:
        # ESM C models use .to() method instead of accelerator.prepare()
        if hasattr(accelerator, 'device'):
            esm_model = esm_model.to(accelerator.device)
        elif hasattr(accelerator, 'local_process_index'):
            # Fallback: try to get device from accelerator
            device = torch.device(f"cuda:{accelerator.local_process_index}" if torch.cuda.is_available() else "cpu")
            esm_model = esm_model.to(device)
    
    if accelerator.is_local_main_process:
        accelerator.print(f"🚀 Using {accelerator.num_processes} GPU(s) for embedding generation")
        accelerator.print(f"📦 Model type: {model_type}, Representation layer: {final_repr_layer}")
    
    # Get max_sequence_length from training config
    max_sequence_length = train_config.get("max_sequence_length", None)
    
    if accelerator.is_local_main_process:
        if max_sequence_length:
            accelerator.print(f"📏 Using max_sequence_length: {max_sequence_length} (sequences will be cropped)")
        else:
            accelerator.print(f"📏 No sequence length limit (using full sequences)")
        accelerator.print(f"📐 Embedding output format: (N, 512, {embed_dim}) - fixed shape for compatibility with reorganize_h5_to_memmap.py")
    
    # Generate embeddings
    generate_embeddings_for_dataset(
        dataset_data=data_list,
        tokenizer=tokenizer,
        esm_model=esm_model,
        batch_converter=batch_converter,
        accelerator=accelerator,
        output_dir=args.output_dir,
        embed_dim=embed_dim,
        batch_size=args.batch_size,
        max_sequence_length=max_sequence_length,
        chunk_size_gb=args.chunk_size_gb,
        repr_layer=final_repr_layer,
        model_type=model_type,
        ac_id_to_ptm_info=ac_id_to_ptm_info if extract_functional_role else None,
    )


if __name__ == "__main__":
    main()

