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
import h5py
import numpy as np
import fcntl
import time
import sys
from typing import Dict, List, Tuple, Optional, Any
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from getters.tokenizer import PTMTokenizer
from getters.freeze_ptm_dataset import get_ptm_dataset
from utils.config import load_config
from utils.esm_utils import make_esm_input_ids, load_esm_model


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
        # Structure: {AC_ID: {'embeddings': [emb1, emb2, ...], 'ranges': [...], 'ori_seqs': [...], 'ptm_seqs': [...]}}
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
                'ptm_seqs': []
            }
            self.ac_id_window_counts[ac_id] = total_windows
            self.ac_id_processed_counts[ac_id] = 0
    
    def add_window(self, ac_id: str, embedding: np.ndarray, window_range: Tuple[int, int], 
                   ori_seq: str, ptm_seq: str):
        """
        Add a window's data to the buffer.
        
        @param ac_id: AC_ID string
        @param embedding: Window embedding (numpy array)
        @param window_range: (start, end) tuple
        @param ori_seq: Original sequence window
        @param ptm_seq: PTM sequence window
        """
        if ac_id not in self.data:
            raise ValueError(f"AC_ID {ac_id} not registered. Call add_ac_id() first.")
        
        if self.embed_dim is None:
            self.embed_dim = embedding.shape[1]
        
        self.data[ac_id]['embeddings'].append(embedding)
        self.data[ac_id]['ranges'].append(window_range)
        self.data[ac_id]['ori_seqs'].append(ori_seq)
        self.data[ac_id]['ptm_seqs'].append(ptm_seq)
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


def append_chunk_to_h5(chunk_data_by_ac_id, chunk_path, embed_dim, max_seq_len):
    """
    Append a chunk of embeddings to HDF5 file using new structure.
    
    New structure:
    - AC_ID (group name) -> {
        'embedding_seq': (num_windows, max_seq_len, embed_dim) - embeddings for each window
        'range_seq': (num_windows, 2) - ranges for each window
        'ori_seq': (num_windows, variable) - original sequence fragments for each window
        'ptm_seq': (num_windows, variable) - PTM sequence fragments for each window
    }
    
    @param {dict} chunk_data_by_ac_id - Dict mapping AC_ID to {
        'embeddings': [emb1, emb2, ...],  # List of numpy arrays
        'ranges': [(start1, end1), (start2, end2), ...]  # List of (start, end) tuples
        'ori_seqs': [seq1, seq2, ...],  # List of original sequence fragments (strings)
        'ptm_seqs': [seq1, seq2, ...],  # List of PTM sequence fragments (strings)
    }
    @param {str} chunk_path - Path to the chunk file
    @param {int} embed_dim - Embedding dimension
    @param {int|None} max_seq_len - Maximum sequence length (None = auto-detect from data)
    """
    if not chunk_data_by_ac_id:
        return max_seq_len
    
    # Auto-detect max_seq_len from data if not provided
    if max_seq_len is None:
        max_len = 0
        for ac_id, data in chunk_data_by_ac_id.items():
            for emb in data['embeddings']:
                max_len = max(max_len, emb.shape[0])
        max_seq_len = max_len
    
    # Use variable-length string type for sequences (no padding needed)
    vlen_str_dtype = h5py.special_dtype(vlen=str)
    
    # Simple append with file lock for multi-process safety
    file_exists = os.path.exists(chunk_path)
    lock_path = chunk_path + '.lock'
    
    # Use blocking file lock to ensure only one process writes at a time
    # Blocking lock is better for multi-process scenarios as it automatically waits
    lock_file = open(lock_path, 'w')
    try:
        # Acquire blocking exclusive lock (will wait until available)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        
        try:
            with h5py.File(chunk_path, 'a') as f:
                # Set attributes if file is new
                if not file_exists:
                    f.attrs['embed_dim'] = embed_dim
                    f.attrs['max_seq_len'] = max_seq_len
                
                # Write each AC_ID as a group
                for ac_id, data in chunk_data_by_ac_id.items():
                    # Encode AC_ID for group name (h5py group names must be strings)
                    group_name = str(ac_id)
                    
                    if group_name in f:
                        # Group exists, append to existing data
                        group = f[group_name]
                        existing_embeddings = group['embedding_seq']
                        existing_ranges = group['range_seq']
                        existing_ori_seqs = group.get('ori_seq')
                        existing_ptm_seqs = group.get('ptm_seq')
                        
                        # Append new embeddings and ranges
                        num_existing = existing_embeddings.shape[0]
                        num_new = len(data['embeddings'])
                        new_size = num_existing + num_new
                        
                        # Resize and append embeddings
                        existing_embeddings.resize((new_size,))
                        for i, emb in enumerate(data['embeddings']):
                            if emb.shape[0] > max_seq_len:
                                raise ValueError(
                                    f"Embedding length {emb.shape[0]} exceeds max_seq_len {max_seq_len} "
                                    f"for AC_ID {ac_id}, window {i}"
                                )
                            # Pad embedding to max_seq_len
                            padded_emb = np.zeros((max_seq_len, embed_dim), dtype=np.float32)
                            padded_emb[:emb.shape[0], :] = emb
                            existing_embeddings[num_existing + i] = padded_emb
                        
                        # Resize and append ranges
                        existing_ranges.resize((new_size, 2))
                        for i, (start, end) in enumerate(data['ranges']):
                            existing_ranges[num_existing + i] = [start, end]
                        
                        # Append ori_seq and ptm_seq if they exist
                        if 'ori_seqs' in data and existing_ori_seqs is not None:
                            existing_ori_seqs.resize((new_size,))
                            for i, seq in enumerate(data['ori_seqs']):
                                existing_ori_seqs[num_existing + i] = seq
                        
                        if 'ptm_seqs' in data and existing_ptm_seqs is not None:
                            existing_ptm_seqs.resize((new_size,))
                            for i, seq in enumerate(data['ptm_seqs']):
                                existing_ptm_seqs[num_existing + i] = seq
                    else:
                        # Create new group
                        group = f.create_group(group_name)
                        
                        # Create datasets for embeddings (variable length, padded to max_seq_len)
                        num_windows = len(data['embeddings'])
                        embedding_data = np.zeros((num_windows, max_seq_len, embed_dim), dtype=np.float32)
                        for i, emb in enumerate(data['embeddings']):
                            if emb.shape[0] > max_seq_len:
                                raise ValueError(
                                    f"Embedding length {emb.shape[0]} exceeds max_seq_len {max_seq_len} "
                                    f"for AC_ID {ac_id}, window {i}"
                                )
                            embedding_data[i, :emb.shape[0], :] = emb
                        
                        group.create_dataset('embedding_seq', data=embedding_data, 
                                            maxshape=(None, max_seq_len, embed_dim),
                                            dtype=np.float32, compression='gzip', compression_opts=9)
                        
                        # Create dataset for ranges
                        range_data = np.array(data['ranges'], dtype=np.int32)
                        group.create_dataset('range_seq', data=range_data, 
                                            maxshape=(None, 2), dtype=np.int32, compression='gzip')
                        
                        # Create datasets for ori_seq and ptm_seq (if available)
                        # Use variable-length string type for efficient storage
                        if 'ori_seqs' in data:
                            ori_seq_data = np.array(data['ori_seqs'], dtype=object)
                            group.create_dataset('ori_seq', data=ori_seq_data,
                                                maxshape=(None,),
                                                dtype=vlen_str_dtype, compression='gzip', compression_opts=9)
                        
                        if 'ptm_seqs' in data:
                            ptm_seq_data = np.array(data['ptm_seqs'], dtype=object)
                            group.create_dataset('ptm_seq', data=ptm_seq_data,
                                                maxshape=(None,),
                                                dtype=vlen_str_dtype, compression='gzip', compression_opts=9)
            
            # Release lock
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            # Release lock on error
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            raise e
    finally:
        lock_file.close()
    
    return max_seq_len


def merge_chunk_files(chunk_files, output_path, accelerator):
    """
    Merge multiple chunk HDF5 files into a single file (simple append mode).
    
    @param {list} chunk_files - List of chunk file paths to merge
    @param {str} output_path - Path to save merged file
    @param {Accelerator} accelerator - Accelerator instance for logging
    """
    if not chunk_files:
        return
    
    # Find global max_seq_len and embed_dim
    embed_dim = None
    global_max_seq_len = 0
    for chunk_file in chunk_files:
        if os.path.exists(chunk_file):
            with h5py.File(chunk_file, 'r') as f:
                global_max_seq_len = max(global_max_seq_len, f.attrs['max_seq_len'])
                if embed_dim is None:
                    embed_dim = f.attrs['embed_dim']
    
    # Simple append: read and append each chunk
    file_created = False
    for chunk_file in chunk_files:
        if not os.path.exists(chunk_file):
            continue
        
        with h5py.File(chunk_file, 'r') as f:
            unique_ids = [uid.decode('utf-8') for uid in f['unique_ids'][:]]
            lengths = f['lengths'][:]
            embeddings = f['embeddings'][:]
            
            # Pad if needed
            if embeddings.shape[1] < global_max_seq_len:
                padded = np.zeros((embeddings.shape[0], global_max_seq_len, embed_dim), dtype=np.float32)
                padded[:, :embeddings.shape[1], :] = embeddings
                embeddings = padded
            
            unique_ids_encoded = [uid.encode('utf-8') for uid in unique_ids]
            lengths_array = np.array(lengths, dtype=np.int32)
        
        # Append to output
        with h5py.File(output_path, 'a') as out_f:
            if not file_created:
                dt = h5py.special_dtype(vlen=bytes)
                out_f.create_dataset('unique_ids', data=unique_ids_encoded, maxshape=(None,), dtype=dt, compression='gzip')
                out_f.create_dataset('lengths', data=lengths_array, maxshape=(None,), dtype=np.int32, compression='gzip')
                out_f.create_dataset('embeddings', data=embeddings, maxshape=(None, global_max_seq_len, embed_dim),
                                   dtype=np.float32, compression='gzip', compression_opts=9)
                out_f.attrs['embed_dim'] = embed_dim
                out_f.attrs['max_seq_len'] = global_max_seq_len
                file_created = True
            else:
                current_size = out_f['unique_ids'].shape[0]
                new_size = current_size + len(unique_ids)
                
                out_f['unique_ids'].resize((new_size,))
                out_f['lengths'].resize((new_size,))
                out_f['embeddings'].resize((new_size, global_max_seq_len, embed_dim))
                
                out_f['unique_ids'][current_size:] = unique_ids_encoded
                out_f['lengths'][current_size:] = lengths_array
                out_f['embeddings'][current_size:] = embeddings
    
    # Clean up
    for chunk_file in chunk_files:
        try:
            os.remove(chunk_file)
        except:
            pass




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
    embeddings_path: str,
    embed_dim: int,
    max_seq_len: Optional[int],
    accelerator: Accelerator,
    chunk_idx: int
) -> Tuple[int, int]:
    """
    Flush complete AC_IDs from buffer to disk.
    
    @param buffer: EmbeddingBuffer instance
    @param embeddings_path: Path to save embeddings
    @param embed_dim: Embedding dimension
    @param max_seq_len: Maximum sequence length
    @param accelerator: Accelerator instance
    @param chunk_idx: Current chunk index
    @return: Tuple of (new_max_seq_len, num_saved_ac_ids)
    """
    complete_data = buffer.pop_complete_data()
    
    if not complete_data:
        return max_seq_len, 0
    
    # Save to disk
    new_max_seq_len = append_chunk_to_h5(
        complete_data, embeddings_path, embed_dim, max_seq_len
    )
    
    num_saved = len(complete_data)
    if accelerator.is_local_main_process:
        accelerator.print(f"💾 Saved chunk {chunk_idx} ({num_saved} complete AC_IDs, "
                         f"buffer memory: {buffer.estimate_memory_gb():.2f}GB)")
    
    return new_max_seq_len, num_saved


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
    dataset,
    tokenizer: PTMTokenizer,
    esm_model,
    batch_converter,
    accelerator,
    output_dir,
    embed_dim: int,
    batch_size=8,
    max_sequence_length=None,
    chunk_size_gb=100,
    original_sequences_dict=None,
    repr_layer=33,
    model_type="esm2",
):
    """
    Generate ESM embeddings for all sequences in the dataset.
    Uses fixed-size chunking to avoid memory explosion.
    
    @param dataset: Dataset dict with train/val/test splits.
    @param tokenizer: Tokenizer instance.
    @param esm_model: ESM model (prepared by accelerator).
    @param batch_converter: ESM alphabet batch converter.
    @param accelerator: Accelerator instance for distributed processing.
    @param output_dir: Directory to save embeddings.
    @param batch_size: Batch size for embedding generation.
    @param max_sequence_length: Maximum sequence length (same as training). If None, no limit.
    @param chunk_size_gb: Target chunk size in GB (default: 100GB)
    @param original_sequences_dict: Optional dict mapping AC_ID to original sequence (without PTM). 
                                    If provided, will use original sequences for ESM input instead of PTM sequences.
    """
    device = accelerator.device
    
    # Create output directory
    if accelerator.is_local_main_process:
        os.makedirs(output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    
    # Process each split
    for split_name in ["train", "val", "test"]:
        if split_name not in dataset or dataset[split_name] is None:
            continue
        
        split_dataset = dataset[split_name]
        if accelerator.is_local_main_process:
            accelerator.print(f"\n📊 Processing {split_name} set ({len(split_dataset)} samples)...")
        
        # All processes write to the same final file (append mode)
        embeddings_path = os.path.join(output_dir, f"{split_name}_embeddings.h5")
        
        # 🔧 Resume functionality: Load already processed AC_IDs
        processed_unique_ids = set()
        if os.path.exists(embeddings_path):
            try:
                with h5py.File(embeddings_path, 'r') as f:
                    # New structure: each AC_ID is a group
                    for ac_id in f.keys():
                        if isinstance(f[ac_id], h5py.Group):
                            processed_unique_ids.add(ac_id)
                    
                    if accelerator.is_local_main_process:
                        accelerator.print(f"📋 Found existing embeddings file with {len(processed_unique_ids)} AC_IDs")
                        accelerator.print(f"🔄 Resuming from checkpoint: {len(processed_unique_ids)} AC_IDs already processed")
            except Exception as e:
                if accelerator.is_local_main_process:
                    accelerator.print(f"⚠️  Warning: Could not read existing file for resume: {e}")
        
        # Distribute samples across processes
        num_processes = accelerator.num_processes
        process_rank = accelerator.process_index
        
        # Calculate indices for this process
        total_samples = len(split_dataset)
        samples_per_process = total_samples // num_processes
        start_idx = process_rank * samples_per_process
        end_idx = start_idx + samples_per_process if process_rank < num_processes - 1 else total_samples
        
        # Estimate chunk size in samples
        estimated_bytes_per_sample = 2 * 1024 * 1024  # 2MB per sample (compressed estimate)
        chunk_size_samples = int(chunk_size_gb * 1024 * 1024 * 1024 / estimated_bytes_per_sample)
        chunk_size_samples = max(1000, min(chunk_size_samples, 50000))  # Between 1K and 50K samples
        
        if accelerator.is_local_main_process:
            accelerator.print(f"📦 Using chunk size: ~{chunk_size_samples} samples per chunk (~{chunk_size_gb}GB)")
            accelerator.print(f"💾 All processes writing directly to: {embeddings_path}")
        
        # Initialize memory buffer
        # max_memory_gb: Safety threshold - flush if memory exceeds this (even if chunk_size not reached)
        max_memory_gb = chunk_size_gb * 1.1  # Allow buffer to grow up to 1.1x chunk_size for safety
        buffer = EmbeddingBuffer(max_memory_gb=max_memory_gb, embed_dim=embed_dim)
        chunk_idx = 0
        max_seq_len = max_sequence_length  # Will be determined on first save if None
        
        # 🔧 Resume statistics
        skipped_sequences = 0
        processed_sequences = 0
        
        # Process in batches
        process_indices = range(start_idx, end_idx)
        if accelerator.is_local_main_process:
            pbar = tqdm(range(0, len(process_indices), batch_size), desc=f"Generating {split_name} embeddings")
        else:
            pbar = range(0, len(process_indices), batch_size)
        
        for batch_offset in pbar:
            batch_samples = []
            batch_indices = []
            
            batch_start = start_idx + batch_offset
            batch_end = min(batch_start + batch_size, end_idx)
            
            for j in range(batch_start, batch_end):
                sample = split_dataset[j]
                batch_samples.append(sample)
                batch_indices.append(j)
            
            # Get sequences and unique_ids (AC_IDs)
            ac_ids = [s.get("unique_id", f"{split_name}_{j}") for j, s in zip(batch_indices, batch_samples)]
            
            # Get PTM sequences (always available from dataset)
            ptm_sequences = [s["sequence"] for s in batch_samples]
            
            # Get original sequences (if available)
            ori_sequences = []
            if original_sequences_dict is not None:
                ori_sequences = [original_sequences_dict.get(ac_id, ptm_seq) 
                                for ac_id, ptm_seq in zip(ac_ids, ptm_sequences)]
            else:
                # If no original sequences dict provided, use PTM sequences as fallback
                ori_sequences = ptm_sequences.copy()
            
            # 🔄 Use original sequences if provided, otherwise use PTM sequences for ESM input
            use_original_seq = original_sequences_dict is not None
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
                for window_info, embedding in zip(window_batch, embeddings):
                    # Use character positions for range (always based on original sequence)
                    char_start = window_info.get('char_start', window_info['window_start'])
                    char_end = window_info.get('char_end', window_info['window_end'])
                    
                    buffer.add_window(
                        ac_id=window_info['ac_id'],
                        embedding=embedding,
                        window_range=(char_start, char_end),  # Character positions in original sequence
                        ori_seq=window_info['ori_seq_window'],
                        ptm_seq=window_info['ptm_seq_window']
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
                    
                    max_seq_len, num_saved = flush_buffer_to_disk(
                        buffer=buffer,
                        embeddings_path=embeddings_path,
                        embed_dim=embed_dim,
                        max_seq_len=max_seq_len,
                        accelerator=accelerator,
                        chunk_idx=chunk_idx
                    )
                    if num_saved > 0:
                        chunk_idx += 1
                        processed_sequences += num_saved
                    
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
            max_seq_len, num_saved = flush_buffer_to_disk(
                buffer=buffer,
                embeddings_path=embeddings_path,
                embed_dim=embed_dim,
                max_seq_len=max_seq_len,
                accelerator=accelerator,
                chunk_idx=chunk_idx
            )
            if num_saved > 0:
                processed_sequences += num_saved
                if accelerator.is_local_main_process:
                    accelerator.print(f"💾 Saved final chunk ({num_saved} complete AC_IDs)")
            
            # Warn if there are incomplete AC_IDs remaining
            remaining_ac_ids = buffer.get_complete_ac_ids()
            if not remaining_ac_ids and buffer.size() > 0:
                incomplete_ac_ids = list(buffer.data.keys())
                if incomplete_ac_ids and accelerator.is_local_main_process:
                    accelerator.print(f"⚠️  Warning: {len(incomplete_ac_ids)} incomplete AC_IDs not saved: {incomplete_ac_ids[:5]}")
        
        # Wait for all processes to finish
        accelerator.wait_for_everyone()
        
        # Print final file info
        if accelerator.is_local_main_process and os.path.exists(embeddings_path):
            file_size_mb = os.path.getsize(embeddings_path) / 1024 / 1024
            accelerator.print(f"✅ Saved {split_name} embeddings to {embeddings_path}")
            accelerator.print(f"   File size: {file_size_mb:.2f} MB ({file_size_mb/1024:.2f} GB)")
            if skipped_sequences > 0:
                accelerator.print(f"   🔄 Resume: Skipped {skipped_sequences} already processed sequences")
                accelerator.print(f"   ✨ Processed {processed_sequences} new sequences")
    
    if accelerator.is_local_main_process:
        accelerator.print(f"\n✅ All embeddings generated and saved to {output_dir}")


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
        "--model",
        type=str,
        default="esm2_15b",
        choices=["esm2_650m", "esm2_15b", "esm3_7b", "esmc_300m", "esmc_600m"],
        help="ESM model to use: 'esm2_650m' (ESM2 650M), 'esm2_15b' (ESM2 15B), 'esm3_7b' (ESM3 7B), 'esmc_300m' (ESM C 300M), or 'esmc_600m' (ESM C 600M). Default: 'esm2_15b'",
    )
    parser.add_argument(
        "--repr_layer",
        type=int,
        default=None,
        help="Representation layer index to extract embeddings from. If not specified, uses model default (last layer).",
    )
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    data_config = cfg["dataset"]
    
    # Load tokenizer
    tokenizer = PTMTokenizer()
    
    # Load dataset
    dataset = get_ptm_dataset(
        tokenizer=tokenizer,
        dataset_location=data_config["dataset_location"],
        sequence_column_name=data_config["sequence_column_name"],
        val_size=data_config.get("val_size", 0),
        test_size=data_config.get("test_size", 0),
        split=data_config.get("split", True),
        subsample_size=data_config.get("subsample_size", None),
        split_seed=data_config.get("split_seed", None),
        max_sequence_length=data_config.get("max_sequence_length", None),
    )
    
    # Initialize accelerator for distributed processing
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    
    # 🔄 Load original sequences if requested
    original_sequences_dict = None
    if args.use_original_sequence:
        import pandas as pd
        if accelerator.is_local_main_process:
            accelerator.print(f"📖 Loading original sequences from column '{args.original_sequence_column}'...")
        
        df = pd.read_csv(data_config["dataset_location"])
        df.drop(df.filter(regex="^Unnamed"), axis=1, inplace=True)
        
        if args.original_sequence_column not in df.columns:
            raise ValueError(
                f"❌ Column '{args.original_sequence_column}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Get AC_ID column
        if "AC_ID" in df.columns:
            ac_ids = df["AC_ID"].tolist()
        else:
            ac_ids = df.index.tolist()
        
        # Create mapping: AC_ID -> original_sequence
        original_sequences_dict = {
            str(ac_id): str(ori_seq) 
            for ac_id, ori_seq in zip(ac_ids, df[args.original_sequence_column].tolist())
        }
        
        if accelerator.is_local_main_process:
            accelerator.print(f"✅ Loaded {len(original_sequences_dict)} original sequences")
    
    # Get representation layer from config or args (args takes precedence)
    train_args = cfg.get("training", {})
    config_repr_layer = train_args.get("repr_layer", None)
    repr_layer_override = args.repr_layer if args.repr_layer is not None else config_repr_layer
    
    # Store source for logging
    if repr_layer_override is not None:
        if args.repr_layer is not None:
            accelerator._repr_layer_source = 'cli'
        else:
            accelerator._repr_layer_source = 'config'
    
    if accelerator.is_local_main_process and repr_layer_override is not None:
        source = "command line" if args.repr_layer is not None else "config"
        accelerator.print(f"📌 Using representation layer from {source}: {repr_layer_override}")
    
    # Load ESM model based on user choice
    esm_model, alphabet, batch_converter, repr_layer, model_type, embed_dim = load_esm_model(
        args.model, accelerator, repr_layer_override=repr_layer_override
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
        accelerator.print(f"📦 Model type: {model_type}, Representation layer: {repr_layer}")
    
    # Get max_sequence_length from training config (should match training setting)
    # (train_args already loaded above)
    max_sequence_length = train_args.get("max_sequence_length", None)
    
    if accelerator.is_local_main_process:
        if max_sequence_length:
            accelerator.print(f"📏 Using max_sequence_length: {max_sequence_length} (sequences will be cropped)")
        else:
            accelerator.print(f"📏 No sequence length limit (using full sequences)")
    
    # Generate embeddings
    generate_embeddings_for_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        esm_model=esm_model,
        batch_converter=batch_converter,
        accelerator=accelerator,
        output_dir=args.output_dir,
        embed_dim=embed_dim,
        batch_size=args.batch_size,
        max_sequence_length=max_sequence_length,
        chunk_size_gb=args.chunk_size_gb,
        original_sequences_dict=original_sequences_dict,
        repr_layer=repr_layer,
        model_type=model_type,
    )


if __name__ == "__main__":
    main()

