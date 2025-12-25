"""
Utility functions for loading pre-generated ESM embeddings.
New structure: AC_ID (group) -> {
    'embedding_seq': [seq1, seq2, seq3] (list of embeddings)
    'range_seq': [[start1, end1], [start2, end2], [start3, end3]] (list of ranges)
}
"""
import os
import h5py
import torch
import numpy as np
from typing import Dict, Optional, List, Tuple


class EmbeddingLoader:
    """
    Loader for pre-generated ESM embeddings stored in HDF5 format.
    New structure: Each AC_ID is a group containing all windows' embeddings and ranges.
    Supports random window selection for sequences with multiple windows.
    If seed is provided, window selection will be reproducible.
    """
    def __init__(self, embeddings_dir: str, random_window_selection: bool = True, seed: Optional[int] = None):
        """
        Initialize embedding loader.
        @param embeddings_dir: Directory containing embedding HDF5 files.
        @param random_window_selection: If True, randomly select a window from multiple windows of the same sequence during training.
        @param seed: Random seed for window selection. If provided, ensures reproducibility.
        """
        self.embeddings_dir = embeddings_dir
        self.embeddings_files = {}
        self.ac_id_to_split = {}  # Map AC_ID -> split_name
        self.random_window_selection = random_window_selection
        # Create random number generator (use seed if provided, otherwise use default)
        self.rng = np.random.RandomState(seed)
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load embeddings index from HDF5 files (new structure)."""
        for split_name in ["train", "val", "test"]:
            embeddings_path = os.path.join(self.embeddings_dir, f"{split_name}_embeddings.h5")
            if os.path.exists(embeddings_path):
                with h5py.File(embeddings_path, 'r') as f:
                    # New structure: each AC_ID is a group
                    ac_ids = []
                    for key in f.keys():
                        if isinstance(f[key], h5py.Group):
                            ac_id = key
                            ac_ids.append(ac_id)
                            self.ac_id_to_split[ac_id] = split_name
                    
                    self.embeddings_files[split_name] = embeddings_path
                    print(f"📦 Loaded index for {len(ac_ids)} AC_IDs from {split_name}_embeddings.h5")
        
        # Log window statistics
        single_window_count = 0
        multi_window_count = 0
        for split_name in ["train", "val", "test"]:
            embeddings_path = self.embeddings_files.get(split_name)
            if embeddings_path and os.path.exists(embeddings_path):
                with h5py.File(embeddings_path, 'r') as f:
                    for ac_id in f.keys():
                        if isinstance(f[ac_id], h5py.Group):
                            group = f[ac_id]
                            if 'embedding_seq' in group:
                                num_windows = group['embedding_seq'].shape[0]
                                if num_windows == 1:
                                    single_window_count += 1
                                else:
                                    multi_window_count += 1
        
        print(f"📊 Window statistics: {single_window_count} sequences with 1 window, {multi_window_count} sequences with multiple windows")
    
    def _select_window_idx(self, num_windows: int, is_training: bool = True) -> int:
        """
        Select a window index for the given number of windows.
        @param num_windows: Number of windows available.
        @param is_training: Whether this is for training (random selection) or evaluation (fixed selection).
        @return: Selected window index (0-based).
        """
        if num_windows == 1:
            return 0
        
        # Multiple windows: select based on mode
        if is_training and self.random_window_selection:
            # Random selection for training (data augmentation)
            return self.rng.randint(0, num_windows)
        else:
            # Fixed selection for evaluation (use first window for consistency)
            return 0
    
    def has_embeddings(self, ac_ids: List[str], is_training: bool = True) -> List[bool]:
        """
        Check which AC_IDs have corresponding embeddings.
        @param ac_ids: List of AC_IDs (original IDs).
        @param is_training: Whether this is for training (random window selection) or evaluation (fixed window selection).
        @return: List of boolean values indicating whether each AC_ID has a corresponding embedding.
        """
        has_emb_list = []
        for ac_id in ac_ids:
            has_emb_list.append(ac_id in self.ac_id_to_split)
        return has_emb_list
    
    def get_embeddings(self, ac_ids: List[str], device: torch.device, is_training: bool = True) -> Optional[Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]]:
        """
        Get embeddings for a batch of samples.
        For sequences with multiple windows, randomly selects one window during training.
        @param ac_ids: List of AC_IDs (original IDs) for the batch samples.
        @param device: Device to place embeddings on.
        @param is_training: Whether this is for training (random window selection) or evaluation (fixed window selection).
        @returns: Tuple of (embeddings tensor with shape (batch_size, max_seq_len, embed_dim), 
                 list of (start, end) ranges for each sample, list of actual sequence lengths),
                 or None if not all found.
        """
        # Group by split to minimize file opens
        split_ac_ids = {}
        found_all = True
        
        for idx, ac_id in enumerate(ac_ids):
            if ac_id in self.ac_id_to_split:
                split_name = self.ac_id_to_split[ac_id]
                if split_name not in split_ac_ids:
                    split_ac_ids[split_name] = []
                split_ac_ids[split_name].append((idx, ac_id))
            else:
                found_all = False
                break
        
        if not found_all:
            return None
        
        # Load embeddings from files
        batch_embeddings = [None] * len(ac_ids)
        batch_ranges = [None] * len(ac_ids)
        seq_lengths = [0] * len(ac_ids)
        
        for split_name, ac_id_list in split_ac_ids.items():
            embeddings_path = self.embeddings_files[split_name]
            with h5py.File(embeddings_path, 'r') as f:
                for batch_idx, ac_id in ac_id_list:
                    if ac_id not in f or not isinstance(f[ac_id], h5py.Group):
                        continue
                    
                    group = f[ac_id]
                    embedding_seq = group['embedding_seq']  # Shape: (num_windows, max_seq_len, embed_dim)
                    range_seq = group['range_seq']  # Shape: (num_windows, 2)
                    
                    num_windows = embedding_seq.shape[0]
                    
                    # Select one window
                    selected_window_idx = self._select_window_idx(num_windows, is_training=is_training)
                    
                    # Get selected window's embedding and range
                    selected_emb = embedding_seq[selected_window_idx]  # Shape: (max_seq_len, embed_dim)
                    selected_range = range_seq[selected_window_idx]  # Shape: (2,) -> [start, end]
                    
                    # Extract actual sequence length from range
                    actual_len = int(selected_range[1] - selected_range[0])
                    
                    # Extract the actual embedding (remove padding)
                    # The embedding is padded to max_seq_len, but we only need the actual window length
                    actual_emb = selected_emb[:actual_len]  # Shape: (actual_len, embed_dim)
                    
                    seq_lengths[batch_idx] = actual_len
                    batch_embeddings[batch_idx] = torch.tensor(actual_emb, device=device)
                    batch_ranges[batch_idx] = (int(selected_range[0]), int(selected_range[1]))
        
        # Pad to same length
        max_len = max(emb.shape[0] for emb in batch_embeddings if emb is not None)
        embed_dim = batch_embeddings[0].shape[1]
        
        padded_embeddings = torch.zeros(len(batch_embeddings), max_len, embed_dim, device=device)
        for i, emb in enumerate(batch_embeddings):
            if emb is not None:
                padded_embeddings[i, :emb.shape[0]] = emb
        
        return padded_embeddings, batch_ranges, seq_lengths
