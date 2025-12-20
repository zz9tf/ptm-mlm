"""
Utility functions for loading pre-generated ESM embeddings.
"""
import os
import h5py
import torch
import numpy as np
from typing import Dict, Optional, List


class EmbeddingLoader:
    """
    Loader for pre-generated ESM embeddings stored in HDF5 format.
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
        self.unique_id_to_index = {}
        self.original_id_to_windows = {}  # Map original_id -> list of window unique_ids
        self.random_window_selection = random_window_selection
        # Create random number generator (use seed if provided, otherwise use default)
        self.rng = np.random.RandomState(seed)
        self._load_embeddings()
    
    def _extract_original_id(self, unique_id: str) -> str:
        """
        Extract original sequence ID from window unique ID.
        @param unique_id: Window unique ID (e.g., "seq_001_window_0" or "seq_001")
        @return: Original sequence ID (e.g., "seq_001")
        """
        if "_window_" in unique_id:
            return unique_id.split("_window_")[0]
        return unique_id
    
    def _load_embeddings(self):
        """Load embeddings index from HDF5 files and build window mapping."""
        for split_name in ["train", "val", "test"]:
            embeddings_path = os.path.join(self.embeddings_dir, f"{split_name}_embeddings.h5")
            if os.path.exists(embeddings_path):
                with h5py.File(embeddings_path, 'r') as f:
                    unique_ids = [uid.decode('utf-8') for uid in f['unique_ids']]
                    # Create mapping from unique_id to index
                    for idx, uid in enumerate(unique_ids):
                        self.unique_id_to_index[uid] = (split_name, idx)
                        
                        # Build mapping from original_id to list of window unique_ids
                        original_id = self._extract_original_id(uid)
                        if original_id not in self.original_id_to_windows:
                            self.original_id_to_windows[original_id] = []
                        self.original_id_to_windows[original_id].append(uid)
                    
                    self.embeddings_files[split_name] = embeddings_path
                    print(f"📦 Loaded index for {len(unique_ids)} embeddings from {split_name}_embeddings.h5")
        
        # Log window statistics
        single_window_count = sum(1 for windows in self.original_id_to_windows.values() if len(windows) == 1)
        multi_window_count = sum(1 for windows in self.original_id_to_windows.values() if len(windows) > 1)
        print(f"📊 Window statistics: {single_window_count} sequences with 1 window, {multi_window_count} sequences with multiple windows")
    
    def _select_window_id(self, original_id: str, is_training: bool = True) -> str:
        """
        Select a window unique_id for the given original sequence ID.
        @param original_id: Original sequence ID.
        @param is_training: Whether this is for training (random selection) or evaluation (fixed selection).
        @return: Selected window unique_id.
        """
        if original_id not in self.original_id_to_windows:
            return original_id  # Return as-is if not found
        
        windows = self.original_id_to_windows[original_id]
        
        if len(windows) == 1:
            # Single window, return as-is
            return windows[0]
        
        # Multiple windows: select based on mode
        if is_training and self.random_window_selection:
            # Random selection for training (data augmentation)
            selected_idx = self.rng.randint(0, len(windows))
            return windows[selected_idx]
        else:
            # Fixed selection for evaluation (use first window for consistency)
            return windows[0]
    
    def get_embeddings(self, unique_ids: List[str], device: torch.device, is_training: bool = True) -> Optional[torch.Tensor]:
        """
        Get embeddings for a batch of samples.
        For sequences with multiple windows, randomly selects one window during training.
        @param unique_ids: List of original unique IDs for the batch samples (may be window IDs or original IDs).
        @param device: Device to place embeddings on.
        @param is_training: Whether this is for training (random window selection) or evaluation (fixed window selection).
        @returns: Tensor of embeddings with shape (batch_size, max_seq_len, embed_dim), or None if not found.
        """
        # Extract original IDs and select windows
        selected_window_ids = []
        for uid in unique_ids:
            original_id = self._extract_original_id(uid)
            window_id = self._select_window_id(original_id, is_training=is_training)
            selected_window_ids.append(window_id)
        
        # Group by split to minimize file opens
        split_indices = {}
        found_all = True
        
        for idx, window_id in enumerate(selected_window_ids):
            if window_id in self.unique_id_to_index:
                split_name, file_idx = self.unique_id_to_index[window_id]
                if split_name not in split_indices:
                    split_indices[split_name] = []
                split_indices[split_name].append((idx, file_idx))
            else:
                found_all = False
                break
        
        if not found_all:
            return None
        
        # Load embeddings from files
        batch_embeddings = [None] * len(selected_window_ids)
        for split_name, indices in split_indices.items():
            embeddings_path = self.embeddings_files[split_name]
            with h5py.File(embeddings_path, 'r') as f:
                embeddings = f['embeddings']
                lengths = f['lengths']
                for batch_idx, file_idx in indices:
                    seq_len = lengths[file_idx]
                    emb = embeddings[file_idx, :seq_len]  # Get actual sequence (without padding)
                    batch_embeddings[batch_idx] = torch.tensor(emb, device=device)
        
        # Pad to same length
        max_len = max(emb.shape[0] for emb in batch_embeddings)
        embed_dim = batch_embeddings[0].shape[1]
        
        padded_embeddings = torch.zeros(len(batch_embeddings), max_len, embed_dim, device=device)
        for i, emb in enumerate(batch_embeddings):
            padded_embeddings[i, :emb.shape[0]] = emb
        
        return padded_embeddings

