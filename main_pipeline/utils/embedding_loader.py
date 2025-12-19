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
    """
    def __init__(self, embeddings_dir: str):
        """
        Initialize embedding loader.
        @param embeddings_dir: Directory containing embedding HDF5 files.
        """
        self.embeddings_dir = embeddings_dir
        self.embeddings_files = {}
        self.unique_id_to_index = {}
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load embeddings index from HDF5 files."""
        for split_name in ["train", "val", "test"]:
            embeddings_path = os.path.join(self.embeddings_dir, f"{split_name}_embeddings.h5")
            if os.path.exists(embeddings_path):
                with h5py.File(embeddings_path, 'r') as f:
                    unique_ids = [uid.decode('utf-8') for uid in f['unique_ids']]
                    # Create mapping from unique_id to index
                    for idx, uid in enumerate(unique_ids):
                        self.unique_id_to_index[uid] = (split_name, idx)
                    self.embeddings_files[split_name] = embeddings_path
                    print(f"📦 Loaded index for {len(unique_ids)} embeddings from {split_name}_embeddings.h5")
    
    def get_embeddings(self, unique_ids: List[str], device: torch.device) -> Optional[torch.Tensor]:
        """
        Get embeddings for a batch of samples.
        @param unique_ids: List of unique IDs for the batch samples.
        @param device: Device to place embeddings on.
        @returns: Tensor of embeddings with shape (batch_size, max_seq_len, embed_dim), or None if not found.
        """
        embeddings_list = []
        found_all = True
        
        # Group by split to minimize file opens
        split_indices = {}
        for idx, unique_id in enumerate(unique_ids):
            if unique_id in self.unique_id_to_index:
                split_name, file_idx = self.unique_id_to_index[unique_id]
                if split_name not in split_indices:
                    split_indices[split_name] = []
                split_indices[split_name].append((idx, file_idx))
            else:
                found_all = False
                break
        
        if not found_all:
            return None
        
        # Load embeddings from files
        batch_embeddings = [None] * len(unique_ids)
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

