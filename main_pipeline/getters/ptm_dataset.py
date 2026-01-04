"""
PTM Dataset for embedding mode only.
Input: original_sequence
Output: original_sequence and ptm_sequence
Loss: computed on all original_sequence positions and all PTM sites.
"""
import os
import re
import pandas as pd
import torch
import h5py
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset as TorchDataset, Sampler
from torch.nn.utils.rnn import pad_sequence


class EmbeddingLoader:
    """
    Loader for pre-generated ESM embeddings stored in HDF5 format.
    Each AC_ID is a group containing all windows' embeddings, ranges, and sequences.
    Supports random window selection for sequences with multiple windows.
    """
    def __init__(self, embeddings_dir: str, random_window_selection: bool = True, seed: Optional[int] = None):
        """
        Initialize embedding loader.
        @param embeddings_dir: Directory containing embedding HDF5 files.
        @param random_window_selection: If True, randomly select a window from multiple windows during training.
        @param seed: Random seed for window selection. If provided, ensures reproducibility.
        """
        self.embeddings_dir = embeddings_dir
        self.embeddings_files = {}
        self.ac_id_to_split = {}  # Map AC_ID -> split_name
        self.random_window_selection = random_window_selection
        self.rng = np.random.RandomState(seed)
        self._load_embeddings()
    
    def _load_embeddings(self):
        """
        Load embeddings index from HDF5 files.
        Compatible with the new structure where each AC_ID is a group containing:
        - embedding_seq: (num_windows, max_seq_len, embed_dim)
        - range_seq: (num_windows, 2)
        - ori_seq: (num_windows,) - variable-length strings
        - ptm_seq: (num_windows,) - variable-length strings
        """
        for split_name in ["train", "val", "test"]:
            embeddings_path = os.path.join(self.embeddings_dir, f"{split_name}_embeddings.h5")
            if os.path.exists(embeddings_path):
                try:
                    with h5py.File(embeddings_path, 'r') as f:
                        ac_ids = []
                        for key in f.keys():
                            if isinstance(f[key], h5py.Group):
                                # Verify it has the expected structure
                                group = f[key]
                                if 'embedding_seq' in group and 'range_seq' in group:
                                    ac_id = str(key)  # Ensure string type
                                    ac_ids.append(ac_id)
                                    self.ac_id_to_split[ac_id] = split_name
                        
                        self.embeddings_files[split_name] = embeddings_path
                        print(f"📦 Loaded index for {len(ac_ids)} AC_IDs from {split_name}_embeddings.h5")
                except Exception as e:
                    print(f"⚠️  Warning: Could not load embeddings from {embeddings_path}: {e}")
        
        # Log window statistics
        single_window_count = 0
        multi_window_count = 0
        total_windows = 0
        for split_name in ["train", "val", "test"]:
            embeddings_path = self.embeddings_files.get(split_name)
            if embeddings_path and os.path.exists(embeddings_path):
                try:
                    with h5py.File(embeddings_path, 'r') as f:
                        for ac_id in f.keys():
                            if isinstance(f[ac_id], h5py.Group):
                                group = f[ac_id]
                                if 'embedding_seq' in group:
                                    num_windows = group['embedding_seq'].shape[0]
                                    total_windows += num_windows
                                    if num_windows == 1:
                                        single_window_count += 1
                                    else:
                                        multi_window_count += 1
                except Exception as e:
                    print(f"⚠️  Warning: Could not read window statistics from {embeddings_path}: {e}")
        
        print(f"📊 Window statistics: {single_window_count} sequences with 1 window, "
              f"{multi_window_count} sequences with multiple windows, "
              f"total {total_windows} windows")
    
    def _select_window_idx(self, num_windows: int, is_training: bool = True) -> int:
        """
        Select a window index for the given number of windows.
        @param num_windows: Number of windows available.
        @param is_training: Whether this is for training (random selection) or evaluation (fixed selection).
        @return: Selected window index (0-based).
        """
        if num_windows == 1:
            return 0
        
        if is_training and self.random_window_selection:
            return self.rng.randint(0, num_windows)
        else:
            return 0
    
    def get_single_sample(
        self, 
        ac_id: str, 
        device: torch.device, 
        is_training: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get a single sample with embedding, original sequence, PTM sequence, and PTM positions.
        Compatible with the new embeddings structure generated by generate_embeddings.py.
        
        Structure expected:
        - AC_ID (group) -> {
            'embedding_seq': (num_windows, max_seq_len, embed_dim) - padded embeddings
            'range_seq': (num_windows, 2) - [start, end] character positions in full sequence
            'ori_seq': (num_windows,) - variable-length strings, original sequence windows
            'ptm_seq': (num_windows,) - variable-length strings, PTM sequence windows
        }
        
        @param ac_id: AC_ID (original ID) for the sample.
        @param device: Device to place embedding on.
        @param is_training: Whether this is for training (random window selection) or evaluation (fixed window selection).
        @returns: Dict with:
            - 'embeddings': torch.Tensor of shape (seq_length, embed_dim)
            - 'original_sequence': str, original sequence for the selected window
            - 'ptm_sequence': str, PTM sequence for the selected window
            - 'ptm_positions': List[int], positions in original_sequence where PTM modifications occur
            - 'range': Tuple[int, int], (start, end) range in the full sequence
            - 'seq_length': int, length of the sequence
            - 'window_idx': int, selected window index
        """
        if ac_id not in self.ac_id_to_split:
            return None
        
        split_name = self.ac_id_to_split[ac_id]
        embeddings_path = self.embeddings_files[split_name]
        
        try:
            with h5py.File(embeddings_path, 'r') as f:
                # Convert ac_id to string to match group name format
                ac_id_str = str(ac_id)
                
                if ac_id_str not in f or not isinstance(f[ac_id_str], h5py.Group):
                    return None
                
                group = f[ac_id_str]
                
                # Verify required datasets exist
                if 'embedding_seq' not in group or 'range_seq' not in group:
                    return None
                
                embedding_seq = group['embedding_seq']  # Shape: (num_windows, max_seq_len, embed_dim)
                range_seq = group['range_seq']  # Shape: (num_windows, 2)
                
                num_windows = embedding_seq.shape[0]
                
                if num_windows == 0:
                    return None
                
                # Select one window randomly (for training) or fixed (for evaluation)
                selected_window_idx = self._select_window_idx(num_windows, is_training=is_training)
                
                # Get selected window's embedding and range
                selected_emb = embedding_seq[selected_window_idx]  # Shape: (max_seq_len, embed_dim)
                selected_range = range_seq[selected_window_idx]  # Shape: (2,) -> [start, end]
                
                # Extract actual sequence length
                # Method 1: Use range (character positions in original sequence)
                # Range represents character positions in the full original sequence
                range_based_len = int(selected_range[1] - selected_range[0])
                
                # Method 2: Detect actual embedding length by finding non-zero padding
                # Embeddings are padded with zeros, so we can detect the actual length
                max_emb_len = selected_emb.shape[0]  # This is max_seq_len (padded length)
                
                # Find actual embedding length by checking for padding (zeros)
                # For ESM embeddings, padding is typically zeros
                # We look for the last non-zero row (with some tolerance for numerical precision)
                non_zero_mask = np.any(np.abs(selected_emb) > 1e-6, axis=1)
                if np.any(non_zero_mask):
                    embedding_based_len = int(np.where(non_zero_mask)[0][-1] + 1)
                else:
                    embedding_based_len = max_emb_len
                
                # Use the smaller of the two (defensive: range should match embedding length)
                # In normal cases, range_based_len should equal embedding_based_len
                actual_len = min(range_based_len, embedding_based_len, max_emb_len)
                
                # Log warning if there's a mismatch (indicates potential data inconsistency)
                if range_based_len != embedding_based_len and range_based_len <= max_emb_len and embedding_based_len <= max_emb_len:
                    # Only warn if both are valid but different (not a critical error)
                    pass  # Could add logging here if needed: print(f"⚠️  Length mismatch for {ac_id}: range={range_based_len}, embedding={embedding_based_len}")
                
                # Extract the actual embedding (remove padding)
                actual_emb = selected_emb[:actual_len]  # Shape: (actual_len, embed_dim)
                
                # Extract original sequence and PTM sequence for the selected window
                # These are stored as variable-length strings in the HDF5 file
                selected_ori_seq = None
                selected_ptm_seq = None
                
                if 'ori_seq' in group:
                    ori_seq_data = group['ori_seq']
                    selected_ori_seq = ori_seq_data[selected_window_idx]
                    
                    # Convert to string (handle different storage formats)
                    if isinstance(selected_ori_seq, bytes):
                        selected_ori_seq = selected_ori_seq.decode('utf-8')
                    elif isinstance(selected_ori_seq, np.ndarray):
                        if selected_ori_seq.dtype == object:
                            item = selected_ori_seq.item()
                            if isinstance(item, bytes):
                                selected_ori_seq = item.decode('utf-8')
                            else:
                                selected_ori_seq = str(item)
                        else:
                            selected_ori_seq = str(selected_ori_seq)
                    elif not isinstance(selected_ori_seq, str):
                        selected_ori_seq = str(selected_ori_seq)
                
                if 'ptm_seq' in group:
                    ptm_seq_data = group['ptm_seq']
                    selected_ptm_seq = ptm_seq_data[selected_window_idx]
                    
                    # Convert to string (handle different storage formats)
                    if isinstance(selected_ptm_seq, bytes):
                        selected_ptm_seq = selected_ptm_seq.decode('utf-8')
                    elif isinstance(selected_ptm_seq, np.ndarray):
                        if selected_ptm_seq.dtype == object:
                            item = selected_ptm_seq.item()
                            if isinstance(item, bytes):
                                selected_ptm_seq = item.decode('utf-8')
                            else:
                                selected_ptm_seq = str(item)
                        else:
                            selected_ptm_seq = str(selected_ptm_seq)
                    elif not isinstance(selected_ptm_seq, str):
                        selected_ptm_seq = str(selected_ptm_seq)
                
                # Fallback: if sequences are not available, use empty strings
                if selected_ori_seq is None:
                    selected_ori_seq = ""
                if selected_ptm_seq is None:
                    selected_ptm_seq = ""
                
                # Extract PTM positions from ptm_sequence
                # PTM markers like <Phosphoserine> appear AFTER the modified amino acid
                ptm_positions = self._extract_ptm_positions(selected_ori_seq, selected_ptm_seq)
                
                return {
                    'embeddings': torch.tensor(actual_emb, dtype=torch.float32, device=device),
                    'original_sequence': selected_ori_seq,
                    'ptm_sequence': selected_ptm_seq,
                    'ptm_positions': ptm_positions,
                    'range': (int(selected_range[0]), int(selected_range[1])),
                    'seq_length': actual_len,
                    'window_idx': selected_window_idx
                }
        except Exception as e:
            print(f"⚠️  Error loading sample for AC_ID {ac_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_ptm_positions(self, ori_seq: str, ptm_seq: str) -> List[int]:
        """
        Extract PTM positions from ptm_seq and map to positions in ori_seq.
        PTM markers like <Phosphoserine> appear AFTER the modified amino acid.
        @param ori_seq: Original sequence without PTM annotations.
        @param ptm_seq: Sequence with PTM annotations like <Phosphoserine>.
        @returns: List of positions (0-based) in ori_seq where PTM modifications occur.
        """
        ptm_positions = []
        pattern = r'<([^>]+)>'
        
        # Find all PTM markers in ptm_seq
        matches = list(re.finditer(pattern, ptm_seq))
        
        # For each marker, find its position in the original sequence
        for match in matches:
            marker_start = match.start()
            
            # Count how many regular characters (not in markers) are before this marker
            # This gives us the position in the original sequence
            char_count = 0
            i = 0
            while i < marker_start:
                if ptm_seq[i] == '<':
                    # Skip the entire marker
                    marker_end = ptm_seq.find('>', i) + 1
                    if marker_end > 0:
                        i = marker_end
                    else:
                        break
                else:
                    char_count += 1
                    i += 1
            
            # The modified AA is at position char_count (0-based) in ori_seq
            # PTM marker appears AFTER the modified AA
            if char_count > 0 and char_count <= len(ori_seq):
                ptm_positions.append(char_count - 1)  # 0-based position
        
        return sorted(list(set(ptm_positions)))  # Remove duplicates and sort


class PTMDataset(TorchDataset):
    """
    PTM Dataset for embedding mode only.
    Input: original_sequence
    Output: original_sequence and ptm_sequence
    """

    def __init__(
        self, 
        samples: List[Dict[str, Any]], 
        embedding_loader: EmbeddingLoader,
        device: Optional[torch.device] = None
    ):
        """
        @param samples: List of sample dicts with 'unique_id' (AC_ID).
        @param embedding_loader: EmbeddingLoader instance.
        @param device: Device to place embeddings on.
        """
        super().__init__()
        self.samples = samples
        self.embedding_loader = embedding_loader
        self.device = device if device is not None else torch.device('cpu')

    def __len__(self) -> int:
        """@returns Dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        @param idx Index of the sample.
        @returns Sample dict containing:
            - 'embeddings': torch.Tensor, shape (seq_length, embed_dim)
            - 'original_sequence': str
            - 'ptm_sequence': str
            - 'ptm_positions': List[int], positions in original_sequence where PTM modifications occur
            - 'unique_id': str, AC_ID
            - 'range': Tuple[int, int], (start, end) range
            - 'seq_length': int
            - 'window_idx': int
        """
        sample = self.samples[idx]
        unique_id = sample.get('unique_id', str(idx))
        is_training = sample.get('split', 'train') == 'train'
        
        embedding_data = self.embedding_loader.get_single_sample(
            ac_id=unique_id,
            device=self.device,
            is_training=is_training
        )
        
        if embedding_data is None:
            raise ValueError(f"Embedding not found for AC_ID: {unique_id}")
        
        # Add unique_id and split info
        embedding_data['unique_id'] = unique_id
        embedding_data['split'] = sample.get('split', 'train')
        
        return embedding_data

    @classmethod
    def from_csv(
        cls,
        dataset_location: str,
        sequence_column_name: str,
        embedding_loader: EmbeddingLoader,
        device: Optional[torch.device] = None,
        subsample_size: Optional[int] = None,
    ) -> "PTMDataset":
        """
        @param dataset_location: CSV path.
        @param sequence_column_name: Column containing raw sequences.
        @param embedding_loader: EmbeddingLoader instance.
        @param device: Device to place embeddings on.
        @param subsample_size: Optional sample cap.
        @returns PTMDataset instance.
        """
        df = pd.read_csv(dataset_location)
        df.drop(df.filter(regex="^Unnamed"), axis=1, inplace=True)
        df.drop_duplicates(subset=sequence_column_name, inplace=True)

        if subsample_size is not None:
            df = df.iloc[:subsample_size]

        # Get unique identifier (AC_ID or index)
        if "AC_ID" in df.columns:
            unique_ids = df["AC_ID"].tolist()
        else:
            unique_ids = df.index.tolist()
        
        samples = []
        for uid in unique_ids:
            samples.append({
                "unique_id": str(uid),
            })

        return cls(samples, embedding_loader=embedding_loader, device=device)

    def split(
        self,
        val_size: int,
        test_size: int,
        split_seed: Optional[int] = None,
    ) -> Dict[str, Optional["PTMDataset"]]:
        """
        @param val_size: Number of samples for validation.
        @param test_size: Number of samples for test.
        @param split_seed: Optional seed for deterministic shuffling.
        @returns Dict with train/val/test PTMDataset; val/test may be None.
        """
        total = len(self.samples)
        if val_size + test_size > total:
            raise ValueError(
                f"val_size + test_size exceeds dataset size ({val_size + test_size} > {total})"
            )

        indices = list(range(total))
        rng = np.random.RandomState(split_seed)
        rng.shuffle(indices)

        test_start = total - test_size
        val_start = test_start - val_size

        train_idx = indices[:val_start] if val_start > 0 else []
        val_idx = indices[val_start:test_start] if val_size > 0 else []
        test_idx = indices[test_start:] if test_size > 0 else []

        def _subset(idxs: List[int], split_name: str) -> Optional["PTMDataset"]:
            if not idxs:
                return None
            subset_samples = [self.samples[i] for i in idxs]
            # Add split information to each sample
            for sample in subset_samples:
                sample["split"] = split_name
            return PTMDataset(
                subset_samples, 
                embedding_loader=self.embedding_loader,
                device=self.device
            ) if subset_samples else None

        splits = {
            "train": _subset(train_idx, "train"),
            "val": _subset(val_idx, "val"),
            "test": _subset(test_idx, "test"),
        }
        
        # Store split mapping for export
        self._split_mapping = {}
        for split_name, idxs in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            for idx in idxs:
                unique_id = self.samples[idx].get("unique_id", str(idx))
                self._split_mapping[unique_id] = split_name
        
        return splits
    
    def get_split_mapping(self) -> Dict[str, str]:
        """
        Get mapping of unique_id to split (train/val/test).
        @returns: Dictionary mapping unique_id to split name.
        """
        if not hasattr(self, "_split_mapping"):
            return {}
        return self._split_mapping.copy()


def get_ptm_dataset(
    dataset_location: str,
    sequence_column_name: str,
    embedding_loader: EmbeddingLoader,
    device: Optional[torch.device] = None,
    val_size: int = 0,
    test_size: int = 0,
    split: bool = True,
    subsample_size: Optional[int] = None,
    split_seed: Optional[int] = None,
) -> Union[PTMDataset, Dict[str, Optional[PTMDataset]]]:
    """
    @param dataset_location: CSV path.
    @param sequence_column_name: Column containing raw sequences.
    @param embedding_loader: EmbeddingLoader instance.
    @param device: Device to place embeddings on.
    @param val_size: Validation set size.
    @param test_size: Test set size.
    @param split: Whether to return train/val/test splits. If False, return full PTMDataset.
    @param subsample_size: Optional sample cap.
    @param split_seed: Optional seed for deterministic split.
    @returns PTMDataset or dict of PTMDataset splits.
    """
    dataset = PTMDataset.from_csv(
        dataset_location=dataset_location,
        sequence_column_name=sequence_column_name,
        embedding_loader=embedding_loader,
        device=device,
        subsample_size=subsample_size,
    )

    if not split:
        return dataset

    splits = dataset.split(
        val_size=val_size,
        test_size=test_size,
        split_seed=split_seed,
    )
    
    # Get split mapping and add to return dict
    split_mapping = dataset.get_split_mapping()
    splits["split_mapping"] = split_mapping
    
    return splits


class DataCollatorWithPadding:
    """
    Data collator that pads embeddings and sequences.
    """

    def __init__(
        self,
        max_tokens: int = None,
        batch_by_tokens: bool = False,
    ) -> None:
        """
        @param max_tokens: Maximum tokens per batch (optional).
        @param batch_by_tokens: Whether to batch by token count.
        """
        self.max_tokens = max_tokens
        self.batch_by_tokens = batch_by_tokens

    def __call__(self, batch):
        """
        Generate a batch of data.
        @param batch: List of dictionaries with keys 'embeddings', 'original_sequence', 'ptm_sequence', etc.
        @returns: Dict with:
            - 'embeddings': Padded tensor, shape (batch_size, max_seq_len, embed_dim)
            - 'pad_mask': Padding mask, shape (batch_size, max_seq_len)
            - 'seq_lengths': List of actual sequence lengths
            - 'original_sequences': List of original sequences
            - 'ptm_sequences': List of PTM sequences
            - 'ptm_positions': List of PTM position lists
            - 'unique_ids': List of unique IDs
        """
        embeddings = [item["embeddings"] for item in batch]
        seq_lengths = [item["seq_length"] for item in batch]
        original_sequences = [item["original_sequence"] for item in batch]
        ptm_sequences = [item["ptm_sequence"] for item in batch]
        ptm_positions = [item["ptm_positions"] for item in batch]
        unique_ids = [item.get("unique_id", None) for item in batch]
        
        # Pad embeddings to same length
        max_len = max(emb.shape[0] for emb in embeddings)
        embed_dim = embeddings[0].shape[1]
        device = embeddings[0].device
        
        padded_embeddings = torch.zeros(len(embeddings), max_len, embed_dim, device=device)
        pad_mask = torch.zeros(len(embeddings), max_len, dtype=torch.bool, device=device)
        
        for i, emb in enumerate(embeddings):
            seq_len = emb.shape[0]
            padded_embeddings[i, :seq_len] = emb
            pad_mask[i, :seq_len] = True
        
        if self.batch_by_tokens and self.max_tokens is not None:
            # Keep a few sequences to make the total number of tokens in the batch <= max_tokens
            total_tokens = padded_embeddings.numel()
            if total_tokens > self.max_tokens:
                max_num_seq = self.max_tokens // (max_len * embed_dim) + 1
                # Randomly select max_num_seq sequences from the batch to keep
                indices = torch.randperm(len(padded_embeddings))[:max_num_seq]
                padded_embeddings = padded_embeddings[indices]
                pad_mask = pad_mask[indices]
                seq_lengths = [seq_lengths[i] for i in indices]
                original_sequences = [original_sequences[i] for i in indices]
                ptm_sequences = [ptm_sequences[i] for i in indices]
                ptm_positions = [ptm_positions[i] for i in indices]
                unique_ids = [unique_ids[i] for i in indices]
        
        result = {
            "embeddings": padded_embeddings,
            "pad_mask": pad_mask,
            "seq_lengths": seq_lengths,
            "original_sequences": original_sequences,
            "ptm_sequences": ptm_sequences,
            "ptm_positions": ptm_positions,
        }
        if any(uid is not None for uid in unique_ids):
            result["unique_ids"] = unique_ids
        return result


class SequenceLengthSampler(Sampler):
    """
    Sampler that sorts sequences by length for efficient batching.
    """

    def __init__(self, dataset, sort: bool = True, sample_len_ascending: bool = True):
        """
        @param dataset: PTMDataset instance.
        @param sort: Whether to sort by sequence length.
        @param sample_len_ascending: If True, sample shorter sequences first.
        """
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        if sort is True:
            def get_length(idx):
                sample = dataset[idx]
                return sample.get("seq_length", 0)
            
            self.indices.sort(
                key=get_length,
                reverse=not sample_len_ascending,
            )

    def __iter__(self):
        """@returns: Iterator over sorted indices."""
        return iter(self.indices)

    def __len__(self):
        """@returns: Number of samples."""
        return len(self.indices)
