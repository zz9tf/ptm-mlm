"""
ESM-C 600M model inference script for generating embeddings.
This script loads ESM-C 600M model and generates embeddings for protein sequences.
This is a shared module used by all downstream tasks.
"""
import torch
import sys
from pathlib import Path
from tqdm import tqdm

try:
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig
    ESMC_AVAILABLE = True
except ImportError:
    ESMC_AVAILABLE = False
    print("⚠️  esm library not found. Please install: pip install fair-esm")


class ESMCInference:
    """
    ESM-C 600M model inference class for generating embeddings.
    This class is shared across all downstream tasks.
    """
    
    def __init__(self, device: str = None, max_sequence_length: int = None, layer_index: int = None):
        """
        Initialize the ESM-C 600M inference model.
        
        @param device: Device to run inference on (None for auto-detect)
        @param max_sequence_length: Maximum sequence length (not used for ESM-C, but kept for compatibility)
        @param layer_index: Layer index to extract (1-based). If None, uses last layer (default: None)
        """
        if not ESMC_AVAILABLE:
            raise ImportError("esm library is required for ESM-C 600M. Install with: pip install fair-esm")
        
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load ESM-C 600M model
        print(f"📦 Loading ESM-C 600M model...")
        try:
            self.esm_model = ESMC.from_pretrained("esmc_600m")
            self.esm_model = self.esm_model.to(self.device)
            self.esm_model.eval()
            for param in self.esm_model.parameters():
                param.requires_grad = False
        except Exception as e:
            raise RuntimeError(
                f"❌ Failed to load ESM-C 600M model: {e}\n"
                f"   Please check:\n"
                f"   1. esm library is correctly installed\n"
                f"   2. esm library version supports ESM-C 600M\n"
                f"   3. Network connection is normal (first use needs to download model)\n"
                f"   Install command: pip install fair-esm"
            )
        
        # Set layer index (1-based for ESM-C)
        self.layer_index = layer_index
        if layer_index is not None:
            print(f"🔍 Will extract embeddings from layer {layer_index} (1-based)")
        else:
            print(f"🔍 Will extract embeddings from last layer (default)")
        
        # ESM-C 600M has 1152 dimensions
        self.hidden_size = 1152
        
        # Set max sequence length (for compatibility, ESM-C handles this internally)
        if max_sequence_length is None:
            self.max_sequence_length = 1024  # ESM-C default
        else:
            self.max_sequence_length = max_sequence_length
        
        print(f"✅ ESM-C 600M model loaded successfully! Hidden size: {self.hidden_size}")
        print(f"📏 Max sequence length: {self.max_sequence_length}")
    
    @torch.no_grad()
    def _compute_esmc_embedding(self, sequences: list):
        """
        Use ESM-C 600M to compute embeddings.
        
        @param sequences: List of protein sequences (strings)
        @returns: ESM-C embeddings tensor with shape (batch_size, seq_len, embed_dim)
        """
        batch_embeddings = []
        
        for seq in sequences:
            try:
                seq_len = len(seq)  # Original sequence length (without special tokens)
                
                # Use ESM-C SDK API
                protein = ESMProtein(sequence=seq)
                protein_tensor = self.esm_model.encode(protein)
                
                if hasattr(protein_tensor, 'error'):
                    raise RuntimeError(f"ESM-C encoding failed: {protein_tensor.error}")
                
                # Get embeddings from specified layer or last layer if None
                if self.layer_index is not None:
                    logits_config = LogitsConfig(
                        sequence=True, 
                        return_embeddings=True,
                        ith_hidden_layer=self.layer_index  # Use specified layer (1-based)
                    )
                else:
                    # Use default (last layer) by not specifying ith_hidden_layer
                    logits_config = LogitsConfig(
                        sequence=True, 
                        return_embeddings=True
                    )
                logits_output = self.esm_model.logits(protein_tensor, logits_config)
                
                # ESM-C returns embeddings as numpy array or torch tensor
                if hasattr(logits_output, 'embeddings'):
                    embeddings = logits_output.embeddings
                    # Convert to torch tensor (if numpy array)
                    if not isinstance(embeddings, torch.Tensor):
                        embeddings = torch.tensor(embeddings, device=self.device)
                    else:
                        embeddings = embeddings.to(self.device)
                    embeddings = embeddings.squeeze(0)
                    
                    # 🔧 Fix: ESM-C embeddings may include special tokens (BOS/EOS)
                    # Remove special tokens to match sequence length
                    emb_len = embeddings.shape[0]
                    if emb_len != seq_len:
                        if emb_len == seq_len + 2:
                            # Remove first and last token (BOS and EOS)
                            embeddings = embeddings[1:-1]
                        else:
                            # Unexpected length mismatch
                            raise RuntimeError(
                                f"❌ Embedding length mismatch: emb_len={emb_len}, seq_len={seq_len}, "
                                f"expected diff=2 (BOS+EOS) but got diff={emb_len - seq_len}"
                            )
                else:
                    raise RuntimeError("ESM-C logits_output does not have embeddings attribute")
                
                batch_embeddings.append(embeddings)
            except Exception as e:
                raise RuntimeError(f"❌ ESM-C 600M processing sequence failed: {e}")
        
        # Align sequence lengths (padding)
        if len(batch_embeddings) == 0:
            raise RuntimeError("No embeddings generated successfully")
        
        max_len = max(emb.shape[0] for emb in batch_embeddings)
        embed_dim = batch_embeddings[0].shape[1]
        batch_size = len(batch_embeddings)
        
        padded_embeddings = torch.zeros(batch_size, max_len, embed_dim, device=self.device)
        for i, emb in enumerate(batch_embeddings):
            emb_len = emb.shape[0]
            # Ensure emb is on correct device
            if isinstance(emb, torch.Tensor):
                if emb.device != self.device:
                    emb = emb.to(self.device)
                padded_embeddings[i, :emb_len, :] = emb[:emb_len, :]
            else:
                emb_tensor = torch.tensor(emb[:emb_len], device=self.device)
                padded_embeddings[i, :emb_len, :] = emb_tensor
        
        return padded_embeddings
    
    @torch.no_grad()
    def generate_embeddings(self, sequences: list, batch_size: int = 32, return_pooled: bool = False, max_sequence_length: int = None):
        """
        Generate embeddings for a list of sequences.
        This method provides compatibility with ModelInference.generate_embeddings interface.
        
        @param sequences: List of protein sequences (strings)
        @param batch_size: Batch size for inference
        @param return_pooled: If True, return pooled embeddings (mean pooling). 
                             If False, return sequence-level embeddings (all token embeddings)
        @param max_sequence_length: Maximum sequence length (for compatibility, not used)
        @returns: Tensor of embeddings with shape (num_sequences, hidden_size) if return_pooled=True,
                 or (num_sequences, seq_len, hidden_size) if return_pooled=False
        """
        all_embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(sequences), batch_size), desc="Generating embeddings"):
            batch_sequences = sequences[i:i + batch_size]
            
            # Compute embeddings using ESM-C
            batch_embeddings = self._compute_esmc_embedding(batch_sequences)  # [batch_size, seq_len, hidden_size]
            
            if return_pooled:
                # Mean pooling over sequence length
                pooled_embeddings = batch_embeddings.mean(dim=1)  # [batch_size, hidden_size]
                all_embeddings.append(pooled_embeddings.cpu())
            else:
                all_embeddings.append(batch_embeddings.cpu())
        
        # Concatenate all batches
        embeddings = torch.cat(all_embeddings, dim=0)
        return embeddings
    
    @torch.no_grad()
    def generate_per_position_embeddings(self, sequences: list, batch_size: int = 32, 
                                         max_sequence_length: int = None, 
                                         use_sliding_window: bool = False,
                                         window_overlap: float = 0.5):
        """
        Generate per-position embeddings for sequences.
        
        @param sequences: List of protein sequences (strings)
        @param batch_size: Batch size for inference
        @param max_sequence_length: Maximum sequence length (for compatibility, not used)
        @param use_sliding_window: If True, use sliding window (not typically needed for ESM-C)
        @param window_overlap: Overlap ratio between windows (unused if sliding window is False)
        @returns: List of tensors, each with shape (seq_len, hidden_size)
        """
        all_embeddings = []
        original_lengths = [len(seq) for seq in sequences]
        
        # Process each sequence
        for seq_idx, sequence in enumerate(tqdm(sequences, desc="Generating per-position embeddings")):
            seq_len = len(sequence)
            
            # ESM-C handles long sequences internally, but we can use sliding window if needed
            if use_sliding_window and seq_len > self.max_sequence_length:
                # Use sliding window (similar to ESM2 implementation)
                windows = self._create_sliding_windows(sequence, self.max_sequence_length, window_overlap)
                window_embeddings_list = []
                
                for i in range(0, len(windows), batch_size):
                    batch_windows = windows[i:i + batch_size]
                    batch_seqs = [w[0] for w in batch_windows]
                    batch_embeddings = self._process_single_window_batch(batch_seqs)
                    window_embeddings_list.extend(batch_embeddings)
                
                # Merge window embeddings
                windows_data = [(emb, start, end) for emb, (_, start, end) in zip(window_embeddings_list, windows)]
                merged_embeddings = self._merge_window_embeddings(windows_data, seq_len, self.hidden_size)
                all_embeddings.append(merged_embeddings)
            else:
                # Process as single sequence
                window_embeddings = self._process_single_window_batch([sequence])
                all_embeddings.append(window_embeddings[0])
        
        return all_embeddings, original_lengths
    
    def _create_sliding_windows(self, sequence: str, window_size: int, overlap: float = 0.5):
        """
        Create sliding windows for a long sequence.
        
        @param sequence: Input sequence string
        @param window_size: Size of each window
        @param overlap: Overlap ratio between windows (0.0 to 1.0, default 0.5 means 50% overlap)
        @returns: List of (window_sequence, start_idx, end_idx) tuples
        """
        windows = []
        seq_len = len(sequence)
        step_size = max(1, int(window_size * (1 - overlap)))
        
        start = 0
        while start < seq_len:
            end = min(start + window_size, seq_len)
            window_seq = sequence[start:end]
            windows.append((window_seq, start, end))
            start += step_size
            
            if start < seq_len and start + window_size > seq_len:
                final_start = max(0, seq_len - window_size)
                if final_start > start - step_size:
                    final_window_seq = sequence[final_start:seq_len]
                    windows.append((final_window_seq, final_start, seq_len))
                break
        
        return windows
    
    def _merge_window_embeddings(self, windows_data: list, full_length: int, hidden_size: int):
        """
        Merge embeddings from multiple sliding windows.
        
        @param windows_data: List of (embeddings_tensor, start_idx, end_idx) tuples
        @param full_length: Full sequence length
        @param hidden_size: Hidden dimension size
        @returns: Merged embeddings tensor of shape (full_length, hidden_size)
        """
        merged_embeddings = torch.zeros(full_length, hidden_size)
        count_tensor = torch.zeros(full_length)
        
        for window_emb, start_idx, end_idx in windows_data:
            window_len = window_emb.shape[0]
            expected_window_len = end_idx - start_idx
            
            # 🔧 Fix: Ensure window embedding length matches expected window length
            # This handles cases where window embedding might have extra tokens
            if window_len != expected_window_len:
                # If embedding is longer, truncate to expected length
                if window_len > expected_window_len:
                    window_emb = window_emb[:expected_window_len]
                    window_len = expected_window_len
                # If embedding is shorter, this shouldn't happen but handle gracefully
                elif window_len < expected_window_len:
                    print(f"⚠️  Warning: Window embedding length ({window_len}) < expected length ({expected_window_len})")
            
            actual_end = min(start_idx + window_len, full_length)
            actual_len = actual_end - start_idx
            
            # Ensure we don't exceed the merged embeddings tensor
            if actual_end > full_length:
                actual_end = full_length
                actual_len = full_length - start_idx
            
            merged_embeddings[start_idx:actual_end] += window_emb[:actual_len]
            count_tensor[start_idx:actual_end] += 1
        
        count_tensor = torch.clamp(count_tensor, min=1.0)
        merged_embeddings = merged_embeddings / count_tensor.unsqueeze(-1)
        
        # 🔧 Verify final length matches expected
        final_len = merged_embeddings.shape[0]
        if final_len != full_length:
            raise RuntimeError(
                f"❌ Merged embedding length ({final_len}) != expected length ({full_length})"
            )
        
        return merged_embeddings
    
    @torch.no_grad()
    def _process_single_window_batch(self, window_sequences: list):
        """
        Process a batch of window sequences and extract per-position embeddings.
        
        @param window_sequences: List of window sequence strings
        @returns: List of embedding tensors, each with shape (window_len, hidden_size)
        """
        # Compute embeddings using ESM-C
        batch_embeddings = self._compute_esmc_embedding(window_sequences)  # [batch_size, seq_len, hidden_size]
        
        # Extract per-position embeddings
        window_embeddings = []
        for emb in batch_embeddings:
            window_embeddings.append(emb.cpu())
        
        return window_embeddings


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings using ESM-C 600M")
    parser.add_argument("--sequences", type=str, nargs="+", help="Input sequences")
    parser.add_argument("--output", type=str, help="Output path to save embeddings")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--layer_index", type=int, default=None, help="Layer index to extract (1-based). If None, uses last layer (default: None)")
    
    args = parser.parse_args()
    
    # Initialize inference model
    inferencer = ESMCInference(layer_index=args.layer_index)
    
    # Generate embeddings
    embeddings, lengths = inferencer.generate_per_position_embeddings(
        args.sequences, 
        batch_size=args.batch_size
    )
    
    # Save embeddings if output path is provided
    if args.output:
        torch.save(embeddings, args.output)
        print(f"✅ Embeddings saved to {args.output}")
    else:
        print(f"📊 Generated embeddings for {len(embeddings)} sequences")

