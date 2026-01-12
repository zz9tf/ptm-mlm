"""
Model inference script for generating embeddings from pre-trained Mamba model.
This script loads a trained checkpoint and generates embeddings for protein sequences.
Supports Mamba-only or Mamba+ESM2-15B combination (matching training setup).

This is a shared module used by all downstream tasks.
"""
import torch
from tqdm import tqdm
import esm

from main_pipeline.getters.tokenizer import PTMTokenizer
from main_pipeline.utils.checkpoint import load_ckpt
from main_pipeline.utils.esm_utils import make_esm_input_ids


class MambaInference:
    """
    Mamba model inference class for generating embeddings from pre-trained Mamba model.
    Supports Mamba-only or Mamba+ESM2-15B combination (matching training setup).
    This class is shared across all downstream tasks.
    """
    
    def __init__(self, checkpoint_path: str, device: str = None, max_sequence_length: int = None,
                 use_esm: bool = True, esm2_model_name: str = "facebook/esm2_t48_15B_UR50D",
                 layer_index: int = None):
        """
        Initialize the inference model.

        @param checkpoint_path: Path to the trained model checkpoint (.ckpt file)
        @param device: Device to run inference on (None for auto-detect)
        @param max_sequence_length: Maximum sequence length for tokenization.
                                   If None, sequences will not be truncated (may cause memory issues).
                                   Default: 512 (matching training config)
        @param use_esm: If True, load ESM2 and use Mamba+ESM2 combination (matching training).
                       If False, use Mamba-only mode.
        @param esm2_model_name: ESM2 model name. Options:
                               - "facebook/esm2_t48_15B_UR50D" (default, 15B parameters)
                               - "facebook/esm2_t33_650M_UR50D" (650M parameters)
        @param layer_index: Layer index to extract embeddings from (0-based).
                           If None, uses last layer (default: None)
        """
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load tokenizer
        self.tokenizer = PTMTokenizer()
        
        # Load model from checkpoint
        print(f"📦 Loading Mamba model from {checkpoint_path}...")
        self.model, _ = load_ckpt(
            checkpoint_path, 
            self.tokenizer, 
            self.device
        )
        
        # Ensure model is on the correct device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Auto-detect ESM2 model type from checkpoint if use_esm is True
        # Check esm_head input dimension to determine which ESM2 model was used during training
        detected_esm_model_name = None
        if use_esm and hasattr(self.model, 'esm_head') and self.model.esm_head is not None:
            esm_embed_dim = self.model.esm_head.in_features
            if esm_embed_dim == 1280:
                detected_esm_model_name = "facebook/esm2_t33_650M_UR50D"
            elif esm_embed_dim == 5120:
                detected_esm_model_name = "facebook/esm2_t48_15B_UR50D"
            else:
                print(f"⚠️  Warning: Unknown ESM embedding dimension {esm_embed_dim} in checkpoint.")
                print(f"   Expected 1280 (ESM2-650M) or 5120 (ESM2-15B).")
        
        # Use detected model name if available, otherwise use provided one
        if detected_esm_model_name:
            if esm2_model_name != detected_esm_model_name:
                print(f"⚠️  Warning: Checkpoint was trained with {detected_esm_model_name}, but {esm2_model_name} was specified.")
                print(f"   Auto-switching to {detected_esm_model_name} to match checkpoint configuration.")
            esm2_model_name = detected_esm_model_name
        
        # Set layer index for extraction
        self.layer_index = layer_index
        if layer_index is not None:
            print(f"🔍 Will extract embeddings from layer {layer_index} (0-based)")

        # Load ESM2 if use_esm is True (matching training setup)
        self.use_esm = use_esm
        self.esm_model = None
        self.batch_converter = None
        self.esm2_model_name = esm2_model_name
        self.esm_repr_layer = 33  # Both ESM2-15B and ESM2-650M use layer 33
        
        if use_esm:
            if esm2_model_name == "facebook/esm2_t48_15B_UR50D":
                print(f"📦 Loading ESM2-15B model...")
                self.esm_model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
            elif esm2_model_name == "facebook/esm2_t33_650M_UR50D":
                print(f"📦 Loading ESM2-650M model...")
                self.esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            else:
                raise ValueError(f"Unknown ESM2 model name: {esm2_model_name}. "
                               f"Supported: 'facebook/esm2_t48_15B_UR50D' or 'facebook/esm2_t33_650M_UR50D'")
            
            self.batch_converter = alphabet.get_batch_converter()
            self.esm_model = self.esm_model.to(self.device)
            self.esm_model.eval()
            for param in self.esm_model.parameters():
                param.requires_grad = False
            print(f"✅ {esm2_model_name} loaded successfully!")
        else:
            print(f"ℹ️  Using Mamba-only mode (no ESM2)")
        
        # Get hidden size from model config
        self.hidden_size = self.model.config.d_model
        
        # Set max sequence length (default to 512 if not provided, matching training config)
        if max_sequence_length is None:
            # Try to get from model config if available, otherwise use default
            self.max_sequence_length = getattr(self.model.config, 'max_sequence_length', 512)
        else:
            self.max_sequence_length = max_sequence_length
        
        print(f"✅ Mamba model loaded successfully! Hidden size: {self.hidden_size}")
        print(f"📏 Max sequence length: {self.max_sequence_length}")
        mode_str = f"Mamba+{esm2_model_name}" if use_esm else "Mamba-only"
        print(f"🔧 Mode: {mode_str}")
    
    @torch.no_grad()
    def generate_embeddings(self, sequences: list, batch_size: int = 32, return_pooled: bool = False, max_sequence_length: int = None):
        """
        Generate embeddings for a list of sequences.
        
        @param sequences: List of protein sequences (strings)
        @param batch_size: Batch size for inference
        @param return_pooled: If True, return pooled embeddings (mean pooling). 
                             If False, return sequence-level embeddings (last token or mean)
        @param max_sequence_length: Maximum sequence length for tokenization. 
                                   If None, uses the instance's max_sequence_length
        @returns: Tensor of embeddings with shape (num_sequences, hidden_size) or 
                 (num_sequences, seq_len, hidden_size) if not pooled
        """
        all_embeddings = []
        
        # Use provided max_sequence_length or fall back to instance default
        max_seq_len = max_sequence_length if max_sequence_length is not None else self.max_sequence_length
        
        # Process in batches
        for i in tqdm(range(0, len(sequences), batch_size), desc="Generating embeddings"):
            batch_sequences = sequences[i:i + batch_size]
            
            # Tokenize sequences with length limit
            input_ids = self.tokenizer(
                batch_sequences,
                add_special_tokens=True,
                return_tensors=True,
                max_sequence_length=max_seq_len
            ).to(self.device)
            
            # Compute ESM2-15B embeddings if use_esm is True (matching training setup)
            esm_embedding = None
            if self.use_esm and self.esm_model is not None:
                # Convert PTM tokens to mask tokens for ESM (matching training)
                esm_input_ids = make_esm_input_ids(input_ids, self.tokenizer)
                
                # Compute ESM2 embeddings using the same method as train.py
                # (matching compute_esm_embedding function logic)
                inputs = [
                    (i, self.tokenizer.decode(input_id.detach().cpu().tolist()))
                    for i, input_id in enumerate(esm_input_ids)
                ]
                batch_labels, batch_strs, batch_tokens = self.batch_converter(inputs)
                batch_tokens = batch_tokens[..., 1:-1].to(self.device)  # remove <cls> and <eos>
                out = self.esm_model(batch_tokens, repr_layers=[self.esm_repr_layer], return_contacts=False)
                esm_embedding = out["representations"][self.esm_repr_layer]  # [batch_size, seq_len, esm_hidden_size]
                
                # Align ESM embedding dimensions with input_ids (handle padding)
                # This matches the alignment logic in train.py
                if esm_embedding.shape[1] < input_ids.shape[1]:
                    pad_size = input_ids.shape[1] - esm_embedding.shape[1]
                    esm_embedding = torch.nn.functional.pad(esm_embedding, (0, 0, 0, pad_size))
                elif esm_embedding.shape[1] > input_ids.shape[1]:
                    esm_embedding = esm_embedding[:, :input_ids.shape[1]]
            
            # Generate embeddings using model with ESM embeddings (matching training)
            # The model will project ESM embeddings through esm_head first, then fuse with Mamba embeddings
            # Note: Use model() instead of model.backbone() to ensure esm_head projection is applied
            model_outputs = self.model(input_ids, embedding=esm_embedding)
            hidden_states = model_outputs.hidden_states
            
            if return_pooled:
                # Mean pooling over sequence length (excluding padding)
                # Get attention mask (non-padding tokens)
                attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
                # Compute mean pooling
                pooled_embeddings = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                all_embeddings.append(pooled_embeddings.cpu())
            else:
                # Return all token embeddings (can be used for per-position prediction)
                all_embeddings.append(hidden_states.cpu())
        
        # Concatenate all batches
        embeddings = torch.cat(all_embeddings, dim=0)
        return embeddings
    
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
        step_size = max(1, int(window_size * (1 - overlap)))  # Ensure step_size >= 1
        
        start = 0
        while start < seq_len:
            end = min(start + window_size, seq_len)
            window_seq = sequence[start:end]
            windows.append((window_seq, start, end))
            
            # Move to next window
            start += step_size
            
            # If we haven't reached the end but the next window would start beyond the sequence,
            # create a final window that ends at the sequence end
            if start < seq_len and start + window_size > seq_len:
                # Create a final window that covers the remaining part
                final_start = max(0, seq_len - window_size)
                if final_start > start - step_size:  # Only add if it's different from previous
                    final_window_seq = sequence[final_start:seq_len]
                    windows.append((final_window_seq, final_start, seq_len))
                break
        
        return windows
    
    def _merge_window_embeddings(self, windows_data: list, full_length: int, hidden_size: int):
        """
        Merge embeddings from multiple sliding windows.
        For overlapping regions, take the average of embeddings.
        
        @param windows_data: List of (embeddings_tensor, start_idx, end_idx) tuples
        @param full_length: Full sequence length
        @param hidden_size: Hidden dimension size
        @returns: Merged embeddings tensor of shape (full_length, hidden_size)
        """
        # Initialize output tensor and count tensor for averaging
        merged_embeddings = torch.zeros(full_length, hidden_size)
        count_tensor = torch.zeros(full_length)
        
        for window_emb, start_idx, end_idx in windows_data:
            window_len = window_emb.shape[0]
            actual_end = min(start_idx + window_len, full_length)
            actual_len = actual_end - start_idx
            
            # Add embeddings to the merged tensor
            merged_embeddings[start_idx:actual_end] += window_emb[:actual_len]
            count_tensor[start_idx:actual_end] += 1
        
        # Average overlapping regions
        # Avoid division by zero
        count_tensor = torch.clamp(count_tensor, min=1.0)
        merged_embeddings = merged_embeddings / count_tensor.unsqueeze(-1)
        
        return merged_embeddings
    
    @torch.no_grad()
    def _process_single_window_batch(self, window_sequences: list):
        """
        Process a batch of window sequences and extract per-position embeddings.
        Uses Mamba+ESM2-15B combination if use_esm=True (matching training setup).
        
        @param window_sequences: List of window sequence strings
        @returns: List of embedding tensors, each with shape (window_len, hidden_size)
        """
        # Store original sequence lengths before tokenization
        original_lengths = [len(seq) for seq in window_sequences]
        
        # Tokenize window sequences
        # 🔧 Fix: For sequences <= max_sequence_length, don't truncate
        # Even if seq_len + 2 > max_sequence_length, we still process without truncation
        # The model can handle it, and we extract embeddings by removing CLS and EOS
        max_seq_len_in_batch = max(len(seq) for seq in window_sequences)
        
        if max_seq_len_in_batch <= self.max_sequence_length:
            # All sequences fit: seq_len <= max_sequence_length
            # Don't truncate, let tokenizer process full sequences (seq_len + 2 special tokens)
            # Even if this exceeds max_sequence_length tokens, the model can handle it
            input_ids = self.tokenizer(
                window_sequences,
                add_special_tokens=True,
                return_tensors=True,
                max_sequence_length=None  # No truncation - let model handle full sequence
            ).to(self.device)
        else:
            # Some sequences are too long (seq_len > max_sequence_length), use truncation
            # This happens for sliding windows where window_size = max_sequence_length
            input_ids = self.tokenizer(
                window_sequences,
                add_special_tokens=True,
                return_tensors=True,
                max_sequence_length=self.max_sequence_length
            ).to(self.device)
        
        # Compute ESM2-15B embeddings if use_esm is True (matching training setup)
        esm_embedding = None
        if self.use_esm and self.esm_model is not None:
            # Convert PTM tokens to mask tokens for ESM (matching training)
            # Note: In inference, we don't need additional masking (pred_mask) like in training
            # because we're not doing MLM task, just generating embeddings
            esm_input_ids = make_esm_input_ids(input_ids, self.tokenizer)
            
            # Compute ESM2 embeddings using the same method as train.py
            # (matching compute_esm_embedding function logic)
            inputs = [
                (i, self.tokenizer.decode(input_id.detach().cpu().tolist()))
                for i, input_id in enumerate(esm_input_ids)
            ]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(inputs)
            batch_tokens = batch_tokens[..., 1:-1].to(self.device)  # remove <cls> and <eos>
            out = self.esm_model(batch_tokens, repr_layers=[self.esm_repr_layer], return_contacts=False)
            esm_embedding = out["representations"][self.esm_repr_layer]  # [batch_size, seq_len, esm_hidden_size]
            
            # Align ESM embedding dimensions with input_ids (handle padding)
            # This matches the alignment logic in train.py
            if esm_embedding.shape[1] < input_ids.shape[1]:
                pad_size = input_ids.shape[1] - esm_embedding.shape[1]
                esm_embedding = torch.nn.functional.pad(esm_embedding, (0, 0, 0, pad_size))
            elif esm_embedding.shape[1] > input_ids.shape[1]:
                esm_embedding = esm_embedding[:, :input_ids.shape[1]]
        
        # Generate embeddings using model with ESM embeddings (matching training)
        # The model will project ESM embeddings through esm_head first, then fuse with Mamba embeddings
        # Note: Use model() instead of model.backbone() to ensure esm_head projection is applied
        model_outputs = self.model(input_ids, embedding=esm_embedding, output_hidden_states=self.layer_index is not None)
        if self.layer_index is not None:
            # Extract from specific layer: layer_index=0 is first transformer layer
            hidden_states = model_outputs.hidden_states[self.layer_index]
        else:
            # Use last hidden state (default behavior)
            hidden_states = model_outputs.hidden_states
        
        # Extract per-position embeddings (remove CLS and EOS tokens)
        # 🔧 Fix: Simply remove CLS (position 0) and EOS, get embeddings for all sequence positions
        # For seq_len=512: tokens are [CLS, seq_0, seq_1, ..., seq_511, EOS]
        # Extract hidden[1:-1] or hidden[1:eos_pos] to get 512 embeddings
        batch_input_ids = input_ids.cpu()
        eos_token_id = self.tokenizer.ids_to_tokens.index("<eos>")
        window_embeddings = []
        
        for idx, (hidden, input_id) in enumerate(zip(hidden_states, batch_input_ids)):
            original_len = original_lengths[idx]
            
            # Find EOS token position
            eos_pos = None
            for pos in range(1, len(input_id)):
                if input_id[pos].item() == eos_token_id:
                    eos_pos = pos
                    break
            
            # Extract embeddings: remove CLS (position 0) and EOS (position eos_pos)
            # hidden[1:eos_pos] gives us embeddings for positions 1 to eos_pos-1
            # This should equal original_len (since eos_pos = original_len + 1)
            if eos_pos is not None:
                seq_embeddings = hidden[1:eos_pos].cpu()
            else:
                # No EOS found, extract all except CLS
                seq_embeddings = hidden[1:].cpu()
            
            if seq_embeddings.shape[0] != original_len:
                if seq_embeddings.shape[0] < original_len:
                    # Pad if shorter (shouldn't happen if no truncation)
                    pad_size = original_len - seq_embeddings.shape[0]
                    padding = torch.zeros(pad_size, seq_embeddings.shape[1])
                    seq_embeddings = torch.cat([seq_embeddings, padding], dim=0)
                else:
                    raise ValueError(f"Embedding length {seq_embeddings.shape[0]} does not match original length {original_len}")
            
            window_embeddings.append(seq_embeddings)
        
        return window_embeddings
    
    @torch.no_grad()
    def generate_per_position_embeddings(self, sequences: list, batch_size: int = 32, 
                                         max_sequence_length: int = None, 
                                         use_sliding_window: bool = True,
                                         window_overlap: float = 0.5):
        """
        Generate per-position embeddings for sequences (useful for site prediction).
        Uses sliding window for long sequences to preserve all positions.
        
        @param sequences: List of protein sequences (strings)
        @param batch_size: Batch size for inference (for window processing)
        @param max_sequence_length: Maximum sequence length for a single window. 
                                   If None, uses the instance's max_sequence_length.
        @param use_sliding_window: If True, use sliding window for sequences longer than max_sequence_length.
                                 If False, truncate long sequences (not recommended for site prediction).
        @param window_overlap: Overlap ratio between windows (0.0 to 1.0, default 0.5 means 50% overlap).
                             Higher overlap provides better context but requires more computation.
        @returns: List of tensors, each with shape (seq_len, hidden_size)
        """
        all_embeddings = []
        original_lengths = [len(seq) for seq in sequences]
        
        # Use provided max_sequence_length or fall back to instance default
        max_seq_len = max_sequence_length if max_sequence_length is not None else self.max_sequence_length
        
        # Process each sequence
        for seq_idx, sequence in enumerate(tqdm(sequences, desc="Generating per-position embeddings")):
            seq_len = len(sequence)
            
            # The model can handle max_seq_len tokens (including special tokens)
            # For seq_len <= max_seq_len (e.g., 512):
            #   - Sequence: seq_len chars
            #   - After tokenization: seq_len + 2 special tokens
            #   - Model processes (seq_len + 2) tokens
            #   - Extract embeddings: remove CLS and EOS, get seq_len embeddings
            #   - This matches the original sequence length and labels
            # For seq_len > max_seq_len, use sliding window
            
            # If sequence fits in one window (seq_len <= max_seq_len) or sliding window is disabled
            if max_seq_len is None or seq_len <= max_seq_len or not use_sliding_window:
                # Process as single window
                # Tokenizer will process full sequence without truncation (max_sequence_length=None)
                # This ensures we get embeddings for all positions
                window_embeddings = self._process_single_window_batch([sequence])
                emb = window_embeddings[0]
                emb_len = emb.shape[0]
                
                # Ensure embedding length matches original sequence length
                # With no truncation, this should always match, but check just in case
                if emb_len != seq_len:
                    if emb_len < seq_len:
                        # Pad to match sequence length (for label alignment)
                        pad_size = seq_len - emb_len
                        emb = torch.cat([emb, torch.zeros(pad_size, emb.shape[1])], dim=0)
                    else:
                        print(f"Warning: Embedding length {emb_len} does not match original length {seq_len}")
                        # Truncate if somehow longer
                        emb = emb[:seq_len]
                
                all_embeddings.append(emb)
            else:
                # Use sliding window for long sequences (seq_len > max_seq_len)
                # Window size should be max_seq_len (e.g., 512)
                # Each window will be processed: window_size + 2 special tokens
                # Tokenizer will handle truncation if needed in _process_single_window_batch
                windows = self._create_sliding_windows(sequence, max_seq_len, window_overlap)
                
                # Process windows in batches
                window_embeddings_list = []
                for i in range(0, len(windows), batch_size):
                    batch_windows = windows[i:i + batch_size]
                    batch_seqs = [w[0] for w in batch_windows]
                    
                    # Process batch of windows
                    batch_embeddings = self._process_single_window_batch(batch_seqs)
                    window_embeddings_list.extend(batch_embeddings)
                
                # Merge window embeddings
                windows_data = [(emb, start, end) for emb, (_, start, end) in zip(window_embeddings_list, windows)]
                merged_embeddings = self._merge_window_embeddings(windows_data, seq_len, self.hidden_size)
                all_embeddings.append(merged_embeddings)
        
        return all_embeddings, original_lengths


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings from pre-trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--sequences", type=str, nargs="+", help="Input sequences")
    parser.add_argument("--output", type=str, help="Output path to save embeddings")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    
    args = parser.parse_args()
    
    # Initialize inference model
    inferencer = MambaInference(args.checkpoint)
    
    # Generate embeddings
    embeddings = inferencer.generate_embeddings(
        args.sequences, 
        batch_size=args.batch_size,
        return_pooled=True
    )
    
    # Save embeddings if output path is provided
    if args.output:
        torch.save(embeddings, args.output)
        print(f"✅ Embeddings saved to {args.output}")
    else:
        print(f"📊 Generated embeddings shape: {embeddings.shape}")

