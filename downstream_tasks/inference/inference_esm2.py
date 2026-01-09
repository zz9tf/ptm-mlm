"""
ESM2 model inference script for generating embeddings.
This script loads ESM2 model and generates embeddings for protein sequences.
This is a shared module used by all downstream tasks.
"""
import torch
import sys
from pathlib import Path
from tqdm import tqdm

try:
    from transformers import EsmModel, EsmTokenizer
    ESM2_AVAILABLE = True
except ImportError:
    ESM2_AVAILABLE = False
    print("⚠️  transformers library not found. Please install: pip install transformers")


class ESM2Inference:
    """
    ESM2 model inference class for generating embeddings.
    This class is shared across all downstream tasks.
    """
    
    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D", device: str = None, max_sequence_length: int = None, layer_index: int = None):
        """
        Initialize the ESM2 inference model.
        
        @param model_name: HuggingFace model name (default: "facebook/esm2_t33_650M_UR50D")
        @param device: Device to run inference on (None for auto-detect)
        @param max_sequence_length: Maximum sequence length for tokenization. 
                                   ESM2 has a max of 1024 tokens by default.
        @param layer_index: Index of layer to extract (0-based). If None, uses last_hidden_state (default: None)
                           For ESM-C 600 layer 30, use layer_index=29 (0-based indexing)
        """
        if not ESM2_AVAILABLE:
            raise ImportError("transformers library is required for ESM2. Install with: pip install transformers")
        
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load tokenizer and model
        print(f"📦 Loading ESM2 model: {model_name}...")
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get hidden size from model config
        self.hidden_size = self.model.config.hidden_size
        
        # Set layer index for extraction
        self.layer_index = layer_index
        if layer_index is not None:
            num_layers = getattr(self.model.config, 'num_hidden_layers', None)
            if num_layers is not None and layer_index >= num_layers:
                raise ValueError(f"layer_index {layer_index} is out of range. Model has {num_layers} layers (0-{num_layers-1})")
            print(f"🔍 Will extract embeddings from layer {layer_index} (0-based)")
        
        # Set max sequence length (ESM2 default is 1024, but can be adjusted)
        if max_sequence_length is None:
            self.max_sequence_length = getattr(self.model.config, 'max_position_embeddings', 1024)
        else:
            self.max_sequence_length = max_sequence_length
        
        print(f"✅ ESM2 model loaded successfully! Hidden size: {self.hidden_size}")
        print(f"📏 Max sequence length: {self.max_sequence_length}")
    
    @torch.no_grad()
    def _tokenize_and_forward(self, sequences: list, max_sequence_length: int = None):
        """
        Internal helper method to tokenize sequences and generate embeddings.
        This method extracts common logic shared by generate_embeddings and _process_single_window_batch.
        
        @param sequences: List of protein sequences (strings)
        @param max_sequence_length: Maximum sequence length for tokenization. 
                                   If None, uses the instance's max_sequence_length
        @returns: Tuple of (hidden_states, attention_mask)
                  hidden_states: [batch_size, seq_len, hidden_size]
                  attention_mask: [batch_size, seq_len]
        """
        max_seq_len = max_sequence_length if max_sequence_length is not None else self.max_sequence_length
        tokenizer_max_length = max_seq_len + 2  # Account for <cls> and <eos> tokens
        
        encoded = self.tokenizer(
            sequences,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=tokenizer_max_length,
            return_tensors="pt"
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Generate embeddings
        # If layer_index is specified, output all hidden states to extract specific layer
        if self.layer_index is not None:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # Extract hidden states from all layers: outputs.hidden_states is a tuple of [batch_size, seq_len, hidden_size]
            # Layer 0 is embedding layer, layer 1 to num_layers are transformer layers
            # For ESM-C 600 layer 30, we want layer_index=29 (0-based), which corresponds to hidden_states[30] (1-based)
            hidden_states = outputs.hidden_states[self.layer_index + 1]  # +1 because hidden_states[0] is embedding layer
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        return hidden_states, attention_mask
    
    @torch.no_grad()
    def generate_embeddings(self, sequences: list, batch_size: int = 32, return_pooled: bool = False, max_sequence_length: int = None):
        """
        Generate embeddings for a list of sequences.
        This method provides compatibility with ModelInference.generate_embeddings interface.
        
        @param sequences: List of protein sequences (strings)
        @param batch_size: Batch size for inference
        @param return_pooled: If True, return pooled embeddings (mean pooling). 
                             If False, return sequence-level embeddings (all token embeddings)
        @param max_sequence_length: Maximum sequence length for tokenization. 
                                   If None, uses the instance's max_sequence_length
        @returns: Tensor of embeddings with shape (num_sequences, hidden_size) if return_pooled=True,
                 or (num_sequences, seq_len, hidden_size) if return_pooled=False
        """
        all_embeddings = []
        
        # Use provided max_sequence_length or fall back to instance default
        max_seq_len = max_sequence_length if max_sequence_length is not None else self.max_sequence_length
        
        # Process in batches
        for i in tqdm(range(0, len(sequences), batch_size), desc="Generating embeddings"):
            batch_sequences = sequences[i:i + batch_size]
            
            # Tokenize and generate embeddings using shared helper method
            hidden_states, attention_mask = self._tokenize_and_forward(batch_sequences, max_seq_len)
            
            if return_pooled:
                # Mean pooling over sequence length (excluding special tokens and padding)
                # Remove <cls> token (first token) and <eos>/<pad> tokens
                # For each sequence, extract embeddings from position 1 to seq_len-1
                pooled_embeddings = []
                for hidden, attn_mask in zip(hidden_states, attention_mask):
                    # Find the actual sequence length (excluding padding)
                    seq_len = attn_mask.sum().item()
                    # Extract sequence embeddings (remove <cls> at position 0 and <eos> at position seq_len-1)
                    if seq_len > 2:
                        seq_embeddings = hidden[1:seq_len-1]  # [seq_len-2, hidden_size]
                        # Mean pooling over sequence length
                        pooled = seq_embeddings.mean(dim=0)  # [hidden_size]
                    elif seq_len == 2:
                        # Only <cls> and <eos> tokens, use <cls> token embedding
                        pooled = hidden[0]  # [hidden_size]
                    else:
                        # seq_len == 1, use the only token
                        pooled = hidden[0]  # [hidden_size]
                    pooled_embeddings.append(pooled)
                
                # Stack into batch tensor
                batch_pooled = torch.stack(pooled_embeddings, dim=0)  # [batch_size, hidden_size]
                all_embeddings.append(batch_pooled.cpu())
            else:
                # Return all token embeddings (remove special tokens)
                batch_embeddings = []
                for hidden, attn_mask in zip(hidden_states, attention_mask):
                    seq_len = attn_mask.sum().item()
                    if seq_len > 2:
                        seq_embeddings = hidden[1:seq_len-1]  # [seq_len-2, hidden_size]
                    elif seq_len == 2:
                        # Only <cls> and <eos> tokens, return empty or use <cls>
                        seq_embeddings = hidden[0:1]  # [1, hidden_size]
                    else:
                        # seq_len == 1, use the only token
                        seq_embeddings = hidden[0:1]  # [1, hidden_size]
                    batch_embeddings.append(seq_embeddings)
                
                # Pad sequences to same length for batching
                max_seq_len_in_batch = max(emb.shape[0] for emb in batch_embeddings)
                padded_embeddings = []
                for emb in batch_embeddings:
                    if emb.shape[0] < max_seq_len_in_batch:
                        padding = torch.zeros(max_seq_len_in_batch - emb.shape[0], self.hidden_size, device=emb.device)
                        emb = torch.cat([emb, padding], dim=0)
                    padded_embeddings.append(emb)
                
                batch_tensor = torch.stack(padded_embeddings, dim=0)  # [batch_size, max_seq_len, hidden_size]
                all_embeddings.append(batch_tensor.cpu())
        
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
        @param max_sequence_length: Maximum sequence length for a single window. 
                                   If None, uses the instance's max_sequence_length
        @param use_sliding_window: If True, use sliding window for sequences longer than max_sequence_length.
                                  For ESM2, usually not needed as it supports up to 1024 tokens.
        @param window_overlap: Overlap ratio between windows (unused if sliding window is False)
        @returns: List of tensors, each with shape (seq_len, hidden_size)
        """
        all_embeddings = []
        original_lengths = [len(seq) for seq in sequences]
        
        # Use provided max_sequence_length or fall back to instance default
        max_seq_len = max_sequence_length if max_sequence_length is not None else self.max_sequence_length
        
        # Process each sequence
        for seq_idx, sequence in enumerate(tqdm(sequences, desc="Generating per-position embeddings")):
            seq_len = len(sequence)
            
            # ESM2 tokenizer adds special tokens, so we need to account for that
            # For sequences longer than max_seq_len, truncate (ESM2 handles this automatically)
            if seq_len > max_seq_len:
                if use_sliding_window:
                    # Use sliding window (similar to Mamba implementation)
                    windows = self._create_sliding_windows(sequence, max_seq_len, window_overlap)
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
                    # Truncate sequence
                    sequence = sequence[:max_seq_len]
                    window_embeddings = self._process_single_window_batch([sequence])
                    all_embeddings.append(window_embeddings[0])
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
            actual_end = min(start_idx + window_len, full_length)
            actual_len = actual_end - start_idx
            
            merged_embeddings[start_idx:actual_end] += window_emb[:actual_len]
            count_tensor[start_idx:actual_end] += 1
        
        count_tensor = torch.clamp(count_tensor, min=1.0)
        merged_embeddings = merged_embeddings / count_tensor.unsqueeze(-1)
        
        return merged_embeddings
    
    @torch.no_grad()
    def _process_single_window_batch(self, window_sequences: list):
        """
        Process a batch of window sequences and extract per-position embeddings.
        
        @param window_sequences: List of window sequence strings
        @returns: List of embedding tensors, each with shape (window_len, hidden_size)
        """
        # Tokenize and generate embeddings using shared helper method
        # Note: Uses instance's max_sequence_length (not parameter) to maintain compatibility
        hidden_states, attention_mask = self._tokenize_and_forward(window_sequences)
        
        # Extract per-position embeddings (remove special tokens: <cls> and <eos>/<pad>)
        window_embeddings = []
        for hidden, attn_mask in zip(hidden_states, attention_mask):
            # Find the actual sequence length (excluding padding)
            seq_len = attn_mask.sum().item()
            # Remove <cls> token (first token) and keep the rest up to seq_len
            # ESM2 uses <cls> at the beginning
            seq_embeddings = hidden[1:seq_len-1].cpu()  # Remove <cls> and <eos>
            window_embeddings.append(seq_embeddings)
        
        return window_embeddings


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings using ESM2")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D", 
                       help="ESM2 model name from HuggingFace")
    parser.add_argument("--sequences", type=str, nargs="+", help="Input sequences")
    parser.add_argument("--output", type=str, help="Output path to save embeddings")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    
    args = parser.parse_args()
    
    # Initialize inference model
    inferencer = ESM2Inference(model_name=args.model_name)
    
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

