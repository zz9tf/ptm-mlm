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
    
    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D", device: str = None, max_sequence_length: int = None):
        """
        Initialize the ESM2 inference model.
        
        @param model_name: HuggingFace model name (default: "facebook/esm2_t33_650M_UR50D")
        @param device: Device to run inference on (None for auto-detect)
        @param max_sequence_length: Maximum sequence length for tokenization. 
                                   ESM2 has a max of 1024 tokens by default.
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
        
        # Set max sequence length (ESM2 default is 1024, but can be adjusted)
        if max_sequence_length is None:
            self.max_sequence_length = getattr(self.model.config, 'max_position_embeddings', 1024)
        else:
            self.max_sequence_length = max_sequence_length
        
        print(f"✅ ESM2 model loaded successfully! Hidden size: {self.hidden_size}")
        print(f"📏 Max sequence length: {self.max_sequence_length}")
    
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
        # Tokenize sequences
        # 🔧 max_length should account for special tokens (<cls> and <eos>)
        # ESM2 tokenizer adds 2 special tokens, so we need to add 2 to the max_sequence_length
        # - Normal case: sequence length <= max_sequence_length (e.g., 512), tokenizer max_length = 512 + 2 = 514
        # - Sliding window case: each window size is max_sequence_length (e.g., 512), tokenizer max_length = 512 + 2 = 514
        # This ensures sequences/windows of max_sequence_length won't be truncated after adding special tokens
        tokenizer_max_length = self.max_sequence_length + 2
        encoded = self.tokenizer(
            window_sequences,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=tokenizer_max_length,
            return_tensors="pt"
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Generate embeddings
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
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

