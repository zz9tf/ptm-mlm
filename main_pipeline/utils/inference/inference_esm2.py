"""
ESM2 model inference script for generating embeddings.
This script loads ESM2 model and generates embeddings for protein sequences.
This is a shared module used by all downstream tasks.
"""
import torch
import sys
from pathlib import Path
from tqdm import tqdm

from transformers import EsmModel, EsmTokenizer


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
        @param max_sequence_length: Maximum token length (includes special tokens).
                                   Defaults to model's max_position_embeddings (typically 1026 for ESM2).
                                   Note: This is token-level, not residue-level.
        @param layer_index: Index of layer to extract (0-based). If None, uses last_hidden_state (default: None)
        """
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
        
        # Set max sequence length
        # Note: max_position_embeddings is token-level max length (includes special tokens)
        if max_sequence_length is None:
            # Use model's max_position_embeddings (token-level, includes special tokens)
            self.max_sequence_length = getattr(self.model.config, 'max_position_embeddings', 1026)
        else:
            # User-provided max_sequence_length: assume it's token-level (includes special tokens)
            self.max_sequence_length = max_sequence_length
        
        print(f"✅ ESM2 model loaded successfully! Hidden size: {self.hidden_size}")
        print(f"📏 Max sequence length: {self.max_sequence_length}")
    
    @torch.inference_mode()
    def _tokenize_and_forward(self, sequences: list, layer_indices: list = None):
        """
        Tokenize sequences and generate embeddings.
        
        @param sequences: List of protein sequences (strings)
        @param layer_indices: List of layer indices to extract (0-based). If None, uses self.layer_index or last layer
        @returns: Tuple of (hidden_states_dict, attention_mask)
                  hidden_states_dict: Dict mapping layer_index to hidden_states tensor [batch_size, seq_len, hidden_size]
                  attention_mask: [batch_size, seq_len]
        """
        # max_sequence_length is already token-level (includes special tokens)
        encoded = self.tokenizer(
            sequences,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors="pt"
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # Determine which layers to extract
        if layer_indices is None:
            layer_indices = [self.layer_index] if self.layer_index is not None else [None]
        
        # Use AMP for faster inference (bfloat16 on CUDA)
        with torch.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu", 
                           dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32):
            # Always request hidden_states if we need multiple layers or specific layer
            need_hidden_states = len(layer_indices) > 1 or (len(layer_indices) == 1 and layer_indices[0] is not None)
            
            if need_hidden_states:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden_states_dict = {}
                for layer_idx in layer_indices:
                    if layer_idx is not None:
                        hidden_states_dict[layer_idx] = outputs.hidden_states[layer_idx + 1]
                    else:
                        hidden_states_dict[None] = outputs.last_hidden_state
            else:
                # Single layer, use optimized path
                if layer_indices[0] is not None:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    hidden_states_dict = {layer_indices[0]: outputs.hidden_states[layer_indices[0] + 1]}
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    hidden_states_dict = {None: outputs.last_hidden_state}
        
        return hidden_states_dict, attention_mask
    
    @torch.inference_mode()
    def _compute_esm2_embedding(self, sequences: list, layer_indices: list = None):
        """
        Compute ESM2 embeddings for a batch of sequences.
        
        Batch converter logic is built-in: automatically handles tokenization, padding, and special tokens.
        External code doesn't need to handle batch_converter conversion or alignment.
        
        @param sequences: List of protein sequences (strings)
        @param layer_indices: List of layer indices to extract (0-based). If None, uses self.layer_index or last layer
        @returns: Dict mapping layer_index to list of embedding tensors
                  Each embedding tensor has shape (seq_len, hidden_size)
                  Includes special tokens (<cls> and <eos>) to match batch_converter behavior
                  If single layer requested, returns list directly for backward compatibility
        """
        hidden_states_dict, attention_mask = self._tokenize_and_forward(sequences, layer_indices)
        
        # Compute lengths on GPU in batch (avoid multiple GPU→CPU syncs)
        lengths = attention_mask.sum(dim=1).to(torch.int32)  # [batch_size] on GPU
        lengths_list = lengths.cpu().tolist()
        
        # Extract embeddings for each layer
        result_dict = {}
        for layer_idx, hidden_states in hidden_states_dict.items():
            # Move to CPU once (single transfer)
            hidden_states = hidden_states.cpu()
            
            # Extract embeddings (all on CPU now, no more GPU syncs)
            embeddings = [hidden_states[i, :lengths_list[i]].contiguous() for i in range(len(sequences))]
            result_dict[layer_idx] = embeddings
        
        # Backward compatibility: if single layer, return list directly
        if len(result_dict) == 1:
            return list(result_dict.values())[0]
        
        return result_dict


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings using ESM2")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D", 
                       help="ESM2 model name from HuggingFace")
    parser.add_argument("--sequences", type=str, nargs="+", help="Input sequences")
    parser.add_argument("--output", type=str, help="Output path to save embeddings")
    parser.add_argument("--layer_index", type=int, default=None, help="Layer index to extract (0-based)")
    
    args = parser.parse_args()
    
    inferencer = ESM2Inference(model_name=args.model_name, layer_index=args.layer_index)
    embeddings = inferencer._compute_esm2_embedding(args.sequences)
    
    if args.output:
        torch.save(embeddings, args.output)
        print(f"✅ Embeddings saved to {args.output}")
    else:
        print(f"📊 Generated embeddings for {len(embeddings)} sequences")

