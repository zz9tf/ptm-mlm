"""
ESM-C 600M model inference script for generating embeddings.
This script loads ESM-C 600M model and generates embeddings for protein sequences.
"""
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


class ESMCInference:
    """
    ESM-C 600M model inference class for generating embeddings.
    """
    
    def __init__(self, device: str = None, layer_index: int = None):
        """
        Initialize the ESM-C 600M inference model for generating embeddings.

        @param device: Device to run inference on (None for auto-detect)
        @param layer_index: Layer index to extract (0-based). If None, uses last layer (default: None)
        """
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
        
        # Set layer index (0-based for ESM-C)
        self.layer_index = layer_index
        if layer_index is not None:
            print(f"🔍 Will extract embeddings from layer {layer_index} (0-based)")
        else:
            print(f"🔍 Will extract embeddings from last layer (default)")
        
        # ESM-C 600M has 1152 dimensions
        self.hidden_size = 1152

        print(f"✅ ESM-C 600M model loaded successfully! Hidden size: {self.hidden_size}")
    
    @torch.inference_mode()
    def _compute_esmc_embedding(self, sequences: list, layer_indices: list = None):
        """
        Compute ESM-C 600M embeddings for a batch of sequences.
        
        Batch converter logic is built-in: automatically handles tokenization, padding, and special tokens.
        External code doesn't need to handle batch_converter conversion or alignment.
        
        🚀 Performance: Uses batch inference via _tokenize() + forward() for ~30x speedup
        compared to SDK API (which processes sequences one by one).
        
        @param sequences: List of protein sequences (strings)
        @param layer_indices: List of layer indices to extract (0-based). If None, uses self.layer_index or last layer
        @returns: Dict mapping layer_index to list of ESM-C embeddings tensors
                  Each embedding tensor has shape (seq_len, embed_dim)
                  Includes special tokens (BOS/EOS) - batch_converter logic is handled internally
                  If single layer requested, returns list directly for backward compatibility
        """
        if not sequences:
            return [] if layer_indices is None or len(layer_indices) == 1 else {}
        
        # Determine which layers to extract
        if layer_indices is None:
            layer_indices = [self.layer_index] if self.layer_index is not None else [None]
        
        # 🚀 Batch processing: Use _tokenize() + forward() for efficient batch inference
        # Batch tokenize all sequences at once
        sequence_tokens = self.esm_model._tokenize(sequences)  # [batch_size, max_seq_len]
        
        # Batch forward pass with AMP for faster inference
        # ESMC.forward() always returns hidden_states, no need for output_hidden_states parameter
        with torch.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu",
                           dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32):
            output = self.esm_model.forward(sequence_tokens=sequence_tokens)
        
        # Compute lengths on GPU in batch (avoid multiple GPU→CPU syncs)
        # ESM-C tokenize behavior: Adds BOS (0) and EOS (2), pads with (1)
        # Find EOS positions for all sequences at once
        eos_token_id = 2
        eos_mask = (sequence_tokens == eos_token_id)  # [batch_size, padded_seq_len]
        eos_positions = eos_mask.int().argmax(dim=1)  # [batch_size] - first EOS position for each seq
        
        # Calculate actual lengths: EOS position + 1 (including EOS)
        # Fallback to expected length if no EOS found
        batch_size = sequence_tokens.shape[0]
        padded_seq_len = sequence_tokens.shape[1]
        has_eos = eos_mask.any(dim=1)  # [batch_size] bool
        eos_lengths = (eos_positions + 1).to(torch.int32)  # [batch_size]
        
        # Expected lengths: len(seq) + 2 (BOS + EOS)
        expected_lengths = torch.tensor([len(seq) + 2 for seq in sequences], 
                                       device=self.device, dtype=torch.int32)
        fallback_lengths = torch.minimum(expected_lengths, 
                                        torch.tensor(padded_seq_len, device=self.device, dtype=torch.int32))
        
        actual_lengths = torch.where(has_eos, eos_lengths, fallback_lengths)  # [batch_size] on GPU
        actual_lengths_list = actual_lengths.cpu().tolist()
        
        # Extract embeddings for each layer
        result_dict = {}
        for layer_idx in layer_indices:
            if layer_idx is not None:
                # layer_index is 0-based, directly use it for hidden_states indexing
                if output.hidden_states is None:
                    raise ValueError("hidden_states is None. Cannot extract layer-specific embeddings.")
                if layer_idx < 0:
                    raise ValueError(f"layer_index must be >= 0 (got {layer_idx})")
                if layer_idx >= output.hidden_states.shape[0]:
                    raise ValueError(
                        f"layer_index {layer_idx} is out of range. "
                        f"Model has {output.hidden_states.shape[0]} layers (0-based: 0-{output.hidden_states.shape[0]-1})."
                    )
                batch_embeddings = output.hidden_states[layer_idx]  # [batch_size, padded_seq_len, embed_dim]
            else:
                # Use final embeddings (last layer)
                if output.embeddings is None:
                    raise ValueError("embeddings is None. Cannot extract embeddings.")
                batch_embeddings = output.embeddings  # [batch_size, padded_seq_len, embed_dim]
            
            if batch_embeddings.device != self.device:
                batch_embeddings = batch_embeddings.to(self.device)
            
            # Move to CPU once (single transfer)
            batch_embeddings = batch_embeddings.cpu()
            
            # Extract embeddings (all on CPU now, no more GPU syncs)
            embeddings = [batch_embeddings[i, :actual_lengths_list[i]].contiguous() 
                         for i in range(batch_size)]
            result_dict[layer_idx] = embeddings
        
        # Backward compatibility: if single layer, return list directly
        if len(result_dict) == 1:
            return list(result_dict.values())[0]
        
        return result_dict
