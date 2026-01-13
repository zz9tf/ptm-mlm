"""
ESM-C 6B model inference script for generating embeddings.
This script loads ESM-C 6B model via Forge API and generates embeddings for protein sequences.
This is a shared module used by all downstream tasks.
"""
import torch
from pathlib import Path
from esm.sdk.forge import ESM3ForgeInferenceClient
from esm.sdk.api import ESMProtein, LogitsConfig


class ESMC6BInference:
    """
    ESM-C 6B model inference class for generating embeddings.
    Uses ESM3ForgeInferenceClient for API access.
    Batch converter logic is built-in: automatically handles tokenization, padding, and special tokens.
    """

    def __init__(self, device: str = None, layer_index: int = None):
        """
        Initialize the ESM-C 6B inference model for generating embeddings.

        @param device: Device to run inference on (not used in Forge API, kept for compatibility)
        @param layer_index: Layer index to extract (1-based). If None, uses last layer (default: None)
        """
        # Get token from .env file
        env_path = Path(__file__).parent.parent.parent.parent / '.env'
        token = None
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    if line.startswith('esmc_token='):
                        token = line.strip().split('=', 1)[1]
                        break

        if not token:
            raise RuntimeError("❌ ESM-C token not found in .env file. Please add 'esmc_token=YOUR_TOKEN' to .env file")

        # Initialize ESM3ForgeInferenceClient for 6B model
        print("📦 Connecting to Forge API with ESM-C 6B model...")
        try:
            self.esm_model = ESM3ForgeInferenceClient(
                model="esmc-6b-2024-12",
                url="https://forge.evolutionaryscale.ai",
                token=token
            )
            print("✅ Connected to Forge API successfully!")
        except Exception as e:
            raise RuntimeError(
                f"❌ Failed to connect to Forge API: {e}\n"
                f"   Please check:\n"
                f"   1. Token is valid and not expired\n"
                f"   2. Internet connection is working\n"
                f"   3. Forge service is available\n"
                f"   4. You have access to ESM-C 6B model (requires appropriate account tier)"
            )

        # Set layer index (1-based for ESM-C)
        self.layer_index = layer_index
        if layer_index is not None:
            print(f"🔍 Will extract embeddings from layer {layer_index} (1-based)")
        else:
            print(f"🔍 Will extract embeddings from last layer (default)")

        # ESM-C 6B has 2560 dimensions
        self.hidden_size = 2560

        print(f"✅ ESM-C 6B model ready! Hidden size: {self.hidden_size}")

    @torch.inference_mode()
    def _compute_esmc_embedding(self, sequences: list, layer_indices: list = None):
        """
        Compute ESM-C 6B embeddings for a batch of sequences.
        
        Batch converter logic is built-in: automatically handles tokenization, padding, and special tokens.
        External code doesn't need to handle batch_converter conversion or alignment.
        
        Note: Forge API processes sequences one by one, but this method provides a unified batch interface.
        For multiple layers, each layer requires a separate API call (API limitation).
        
        @param sequences: List of protein sequences (strings)
        @param layer_indices: List of layer indices to extract (1-based). If None, uses self.layer_index or last layer
        @returns: Dict mapping layer_index to list of ESM-C 6B embeddings tensors
                  Each embedding tensor has shape (seq_len, embed_dim)
                  Includes special tokens (BOS/EOS) - batch_converter logic is handled internally
                  If single layer requested, returns list directly for backward compatibility
        """
        if not sequences:
            return [] if layer_indices is None or len(layer_indices) == 1 else {}
        
        # Determine which layers to extract
        if layer_indices is None:
            layer_indices = [self.layer_index] if self.layer_index is not None else [None]
        
        # Check if we need multiple layers
        multiple_layers = len(layer_indices) > 1
        
        if multiple_layers:
            # Multiple layers: collect embeddings for each layer
            # Note: Forge API doesn't support multiple layers in one call, so we need separate calls
            result_dict = {}
            
            for layer_idx in layer_indices:
                batch_embeddings = []
                
                for seq in sequences:
                    try:
                        seq_len = len(seq)  # Original sequence length

                        # Create ESMProtein
                        protein = ESMProtein(sequence=seq)
                        protein_tensor = self.esm_model.encode(protein)

                        # Configure logits output for 6B model
                        logits_config = LogitsConfig(sequence=True, return_embeddings=True)
                        if layer_idx is not None:
                            # Set layer index if specified (1-based)
                            logits_config.ith_hidden_layer = layer_idx
                        # If layer_idx is None, use default (last layer)

                        logits_output = self.esm_model.logits(protein_tensor, logits_config)

                        # Extract embeddings
                        if hasattr(logits_output, 'embeddings'):
                            embeddings = logits_output.embeddings
                            # Convert to torch tensor if needed
                            if not isinstance(embeddings, torch.Tensor):
                                embeddings = torch.tensor(embeddings)
                            embeddings = embeddings.squeeze(0)  # Remove batch dimension if present

                            # ESM-C 6B includes BOS and EOS tokens, keep them to match batch_converter behavior
                            # The embeddings already include special tokens, so we keep them
                            batch_embeddings.append(embeddings)
                        else:
                            raise RuntimeError("Logits output does not have embeddings attribute")

                    except Exception as e:
                        raise RuntimeError(f"ESM-C 6B processing failed for sequence (length={len(seq)}): {e}")
                
                result_dict[layer_idx] = batch_embeddings
            
            return result_dict
        else:
            # Single layer: original behavior
            batch_embeddings = []
            layer_idx = layer_indices[0]
            
            for seq in sequences:
                try:
                    seq_len = len(seq)  # Original sequence length

                    # Create ESMProtein
                    protein = ESMProtein(sequence=seq)
                    protein_tensor = self.esm_model.encode(protein)

                    # Configure logits output for 6B model
                    logits_config = LogitsConfig(sequence=True, return_embeddings=True)
                    if layer_idx is not None:
                        # Set layer index if specified (1-based)
                        logits_config.ith_hidden_layer = layer_idx
                    # If layer_idx is None, use default (last layer)

                    logits_output = self.esm_model.logits(protein_tensor, logits_config)

                    # Extract embeddings
                    if hasattr(logits_output, 'embeddings'):
                        embeddings = logits_output.embeddings
                        # Convert to torch tensor if needed
                        if not isinstance(embeddings, torch.Tensor):
                            embeddings = torch.tensor(embeddings)
                        embeddings = embeddings.squeeze(0)  # Remove batch dimension if present

                        # ESM-C 6B includes BOS and EOS tokens, keep them to match batch_converter behavior
                        # The embeddings already include special tokens, so we keep them
                        batch_embeddings.append(embeddings)
                    else:
                        raise RuntimeError("Logits output does not have embeddings attribute")

                except Exception as e:
                    raise RuntimeError(f"ESM-C 6B processing failed for sequence (length={len(seq)}): {e}")
            
            return batch_embeddings


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings using ESM-C 6B via Forge API")
    parser.add_argument("--sequences", type=str, nargs="+", help="Input sequences")
    parser.add_argument("--output", type=str, help="Output path to save embeddings")
    parser.add_argument("--layer_index", type=int, default=None, help="Layer index to extract (1-based)")
    
    args = parser.parse_args()
    
    inferencer = ESMC6BInference(layer_index=args.layer_index)
    embeddings = inferencer._compute_esmc_embedding(args.sequences)
    
    if args.output:
        torch.save(embeddings, args.output)
        print(f"✅ Embeddings saved to {args.output}")
    else:
        print(f"📊 Generated embeddings for {len(embeddings)} sequences")