import torch
import esm
from accelerate import Accelerator
from typing import Optional, Tuple, Any


def make_esm_input_ids(input_ids, tokenizer):
    """
    Replace PTM tokens with mask token for ESM input.
    @param input_ids: Input token IDs tensor.
    @param tokenizer: Tokenizer instance.
    @returns: ESM input IDs with PTM tokens replaced by mask token.
    """
    device = input_ids.device
    is_ptm_mask = tokenizer.is_ptm_token(input_ids).to(device)
    esm_input_ids = input_ids.clone()
    esm_input_ids[is_ptm_mask] = tokenizer.mask_token_id
    return esm_input_ids


@torch.no_grad()
def compute_esm_embedding(tokenizer, esm_model, batch_converter, masked_input_ids, accelerator):
    """
    Compute ESM embeddings for masked input sequences.
    @param tokenizer: Tokenizer instance.
    @param esm_model: ESM model (already prepared by accelerator).
    @param batch_converter: ESM alphabet batch converter.
    @param masked_input_ids: Masked input token IDs.
    @param accelerator: Accelerator instance for device management.
    @returns: ESM embeddings tensor.
    """
    device = accelerator.device
    # ESM model is already on correct device after accelerator.prepare()
    inputs = [
        (i, tokenizer.decode(input_id.detach().cpu().tolist()))
        for i, input_id in enumerate(masked_input_ids)
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(inputs)
    target_seq_len = masked_input_ids.shape[1]
    
    batch_tokens = batch_tokens.to(device)
    out = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
    embedding = out["representations"][33]
    
    # Intelligent length alignment
    embedding_len = embedding.shape[1]
    length_diff = embedding_len - target_seq_len
    
    if length_diff == 0:
        # Perfect match - no adjustment needed
        pass
    elif length_diff == 2:
        # ESM added 2 tokens (<cls> and <eos>) - remove them
        embedding = embedding[:, 1:-1, :]
    elif length_diff > 0:
        # ESM added more tokens than expected - remove from both ends
        # Usually the first and last tokens are special tokens
        remove_from_start = length_diff // 2
        remove_from_end = length_diff - remove_from_start
        if remove_from_end > 0:
            embedding = embedding[:, remove_from_start:-remove_from_end, :]
        else:
            embedding = embedding[:, remove_from_start:, :]
        print(f"Warning: ESM embedding is longer than target sequence length. Removing extra tokens. target_seq_len: {target_seq_len}, embedding_len: {embedding_len}")
    else:
        # Embedding is shorter than target (shouldn't happen, but handle gracefully)
        # This might happen if ESM's tokenization is different
        batch_size = embedding.shape[0]
        embed_dim = embedding.shape[2]
        pad_len = target_seq_len - embedding_len
        padding = torch.zeros(
            batch_size, pad_len, embed_dim,
            device=embedding.device, dtype=embedding.dtype
        )
        embedding = torch.cat([embedding, padding], dim=1)
        print(f"Warning: ESM embedding is shorter than target sequence length. Padding with zeros. target_seq_len: {target_seq_len}, embedding_len: {embedding_len}")
    
    return embedding


def get_esm_embed_dim(model_name: str, repr_layer: Optional[int] = None) -> int:
    """
    Get ESM embedding dimension for a given model name.
    Uses known dimensions for common models (fast, no model loading needed).
    
    @param model_name: Model name ('esm2_650m', 'esm2_15b', 'esm3_7b', or 'esmc_300m')
    @param repr_layer: Optional layer index to extract embeddings from (None = use default, not used for known models)
    @return: Embedding dimension
    """
    # Known embedding dimensions for common models (for quick lookup)
    # These are the standard dimensions for each model
    known_dims = {
        "esm2_650m": 1280,
        "esm2_15b": 5120,
        "esm3_7b": 2560,  # ESM3 7B typically has 2560 dimensions
        "esmc_300m": 640,  # ESM C 300M typically has 640 dimensions
        "esmc_600m": 1152,  # ESM C 600M has 1152 dimensions (not 640!)
    }
    
    # Return known dimension (fast, no model loading needed)
    if model_name in known_dims:
        return known_dims[model_name]
    
    # If not in known list, raise error (should not happen for supported models)
    raise ValueError(f"Unknown model name: {model_name}. Supported: {list(known_dims.keys())}")


def load_esm_model(model_name: str, accelerator: Accelerator, repr_layer_override: Optional[int] = None) -> Tuple[Any, Any, Any, Optional[int], str, int]:
    """
    Load ESM model (ESM2 650M, ESM2 15B, ESM3 7B, ESM C 300M, or ESM C 600M) based on model name.
    
    @param model_name: Model name ('esm2_650m', 'esm2_15b', 'esm3_7b', 'esmc_300m', or 'esmc_600m')
    @param accelerator: Accelerator instance for logging
    @param repr_layer_override: Optional layer index to override default (None = use default)
    @return: Tuple of (esm_model, alphabet, batch_converter, repr_layer, model_type, embed_dim)
    """
    if accelerator.is_local_main_process:
        accelerator.print(f"🔄 Loading {model_name} model...")
    
    if model_name == "esm2_650m":
        # Load ESM2 650M model
        esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        default_repr_layer = 33  # ESM2 650M has 33 layers, use last layer (index 33)
        model_type = "esm2"
    elif model_name == "esm2_15b":
        # Load ESM2 15B model
        esm_model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
        default_repr_layer = 33  # ESM2 15B uses layer 33 for embeddings
        model_type = "esm2"
    elif model_name == "esm3_7b":
        # Load ESM3 7B model
        # Try different possible model identifiers for ESM3 7B
        try:
            # Try esm3-sm-open-v1 (7B model)
            esm_model, alphabet = esm.pretrained.load_model_and_alphabet("esm3-sm-open-v1")
            default_repr_layer = None  # Will use the last layer
            model_type = "esm3"
        except Exception as e1:
            try:
                # Try esm3-medium-2024-08 (alternative identifier)
                esm_model, alphabet = esm.pretrained.load_model_and_alphabet("esm3-medium-2024-08")
                default_repr_layer = None
                model_type = "esm3"
            except Exception as e2:
                if accelerator.is_local_main_process:
                    accelerator.print(f"❌ Failed to load ESM3 7B model")
                    accelerator.print(f"   Error 1 (esm3-sm-open-v1): {e1}")
                    accelerator.print(f"   Error 2 (esm3-medium-2024-08): {e2}")
                raise RuntimeError(f"Could not load ESM3 7B model. Please check if the model is available.")
    elif model_name == "esmc_300m":
        # Load ESM C 300M model
        try:
            from esm.models.esmc import ESMC
            esm_model = ESMC.from_pretrained("esmc_300m")
            # ESM C uses a different API, we'll handle it separately
            alphabet = None  # ESM C doesn't use alphabet in the same way
            default_repr_layer = None  # ESM C uses last layer by default
            model_type = "esmc"
        except Exception as e:
            if accelerator.is_local_main_process:
                accelerator.print(f"❌ Failed to load ESM C 300M model: {e}")
            raise RuntimeError(f"Could not load ESM C 300M model. Please check if the model is available.")
    elif model_name == "esmc_600m":
        # Load ESM C 600M model
        try:
            from esm.models.esmc import ESMC
            esm_model = ESMC.from_pretrained("esmc_600m")
            # ESM C uses a different API, we'll handle it separately
            alphabet = None  # ESM C doesn't use alphabet in the same way
            default_repr_layer = None  # ESM C uses last layer by default
            model_type = "esmc"
        except Exception as e:
            if accelerator.is_local_main_process:
                accelerator.print(f"❌ Failed to load ESM C 600M model: {e}")
            raise RuntimeError(f"Could not load ESM C 600M model. Please check if the model is available.")
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported: 'esm2_650m', 'esm2_15b', 'esm3_7b', 'esmc_300m', 'esmc_600m'")
    
    # Initialize embed_dim as None - will be determined dynamically
    embed_dim = None
    
    # Use override if provided, otherwise use default
    repr_layer = repr_layer_override if repr_layer_override is not None else default_repr_layer
    
    if model_type != "esmc":
        # ESM2/ESM3 models
        esm_model.eval()
        for param in esm_model.parameters():
            param.requires_grad = False
        
        batch_converter = alphabet.get_batch_converter()
    else:
        # ESM C models - different API
        esm_model.eval()
        for param in esm_model.parameters():
            param.requires_grad = False
        batch_converter = None  # ESM C uses different encoding method
    
    # Determine the actual representation layer for ESM3
    if model_type == "esm3" and repr_layer is None:
        # ESM3 typically uses the last layer, get the number of layers
        try:
            # Try to get layer count from model structure
            if hasattr(esm_model, "num_layers"):
                repr_layer = esm_model.num_layers - 1
            elif hasattr(esm_model, "encoder") and hasattr(esm_model.encoder, "num_layers"):
                repr_layer = esm_model.encoder.num_layers - 1
            elif hasattr(esm_model, "layers"):
                repr_layer = len(esm_model.layers) - 1
            else:
                # Default: use -1 to indicate last layer (will be determined at runtime)
                repr_layer = -1
        except Exception:
            # If we can't determine, use -1 to indicate last layer
            repr_layer = -1
    
    # For ESM C, determine layer count if needed
    # Note: ESM C 300M has 30 layers, but we can't easily detect this from model structure
    # We'll use None to indicate default (last layer), and extract from hidden_states when needed
    if model_type == "esmc" and repr_layer is None:
        # ESM C: None means use default (last layer) via embeddings
        # When repr_layer is specified, we'll extract from hidden_states
        repr_layer = None  # Use default (last layer via embeddings)
    
    # Determine embedding dimension dynamically by running a test forward pass
    try:
        if model_type == "esmc":
            # ESM C: Use SDK API to get embedding dimension
            from esm.sdk.api import ESMProtein, LogitsConfig
            dummy_protein = ESMProtein(sequence="M")
            dummy_tensor = esm_model.encode(dummy_protein)
            if hasattr(dummy_tensor, 'error'):
                raise RuntimeError(f"Failed to encode test sequence: {dummy_tensor.error}")
            logits_config = LogitsConfig(sequence=True, return_embeddings=True)
            logits_output = esm_model.logits(dummy_tensor, logits_config)
            embed_dim = logits_output.embeddings.shape[-1]  # Get dimension from last axis
        else:
            # ESM2/ESM3: Use batch converter and forward pass
            dummy_seq = "M"
            if batch_converter is not None:
                _, _, dummy_tokens = batch_converter([(0, dummy_seq)])
                with torch.no_grad():
                    dummy_tokens = dummy_tokens.to(next(esm_model.parameters()).device)
                    # Determine which layer to use
                    test_layer = repr_layer if repr_layer is not None and repr_layer >= 0 else default_repr_layer
                    if test_layer is None:
                        # Try to determine last layer
                        if hasattr(esm_model, "num_layers"):
                            test_layer = esm_model.num_layers - 1
                        elif hasattr(esm_model, "encoder") and hasattr(esm_model.encoder, "num_layers"):
                            test_layer = esm_model.encoder.num_layers - 1
                        else:
                            test_layer = 33  # Default fallback for ESM2
                    
                    # Run forward pass
                    out = esm_model(dummy_tokens, repr_layers=[test_layer])
                    embed_dim = out["representations"][test_layer].shape[-1]
            else:
                raise RuntimeError("Cannot determine embedding dimension: batch_converter is None")
    except Exception as e:
        if accelerator.is_local_main_process:
            accelerator.print(f"❌ Failed to determine embedding dimension dynamically: {e}")
            import traceback
            traceback.print_exc()
        raise RuntimeError(f"Could not determine embedding dimension for {model_name}. Error: {e}")
    
    if accelerator.is_local_main_process:
        accelerator.print(f"✅ Loaded {model_name} model")
        accelerator.print(f"   Embedding dimension: {embed_dim}")
        if repr_layer_override is not None:
            layer_source = "command line" if hasattr(accelerator, '_repr_layer_source') and accelerator._repr_layer_source == 'cli' else "config"
            accelerator.print(f"   Using representation layer: {repr_layer} (from {layer_source})")
        elif repr_layer is not None:
            accelerator.print(f"   Using representation layer: {repr_layer} (default)")
        else:
            accelerator.print(f"   Using representation layer: default (last layer)")
    
    return esm_model, alphabet, batch_converter, repr_layer, model_type, embed_dim

