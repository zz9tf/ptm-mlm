from typing import Optional


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
        "esmc_600m": 1152,  # ESM C 600M has 1152 dimensions
        "esmc_6b": 2560,  # ESM C 6B has 2560 dimensions
    }
    
    # Return known dimension (fast, no model loading needed)
    if model_name in known_dims:
        return known_dims[model_name]
    
    # If not in known list, raise error (should not happen for supported models)
    raise ValueError(f"Unknown model name: {model_name}. Supported: {list(known_dims.keys())}")

