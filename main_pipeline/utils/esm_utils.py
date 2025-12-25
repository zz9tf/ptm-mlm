import torch
from accelerate import Accelerator


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

