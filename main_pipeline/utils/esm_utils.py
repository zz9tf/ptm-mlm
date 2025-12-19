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
    batch_tokens = batch_tokens[..., 1:-1].to(
        device
    )  # remove <cls> and <eos> from ESM encoding
    out = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
    embedding = out["representations"][33]
    return embedding

