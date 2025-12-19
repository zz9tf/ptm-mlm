import torch.nn.functional as F


def mlm_loss(outputs, input_ids, mask):
    """
    Compute masked language modeling loss.
    @param outputs: Model logits of shape (batch_size, seq_len, vocab_size).
    @param input_ids: Target token IDs of shape (batch_size, seq_len).
    @param mask: Boolean mask of shape (batch_size, seq_len) indicating positions to predict.
    @returns: Scalar loss tensor.
    """
    return F.cross_entropy(
        outputs[mask],
        input_ids[mask],
    )

