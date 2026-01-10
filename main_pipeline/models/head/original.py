"""
Original head: Generates original sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict

class OriginalHead(nn.Module):
    """
    Head for generating original sequence.
    Loss is computed at all non-padding positions.
    """
    
    def __init__(self, d_model: int, vocab_size: int, device=None, dtype=None, **kwargs):
        """
        Initialize original head.
        
        @param d_model: Model dimension for input features
        @param vocab_size: Vocabulary size for output sequences
        @param device: Device to place the model on
        @param dtype: Data type for the model
        @param **kwargs: Additional arguments (unused)
        """
        super().__init__()
        self.type = "original"
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)
    
    def forward(self, features: torch.Tensor, processed: Dict[str, Any], **kwargs) -> torch.Tensor:
        """
        Generate original sequence logits.

        @param features: Processed features of shape (batch_size, seq_len, d_model)
        @param processed: Dictionary of processed features from previous heads
        @param **kwargs: Additional arguments (unused)
        @returns: Logits of shape (batch_size, seq_len, vocab_size)
        """

        def _check(name, x):
            if x is None:
                raise RuntimeError(f"{name} is None")
            if not torch.isfinite(x).all():
                bad = (~torch.isfinite(x)).nonzero(as_tuple=False)[:5]
                raise RuntimeError(f"{name} has NaN/Inf, examples idx={bad.tolist()}")
            m = x.detach().abs().max().item()
            if m > 1e4:
                print(f"[warn] {name} abs_max={m:.2e} (may overflow in fp16/bf16)")

        _check("original_input_features", features)
        logits = self.head(features)
        _check("original_logits", logits)

        return {
            "logits": logits
        }
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        device: torch.device,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute loss at all non-padding positions.
        
        @param logits: Head output logits of shape (batch_size, seq_len, vocab_size)
        @param input_ids: Target token IDs of shape (batch_size, seq_len)
        @param device: Device for tensor operations
        @param **kwargs: Additional arguments (unused)
        @returns: Scalar loss tensor
        """
        non_padding_mask = input_ids != 0
        if non_padding_mask.any():
            return F.cross_entropy(
                logits[non_padding_mask],
                input_ids[non_padding_mask],
            )
        else:
            return torch.tensor(0.0, device=device)

